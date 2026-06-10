"""Use case #2 (NLP triage) — two-stage pipeline.

Stage 1 (deterministic): spaCy NER + regex/rule extraction. Always runs, no LLM,
no AWS. Extracts ATA chapter, FIN/PN/AMM refs, severity, component, symptom,
action, and a criticality heuristic.

Stage 2 (LLM): Bedrock (Titan via the Converse API) enriches/overrides the
Stage-1 fields with a strict-JSON classification. Optional — only fires when
BEDROCK_MODEL_ID is set and a bedrock-runtime client is reachable. Degrades
silently to Stage-1 results otherwise; never raises.

Recall-first if treated as sensitive: bias toward over-flagging identifiers.
"""
import json
import re
from dataclasses import dataclass, field

try:  # works both as a package module and as a standalone script
    from .common import BEDROCK_MODEL_ID, bedrock_runtime
except ImportError:  # pragma: no cover - executed only when run as a script
    from common import BEDROCK_MODEL_ID, bedrock_runtime

_NLP = None


def nlp():
    """Lazy-load the French spaCy model (fr_core_news_lg)."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("fr_core_news_lg")
    return _NLP


@dataclass
class TriagedReport:
    raw: str
    ata_chapter: str = ""
    component: str = ""
    symptom: str = ""
    action: str = ""
    criticality: str = ""           # e.g. low / medium / high
    duplicate_of: int | None = None
    entities: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# ATA knowledge base — consistent with the synthetic corpus (chapters present:
# 21, 24, 27, 29, 32, 34, 49, 52). Used both to label the system name from an
# extracted chapter and to *infer* a chapter from keywords when no ATA/AMM ref
# is present in the narrative.
# --------------------------------------------------------------------------- #
ATA_KB: dict[str, dict] = {
    "21": {
        "system": "Air Conditioning & Pressurization",
        "components": ["pack", "pack valve", "trim air valve", "cabin fan", "outflow valve"],
        "symptoms": ["cabin temp drift", "pack fault", "pressurization", "depressurization",
                     "cabin pressure", "overtemp"],
    },
    "24": {
        "system": "Electrical Power",
        "components": ["IDG", "GCU", "bus tie contactor", "battery", "TR unit", "generator"],
        "symptoms": ["battery low charge", "bus tie contactor fail", "gen fault", "bus fault",
                     "wiring"],
    },
    "27": {
        "system": "Flight Controls",
        "components": ["aileron actuator", "spoiler", "elevator", "rudder", "flap", "slat",
                       "PCU", "FCU"],
        "symptoms": ["surface jam", "flap asymmetry", "spoiler fault", "actuator leak",
                     "control restriction"],
    },
    "29": {
        "system": "Hydraulic Power",
        "components": ["yellow electric pump", "green pump", "blue pump", "PTU", "reservoir",
                       "accumulator", "EDP"],
        "symptoms": ["reservoir low level", "low pressure", "overheat", "pump fault",
                     "hydraulic leak"],
    },
    "32": {
        "system": "Landing Gear",
        "components": ["MLG shock absorber", "NLG", "WoW sensor", "tyre", "brake", "wheel",
                       "actuator"],
        "symptoms": ["WoW disagree", "gear unsafe indication", "brake temp high", "tyre wear",
                     "gear retract fault"],
    },
    "34": {
        "system": "Navigation",
        "components": ["ADIRU", "pitot probe", "GPS antenna", "radio altimeter", "ILS receiver",
                       "transponder"],
        "symptoms": ["nav data drift", "ADR fault", "radio alt fault", "position discrepancy",
                     "ils fault"],
    },
    "49": {
        "system": "APU",
        "components": ["APU generator", "ECB", "starter", "fuel control unit", "APU"],
        "symptoms": ["APU gen fault", "slow start", "no start", "auto shutdown", "egt high"],
    },
    "52": {
        "system": "Doors",
        "components": ["cargo door actuator", "door warning sensor", "latch mechanism",
                       "slide", "proximity sensor"],
        "symptoms": ["slide pressure low", "door not closed indication", "door warning",
                     "latch fault"],
    },
}

# Chapters whose nature is intrinsically safety-critical (recall-first bias).
_HIGH_CRIT_CHAPTERS = {"21", "27", "32"}

# Keyword -> criticality escalators. recall-first: if any appears, flag HIGH.
_HIGH_CRIT_KEYWORDS = (
    "aog", "overheat", "unsafe", "depressuriz", "depressur", "fire", "smoke",
    "jam", "asymmetry", "uncommanded", "loss of", "dual fault", "rejected takeoff",
)
_MEDIUM_CRIT_KEYWORDS = (
    "major", "fault", "leak", "fail", "warning", "ecam", "repetitive",
)

# --------------------------------------------------------------------------- #
# Regex extractors
# --------------------------------------------------------------------------- #
# AMM / ATA reference like 24-31-15 or "ATA 24" -> capture chapter "24".
_RE_AMM = re.compile(r"\b(\d{2})-\d{2}-\d{2,3}\b")
_RE_ATA = re.compile(r"\bATA\s*[:\-]?\s*(\d{2})\b", re.IGNORECASE)
_RE_FIN = re.compile(r"\bFIN\s*[:\-]?\s*([0-9]{1,3}[A-Z]{1,2}[0-9]{1,2})\b", re.IGNORECASE)
_RE_PN = re.compile(r"\b(\d{3}-\d{3,5}(?:-\d+)?)\b")            # P/N like 980-1234
_RE_SEVERITY = re.compile(r"\b(AOG|MAJOR|MINOR|INFO)\b", re.IGNORECASE)


def _extract_ata(text: str) -> str:
    """Pull the ATA chapter from an explicit 'ATA NN' or an AMM 'NN-NN-NN' ref.

    Prefers an explicit ATA mention, then an AMM ref whose chapter is one we
    know about, then any AMM ref. Returns "" if none found.
    """
    m = _RE_ATA.search(text)
    if m:
        return m.group(1)
    known = [c for c in _RE_AMM.findall(text) if c in ATA_KB]
    if known:
        return known[0]
    any_amm = _RE_AMM.findall(text)
    return any_amm[0] if any_amm else ""


def _match_kb(text_low: str, key: str) -> str:
    """Return the first KB phrase of `key` (component|symptom) found in text."""
    for chap in ATA_KB.values():
        for phrase in chap[key]:
            if phrase.lower() in text_low:
                return phrase
    return ""


def _infer_ata_from_keywords(text_low: str) -> str:
    """Guess the ATA chapter from component/symptom keywords in the text."""
    best_chap, best_hits = "", 0
    for chap, kb in ATA_KB.items():
        hits = sum(1 for p in kb["components"] + kb["symptoms"] if p.lower() in text_low)
        if hits > best_hits:
            best_chap, best_hits = chap, hits
    return best_chap


def _criticality(text_low: str, ata: str, severity: str) -> str:
    """Heuristic criticality. Recall-first: bias toward over-flagging."""
    sev = (severity or "").upper()
    if sev == "AOG":
        return "high"
    if any(kw in text_low for kw in _HIGH_CRIT_KEYWORDS):
        return "high"
    if ata in _HIGH_CRIT_CHAPTERS and any(kw in text_low for kw in _MEDIUM_CRIT_KEYWORDS):
        return "high"
    if sev == "MAJOR":
        return "high"
    if any(kw in text_low for kw in _MEDIUM_CRIT_KEYWORDS):
        return "medium"
    if sev == "MINOR":
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# Stage 1 — deterministic
# --------------------------------------------------------------------------- #
def _stage1(report: str) -> TriagedReport:
    text = report or ""
    text_low = text.lower()

    # NER
    entities: list[dict] = []
    try:
        doc = nlp()(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    except Exception:  # noqa: BLE001 — spaCy must never crash the pipeline
        entities = []

    # Identifiers (recall-first: collected as entities too, so nothing is missed)
    for label, rx in (("FIN", _RE_FIN), ("PN", _RE_PN), ("AMM", _RE_AMM)):
        for m in rx.finditer(text):
            entities.append({"text": m.group(0), "label": label})

    ata = _extract_ata(text) or _infer_ata_from_keywords(text_low)
    component = _match_kb(text_low, "components")
    symptom = _match_kb(text_low, "symptoms")

    sev_m = _RE_SEVERITY.search(text)
    severity = sev_m.group(1).upper() if sev_m else ""

    # naive action extraction: clause after the symptom/checked verbs
    action = ""
    m_act = re.search(
        r"\b(reset|replaced?|deactivated?|bled|lubricated|checked|borescope|troubleshoot\w*|"
        r"cleaned|adjusted|inspected|swapped)\b[^.;\[]*",
        text_low,
    )
    if m_act:
        action = m_act.group(0).strip()

    criticality = _criticality(text_low, ata, severity)

    return TriagedReport(
        raw=report,
        ata_chapter=ata,
        component=component,
        symptom=symptom,
        action=action,
        criticality=criticality,
        entities=entities,
    )


# --------------------------------------------------------------------------- #
# Stage 2 — optional LLM enrichment via Bedrock Converse API
# --------------------------------------------------------------------------- #
_STAGE2_SYSTEM = (
    "You are an aircraft maintenance triage assistant. Given a noisy free-text "
    "maintenance log (mix of French/English, ATA codes, acronyms, FIN/PN/AMM "
    "references), extract structured fields. Known ATA chapters and systems: "
    + ", ".join(f"{c}={kb['system']}" for c, kb in ATA_KB.items())
    + ". Respond with a STRICT JSON object and nothing else, with exactly these "
    "keys: ata_chapter (2-digit string), component, symptom, action, "
    "criticality (one of low/medium/high). If unsure, infer the most likely "
    "value. Bias criticality HIGH when safety-relevant (AOG, overheat, unsafe, "
    "depressurization, jam, asymmetry)."
)


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of the first JSON object in `text`."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _stage2(report: str, base: TriagedReport) -> TriagedReport:
    """Enrich `base` with a Bedrock Converse classification. Never raises."""
    if not BEDROCK_MODEL_ID:
        return base
    try:
        client = bedrock_runtime()
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": _STAGE2_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": report}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.0},
        )
        out = resp["output"]["message"]["content"][0]["text"]
        data = _parse_json_object(out)
    except Exception:  # noqa: BLE001 — no creds / unset model / API error -> fall back
        return base

    if not data:
        return base

    return TriagedReport(
        raw=base.raw,
        ata_chapter=str(data.get("ata_chapter") or base.ata_chapter),
        component=str(data.get("component") or base.component),
        symptom=str(data.get("symptom") or base.symptom),
        action=str(data.get("action") or base.action),
        criticality=str(data.get("criticality") or base.criticality),
        duplicate_of=base.duplicate_of,
        entities=base.entities,
    )


def triage(report: str) -> TriagedReport:
    """Run the two-stage pipeline on one free-text entry.

    Stage 1 (deterministic) always runs; Stage 2 (Bedrock LLM) enriches the
    result when available and degrades silently otherwise.
    """
    base = _stage1(report)
    return _stage2(report, base)


# --------------------------------------------------------------------------- #
# Duplicate detection (deterministic, set-level)
# --------------------------------------------------------------------------- #
_RE_NONWORD = re.compile(r"[^a-z0-9 ]+")
_RE_WS = re.compile(r"\s+")
# Bracketed tags / identifiers stripped before fingerprinting so that two reports
# that differ only by reg/FIN/MEL noise still collide.
_RE_BRACKET = re.compile(r"\[[^\]]*\]")
_RE_REG = re.compile(r"\bF-[A-Z]{4}\b")


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = _RE_BRACKET.sub(" ", t)
    t = _RE_REG.sub(" ", t)
    t = _RE_FIN.sub(" ", t)
    t = _RE_PN.sub(" ", t)
    t = _RE_NONWORD.sub(" ", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def _shingles(norm: str, n: int = 3) -> set:
    toks = norm.split()
    if len(toks) < n:
        return {norm} if norm else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def triage_file(path: str = "input/maintenance_logs.jsonl", limit: int | None = None
                ) -> list[TriagedReport]:
    """Triage every narrative in the jsonl and detect duplicates across the set.

    Sets `duplicate_of` to the index of the earliest matching earlier report.
    """
    reports: list[TriagedReport] = []
    narratives: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            narratives.append(row.get("narrative", ""))

    for narrative in narratives:
        reports.append(triage(narrative))

    # Duplicate detection: Jaccard over 3-shingles of normalized text.
    fingerprints = [_shingles(_normalize(n)) for n in narratives]
    threshold = 0.6
    for i in range(len(reports)):
        for j in range(i):
            if reports[j].duplicate_of is not None:
                continue  # link to the canonical (earliest) report
            if _jaccard(fingerprints[i], fingerprints[j]) >= threshold:
                reports[i].duplicate_of = j
                break

    return reports


# --------------------------------------------------------------------------- #
# Evaluation — Stage-1 ATA accuracy vs ground truth (zero LLM cost)
# --------------------------------------------------------------------------- #
def evaluate(path: str = "input/maintenance_logs.jsonl") -> dict:
    """Compare Stage-1 (deterministic) ATA extraction to ground truth.

    Returns {n, ata_accuracy, n_duplicates}. Stage 1 only — no Bedrock calls,
    so this runs anywhere and quantifies the no-LLM-cost classification quality.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    correct = 0
    for row in rows:
        pred = _stage1(row.get("narrative", "")).ata_chapter
        if pred == str(row.get("ata_chapter", "")):
            correct += 1

    n = len(rows)
    triaged = triage_file(path)  # uses Stage-2 if available, but dup count is deterministic
    n_dupes = sum(1 for r in triaged if r.duplicate_of is not None)

    return {
        "n": n,
        "ata_accuracy": round(correct / n, 4) if n else 0.0,
        "n_duplicates": n_dupes,
    }


# --------------------------------------------------------------------------- #
# Demo entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PATH = "input/maintenance_logs.jsonl"
    print("=== NLP triage demo (first 10 records) ===\n")
    for idx, rep in enumerate(triage_file(PATH, limit=10)):
        dup = f"  [DUPLICATE of #{rep.duplicate_of}]" if rep.duplicate_of is not None else ""
        print(f"#{idx}{dup}")
        print(f"  ATA        : {rep.ata_chapter}")
        print(f"  component  : {rep.component}")
        print(f"  symptom    : {rep.symptom}")
        print(f"  action     : {rep.action}")
        print(f"  criticality: {rep.criticality}")
        ents = ", ".join(f"{e['text']}({e['label']})" for e in rep.entities[:6])
        print(f"  entities   : {ents}")
        print(f"  narrative  : {rep.raw[:90]}")
        print()

    print("=== evaluate() — Stage-1, zero LLM cost ===")
    print(json.dumps(evaluate(PATH), indent=2))

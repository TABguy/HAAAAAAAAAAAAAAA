"""Synthetic Airbus maintenance data generator (FIRST DELIVERABLE).

Produces realistic, noisy aircraft-maintenance content so the Docling pipeline and
Streamlit demo have something to parse from minute one — no real Airbus data needed.

Outputs (all into input/):
  - maintenance_logs.csv        structured records
  - maintenance_logs.jsonl      same, one JSON object per line
  - workorder_*.pdf             free-text work orders (for Docling PDF parsing)
  - procedure_ATA*.pdf          mini technical procedures (for RAG / retrieval demos)

PDF generation uses reportlab if available; otherwise falls back to .html
(Docling parses HTML too). Deterministic via --seed for reproducible demos.

Usage:
    python src/gen_data.py --n 80 --seed 42 --docs 4
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path("input")

# --- Domain knowledge: ATA chapters -> system, components, symptoms, actions ---
ATA = {
    "21": {
        "system": "Air Conditioning & Pressurization",
        "components": ["pack flow control valve", "cabin pressure controller", "trim air valve", "recirculation fan"],
        "symptoms": ["cabin temp drift", "pack overheat warning", "slow cabin depressurization", "fan vibration"],
        "actions": ["replaced flow control valve", "reset controller via BITE", "cleaned air filter", "no fault found on ground"],
    },
    "24": {
        "system": "Electrical Power",
        "components": ["IDG", "TR unit", "battery", "GCU"],
        "symptoms": ["IDG low oil pressure", "intermittent ELEC fault on ECAM", "battery low charge", "bus tie contactor fail"],
        "actions": ["replaced battery P/N 980-xxxx", "reset GCU", "checked wiring per AMM", "NFF, BITE clear"],
    },
    "27": {
        "system": "Flight Controls",
        "components": ["spoiler servo", "rudder travel limiter", "slat actuator", "ELAC"],
        "symptoms": ["spoiler 3 fault", "rudder limiter warning", "slat asymmetry", "F/CTL ELAC 1 fault"],
        "actions": ["swapped ELAC 1/2 for trouble-shooting", "lubricated actuator", "replaced servo", "rigging check OK"],
    },
    "29": {
        "system": "Hydraulic Power",
        "components": ["green system EDP", "yellow electric pump", "reservoir", "PTU"],
        "symptoms": ["green sys low pressure", "PTU noisy", "reservoir low level", "yellow pump overheat"],
        "actions": ["topped up reservoir", "deactivated PTU per MEL", "replaced EDP", "leak check performed"],
    },
    "32": {
        "system": "Landing Gear",
        "components": ["MLG shock absorber", "brake assembly", "tyre", "WoW sensor"],
        "symptoms": ["brake temp high", "tyre worn beyond limits", "gear unsafe indication", "WoW disagree"],
        "actions": ["replaced tyre", "bled brakes", "replaced WoW sensor", "serviced shock absorber nitrogen"],
    },
    "34": {
        "system": "Navigation",
        "components": ["ADIRU", "radio altimeter", "GPS receiver", "pitot probe"],
        "symptoms": ["IR 1 align fault", "RA disagree below 200ft", "GPS primary lost", "ADR fault"],
        "actions": ["replaced ADIRU", "cleaned pitot drain", "reset via BITE", "NFF after ground run"],
    },
    "49": {
        "system": "APU",
        "components": ["APU fuel control unit", "APU generator", "EGT thermocouple", "starter"],
        "symptoms": ["APU auto shutdown", "EGT overtemp", "slow start", "APU gen fault"],
        "actions": ["replaced thermocouple", "borescope inspection performed", "reset ECB", "no fault found"],
    },
    "52": {
        "system": "Doors",
        "components": ["cargo door actuator", "door warning sensor", "escape slide", "latch"],
        "symptoms": ["fwd cargo door not closed indication", "door warning on ECAM", "slide pressure low"],
        "actions": ["adjusted proximity sensor", "lubricated latch mechanism", "replaced micro-switch", "rigging adjusted"],
    },
}

AIRCRAFT = ["A320-214", "A321neo", "A330-300", "A350-941", "A319-112"]
SEVERITY = ["INFO", "MINOR", "MAJOR", "AOG"]
STATIONS = ["CDG", "TLS", "MUC", "LHR", "FRA", "MAD", "AMS"]
PHASES = ["pre-flight", "transit check", "A-check", "line maintenance", "post-flight"]
NOISE_TAGS = ["[crew report]", "[BITE]", "[deferred per MEL]", "[repetitive defect]", ""]


def _fin() -> str:
    return f"{random.randint(1,99)}{random.choice('VWXYZ')}{random.randint(1,9)}"


def make_record(rng: random.Random, base_date: datetime) -> dict:
    chapter = rng.choice(list(ATA))
    info = ATA[chapter]
    comp = rng.choice(info["components"])
    symptom = rng.choice(info["symptoms"])
    action = rng.choice(info["actions"])
    sev = rng.choices(SEVERITY, weights=[3, 5, 3, 1])[0]
    ac = rng.choice(AIRCRAFT)
    date = base_date + timedelta(days=rng.randint(0, 120), hours=rng.randint(0, 23))
    noise = rng.choice(NOISE_TAGS)

    # Noisy free-text narrative, the way a mechanic/pilot would actually write it
    templates = [
        f"{noise} {ac} reg F-{rng.choice('GHIK')}{''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ',k=3))}: "
        f"crew reported {symptom}. ATA {chapter} ({info['system']}). FIN {_fin()}. {action}.",
        f"ECAM msg related to {comp}. {symptom} during {rng.choice(PHASES)} at {rng.choice(STATIONS)}. "
        f"Troubleshooting per AMM {chapter}-{rng.randint(10,99)}-{rng.randint(10,99)}. {action}. {noise}",
        f"{symptom} on {comp}. checked, {action}. ops ck good. {noise}",
    ]
    narrative = rng.choice(templates).strip().replace("  ", " ")

    return {
        "report_id": f"MR-{date:%Y%m%d}-{rng.randint(1000,9999)}",
        "date": date.strftime("%Y-%m-%d %H:%M"),
        "aircraft": ac,
        "station": rng.choice(STATIONS),
        "ata_chapter": chapter,
        "system": info["system"],
        "component": comp,
        "symptom": symptom,
        "action": action,
        "severity": sev,
        "narrative": narrative,
    }


def write_structured(records: list[dict]) -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with (INPUT_DIR / "maintenance_logs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    with (INPUT_DIR / "maintenance_logs.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _try_pdf(path: Path, title: str, blocks: list[tuple[str, str]]) -> bool:
    """Render a simple document as PDF via reportlab. Returns False if unavailable."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        return False
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    flow = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for heading, body in blocks:
        if heading:
            flow.append(Paragraph(heading, styles["Heading2"]))
        flow.append(Paragraph(body, styles["BodyText"]))
        flow.append(Spacer(1, 8))
    doc.build(flow)
    return True


def _write_html(path: Path, title: str, blocks: list[tuple[str, str]]) -> None:
    parts = [f"<h1>{title}</h1>"]
    for heading, body in blocks:
        if heading:
            parts.append(f"<h2>{heading}</h2>")
        parts.append(f"<p>{body}</p>")
    path.with_suffix(".html").write_text("\n".join(parts), encoding="utf-8")


def emit_doc(stem: str, title: str, blocks: list[tuple[str, str]]) -> None:
    pdf_path = INPUT_DIR / f"{stem}.pdf"
    if not _try_pdf(pdf_path, title, blocks):
        _write_html(INPUT_DIR / stem, title, blocks)


def make_workorder(rng: random.Random, records: list[dict]) -> None:
    r = rng.choice(records)
    blocks = [
        ("Aircraft", f"{r['aircraft']} — station {r['station']} — {r['date']}"),
        ("Reported defect", f"ATA {r['ata_chapter']} {r['system']}: {r['symptom']} on {r['component']}."),
        ("Narrative", r["narrative"]),
        ("Corrective action", f"{r['action']}. Operational check satisfactory."),
        ("Severity", r["severity"]),
        ("Sign-off", f"Certifying staff stamp / Auth No. {rng.randint(100000,999999)}"),
    ]
    emit_doc(f"workorder_{r['report_id']}", f"Work Order {r['report_id']}", blocks)


def make_procedure(rng: random.Random) -> None:
    chapter = rng.choice(list(ATA))
    info = ATA[chapter]
    comp = rng.choice(info["components"])
    blocks = [
        ("Scope", f"This procedure covers the inspection and replacement of the {comp} "
                  f"on the {rng.choice(AIRCRAFT)} ({info['system']}, ATA {chapter})."),
        ("Warning", "Ensure aircraft is electrically and hydraulically safe before starting. "
                    "Refer to applicable safety precautions in the AMM."),
        ("Procedure", f"1. Access the {comp}. 2. Disconnect electrical/hydraulic connections. "
                      f"3. Remove attaching hardware. 4. Install serviceable unit (note P/N and S/N). "
                      f"5. Reconnect and torque to specification. 6. Perform operational test via BITE."),
        ("Close-up", "Restore access panels, remove warning notices, complete the work order and update the technical log."),
    ]
    emit_doc(f"procedure_ATA{chapter}", f"Maintenance Procedure — ATA {chapter} {info['system']}", blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="number of log records")
    ap.add_argument("--docs", type=int, default=4, help="number of PDF work orders")
    ap.add_argument("--procedures", type=int, default=3, help="number of procedure docs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = datetime(2026, 1, 1)
    records = [make_record(rng, base) for _ in range(args.n)]
    write_structured(records)
    for _ in range(args.docs):
        make_workorder(rng, records)
    for _ in range(args.procedures):
        make_procedure(rng)

    pdfs = list(INPUT_DIR.glob("*.pdf"))
    htmls = list(INPUT_DIR.glob("*.html"))
    print(f"Generated {len(records)} log records -> input/maintenance_logs.{{csv,jsonl}}")
    print(f"Generated {len(pdfs)} PDF and {len(htmls)} HTML documents in input/")


if __name__ == "__main__":
    main()

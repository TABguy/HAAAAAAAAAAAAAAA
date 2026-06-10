"""Generate the HAKS 2026 demo-day runbook as a PDF.

A single printable cheat-sheet: setup, the three use cases (how to run + how to
adapt to the real brief), the decision guide, a demo script, business-value
numbers, and troubleshooting. Regenerate any time:

    python src/make_runbook.py            # -> docs/HAKS2026_runbook.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

OUT = Path("docs/HAKS2026_runbook.pdf")
NAVY = colors.HexColor("#0b2545")
BLUE = colors.HexColor("#13558c")
LIGHT = colors.HexColor("#eef3f8")
GREEN = colors.HexColor("#1b7a3d")
AMBER = colors.HexColor("#b5651d")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Cover", parent=s["Title"], fontSize=26, textColor=NAVY, leading=30))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=12, textColor=BLUE,
                          alignment=TA_CENTER, leading=16))
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=15, textColor=colors.white,
                          backColor=NAVY, leading=20, spaceBefore=14, spaceAfter=8,
                          leftIndent=6, borderPadding=(4, 4, 4, 4)))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=12, textColor=BLUE,
                          spaceBefore=10, spaceAfter=4))
    s.add(ParagraphStyle("Body", parent=s["BodyText"], fontSize=9.5, leading=13, spaceAfter=4))
    s.add(ParagraphStyle("Bul", parent=s["BodyText"], fontSize=9.5, leading=13, spaceAfter=2))
    s.add(ParagraphStyle("Mono", parent=s["Code"], fontSize=8.5, leading=11,
                          backColor=LIGHT, textColor=NAVY, leftIndent=6, spaceBefore=2,
                          spaceAfter=6, borderPadding=(5, 5, 5, 5)))
    s.add(ParagraphStyle("Small", parent=s["Normal"], fontSize=8, textColor=colors.grey))
    return s


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    S = _styles()
    F = []  # flowables

    def h1(t): F.append(Paragraph(t, S["H1"]))
    def h2(t): F.append(Paragraph(t, S["H2"]))
    def p(t): F.append(Paragraph(t, S["Body"]))
    def code(t): F.append(Paragraph(t.replace("\n", "<br/>"), S["Mono"]))
    def gap(h=6): F.append(Spacer(1, h))
    def bullets(items):
        F.append(ListFlowable(
            [ListItem(Paragraph(x, S["Bul"]), leftIndent=10) for x in items],
            bulletType="bullet", start="•", leftIndent=12))

    cell = ParagraphStyle("Cell", parent=S["Normal"], fontSize=8.5, leading=11)

    def table(rows, widths, header=True):
        wrapped = []
        for i, row in enumerate(rows):
            if header and i == 0:
                wrapped.append(row)  # header stays plain (styled white-on-blue)
            else:
                wrapped.append([c if not isinstance(c, str) else Paragraph(c, cell) for c in row])
        t = Table(wrapped, colWidths=widths)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), BLUE),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        F.append(t)

    # ---------------------------------------------------------------- COVER
    gap(60)
    F.append(Paragraph("HAKS 2026 — Demo-Day Runbook", S["Cover"]))
    gap(6)
    F.append(Paragraph("IBM Bob × Airbus × AWS · Aircraft-maintenance AI", S["Sub"]))
    gap(10)
    F.append(Paragraph("Project <b>JacquesLePluBo</b> — one-day hackathon · 5h build (14:00–19:00) · "
                       "pitch 19:00", S["Sub"]))
    gap(20)
    F.append(HRFlowable(width="60%", thickness=1.2, color=BLUE, hAlign="CENTER"))
    gap(16)
    p("<b>What this is:</b> three Airbus maintenance use cases, already wired end-to-end against "
      "realistic synthetic data, so whichever brief Airbus reveals at 13:00 you start from a "
      "running demo — not a blank repo. Two of the three run with <b>zero AWS dependency</b>; "
      "the third (RAG) has an offline fallback so it demos even if Bedrock isn't configured.")
    gap(6)
    p("<b>Judging criteria (optimise in order):</b> quantified business value · a live demo that "
      "actually runs · scalability &amp; data sovereignty.")
    gap(10)
    table([
        ["Use case", "Runs offline?", "Headline metric (synthetic)"],
        ["#1 RAG doc assistant", "Yes (TF-IDF); Bedrock optional", "Cited answers, source traceability"],
        ["#2 NLP maintenance triage", "Yes (spaCy + rules)", "100% ATA classification, dedup, priority queue"],
        ["#3 Predictive maintenance", "Yes (scikit-learn)", "Recall 0.55–0.92 vs labelled anomalies; RUL est."],
    ], [5.2 * cm, 5 * cm, 6 * cm])
    F.append(PageBreak())

    # ---------------------------------------------------------------- QUICKSTART
    h1("1 · Setup — get running in ~10 minutes")
    p("Each teammate, on their own laptop (Linux/macOS). The <b>venv is not in git</b> — create it.")
    code("git clone git@github.com:TABguy/JacquesLePluBo.git\n"
         "cd JacquesLePluBo\n"
         "python3 -m venv venv &amp;&amp; source venv/bin/activate\n"
         "pip install --upgrade pip &amp;&amp; pip install -r requirements.txt\n"
         "python -m spacy download fr_core_news_lg      # French NER model (~560 MB)\n"
         "cp .env.example .env                          # fill in Bedrock IDs (only #1 needs them)")
    h2("Generate data + launch (all offline)")
    code("python src/gen_data.py        # maintenance logs + work-order/procedure PDFs -> input/\n"
         "python src/gen_sensors.py     # labelled sensor time series -> input/sensors/\n"
         "python src/ingest.py --offline  # build the TF-IDF index for RAG (no AWS)\n"
         "streamlit run app.py          # the 3-tab demo dashboard")
    bullets([
        "<b>Verify AWS (only for #1's Bedrock path):</b> <font face='Courier'>python src/common.py</font> — "
        "if it prints a model count, Bedrock is reachable in eu-west-1.",
        "Data is <b>deterministic</b> (seeded) — every laptop produces identical data.",
        "<b>Never commit</b> <font face='Courier'>.env</font> or AWS keys.",
    ])
    gap(4)
    h2("Repo map")
    table([
        ["File", "Role"],
        ["src/gen_data.py", "Synthetic maintenance logs (CSV/JSONL) + work-order/procedure PDFs"],
        ["src/gen_sensors.py", "Synthetic sensor series with injected, labelled anomalies (#3)"],
        ["src/common.py", "Config (.env) + Bedrock client + connectivity check"],
        ["src/ingest.py", "#1 build index: TF-IDF (offline) or Bedrock→Chroma (--offline flag)"],
        ["src/rag.py", "#1 retrieval + cited answer; auto-picks Bedrock or offline backend"],
        ["src/nlp_triage.py", "#2 spaCy NER + rules + optional Bedrock; dedup, priority, summarise"],
        ["src/anomaly.py", "#3 IsolationForest + fleet health + trend/RUL + metrics"],
        ["app.py", "Streamlit UI — one tab per use case"],
    ], [4.2 * cm, 12 * cm])
    F.append(PageBreak())

    # ---------------------------------------------------------------- THE THREE
    h1("2 · The three use cases (what to run, how to adapt)")

    # --- #1
    h2("#1 · RAG assistant over technical documentation  —  recommended priority")
    p("<b>Story:</b> a mechanic asks in natural language and gets the right procedure with its "
      "source reference / ATA chapter. Immobilisation time = money; traceability sells to the jury.")
    p("<b>Run:</b> <font face='Courier'>python src/ingest.py --offline</font> then ask in the RAG tab "
      "(works with no AWS). With <font face='Courier'>BEDROCK_MODEL_ID</font>+"
      "<font face='Courier'>EMBED_MODEL_ID</font> set, <font face='Courier'>python src/ingest.py</font> "
      "builds a Bedrock/Chroma index and answers become LLM-generated.")
    p("<b>Adapt to the real brief:</b> drop the real (or better synthetic) PDFs into "
      "<font face='Courier'>input/</font> and re-run ingest — the loader already reads any PDF. "
      "Tune chunk size / top-k in <font face='Courier'>ingest.py</font> / "
      "<font face='Courier'>rag.py</font>. For the IBM angle, the optional "
      "<font face='Courier'>src/ingest_docling.py</font> swaps in Docling layout-aware parsing "
      "(heavy — see §5).")

    # --- #2
    h2("#2 · NLP triage of maintenance reports / logbooks")
    p("<b>Story:</b> noisy free-text PIREP/logbook entries → auto-classified by ATA chapter, "
      "entities extracted, duplicates merged, prioritised by criticality. Two-stage "
      "(deterministic rules first, LLM enrichment second) signals reliability, not just 'all-LLM'.")
    p("<b>Run (offline):</b> Triage tab → 'Triage full corpus'. Stage 1 hits <b>100% ATA accuracy</b> "
      "on the synthetic set at zero LLM cost; Bedrock Stage 2 is optional enrichment.")
    p("<b>Adapt:</b> edit the <font face='Courier'>ATA_KB</font> dict and priority weights in "
      "<font face='Courier'>nlp_triage.py</font> to match the brief's taxonomy; point "
      "<font face='Courier'>triage_file()</font> at the real input. Outputs a prioritised "
      "<font face='Courier'>input/triaged.csv</font>.")

    # --- #3
    h2("#3 · Predictive maintenance / anomaly detection")
    p("<b>Story:</b> sensor series → anomalies flagged, component health scored, rough "
      "remaining-useful-life (RUL) estimated, fleet ranked worst-first. High storytelling "
      "(safety + cost) but <b>data-risky</b> — which is why the generator injects labelled "
      "anomalies so the demo shows real precision/recall.")
    p("<b>Run (offline):</b> <font face='Courier'>python src/gen_sensors.py</font> then the "
      "Predictive tab — fleet table + health bars + per-sensor anomaly plot.")
    p("<b>Adapt:</b> add sensors / change baselines, degradation and failure thresholds in "
      "<font face='Courier'>gen_sensors.py</font>; swap IsolationForest or tune "
      "<font face='Courier'>contamination</font> in <font face='Courier'>anomaly.py</font>. If "
      "the brief brings real sensor data, point <font face='Courier'>fleet_summary()</font> at it.")
    F.append(PageBreak())

    # ---------------------------------------------------------------- DECISION + DEMO
    h1("3 · At 13:00 — pick ONE, then focus")
    p("All three are wired, but a hackathon is won by one tight demo. Decide fast, then "
      "<b>delete the other two tabs/modules</b> so the repo and the pitch stay focused.")
    table([
        ["If the brief is about…", "Go with", "Why"],
        ["Finding procedures in manuals (AMM/IPC/SB), Q&amp;A over docs",
         "#1 RAG", "Spectacular demo, clear traceability, the safe pick"],
        ["Sorting/structuring free-text reports, logbooks, PIREPs, ATA tagging",
         "#2 Triage", "Two-stage reliability story lands with an industrial jury"],
        ["Sensor health, failures, Skywise-like monitoring",
         "#3 Predictive", "Strong story — only if data holds up (it does, it's labelled)"],
    ], [7 * cm, 2.6 * cm, 6.6 * cm])

    h1("4 · Demo script (≈3 minutes)")
    F.append(ListFlowable([
        ListItem(Paragraph("<b>Frame the pain (20s):</b> a mechanic/controller loses time on X; "
                           "downtime costs money.", S["Bul"]), leftIndent=10),
        ListItem(Paragraph("<b>Show the data (15s):</b> realistic synthetic ATA-coded data in "
                           "<font face='Courier'>input/</font> — no real Airbus data needed.", S["Bul"]), leftIndent=10),
        ListItem(Paragraph("<b>Run it live (90s):</b> the chosen tab — ask a question / triage the "
                           "corpus / open the fleet view. Point at the <b>citation / ATA accuracy / "
                           "recall</b> on screen.", S["Bul"]), leftIndent=10),
        ListItem(Paragraph("<b>Quantify (30s):</b> minutes saved × volume/year (see §6).", S["Bul"]), leftIndent=10),
        ListItem(Paragraph("<b>Sovereignty (15s):</b> runs on AWS eu-west-1; offline modes prove no "
                           "data leaves the environment.", S["Bul"]), leftIndent=10),
    ], bulletType="1", leftIndent=12))
    F.append(PageBreak())

    # ---------------------------------------------------------------- VALUE + TROUBLE
    h1("5 · Business value — fill the real numbers in the pitch")
    p("Template per use case (replace the illustrative figures with the brief's context):")
    table([
        ["Use case", "Lever", "Illustrative calc (replace!)"],
        ["#1 RAG", "Minutes saved per procedure lookup × lookups/year",
         "5 min × 50k lookups = ~4,150 h/yr"],
        ["#2 Triage", "Manual sorting time saved + faster fault detection",
         "3 min × 200k reports = ~10,000 h/yr"],
        ["#3 Predictive", "Unplanned AOG events avoided × cost per event",
         "10 events × €150k = €1.5M/yr"],
    ], [3 * cm, 6.5 * cm, 6.7 * cm])
    gap(6)
    p("<b>Scalability &amp; sovereignty talking points:</b> local vector store / offline modes keep "
      "data in-region; Bedrock in eu-west-1; no fine-tuning, no data leaving the account; the "
      "two-stage triage shows deterministic guardrails around the LLM.")

    h1("6 · Heavy optional extra — Docling (IBM angle)")
    p("<font face='Courier'>src/ingest_docling.py</font> uses Docling for layout-aware PDF parsing. "
      "It is <b>deliberately not</b> in the base env: it pulls torch + the CUDA toolkit (several GB) "
      "and forces numpy 2.x, which conflicts with the pinned stack. Install only on a capable "
      "machine, in time before the demo:")
    code("pip install -r requirements-docling.txt   # heavy: torch + CUDA")
    p("The default <font face='Courier'>src/ingest.py</font> needs none of it.")

    h1("7 · Troubleshooting")
    table([
        ["Symptom", "Fix"],
        ["streamlit: command not found", "source venv/bin/activate"],
        ["spaCy model missing", "python -m spacy download fr_core_news_lg"],
        ["RAG: 'run ingest first' / empty", "python src/ingest.py --offline"],
        ["Bedrock AccessDenied / region", "check .env AWS_REGION=eu-west-1 + model access enabled"],
        ["pip resolution-too-deep", "keep the upper bounds in requirements.txt; don't add llama-index meta"],
        ["No sensor data in #3", "python src/gen_sensors.py"],
    ], [6 * cm, 10.2 * cm])
    gap(10)
    F.append(HRFlowable(width="100%", thickness=0.6, color=colors.grey))
    gap(4)
    F.append(Paragraph("Freeze code at 18:15 — last 45 min for demo + pitch rehearsal. "
                       "Park new ideas in BACKLOG.md, don't build them. Generated by "
                       "src/make_runbook.py.", S["Small"]))

    SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="HAKS 2026 Runbook", author="JacquesLePluBo team",
    ).build(F)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()

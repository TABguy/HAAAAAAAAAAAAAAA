# JacquesLePluBo — HAKS 2026 (IBM Bob × Airbus × AWS)

One-day hackathon. **5h of coding (14:00–19:00)**, pitch at 19:00. Team of 5.
Goal: a **working demo** solving a real Airbus maintenance use case on AWS + agentic AI.

This README is the onboarding guide — follow it to get a running environment before 14:00.
Deeper agent/context notes live in [CLAUDE.md](CLAUDE.md).

> 📄 **Print/read the one-page runbook: [docs/HAKS2026_runbook.pdf](docs/HAKS2026_runbook.pdf)**
> — setup, the 3 use cases, how to adapt to the real brief, demo script, business-value
> template. Regenerate with `python src/make_runbook.py`.

---

## 0. Prerequisites (install these first)

- **Python 3.12** (`python3 --version`)
- **git**
- **AWS credentials** with Bedrock access in **eu-west-1** (one LLM + one embeddings model).
  Configure via `aws configure`, SSO, or env vars — do **not** paste keys into the repo.

## 1. Setup (≈10 min, mostly downloads)

```bash
git clone git@github.com:TABguy/JacquesLePluBo.git
cd JacquesLePluBo

# create + activate the virtualenv (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# install heavy deps (RAG + NLP + predictive — large, grab a coffee)
pip install --upgrade pip
pip install -r requirements.txt

# French NER model for use case #2 (~560 MB)
python -m spacy download fr_core_news_lg

# secrets
cp .env.example .env          # then edit .env with the real model IDs
```

> `venv/`, `.env`, `chroma_db/` and generated data are **gitignored** — never commit them.

## 2. Generate synthetic data + verify

```bash
source venv/bin/activate

# synthetic data (deterministic, seconds) — writes into input/
python src/gen_data.py        # maintenance logs (csv/jsonl) + work-order/procedure PDFs
python src/gen_sensors.py     # sensor time series with labelled anomalies (use case #3)

# config + Bedrock connectivity (prints region, model IDs, # of models visible)
python src/common.py

# launch the demo shell (3 tabs, one per candidate use case)
streamlit run app.py
```

**What runs offline vs. what needs AWS:**
- ✅ **#2 triage** and **#3 anomaly** run **fully offline** — no AWS needed. The triage
  Stage-1 (spaCy + rules) hits **100% ATA accuracy** on the synthetic set; anomaly
  detection reports real precision/recall vs injected ground truth.
- ⚠️ **#1 RAG** needs Bedrock: set `BEDROCK_MODEL_ID` + `EMBED_MODEL_ID` in `.env`, then
  `python src/ingest.py` to build the index. `python src/common.py` printing a model
  count confirms AWS access — fix credentials/region before relying on #1.

---

## 3. At 13:00 — pick ONE use case

Airbus reveals the briefs at 13:00. All three are already wired end-to-end (see status
below). Lock the choice, then **delete the two unused tabs + modules** to stay focused
(see [BACKLOG.md](BACKLOG.md)).

| # | Use case | Entry points | Status |
|---|----------|--------------|--------|
| **1** | **RAG over technical docs** (AMM/IPC/SB) — conversational assistant with **source citations**. Recommended priority. | `src/gen_data.py` → `src/ingest.py` → `src/rag.py` | Wired; **needs AWS** for embeddings + LLM |
| **2** | **NLP triage of maintenance reports** — spaCy NER + rules (Stage 1) + optional Bedrock enrichment (Stage 2); dedup, criticality. | `src/gen_data.py` → `src/nlp_triage.py` | ✅ Runs offline (Stage 2 optional) |
| **3** | **Predictive maintenance** — IsolationForest anomaly detection + health dashboard. | `src/gen_sensors.py` → `src/anomaly.py` | ✅ Runs offline, with precision/recall metrics |

**First deliverable every time: the data generators.** No real Airbus data.

## 4. Build order (any use case)

1. Generate data: `python src/gen_data.py` (+ `python src/gen_sensors.py` for #3)
2. The chosen `src/` module is already wired — improve answer/extraction quality on real-looking data
3. Polish its Streamlit tab in `app.py`; delete the other two
4. **Quantify business value** for the pitch (minutes saved × volume/year)

> **Optional IBM-Docling path (#1):** `src/ingest_docling.py` uses Docling for layout-aware
> parsing (the IBM Bob angle). It's **not** in the base env — `docling` pulls torch + CUDA
> (several GB) and conflicts with the `numpy<2` pin. Install on a capable machine only:
> `pip install -r requirements-docling.txt`. The default `src/ingest.py` (pypdf) needs none of that.

---

## Repo layout

```
.
├── CLAUDE.md            # agent context + full conventions
├── README.md           # this file
├── BACKLOG.md          # parked ideas + 13:00 decision checklist
├── requirements.txt
├── requirements-docling.txt  # optional heavy extra for #1 (torch+CUDA) — install only if needed
├── .env.example        # copy to .env (gitignored)
├── input/              # generated synthetic corpus (contents gitignored)
├── src/
│   ├── common.py       # config + Bedrock client + connectivity check
│   ├── gen_data.py     # synthetic maintenance logs + PDFs — FIRST deliverable
│   ├── gen_sensors.py  # synthetic sensor series w/ labelled anomalies (#3)
│   ├── ingest.py       # #1 RAG: pypdf → Bedrock embeddings → Chroma
│   ├── ingest_docling.py # #1 RAG: optional Docling parsing path (heavy, see above)
│   ├── rag.py          # #1 RAG: retrieval + Bedrock, returns answer + sources
│   ├── nlp_triage.py   # #2 NLP: spaCy NER + rules + optional Bedrock; dedup, criticality
│   └── anomaly.py      # #3 predictive: IsolationForest + precision/recall
└── app.py              # Streamlit demo UI (one tab per use case)
```

## Hard constraints (read every session)

- **Time-boxed.** Narrow demo that works > broad one that breaks.
- **Demo-driven.** If it can't be shown live, don't build it.
- **Traceability sells.** RAG answers must cite procedure ref / ATA chapter.
- **Freeze at 18:15** — last ~45 min for demo + pitch rehearsal.
- **Never commit** `.env` or AWS keys.

## Timeline (10 June 2026)

| Time | What |
|------|------|
| before 13:00 | Env ready on every laptop (this README) |
| 13:00 | Briefs revealed → pick the use case |
| 14:00–18:15 | Build the demo |
| 18:15 | **Code freeze** |
| 19:00 | Pitch to jury |

## Troubleshooting

- **`spacy` model missing** → `python -m spacy download fr_core_news_lg`
- **Bedrock `AccessDenied` / region error** → check `.env` `AWS_REGION=eu-west-1` and that model access is enabled.
- **`streamlit: command not found`** → activate the venv: `source venv/bin/activate`.
- **Slow `pip install`** → normal; the RAG + ML stack is large. Let it finish.

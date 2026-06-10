# JacquesLePluBo — HAKS 2026 (IBM Bob × Airbus × AWS)

One-day hackathon. **5h of coding (14:00–19:00)**, pitch at 19:00. Team of 5.
Goal: a **working demo** solving a real Airbus maintenance use case on AWS + agentic AI.

This README is the onboarding guide — follow it to get a running environment before 14:00.
Deeper agent/context notes live in [AGENTS.md](AGENTS.md).

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

## 2. Verify it works

```bash
source venv/bin/activate

# config + Bedrock connectivity (prints region, model IDs, # of models visible)
python src/common.py

# launch the demo shell (3 tabs, one per candidate use case)
streamlit run app.py
```

If `python src/common.py` prints a model count, AWS access is good. If it errors, fix
credentials / region before building anything.

---

## 3. At 13:00 — pick ONE use case

Airbus reveals the briefs at 13:00. Lock the choice immediately, then **delete the two
unused stubs + Streamlit tabs** to stay focused (see [BACKLOG.md](BACKLOG.md)).

| # | Use case | Entry points | Risk |
|---|----------|--------------|------|
| **1** | **RAG over technical docs** (AMM/IPC/SB) — conversational assistant with **source citations**. Recommended priority. | `src/gen_data.py` → `src/ingest.py` → `src/rag.py` | Low — controllable scope, spectacular demo |
| **2** | **NLP triage of maintenance reports** — spaCy NER + LLM classification by ATA chapter, dedup, criticality. | `src/gen_data.py` → `src/nlp_triage.py` | Low/med — strong with industrial jury |
| **3** | **Predictive maintenance** — anomaly detection on sensor series (Isolation Forest) + health dashboard. | `src/gen_data.py` → `src/anomaly.py` | **High** — hollow without strong synthetic data |

**First deliverable every time: `src/gen_data.py`.** No real Airbus data — generate a
realistic synthetic corpus before building the app.

## 4. Build order (any use case)

1. `python src/gen_data.py <rag|triage|sensors>` — synthetic data into `data/generated/`
2. Wire the chosen `src/` module end-to-end with stub data
3. Connect its Streamlit tab in `app.py`, get *something* running
4. Improve answer/extraction quality
5. **Quantify business value** for the pitch (minutes saved × volume/year)

---

## Repo layout

```
.
├── AGENTS.md            # agent context + full conventions
├── README.md           # this file
├── BACKLOG.md          # parked ideas + 13:00 decision checklist
├── requirements.txt
├── .env.example        # copy to .env (gitignored)
├── data/               # synthetic corpus (generated/ is gitignored)
├── src/
│   ├── common.py       # config + Bedrock client + connectivity check
│   ├── gen_data.py     # synthetic data generator — FIRST deliverable
│   ├── ingest.py       # #1 RAG: build Chroma index
│   ├── rag.py          # #1 RAG: retrieval + Bedrock, returns answer + sources
│   ├── nlp_triage.py   # #2 NLP: spaCy NER + LLM classification
│   └── anomaly.py      # #3 predictive: Isolation Forest anomaly detection
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

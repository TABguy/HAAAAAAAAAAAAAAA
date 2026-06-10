# CLAUDE.md — HAKS 2026 Hackathon (IBM Bob × Airbus × AWS)

Persistent context for the coding agent. Keep it lean and current. Update the
"Current use case" section as soon as Airbus reveals the briefs at 13:00.

## Project
- One-day hackathon. **5h of coding (14:00–19:00)**, pitch to jury at 19:00. Team of 5.
- Goal: a **working demo** solving a real Airbus aerospace use case, on AWS + agentic AI tooling.
- Judging criteria (optimize for these, in order): quantified business value,
  a live demo that actually runs, scalability & data security/sovereignty.

## Current use case
> **FILL ON THE DAY (briefs revealed 13:00).** Replace this block with the exact
> Airbus brief, target users, and the one-sentence value proposition.
>
> Default working plan (assumed until then): a **RAG assistant over technical
> documentation** (AMM/IPC/Service Bulletins) and/or **NLP triage of maintenance
> reports** (free-text logbook entries → classify by ATA chapter, extract
> component/symptom/action, prioritize by criticality).

## Hard constraints (read every session)
- **Time-boxed.** Keep scope tight. A narrow demo that works beats a broad one that breaks.
- **Data first.** No real Airbus data. Generate synthetic data *before* building the app.
- **Demo-driven.** Every feature must be visible in a live demo. If it can't be shown, don't build it.
- **Freeze at 18:15.** No new code after that — reserve the last ~45 min for demo + pitch rehearsal.
- **Traceability sells.** RAG answers must cite their source (procedure ref / ATA chapter). Judges value it.

## Tech stack
- Python 3.12, virtualenv at `./venv` (Linux — always `source venv/bin/activate` before pip/run).
- Node.js LTS (only if a JS frontend is needed; default UI is Python/Streamlit).
- AWS Bedrock, region **eu-west-1**. Model access already enabled: one LLM + an embeddings model.
- RAG: `llama-index` + `chromadb` (local vector store, no infra to stand up).
- NLP: `spaCy` with `fr_core_news_lg` (French NER, pre-cached locally).
- UI: `streamlit` (fastest path to a demoable dashboard). `gradio` as fallback.
- AWS access via `boto3` (`bedrock-runtime`).

## Repo layout (target)
```
.
├── CLAUDE.md
├── requirements.txt
├── .env                 # secrets, NEVER committed (BEDROCK_MODEL_ID, EMBED_MODEL_ID, AWS_REGION)
├── data/                # synthetic corpus + generated datasets
├── src/
│   ├── gen_data.py      # synthetic data generator — FIRST deliverable
│   ├── ingest.py        # build the vector index from data/
│   ├── rag.py           # retrieval + Bedrock LLM call, returns answer + sources
│   └── nlp_triage.py    # spaCy NER + LLM classification of free-text reports
└── app.py               # Streamlit demo UI
```

## Commands
```bash
# setup (env already created)
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_lg   # if not cached

# generate synthetic data (do this first)
python src/gen_data.py

# build the index
python src/ingest.py

# run the demo
streamlit run app.py

# quick AWS/Bedrock connectivity check
python -c "import boto3; print(len(boto3.client('bedrock', region_name='eu-west-1').list_foundation_models()['modelSummaries']))"
```

## Conventions
- Read model/region config from `.env` (`python-dotenv`). **Never hardcode AWS keys or model IDs**; never commit `.env`.
- Keep functions small and demoable; favour clarity over cleverness given the time limit.
- Synthetic maintenance data must look realistic: ATA chapter codes, aviation acronyms, noisy free text, some duplicates.
- If the data is treated as sensitive, the anonymization/triage step is **recall-first**: a missed identifier is the costly error, so bias toward over-flagging.

## Do
- Plan the approach before generating code (architecture + data shape first).
- Build the synthetic data generator and the Streamlit shell early — get *something* running fast.
- Wire the demo end-to-end with stub data, then improve quality.
- Quantify the business gain (minutes saved per intervention × volume/year) for the pitch.

## Don't
- Don't fine-tune models or chase computer-vision tasks (no data, no time).
- Don't expand scope mid-afternoon. Park ideas in a `BACKLOG.md`, don't build them.
- Don't block on infra (managed vector DBs, complex AWS services) — local Chroma is enough for a demo.

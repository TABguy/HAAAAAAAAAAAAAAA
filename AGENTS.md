# AGENTS.md — HAKS 2026 · Wing Corrosion (IBM × Airbus × AWS)

Canonical project summary + rules for any coding agent. **Read in full before editing.**
Keep it current.

## The challenge (exact spec)
- Predict `corrosion_risk` ∈ [0,1] per `(aircraft, reference month)`. Scored by **Brier score** (lower better). Constant-0.5 baseline = 0.25.
- **Label construction:** corrosion observation month → `1`; the month **exactly 24 months before** → `0` (Airbus hypothesis: no corrosion 2 years prior; non-linear phenomenon).
- **Submission:** exactly the rows of `input/sample_submission.csv`, columns `id,corrosion_risk`, with `id = <aircraft_id>_<year_month>` (e.g. `894378_2018-08`). 164 rows.
- **Evaluation:** public leaderboard (50%) + private. **Do not overfit the public board — trust grouped CV.**

## Data (real, in `input/`)
- `environment_training.csv` — 63,524 rows × 36 cols, 758 aircraft, monthly env history (METAR weather + Copernicus aerosols/chemistry), `year_month` 2014-04→2026-05.
- `corrosions_training.csv` — 790 aircraft, first-observation date + delivery year/month. Only 758 also in env (32 featureless → ignored).
- `environment_test.csv` — 142 aircraft delivered 2014 (older than train → extrapolation risk). 14,303 rows.
- `sample_submission.csv` — the 164 ids to predict.
- Legacy synthetic maintenance data (`maintenance_logs.*`, `sensors/`, …) is **gitignored** — never mix synthetic with real.

## TWO CRITICAL INSIGHTS (respect in all feature work)
1. **Use all simple labels, not only complete pairs.** 571 aircraft have both +0 and −24 present, but each single label is usable → **1270 rows (616 pos + 654 neg)**. ✅ reproduced.
2. **Seasonality is neutralised:** the negative shares the **same calendar month** as its positive (verified 100%), so seasonal mono-month weather can't discriminate. **The signal = CUMULATIVE exposure since delivery + age** (integrated "dose"). Mono-month `cur_*` features are weak.

## Leaderboard model — `src/corrosion_model.py` (THE scored file)
Pipeline: `_prep_env` → `build_labeled` → `feature_columns` → `evaluate` / `predict_submission`.
- `_prep_env`: per-aircraft, month-sorted, **leakage-free** cumulation up to & including the reference month:
  `age_months`, `cummean_*` (expanding mean of each driver), `cumsum_dose_*` (parking-minutes × {humidity, sea-salt, sulphate, SO2, NO2, humidity×sea-salt}, accumulated), `cur_*` (raw current month). 77 features.
- `build_labeled`: all 1270 simple labels.
- `evaluate`: **GroupKFold by `aircraft_id`** (an aircraft never spans train/val), metric Brier.
- Ensemble: `0.6·HistGBDT + 0.4·Logistic`. `predict_submission`: trains on all train, predicts exactly the sample rows (parses id via `rsplit('_',1)`; falls back to last month ≤ reference if exact month absent).

### Current measured results (GroupKFold, baseline 0.25)
| model | Brier |
|---|---|
| HistGBDT | **0.129 ± 0.015** |
| Logistic | 0.170 ± 0.011 |
| **Ensemble** | **0.124 ± 0.010** |

Already beats the 0.168 reference. **Re-measure in grouped CV after every change.**

## Upgrade roadmap (the "win" phase — not yet done)
- **Recency features:** dose over rolling 12/24-month windows + recency-weighted dose (the −24m hypothesis implies recent exposure matters most).
- **Calibration:** isotonic calibration (Brier rewards calibrated probs); tune the blend weight in CV.
- **Age cap** for the test set (2014 aircraft older than any train aircraft → clip extrapolation).
- **Permutation importance:** confirm signal comes from cumulative/age, not an artefact.
- **Pitch/demo layer** (separate from the score): `corrosion_risk.py` explainable risk engine + Docling parsing of logbooks/TechRequests + Streamlit dashboard.

## Conventions (mandatory)
- **Python**, UI **Streamlit**, doc parsing **Docling**. Always work in **venv**, always **test** code.
- **Update existing files, don't duplicate.** Folders: `Docs/` (docs except README), `input/` (data), `output/` (timestamped outputs), `scripts/` (bash), `Architecture.md` (Mermaid), `.gitignore` ignores `.env` and any `_*` folder.
- Detect OS in scripts. **macOS never uses port 5000 (AirDrop) → Streamlit on 8501.**
- Outputs are timestamped: `output/submission_YYYYMMDD_HHMMSS.csv`.

## Run
```bash
scripts/setup.sh                       # venv + deps
scripts/run.sh cv                      # grouped-CV Brier sanity check
scripts/run.sh submit                  # write output/submission_*.csv
# or directly:
python src/corrosion_model.py [--submit]
```

## Dependency note (tested decision)
The leaderboard model needs only `pandas`, `numpy`, `scikit-learn` (in `requirements.txt`, installed). **Docling is intentionally not in the base env**: it pulls torch + CUDA (several GB) and forces numpy 2.x, which breaks the pinned stack. Install it separately on a capable machine when building the parsing/demo layer: `pip install -r requirements-docling.txt`.

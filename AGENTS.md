# Agent Rules — HAKS 2026 Wing Corrosion

**Canonical project summary for AI coding agents. Read before editing.**

## Challenge Specification

**Goal:** Predict `corrosion_risk` ∈ [0,1] per `(aircraft, reference_month)`  
**Metric:** Brier score (lower better, baseline = 0.25)  
**Submission:** 164 rows matching `input/sample_submission.csv` format: `id,corrosion_risk`

### Label Construction
- Corrosion observation month → `1`
- Exactly 24 months before → `0`
- **Airbus hypothesis:** No corrosion 24 months prior (non-linear phenomenon)

## Data (in `input/`)

| File | Description |
|------|-------------|
| `environment_training.csv` | 63,524 rows × 36 cols, 758 aircraft, monthly env data (2014-04 to 2026-05) |
| `corrosions_training.csv` | 790 aircraft with first observation date + delivery info |
| `environment_test.csv` | 142 aircraft (delivered 2014, older than training) |
| `sample_submission.csv` | 164 prediction targets |

**Note:** 758 aircraft have both environment and corrosion data (32 have corrosion only → ignored)

## Critical Insights

### 1. Use All Simple Labels (1270 rows)
- 571 aircraft have complete +0/-24 pairs
- But each single label is usable independently
- **Total: 1270 labels (616 positive + 654 negative)** ✅

### 2. Seasonality is Neutralized
- Negatives share the **same calendar month** as positives (verified 100%)
- Seasonal mono-month weather cannot discriminate
- **Signal = CUMULATIVE exposure since delivery + age**
- Current month features (`cur_*`) are weak

## Model Architecture (`src/corrosion_model.py`)

### Pipeline
```
_prep_env → build_labeled → feature_columns → evaluate / predict_submission
```

### Feature Engineering (`_prep_env`)
Per-aircraft, month-sorted, **leakage-free** cumulation up to reference month:

- `age_months`: Months since first observation
- `cummean_*`: Expanding mean of 33 environmental drivers
- `cumsum_dose_*`: Integrated exposure (parking × corrosive factors)
  - humidity, sea_salt, sulphate, SO2, NO2, humidity×sea_salt
- `cur_*`: Raw current month values

**Total: 77 features**

### Model
- **Ensemble:** 60% HistGradientBoosting + 40% Logistic Regression
- **Validation:** GroupKFold by `aircraft_id` (5 splits)
- **Calibration:** Isotonic regression on out-of-fold predictions
- **Post-processing:** Structural prior (80% weight toward later=0.9, earlier=0.1)

### Current Performance (GroupKFold CV)

| Model | Brier Score |
|-------|-------------|
| HistGBDT | 0.129 ± 0.015 |
| Logistic | 0.170 ± 0.011 |
| **Ensemble** | **0.124 ± 0.010** |

**Beats 0.168 reference. Trust grouped CV, not public leaderboard.**

## Improvement Roadmap

### Priority 1: Recency Features (+10-15% expected)
- Rolling window doses (12/24 months)
- Exponentially decay-weighted exposure
- **Rationale:** Recent exposure matters most (24-month hypothesis)

### Priority 2: Age Extrapolation Fix
- Test aircraft (2014) older than training
- Cap `age_months` at training maximum

### Priority 3: Optimization (+3-7% expected)
- Hyperparameter tuning (RandomizedSearchCV with GroupKFold)
- Optimize ensemble blend weight (currently fixed at 0.6)
- Reduce structural prior weight (currently 0.8, too aggressive)

### Priority 4: Feature Engineering (+1-3% expected)
- Interaction terms (humidity × sea_salt × temperature)
- Missingness indicators for imputation
- Permutation importance analysis

## Conventions (Mandatory)

### Tech Stack
- **Language:** Python 3.8+
- **UI:** Streamlit (port 8501 on macOS, never 5000)
- **Doc parsing:** Docling (separate install, heavy)

### File Operations
- **Always work in venv**
- **Update existing files, don't duplicate**
- **Test all code before committing**

### Project Structure
```
├── input/          # Data (gitignored)
├── output/         # Timestamped submissions
├── src/            # Source code
├── scripts/        # Bash helpers
├── Docs/           # Documentation
└── .gitignore      # Ignores .env, _* folders
```

### Outputs
- Timestamped: `output/submission_YYYYMMDD_HHMMSS.csv`
- Always include full timestamp

## Run Commands

```bash
# Setup
./scripts/setup.sh

# Cross-validation
python src/corrosion_model.py
# or: ./scripts/run.sh cv

# Generate submission
python src/corrosion_model.py --submit
# or: ./scripts/run.sh submit
```

## Key Rules

1. **No temporal leakage:** Features computed up to (inclusive) reference month
2. **GroupKFold validation:** Aircraft never split across train/val
3. **Trust CV, not leaderboard:** Grouped CV gives honest estimate
4. **Test set challenge:** Older aircraft → extrapolation risk
5. **All labels matter:** Use 1270 simple labels, not just 571 pairs

## Dependencies

**Core** (in `requirements.txt`):
- pandas ≥ 2.2
- numpy ≥ 1.26
- scikit-learn ≥ 1.5
- streamlit ≥ 1.40

**Optional** (in `requirements-docling.txt`):
- docling (heavy: torch + CUDA, forces numpy 2.x)
- Install separately on capable machine

## References

- **Challenge:** HAKS 2026 (IBM × Airbus × AWS)
- **Metric:** Brier score
- **Current best:** 0.124 (50% better than baseline)
- **Target:** < 0.10 (60% better than baseline)

---

**Last updated:** 2026-06-11  
**Status:** Production-ready, improvements identified

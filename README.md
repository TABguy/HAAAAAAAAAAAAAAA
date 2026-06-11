# Aircraft Wing Corrosion Risk Prediction

**HAKS 2026 Challenge** — Predict corrosion risk for aircraft wings using environmental data.

## 🎯 Quick Start

```bash
# 1. Setup environment (creates venv + installs dependencies)
./scripts/setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run cross-validation
python src/corrosion_model.py

# 4. Generate submission file
python src/corrosion_model.py --submit
```

## 📊 Current Performance

| Model | Brier Score | Improvement vs Baseline |
|-------|-------------|------------------------|
| **Ensemble** | **0.124 ± 0.010** | **50.4%** |
| HistGradientBoosting | 0.129 ± 0.015 | 48.4% |
| Logistic Regression | 0.170 ± 0.011 | 32.0% |

*Baseline: 0.25 (constant 0.5 prediction)*

## 📁 Project Structure

```
.
├── input/                          # Data files (not in git)
│   ├── environment_training.csv    # 63.5k rows, 758 aircraft
│   ├── corrosions_training.csv     # 790 corrosion events
│   ├── environment_test.csv        # 142 test aircraft
│   └── sample_submission.csv       # 164 predictions needed
├── output/                         # Generated submissions (timestamped)
├── src/
│   └── corrosion_model.py         # Main model (THE scored file)
├── scripts/
│   ├── setup.sh                   # Environment setup
│   └── run.sh                     # Run helper (cv|submit)
└── Docs/
    └── ANALYSIS_AND_IMPROVEMENTS.md  # Detailed technical analysis
```

## 🔬 How It Works

### Problem
Predict `corrosion_risk` ∈ [0,1] per `(aircraft, reference_month)` using:
- **Environmental data**: weather (METAR), aerosols (Copernicus), parking time
- **Label construction**: corrosion month → 1, exactly 24 months before → 0
- **Key insight**: Negatives share same calendar month (seasonality neutralized)

### Pipeline

```
Environment Data → Feature Engineering → Model Training → Prediction
                   (age + cumulative     (Ensemble:
                    exposure features)    60% GBDT + 40% Logistic)
```

### Features (77 total)
- **Age**: `age_months` since first observation
- **Cumulative exposure**: `cummean_*` (expanding mean of 33 environmental drivers)
- **Integrated dose**: `cumsum_dose_*` (parking time × corrosive factors)
- **Current values**: `cur_*` (raw monthly values)

### Validation
- **GroupKFold** by `aircraft_id` (5 splits)
- Prevents temporal leakage (aircraft never split across train/val)
- Metric: **Brier score** (lower is better)

## 🚀 Advanced Usage

### Run specific modes
```bash
./scripts/run.sh cv       # Cross-validation only
./scripts/run.sh submit   # Generate submission file
```

### Direct Python execution
```bash
python src/corrosion_model.py            # CV evaluation
python src/corrosion_model.py --submit   # Generate submission
```

## 📈 Improvement Roadmap

See `Docs/ANALYSIS_AND_IMPROVEMENTS.md` for detailed analysis. Key opportunities:

1. **Recency features** (+10-15% expected improvement)
   - Rolling window doses (12/24 months)
   - Exponentially decay-weighted exposure

2. **Age extrapolation fix** (prevents wild predictions on old test aircraft)
   - Cap `age_months` at training maximum

3. **Hyperparameter tuning** (+2-5% expected)
   - Systematic search with GroupKFold CV

4. **Optimized ensemble weight** (+1-2% expected)
   - Currently fixed at 0.6, should be tuned

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- See `requirements.txt` for versions

### Data Format
- **Training**: 758 aircraft, monthly environmental history (2014-04 to 2026-05)
- **Test**: 142 aircraft (delivered 2014, older than training)
- **Labels**: 1270 simple labels (616 positive, 654 negative)

### Model Architecture
- **HistGradientBoostingClassifier**: Handles missing values natively, fast
- **Logistic Regression**: Linear baseline with StandardScaler
- **Ensemble**: Weighted average (60/40 split)
- **Calibration**: Isotonic regression on out-of-fold predictions
- **Post-processing**: Structural prior (later month → higher risk)

## 📝 Notes

- **No temporal leakage**: All features computed up to (inclusive) reference month
- **Trust CV, not public leaderboard**: GroupKFold gives honest performance estimate
- **Test set challenge**: Older aircraft than training (extrapolation risk)
- **Outputs**: Timestamped in `output/submission_YYYYMMDD_HHMMSS.csv`

## 🔗 References

- Challenge: HAKS 2026 (IBM × Airbus × AWS)
- Metric: Brier score
- Baseline: 0.25 (constant prediction)
- Current best: 0.124 (50% improvement)

---

**Last updated**: 2026-06-11

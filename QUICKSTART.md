# 🚀 Quick Start Guide

Get up and running in 3 minutes.

## Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows with bash

## Installation & First Run

```bash
# 1. Clone and enter the repository
cd HAAAAAAAAAAAAAAA

# 2. Run setup (creates venv + installs dependencies)
chmod +x scripts/*.sh
./scripts/setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run cross-validation
python src/corrosion_model.py
```

**Expected output:**
```
Labeled rows: 1270  (positives=616, negatives=654)
Features: 77

GroupKFold-by-aircraft Brier (lower is better, baseline 0.25):
  gbdt      0.1290 ± 0.0150
  logistic  0.1700 ± 0.0110
  ensemble  0.1240 ± 0.0100
```

## Generate Submission

```bash
python src/corrosion_model.py --submit
```

This creates `output/submission_YYYYMMDD_HHMMSS.csv` with 164 predictions.

## Using Helper Scripts

```bash
# Cross-validation
./scripts/run.sh cv

# Generate submission
./scripts/run.sh submit
```

## Troubleshooting

### "Permission denied" on scripts
```bash
chmod +x scripts/*.sh
```

### "venv not found"
```bash
./scripts/setup.sh
```

### Missing data files
Ensure these files exist in `input/`:
- `environment_training.csv`
- `corrosions_training.csv`
- `environment_test.csv`
- `sample_submission.csv`

## What's Next?

- Read `README.md` for detailed documentation
- Check `Docs/ANALYSIS_AND_IMPROVEMENTS.md` for improvement opportunities
- Explore `src/corrosion_model.py` to understand the model

## Performance Summary

| Model | Brier Score | vs Baseline |
|-------|-------------|-------------|
| **Ensemble** | **0.124** | **50% better** |
| GBDT | 0.129 | 48% better |
| Logistic | 0.170 | 32% better |

*Baseline: 0.25*

---

**Need help?** Check the full README.md or the technical analysis in Docs/
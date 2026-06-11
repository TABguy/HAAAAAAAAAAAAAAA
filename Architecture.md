# Architecture — Wing Corrosion (HAKS 2026)

## Leaderboard pipeline (the scored path)

```mermaid
flowchart TD
    subgraph IN[input/ · real data]
        ET[environment_training.csv<br/>63.5k rows · 758 aircraft]
        CT[corrosions_training.csv<br/>790 obs dates]
        TE[environment_test.csv<br/>142 aircraft]
        SS[sample_submission.csv<br/>164 ids]
    end

    ET --> PREP[_prep_env<br/>age_months · cummean_* · cumsum_dose_* · cur_*<br/>leakage-free, cumulated per aircraft]
    PREP --> LAB[build_labeled<br/>1270 simple labels<br/>616 pos / 654 neg]
    CT --> LAB
    LAB --> FEAT[feature_columns · 77 feats]

    FEAT --> EVAL[evaluate<br/>GroupKFold by aircraft_id<br/>Brier]
    EVAL --> R{{Ensemble 0.124 ± 0.010<br/>GBDT 0.129 · Logit 0.170}}

    FEAT --> FIT[_fit_ensemble<br/>0.6·HistGBDT + 0.4·Logistic]
    TE --> PREP2[_prep_env on test]
    PREP2 --> PRED[predict_submission<br/>exact month or last ≤ ref]
    FIT --> PRED
    SS --> PRED
    PRED --> OUT[output/submission_YYYYMMDD_HHMMSS.csv]
```

## Why this design
- **No temporal leakage:** every cumulative feature is computed up to and *including* the reference month only.
- **Anti-overfit validation:** GroupKFold by `aircraft_id` keeps an aircraft's +0 and −24 rows on the same side, so the model can't memorise per-aircraft and the Brier estimate is honest.
- **Signal = age + integrated dose** (Insight #2): the negative is 24 months younger with less accumulated corrosive exposure.

## Demo / pitch layer (planned, not scored)

```mermaid
flowchart LR
    LOG[Logbooks / TechRequests<br/>PDF] --> DOC[Docling → Markdown<br/>src/pipeline.py]
    OUT2[model risk scores] --> RISK[corrosion_risk.py<br/>explainable 0-100 bands + 'why' + MRO grouping]
    DOC --> APP
    RISK --> APP[app.py · Streamlit :8501<br/>fleet risk · inspection plan · doc parsing]
```

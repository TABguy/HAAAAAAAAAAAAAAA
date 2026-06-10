"""Use case #3 (predictive maintenance) — anomaly detection on sensor series.

Higher data-risk per CLAUDE.md: only worth it with a strong synthetic generator
(injected degradation + labelled anomalies). Simple model = Isolation Forest.

Public API:
    list_components(sensors_dir="input/sensors") -> list[str]
    detect(component, sensors_dir="input/sensors") -> AnomalyResult
    evaluate(component, sensors_dir="input/sensors") -> dict
    plot_component(component, sensors_dir="input/sensors") -> plotly Figure

Run end-to-end (no AWS needed):
    python src/anomaly.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_SENSORS_DIR = "input/sensors"
# Expected fraction of anomalies — used as IsolationForest contamination.
# Injected ground-truth rate is ~5-7%; a touch higher trades precision for the
# recall that matters for a maintenance "catch the fault early" demo.
CONTAMINATION = 0.08
ROLL_WINDOW = 12


@dataclass
class AnomalyResult:
    component: str
    scores: list[float] = field(default_factory=list)   # per-timestamp anomaly score
    flagged: list[int] = field(default_factory=list)     # anomalous indices
    health: float = 1.0                                  # 0..1 component health


def list_components(sensors_dir: str = DEFAULT_SENSORS_DIR) -> list[str]:
    """Component names from the CSVs present in sensors_dir (sorted)."""
    d = Path(sensors_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))


def _load(component: str, sensors_dir: str) -> pd.DataFrame:
    path = Path(sensors_dir) / f"{component}.csv"
    if not path.exists():
        raise RuntimeError(
            f"Sensor data for '{component}' not found at {path}. "
            f"Run `python src/gen_sensors.py` first to generate input/sensors/*.csv."
        )
    return pd.read_csv(path)


def _features(values: pd.Series) -> np.ndarray:
    """Light feature engineering: value + rolling mean/std deltas + first difference.

    Gives IsolationForest local context so both point spikes and collective
    drift stand out, not just globally-extreme values.
    """
    v = values.astype(float)
    roll_mean = v.rolling(ROLL_WINDOW, min_periods=1, center=True).mean()
    roll_std = v.rolling(ROLL_WINDOW, min_periods=1, center=True).std().fillna(0.0)
    delta_mean = v - roll_mean          # deviation from local trend
    diff = v.diff().fillna(0.0)         # jump vs previous sample
    feats = np.column_stack([v.to_numpy(), delta_mean.to_numpy(),
                             roll_std.to_numpy(), diff.to_numpy()])
    return feats


def detect(component: str, sensors_dir: str = DEFAULT_SENSORS_DIR) -> AnomalyResult:
    """Fit IsolationForest on a component's series, return scored anomalies."""
    df = _load(component, sensors_dir)
    feats = _features(df["value"])

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )
    preds = model.fit_predict(feats)            # -1 anomaly, 1 normal
    # score_samples: higher = more normal. Negate so higher = more anomalous.
    raw = model.score_samples(feats)
    anomaly_score = -raw

    flagged = [int(i) for i in np.where(preds == -1)[0]]

    # Health: blend "how many flagged" with "how bad the worst margin is".
    frac_flagged = len(flagged) / len(df)
    # decision_function < 0 for anomalies; mean negative margin -> severity.
    margins = model.decision_function(feats)
    severity = float(np.clip(-margins[margins < 0].mean() if (margins < 0).any() else 0.0, 0.0, 1.0))
    health = float(np.clip(1.0 - (0.5 * frac_flagged * 10.0) - 0.3 * severity, 0.0, 1.0))

    return AnomalyResult(
        component=component,
        scores=[float(s) for s in anomaly_score],
        flagged=flagged,
        health=health,
    )


def evaluate(component: str, sensors_dir: str = DEFAULT_SENSORS_DIR) -> dict:
    """Compare flagged anomalies vs ground-truth `anomaly` column.

    Returns {precision, recall, f1, n_flagged, n_true}.
    """
    df = _load(component, sensors_dir)
    res = detect(component, sensors_dir)

    n = len(df)
    pred = np.zeros(n, dtype=int)
    pred[res.flagged] = 1
    truth = df["anomaly"].astype(int).to_numpy() if "anomaly" in df else np.zeros(n, dtype=int)

    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "n_flagged": int(pred.sum()),
        "n_true": int(truth.sum()),
    }


def plot_component(component: str, sensors_dir: str = DEFAULT_SENSORS_DIR):
    """Plotly figure: the series with detected anomalies highlighted.

    app.py can pass the returned figure straight to st.plotly_chart(...).
    """
    import plotly.graph_objects as go

    df = _load(component, sensors_dir)
    res = detect(component, sensors_dir)
    ts = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=df["value"], mode="lines", name=component,
        line=dict(color="#2c7fb8", width=1.5),
    ))
    if res.flagged:
        fig.add_trace(go.Scatter(
            x=ts.iloc[res.flagged], y=df["value"].iloc[res.flagged],
            mode="markers", name="anomaly",
            marker=dict(color="#d7301f", size=8, symbol="x"),
        ))
    fig.update_layout(
        title=f"{component} — health {res.health:.2f} ({len(res.flagged)} anomalies flagged)",
        xaxis_title="time", yaxis_title="value",
        template="plotly_white", margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def main() -> None:
    comps = list_components()
    if not comps:
        raise RuntimeError(
            "No sensor CSVs found in input/sensors. Run `python src/gen_sensors.py` first."
        )
    header = f"{'component':<20} {'health':>7} {'flagged':>8} {'true':>5} {'prec':>6} {'recall':>7} {'f1':>6}"
    print(header)
    print("-" * len(header))
    for c in comps:
        res = detect(c)
        ev = evaluate(c)
        print(
            f"{c:<20} {res.health:>7.2f} {ev['n_flagged']:>8} {ev['n_true']:>5} "
            f"{ev['precision']:>6.3f} {ev['recall']:>7.3f} {ev['f1']:>6.3f}"
        )


if __name__ == "__main__":
    main()

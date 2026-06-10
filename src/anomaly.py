"""Use case #3 (predictive maintenance) — anomaly detection on sensor series.

Higher data-risk per CLAUDE.md: only worth it with a strong synthetic generator
(injected degradation + labelled anomalies). Simple model = Isolation Forest.

Public API:
    list_components(sensors_dir="input/sensors") -> list[str]
    detect(component, sensors_dir="input/sensors", contamination=0.08) -> AnomalyResult
    evaluate(component, sensors_dir="input/sensors") -> dict
    fleet_summary(sensors_dir="input/sensors") -> pandas.DataFrame
    plot_component(component, sensors_dir="input/sensors") -> plotly Figure
    plot_fleet(sensors_dir="input/sensors") -> plotly Figure

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

# Health bands -> planner status. Worst (lowest health) first.
ALERT_BELOW = 0.55     # health < 0.55 -> "ALERT" (act now)
MONITOR_BELOW = 0.80   # 0.55 <= health < 0.80 -> "MONITOR"; >= 0.80 -> "OK"

# Per-sensor failure limits (engineering units) used to estimate RUL. Sourced
# from gen_sensors.FAILURE_THRESHOLDS so detection stays consistent with the
# synthetic generator; None = stable component (no wear limit, no RUL).
try:  # works whether run as `python src/anomaly.py` or imported as `src.anomaly`
    from gen_sensors import FAILURE_THRESHOLDS as _RAW_THRESHOLDS
except ImportError:  # pragma: no cover - import-path fallback
    from src.gen_sensors import FAILURE_THRESHOLDS as _RAW_THRESHOLDS
FAILURE_THRESHOLDS: dict[str, float | None] = dict(_RAW_THRESHOLDS)

# A trend is only "real degradation" if its per-step slope magnitude exceeds this
# (units/step). Below it, drift is noise and RUL is left undefined (None).
TREND_EPS = 1e-3


@dataclass
class AnomalyResult:
    component: str
    scores: list[float] = field(default_factory=list)   # per-timestamp anomaly score
    flagged: list[int] = field(default_factory=list)     # anomalous indices
    health: float = 1.0                                  # 0..1 component health
    trend: float = 0.0                                   # slope (units/step) of linear fit
    rul_cycles: float | None = None                      # est. steps until threshold; None if not degrading
    status: str = ""                                     # "OK" / "MONITOR" / "ALERT"


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


def _status_from_health(health: float) -> str:
    """Map a 0..1 health score to a planner-facing status band."""
    if health < ALERT_BELOW:
        return "ALERT"
    if health < MONITOR_BELOW:
        return "MONITOR"
    return "OK"


def _estimate_rul(component: str, smoothed: pd.Series, trend: float) -> float | None:
    """Rough remaining-useful-life: steps until the smoothed value crosses the
    per-sensor failure limit, extrapolating the current linear trend.

    Returns None when the component has no failure limit, is not meaningfully
    degrading, or is already trending *away* from its limit. This is a coarse
    "order of magnitude" indicator for planning, NOT a guaranteed time-to-failure:
    it assumes the present trend simply continues.
    """
    threshold = FAILURE_THRESHOLDS.get(component)
    if threshold is None or abs(trend) < TREND_EPS:
        return None

    current = float(smoothed.iloc[-1])
    gap = threshold - current  # distance still to cover to reach the limit

    # The component must be moving toward the limit (gap and trend same sign).
    if gap == 0:
        return 0.0
    if (gap > 0) != (trend > 0):
        return None  # improving / moving away from the limit -> no RUL

    rul = gap / trend
    return float(max(0.0, round(rul, 1)))


def detect(
    component: str,
    sensors_dir: str = DEFAULT_SENSORS_DIR,
    contamination: float = CONTAMINATION,
) -> AnomalyResult:
    """Fit IsolationForest on a component's series, return scored anomalies.

    `contamination` is the expected anomaly fraction (IsolationForest knob); the
    default reproduces the original behaviour. Also computes a degradation `trend`
    (slope per step), a planner `status`, and a rough `rul_cycles` estimate.
    """
    df = _load(component, sensors_dir)
    feats = _features(df["value"])

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
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

    # Degradation trend: slope of a linear fit (value ~ a*step + b), per step.
    # Fit on a smoothed series so injected spikes don't dominate the slope.
    values = df["value"].astype(float)
    smoothed = values.rolling(ROLL_WINDOW, min_periods=1, center=True).mean()
    steps = np.arange(len(smoothed), dtype=float)
    slope = float(np.polyfit(steps, smoothed.to_numpy(), 1)[0]) if len(smoothed) > 1 else 0.0

    status = _status_from_health(health)
    rul = _estimate_rul(component, smoothed, slope)

    return AnomalyResult(
        component=component,
        scores=[float(s) for s in anomaly_score],
        flagged=flagged,
        health=health,
        trend=round(slope, 6),
        rul_cycles=rul,
        status=status,
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


def fleet_summary(sensors_dir: str = DEFAULT_SENSORS_DIR) -> pd.DataFrame:
    """Fleet health + prognostics table — the headline planner view.

    One row per component, sorted worst-health first, with the detection health,
    planner status, anomaly count, degradation trend, rough RUL (steps to the
    failure limit), and detection recall/precision against ground truth.
    """
    rows = []
    for c in list_components(sensors_dir):
        res = detect(c, sensors_dir)
        ev = evaluate(c, sensors_dir)
        rows.append({
            "component": c,
            "health": round(res.health, 3),
            "status": res.status,
            "n_flagged": len(res.flagged),
            "trend": res.trend,
            "rul_cycles": res.rul_cycles,
            "recall": ev["recall"],
            "precision": ev["precision"],
        })

    cols = ["component", "health", "status", "n_flagged", "trend",
            "rul_cycles", "recall", "precision"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values("health", ascending=True).reset_index(drop=True)
    return df


def plot_fleet(sensors_dir: str = DEFAULT_SENSORS_DIR):
    """Plotly figure: fleet health overview as a horizontal bar chart.

    Bars are coloured by status (ALERT/MONITOR/OK) and annotated with RUL so a
    planner sees at a glance what to act on and roughly when. app.py can pass the
    returned figure straight to st.plotly_chart(...).
    """
    import plotly.graph_objects as go

    df = fleet_summary(sensors_dir)
    color_map = {"ALERT": "#d7301f", "MONITOR": "#fc8d59", "OK": "#1a9850"}

    # Worst at the top of the bar chart -> plot best-first so bars read top-down.
    df = df.sort_values("health", ascending=False)
    colors = [color_map.get(s, "#999999") for s in df["status"]]

    def _rul_label(r) -> str:
        if r["rul_cycles"] is None or (isinstance(r["rul_cycles"], float) and np.isnan(r["rul_cycles"])):
            return f"{r['status']}"
        return f"{r['status']} · RUL≈{r['rul_cycles']:.0f}"

    labels = [_rul_label(r) for _, r in df.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["health"], y=df["component"], orientation="h",
        marker=dict(color=colors), text=labels, textposition="outside",
        hovertemplate="%{y}<br>health=%{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=ALERT_BELOW, line=dict(color="#d7301f", dash="dot", width=1))
    fig.add_vline(x=MONITOR_BELOW, line=dict(color="#fc8d59", dash="dot", width=1))
    fig.update_layout(
        title="Fleet health — act-on / monitor / OK (RUL = est. steps to limit)",
        xaxis=dict(title="health (0..1)", range=[0, 1.05]),
        yaxis_title="component",
        template="plotly_white", margin=dict(l=40, r=80, t=50, b=40),
        showlegend=False,
    )
    return fig


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
    header = f"{'component':<20} {'health':>7} {'status':>8} {'flagged':>8} {'true':>5} {'prec':>6} {'recall':>7} {'f1':>6} {'trend':>10} {'rul':>8}"
    print(header)
    print("-" * len(header))
    for c in comps:
        res = detect(c)
        ev = evaluate(c)
        rul = "-" if res.rul_cycles is None else f"{res.rul_cycles:.1f}"
        print(
            f"{c:<20} {res.health:>7.2f} {res.status:>8} {ev['n_flagged']:>8} {ev['n_true']:>5} "
            f"{ev['precision']:>6.3f} {ev['recall']:>7.3f} {ev['f1']:>6.3f} "
            f"{res.trend:>10.5f} {rul:>8}"
        )

    print("\nFleet summary (worst health first):")
    summary = fleet_summary()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

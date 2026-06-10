"""Use case #3 (predictive maintenance) — anomaly detection on sensor series.

Higher data-risk per CLAUDE.md: only worth it with a strong synthetic generator
(injected degradation + labelled anomalies). Simple model = Isolation Forest.
"""
from dataclasses import dataclass, field


@dataclass
class AnomalyResult:
    component: str
    scores: list[float] = field(default_factory=list)   # per-timestamp anomaly score
    flagged: list[int] = field(default_factory=list)     # anomalous indices
    health: float = 1.0                                  # 0..1 component health


def detect(component: str):
    """Fit IsolationForest on a component's series, return scored anomalies.

    Sketch (wire after brief):
        from sklearn.ensemble import IsolationForest
        ...
    """
    raise NotImplementedError("Fill after brief: anomaly detection.")

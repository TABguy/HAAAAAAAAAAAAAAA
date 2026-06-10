"""Synthetic aircraft sensor time series generator (USE CASE #3 — predictive maintenance).

Produces, per component/sensor, a realistic time series with:
  - a healthy baseline + Gaussian noise,
  - for some components a slow DEGRADATION trend,
  - injected point (spike) and collective (drift) anomalies,
  - a ground-truth integer column `anomaly` (1 = injected, 0 = normal).

Outputs one CSV per component into input/sensors/<component>.csv with columns:
    timestamp, value, anomaly

Fully deterministic via --seed (np.random.default_rng) and a fixed base date —
no datetime.now() / unseeded random — so demos are reproducible.

Usage:
    python src/gen_sensors.py --points 500 --seed 42
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

INPUT_DIR = Path("input")
SENSORS_DIR = INPUT_DIR / "sensors"
BASE_DATE = datetime(2026, 1, 1)
SAMPLE_MINUTES = 60  # one sample per flight-hour-ish step

# Per-sensor profile. baseline/noise in engineering units. `degrade` is the total
# drift (in units) applied linearly across the whole series for components that
# wear over time; 0.0 = stable component. ATA chapter aligns to the maintenance docs.
SENSORS = {
    # APU exhaust gas temperature — ATA 49. Hot, slowly degrades (coking/wear).
    "APU_EGT": {"unit": "degC", "baseline": 480.0, "noise": 6.0, "degrade": 55.0, "ata": "49"},
    # Engine 1 oil pressure — ATA 71/79 (engine). Drops slowly as pumps/seals wear.
    "ENG1_oil_pressure": {"unit": "psi", "baseline": 65.0, "noise": 1.5, "degrade": -8.0, "ata": "71"},
    # Main landing gear brake temperature — ATA 32. Stable baseline, transient spikes.
    "MLG_brake_temp": {"unit": "degC", "baseline": 120.0, "noise": 8.0, "degrade": 0.0, "ata": "32"},
    # Green hydraulic system pressure — ATA 29. Stable, occasional pressure dips.
    "HYD_green_pressure": {"unit": "psi", "baseline": 3000.0, "noise": 25.0, "degrade": 0.0, "ata": "29"},
    # Integrated Drive Generator vibration — ATA 24. Rises slowly as bearings wear.
    "IDG_vibration": {"unit": "ips", "baseline": 1.2, "noise": 0.08, "degrade": 0.9, "ata": "24"},
}


def _make_series(rng: np.random.Generator, profile: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, anomaly_labels) for one sensor."""
    baseline = profile["baseline"]
    noise = profile["noise"]
    degrade = profile["degrade"]

    t = np.arange(n)
    # Healthy baseline + gentle sinusoidal duty cycle + Gaussian noise.
    duty = 0.4 * noise * np.sin(2 * np.pi * t / max(24, n / 8))
    values = baseline + duty + rng.normal(0.0, noise, n)

    # Slow degradation trend (ramps in over the back ~60% of the series).
    if degrade:
        ramp = np.clip((t - 0.4 * n) / (0.6 * n), 0.0, 1.0)
        values = values + degrade * ramp

    labels = np.zeros(n, dtype=int)

    # --- Injected POINT anomalies (sharp spikes) ---
    n_points = max(3, n // 80)
    pts = rng.choice(np.arange(n), size=n_points, replace=False)
    for idx in pts:
        sign = 1.0 if rng.random() < 0.7 else -1.0
        values[idx] += sign * rng.uniform(6.0, 11.0) * noise
        labels[idx] = 1

    # --- Injected COLLECTIVE anomalies (drift / step bursts) ---
    n_bursts = max(1, n // 250)
    for _ in range(n_bursts):
        length = int(rng.integers(6, 16))
        start = int(rng.integers(0, max(1, n - length)))
        sign = 1.0 if rng.random() < 0.6 else -1.0
        # ramp up to a sustained offset then back, like a developing fault.
        offset = sign * rng.uniform(4.0, 7.0) * noise
        shape = np.sin(np.linspace(0, np.pi, length))
        values[start:start + length] += offset * shape
        labels[start:start + length] = 1

    return values, labels


def write_component(name: str, timestamps: list[str], values: np.ndarray, labels: np.ndarray) -> None:
    SENSORS_DIR.mkdir(parents=True, exist_ok=True)
    path = SENSORS_DIR / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "anomaly"])
        for ts, v, a in zip(timestamps, values, labels):
            w.writerow([ts, f"{v:.4f}", int(a)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=500, help="samples per component")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.points
    timestamps = [
        (BASE_DATE + timedelta(minutes=SAMPLE_MINUTES * i)).strftime("%Y-%m-%d %H:%M")
        for i in range(n)
    ]

    print(f"Generating {len(SENSORS)} sensor series x {n} points (seed={args.seed})")
    for name, profile in SENSORS.items():
        values, labels = _make_series(rng, profile, n)
        write_component(name, timestamps, values, labels)
        trend = "degrading" if profile["degrade"] else "stable"
        print(
            f"  {name:<20} ATA{profile['ata']:<3} {trend:<9} "
            f"baseline={profile['baseline']:>7.1f} {profile['unit']:<4} "
            f"anomalies={int(labels.sum()):>4} -> input/sensors/{name}.csv"
        )

    total = n * len(SENSORS)
    print(f"Done. {len(SENSORS)} CSVs, {total} rows total in {SENSORS_DIR}/")


if __name__ == "__main__":
    main()

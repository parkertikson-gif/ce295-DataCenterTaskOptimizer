from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_HOURS = 168  # one week


def _diurnal_24h(seed: int) -> np.ndarray:
    """Single-day diurnal shape, mean=1.0, with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    # Peak mid-afternoon (hour 15), min pre-dawn (hour 3). Amplitude ~0.3.
    shape = 1.0 + 0.30 * np.sin(2 * np.pi * (hours - 9) / 24)
    noise = rng.normal(0.0, 0.03, size=24)
    return np.clip(shape + noise, 0.1, None)


def _week_profile(seed: int) -> np.ndarray:
    """168h profile: 24h diurnal tiled 7x with day-of-week multiplicative noise."""
    rng = np.random.default_rng(seed)
    base = _diurnal_24h(seed)
    # Day-of-week factor: weekdays slightly higher than weekends.
    dow = np.array([1.02, 1.03, 1.03, 1.02, 1.00, 0.94, 0.93])
    dow_noise = rng.normal(0.0, 0.02, size=7)
    factors = dow + dow_noise
    week = np.concatenate([base * factors[d] for d in range(7)])
    # Normalize so that the weekly mean equals 1.0.
    return week / week.mean()


def generate_demand_profile(config: dict) -> pd.DataFrame:
    """Return a DataFrame indexed by hour (0..167) with one column per task name.

    Each column's mean over the week equals share_of_demand * total_capacity_mw.
    """
    seed = int(config.get("seed", 42))
    total_capacity = float(config["total_capacity_mw"])
    shape = _week_profile(seed)  # mean=1.0

    data = {}
    for task in config["tasks"]:
        share = float(task["share_of_demand"])
        per_task_mean = share * total_capacity
        # Give each task a slightly different phase/noise so they aren't identical.
        task_seed = seed + hash(task["name"]) % 1000
        task_shape = _week_profile(task_seed)
        data[task["name"]] = task_shape * per_task_mean

    df = pd.DataFrame(data)
    df.index.name = "hour"
    return df


def write_demand_csv(demand_df: pd.DataFrame, out_path: str | Path) -> None:
    demand_df.to_csv(out_path)


def baseline_fractions(flexibility_hours: int) -> np.ndarray:
    """Flat schedule: spread demand evenly over [0, W] shift offsets."""
    w = int(flexibility_hours)
    return np.full(w + 1, 1.0 / (w + 1))


def baseline_power_matrix(
    demand_df: pd.DataFrame,
    config: dict,
) -> dict[str, np.ndarray]:
    """Baseline = tasks run at their release hour (no spreading).

    This is the realistic "no optimization" counterfactual: each task's
    power at hour t is simply its released demand at hour t.
    """
    return {
        task["name"]: demand_df[task["name"]].to_numpy(dtype=float)
        for task in config["tasks"]
    }

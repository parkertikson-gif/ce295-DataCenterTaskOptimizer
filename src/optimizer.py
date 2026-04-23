from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


HORIZON_HOURS = 168


@dataclass
class OptResult:
    alpha: float
    season: str
    power_by_task: dict[str, np.ndarray]   # inner 168h per task (MW)
    total_power: np.ndarray                # inner 168h (MW)
    lmp_inner: np.ndarray                  # inner 168h ($/MWh)
    moer_inner: np.ndarray                 # inner 168h (lb/MWh)
    lmp_norm_inner: np.ndarray
    moer_norm_inner: np.ndarray
    status: str


def tile_signal(signal_24: np.ndarray, length: int, start_offset: int = 0) -> np.ndarray:
    """Tile a 24h signal over `length` hours, aligning hour-of-day via start_offset."""
    hours = (np.arange(length) + start_offset) % 24
    return signal_24[hours]


def _max_window(config: dict) -> int:
    return max(int(t["flexibility_hours"]) for t in config["tasks"])


def solve(
    demand_df: pd.DataFrame,
    config: dict,
    lmp_24: np.ndarray,
    moer_24: np.ndarray,
    alpha: float,
    season: str,
) -> OptResult:
    """Solve the LP for a single (season, alpha).

    Padded horizon = HORIZON_HOURS + 2 * W_max; inner hours are
    reported. Signals are tiled from the 24h diurnal pattern.
    """
    n_inner = HORIZON_HOURS
    w_max = _max_window(config)
    h_pad = n_inner + 2 * w_max

    # The inner horizon starts at padded index w_max. Align signal hour-of-day
    # so that inner hour 0 corresponds to hour-of-day 0 of the 24h pattern.
    lmp_pad = tile_signal(lmp_24, h_pad, start_offset=-w_max)
    moer_pad = tile_signal(moer_24, h_pad, start_offset=-w_max)
    lmp_norm = lmp_pad / max(lmp_24.max(), 1e-9)
    moer_norm = moer_pad / max(moer_24.max(), 1e-9)

    total_capacity = float(config["total_capacity_mw"])
    p_max = float(config.get("peak_multiplier", 1.15)) * total_capacity

    # Build per-task variables and accumulate padded power expression.
    task_names = [t["name"] for t in config["tasks"]]
    variables: dict[str, cp.Variable] = {}
    per_task_power_padded: dict[str, cp.Expression] = {}
    constraints: list = []
    p_padded = cp.Constant(np.zeros(h_pad))

    for task in config["tasks"]:
        name = task["name"]
        w = int(task["flexibility_hours"])
        d_k = demand_df[name].to_numpy(dtype=float)  # (168,)
        X = cp.Variable((n_inner, w + 1), nonneg=True, name=f"X_{name}")
        variables[name] = X
        constraints.append(cp.sum(X, axis=1) == 1)

        # Energy conservation: forbid offsets that would push dispatch past
        # the inner horizon. For release hour t, only s <= n_inner-1-t is valid.
        if w > 0:
            valid_mask = np.zeros((n_inner, w + 1))
            for t in range(n_inner):
                max_s = min(w, n_inner - 1 - t)
                valid_mask[t, : max_s + 1] = 1.0
            invalid_mask = 1.0 - valid_mask
            if invalid_mask.any():
                constraints.append(cp.multiply(invalid_mask, X) == 0)

        # Accumulate power at padded hours.
        task_power = cp.Constant(np.zeros(h_pad))
        for s in range(w + 1):
            contribution = cp.multiply(d_k, X[:, s])  # (168,)
            left_pad = w_max + s
            right_pad = h_pad - left_pad - n_inner
            pieces: list = []
            if left_pad > 0:
                pieces.append(np.zeros(left_pad))
            pieces.append(contribution)
            if right_pad > 0:
                pieces.append(np.zeros(right_pad))
            padded = cp.hstack(pieces) if len(pieces) > 1 else pieces[0]
            task_power = task_power + padded
        per_task_power_padded[name] = task_power
        p_padded = p_padded + task_power

        # Per-task hourly cap (full padded horizon).
        share = float(task["share_of_demand"])
        mult = float(task.get("max_power_multiplier", 2.0))
        per_task_cap = mult * share * total_capacity
        constraints.append(task_power <= per_task_cap)

    constraints.append(p_padded <= p_max)

    objective = cp.Minimize(
        alpha * cp.sum(cp.multiply(lmp_norm, p_padded))
        + (1.0 - alpha) * cp.sum(cp.multiply(moer_norm, p_padded))
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    # Extract inner 168h results.
    inner = slice(w_max, w_max + n_inner)
    power_by_task = {}
    for name in task_names:
        vals = per_task_power_padded[name].value
        power_by_task[name] = np.asarray(vals[inner]).flatten()
    total_power = np.asarray(p_padded.value[inner]).flatten()

    return OptResult(
        alpha=alpha,
        season=season,
        power_by_task=power_by_task,
        total_power=total_power,
        lmp_inner=lmp_pad[inner],
        moer_inner=moer_pad[inner],
        lmp_norm_inner=lmp_norm[inner],
        moer_norm_inner=moer_norm[inner],
        status=prob.status,
    )

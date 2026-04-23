from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    ensure_grid_data,
    load_task_config,
    season_24h,
)
from src.workload import (
    baseline_power_matrix,
    generate_demand_profile,
    write_demand_csv,
)
from src.optimizer import solve, tile_signal
from src.visualize import (
    plot_demand_before_after,
    plot_overlay_48h,
    plot_pareto,
    plot_signals,
    plot_stacked_48h,
    plot_total_overlay,
    SEASON_ORDER,
)


ALPHAS = [round(x, 1) for x in np.arange(0.0, 1.0001, 0.1)]
REPORT_ALPHA = 0.5


def normalized_totals(power_inner: np.ndarray, lmp_norm_inner: np.ndarray, moer_norm_inner: np.ndarray) -> tuple[float, float]:
    cost_norm = float(np.sum(lmp_norm_inner * power_inner))
    carbon_norm = float(np.sum(moer_norm_inner * power_inner))
    return cost_norm, carbon_norm


def main() -> None:
    inputs_dir = ROOT / "inputs"
    outputs_dir = ROOT / "outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = load_task_config(inputs_dir / "task_config.yaml")
    lmp_df, moer_df = ensure_grid_data(inputs_dir)

    demand_df = generate_demand_profile(config)
    write_demand_csv(demand_df, inputs_dir / "demand_profile.csv")

    baseline_by_task = baseline_power_matrix(demand_df, config)
    baseline_total = np.sum(list(baseline_by_task.values()), axis=0)

    w_max = max(int(t["flexibility_hours"]) for t in config["tasks"])
    h_pad = 168 + 2 * w_max

    pareto_rows = []
    schedule_rows = []
    kpi_rows = []
    baseline_power_by_season: dict[str, dict[str, np.ndarray]] = {}
    optimized_power_by_season: dict[str, dict[str, np.ndarray]] = {}
    kpi_at_report_alpha: dict[str, dict] = {}

    for season in SEASON_ORDER:
        lmp_24 = season_24h(lmp_df, season)
        moer_24 = season_24h(moer_df, season)

        lmp_pad = tile_signal(lmp_24, h_pad, start_offset=-w_max)
        moer_pad = tile_signal(moer_24, h_pad, start_offset=-w_max)
        lmp_inner = lmp_pad[w_max:w_max + 168]
        moer_inner = moer_pad[w_max:w_max + 168]
        lmp_norm_inner = lmp_inner / max(lmp_24.max(), 1e-9)
        moer_norm_inner = moer_inner / max(moer_24.max(), 1e-9)

        cost_norm_base, carbon_norm_base = normalized_totals(
            baseline_total, lmp_norm_inner, moer_norm_inner
        )

        baseline_power_by_season[season] = baseline_by_task
        print(f"[{season}] baseline cost_norm={cost_norm_base:.2f}  carbon_norm={carbon_norm_base:.2f}")

        # Real-unit baseline KPIs (same for every α within a season).
        cost_base_usd = float(np.sum(baseline_total * lmp_inner))
        carbon_base_lbs = float(np.sum(baseline_total * moer_inner))

        for alpha in ALPHAS:
            res = solve(demand_df, config, lmp_24, moer_24, alpha, season)
            cost_norm_opt, carbon_norm_opt = normalized_totals(
                res.total_power, lmp_norm_inner, moer_norm_inner
            )
            pareto_rows.append({
                "season": season,
                "alpha": alpha,
                "total_cost_norm_baseline": cost_norm_base,
                "total_cost_norm_optimized": cost_norm_opt,
                "total_carbon_norm_baseline": carbon_norm_base,
                "total_carbon_norm_optimized": carbon_norm_opt,
                "solver_status": res.status,
            })
            print(
                f"  α={alpha:.1f}  status={res.status:10s}  "
                f"cost_norm={cost_norm_opt:.2f}  carbon_norm={carbon_norm_opt:.2f}"
            )

            cost_opt_usd = float(np.sum(res.total_power * lmp_inner))
            carbon_opt_lbs = float(np.sum(res.total_power * moer_inner))
            cost_sav_usd = cost_base_usd - cost_opt_usd
            carbon_sav_lbs = carbon_base_lbs - carbon_opt_lbs
            kpi = {
                "season": season,
                "alpha": alpha,
                "cost_baseline_usd": cost_base_usd,
                "cost_optimized_usd": cost_opt_usd,
                "cost_savings_usd": cost_sav_usd,
                "cost_savings_pct": 100.0 * cost_sav_usd / cost_base_usd if cost_base_usd else 0.0,
                "cost_savings_annualized_usd": cost_sav_usd * 52,
                "carbon_baseline_lbs": carbon_base_lbs,
                "carbon_optimized_lbs": carbon_opt_lbs,
                "carbon_savings_lbs": carbon_sav_lbs,
                "carbon_savings_pct": 100.0 * carbon_sav_lbs / carbon_base_lbs if carbon_base_lbs else 0.0,
                "carbon_savings_tons": carbon_sav_lbs / 2000.0,
                "carbon_savings_annualized_tons": (carbon_sav_lbs / 2000.0) * 52,
            }
            kpi_rows.append(kpi)

            if abs(alpha - REPORT_ALPHA) < 1e-6:
                kpi_at_report_alpha[season] = kpi
                optimized_power_by_season[season] = res.power_by_task
                for t in range(168):
                    for task in config["tasks"]:
                        name = task["name"]
                        p_base = baseline_by_task[name][t]
                        p_opt = res.power_by_task[name][t]
                        schedule_rows.append({
                            "hour": t,
                            "season": season,
                            "task_name": name,
                            "power_mw_baseline": p_base,
                            "power_mw_optimized": p_opt,
                            "lmp": lmp_inner[t],
                            "moer": moer_inner[t],
                            "cost_baseline": p_base * lmp_inner[t],
                            "cost_optimized": p_opt * lmp_inner[t],
                            "carbon_baseline": p_base * moer_inner[t],
                            "carbon_optimized": p_opt * moer_inner[t],
                        })

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.to_csv(outputs_dir / "pareto_frontier.csv", index=False)

    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df.to_csv(outputs_dir / "schedule.csv", index=False)

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(outputs_dir / "kpi_summary.csv", index=False)

    # Per-task hourly cap verification.
    total_capacity = float(config["total_capacity_mw"])
    max_ratio = 0.0
    for task in config["tasks"]:
        name = task["name"]
        mult = float(task.get("max_power_multiplier", 2.0))
        share = float(task["share_of_demand"])
        cap = mult * share * total_capacity
        task_rows = schedule_df[schedule_df["task_name"] == name]
        observed = task_rows["power_mw_optimized"].max()
        ratio = observed / (share * total_capacity) if share > 0 else 0.0
        max_ratio = max(max_ratio, ratio)
        ok = "OK" if observed <= cap + 1e-6 else "FAIL"
        print(f"[cap] {name:20s} cap={cap:7.2f} MW  observed_max={observed:7.2f} MW  ratio={ratio:.3f}x  {ok}")
    print(f"[cap] max ratio over all tasks: {max_ratio:.3f}x  (limit = 2.000x)")

    print("\n=== KPI Summary (α=0.5, one week) ===")
    for season in SEASON_ORDER:
        k = kpi_at_report_alpha[season]
        print(
            f"{season.capitalize():9s} "
            f"cost saved {k['cost_savings_pct']:5.2f}%  "
            f"(${k['cost_savings_usd']:,.0f}/week, ${k['cost_savings_annualized_usd']:,.0f}/yr)  |  "
            f"carbon avoided {k['carbon_savings_pct']:5.2f}%  "
            f"({k['carbon_savings_tons']:,.1f} tons/week, {k['carbon_savings_annualized_tons']:,.1f} tons/yr)"
        )

    plot_signals(lmp_df, moer_df, plots_dir / "signals.png")
    plot_pareto(pareto_df, plots_dir / "pareto_frontier.png")
    plot_demand_before_after(
        baseline_power_by_season, optimized_power_by_season, config,
        plots_dir / "demand_before_after.png",
    )
    plot_total_overlay(
        baseline_power_by_season, optimized_power_by_season,
        plots_dir / "demand_total_overlay.png",
    )
    plot_stacked_48h(
        baseline_power_by_season, optimized_power_by_season, config,
        plots_dir / "demand_stacked_48h.png",
    )
    plot_overlay_48h(
        baseline_power_by_season, optimized_power_by_season,
        plots_dir / "demand_overlay_48h.png",
    )

    print(f"\nWrote: {outputs_dir/'schedule.csv'}")
    print(f"Wrote: {outputs_dir/'pareto_frontier.csv'}")
    print(f"Wrote: {outputs_dir/'kpi_summary.csv'}")
    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()

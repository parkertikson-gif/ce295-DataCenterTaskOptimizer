from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEASON_ORDER = ["summer", "winter", "shoulder"]


def _task_colors(config: dict) -> dict[str, str]:
    return {t["name"]: t["color"] for t in config["tasks"]}


def plot_demand_before_after(
    baseline_by_season: dict[str, dict[str, np.ndarray]],
    optimized_by_season: dict[str, dict[str, np.ndarray]],
    config: dict,
    out_path: str | Path,
) -> None:
    colors = _task_colors(config)
    names = [t["name"] for t in config["tasks"]]
    hours = np.arange(168)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True, sharex=True)
    for col, season in enumerate(SEASON_ORDER):
        for row, (label, bundle) in enumerate(
            [("Baseline", baseline_by_season[season]), ("Optimized", optimized_by_season[season])]
        ):
            ax = axes[row, col]
            stacks = np.vstack([bundle[n] for n in names])
            ax.stackplot(
                hours, stacks, labels=names,
                colors=[colors[n] for n in names], alpha=0.85,
            )
            ax.set_title(f"{season.capitalize()} — {label}")
            ax.set_xlim(0, 167)
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("Hour of week")
            if col == 0:
                ax.set_ylabel("Power (MW)")
    axes[0, -1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Data center demand: baseline vs. optimized (α=0.5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_total_overlay(
    baseline_by_season: dict[str, dict[str, np.ndarray]],
    optimized_by_season: dict[str, dict[str, np.ndarray]],
    out_path: str | Path,
) -> None:
    hours = np.arange(168)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for col, season in enumerate(SEASON_ORDER):
        ax = axes[col]
        base_total = np.sum(list(baseline_by_season[season].values()), axis=0)
        opt_total = np.sum(list(optimized_by_season[season].values()), axis=0)
        ax.fill_between(hours, base_total, color="#b0b0b0", alpha=0.85, label="Baseline")
        ax.fill_between(hours, opt_total, color="#1f77b4", alpha=0.55, label="Optimized (α=0.5)")
        ax.set_title(season.capitalize())
        ax.set_xlabel("Hour of week")
        ax.set_xlim(0, 167)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Total power (MW)")
    axes[-1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Total demand: baseline vs. optimized (α=0.5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _apply_48h_xaxis(ax, start: int) -> None:
    # Show hour-of-day (0–23) repeated; vertical dashed line at day boundary.
    hours = np.arange(start, start + 48)
    ax.set_xlim(start, start + 47)
    tick_positions = [start + k for k in (0, 6, 12, 18, 24, 30, 36, 42)]
    tick_labels = [f"{((start + k) % 24):02d}" for k in (0, 6, 12, 18, 24, 30, 36, 42)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.axvline(start + 24, color="black", linestyle="--", linewidth=1.0, alpha=0.6)


def plot_stacked_48h(
    baseline_by_season: dict[str, dict[str, np.ndarray]],
    optimized_by_season: dict[str, dict[str, np.ndarray]],
    config: dict,
    out_path: str | Path,
    window: tuple[int, int] = (72, 120),
) -> None:
    colors = _task_colors(config)
    names = [t["name"] for t in config["tasks"]]
    start, end = window
    hours = np.arange(start, end)

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True, sharex=True)
    for col, season in enumerate(SEASON_ORDER):
        for row, (label, bundle) in enumerate(
            [("Baseline", baseline_by_season[season]), ("Optimized", optimized_by_season[season])]
        ):
            ax = axes[row, col]
            stacks = np.vstack([bundle[n][start:end] for n in names])
            ax.stackplot(
                hours, stacks, labels=names,
                colors=[colors[n] for n in names], alpha=0.85,
            )
            ax.set_title(f"{season.capitalize()} — {label}")
            _apply_48h_xaxis(ax, start)
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("Hour of day")
            if col == 0:
                ax.set_ylabel("Power (MW)")
    axes[0, -1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Stacked demand, hours {start}–{end - 1}: baseline vs. optimized (α=0.5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlay_48h(
    baseline_by_season: dict[str, dict[str, np.ndarray]],
    optimized_by_season: dict[str, dict[str, np.ndarray]],
    out_path: str | Path,
    window: tuple[int, int] = (72, 120),
) -> None:
    start, end = window
    hours = np.arange(start, end)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for col, season in enumerate(SEASON_ORDER):
        ax = axes[col]
        base_total = np.sum(list(baseline_by_season[season].values()), axis=0)[start:end]
        opt_total = np.sum(list(optimized_by_season[season].values()), axis=0)[start:end]
        ax.fill_between(hours, base_total, color="#b0b0b0", alpha=0.85, label="Baseline")
        ax.fill_between(hours, opt_total, color="#1f77b4", alpha=0.55, label="Optimized (α=0.5)")
        ax.set_title(season.capitalize())
        ax.set_xlabel("Hour of day")
        _apply_48h_xaxis(ax, start)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Total power (MW)")
    axes[-1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(f"Total demand, hours {start}–{end - 1}: baseline vs. optimized (α=0.5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signals(
    lmp_df: pd.DataFrame,
    moer_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    season_colors = {"summer": "#e74c3c", "winter": "#2980b9", "shoulder": "#27ae60"}

    for season in SEASON_ORDER:
        sub = lmp_df[lmp_df["season"] == season].sort_values("hour")
        axes[0].plot(sub["hour"], sub["value"], label=season, color=season_colors[season], linewidth=2)
        sub = moer_df[moer_df["season"] == season].sort_values("hour")
        axes[1].plot(sub["hour"], sub["value"], label=season, color=season_colors[season], linewidth=2)

    axes[0].set_title("CAISO NP15 Day-Ahead LMP")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("LMP ($/MWh)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("CAISO Grid Carbon Intensity (MOER)")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("MOER (lb CO₂ / MWh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(
    pareto_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    season_colors = {"summer": "#e74c3c", "winter": "#2980b9", "shoulder": "#27ae60"}

    for season in SEASON_ORDER:
        sub = pareto_df[pareto_df["season"] == season].sort_values("alpha")
        ax.plot(
            sub["total_cost_norm_optimized"],
            sub["total_carbon_norm_optimized"],
            marker="o", label=f"{season} (swept α)",
            color=season_colors[season], linewidth=2,
        )
        base_row = sub.iloc[0]
        ax.scatter(
            base_row["total_cost_norm_baseline"],
            base_row["total_carbon_norm_baseline"],
            marker="*", s=220, edgecolor="black", linewidth=1.0,
            color=season_colors[season], label=f"{season} baseline", zorder=5,
        )

    ax.set_xlabel("Normalized cost  Σ λ̃·P  (unitless)")
    ax.set_ylabel("Normalized carbon  Σ m̃·P  (unitless)")
    ax.set_title("Cost–carbon Pareto frontier by season")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

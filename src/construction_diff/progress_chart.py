"""Matplotlib charts for 4D construction progress visualisation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from construction_diff.timeline import ProgressReport


def plot_progress(report: ProgressReport, output_path: str | Path) -> None:
    """Generate a multi-panel construction progress chart.

    The chart contains three panels:

    1. **Cumulative new points** -- line chart showing total new construction
       points over time (proxy for construction progress).
    2. **New / Removed per step** -- bar chart showing the per-step change
       counts.
    3. **Progress rate** -- trend line of new points per hour.

    Parameters
    ----------
    report:
        A :class:`ProgressReport` produced by
        :meth:`ConstructionTimeline.compute_progress`.
    output_path:
        File path for the saved chart image (e.g. ``progress.png``).
    """
    output_path = Path(output_path)

    # Parse timestamps.
    all_ts = [datetime.fromisoformat(t) for t in report.timestamps]
    # Step timestamps are the midpoints or simply the "to" timestamps.
    step_ts = all_ts[1:]
    n_steps = len(step_ts)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Panel 1: Cumulative new points (construction progress proxy) ---
    ax1 = axes[0]
    ax1.plot(
        step_ts,
        report.cumulative_new,
        marker="o",
        linewidth=2,
        color="#2196F3",
        label="Cumulative new points",
    )
    ax1.fill_between(step_ts, report.cumulative_new, alpha=0.15, color="#2196F3")
    ax1.set_ylabel("Cumulative New Points")
    ax1.set_title("Construction Progress Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Per-step new / removed bar chart ---
    ax2 = axes[1]
    bar_width_days = 0.35
    if n_steps > 1:
        # Compute bar width as fraction of average step interval.
        avg_interval = (step_ts[-1] - step_ts[0]).total_seconds() / (n_steps - 1)
        from matplotlib.dates import date2num

        bar_width = avg_interval / 86400.0 * 0.3  # 30% of interval in days
    else:
        bar_width = 1.0

    from matplotlib.dates import date2num

    step_nums = date2num(step_ts)
    offset = bar_width / 2

    bars_new = ax2.bar(
        [s - offset for s in step_nums],
        report.new_points_per_step,
        width=bar_width,
        color="#4CAF50",
        label="New (constructed)",
        align="center",
    )
    bars_removed = ax2.bar(
        [s + offset for s in step_nums],
        report.removed_per_step,
        width=bar_width,
        color="#F44336",
        label="Removed",
        align="center",
    )
    ax2.xaxis_date()
    ax2.set_ylabel("Points per Step")
    ax2.set_title("Per-Step Changes")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Progress rate trend ---
    ax3 = axes[2]
    ax3.plot(
        step_ts,
        report.progress_rate,
        marker="s",
        linewidth=2,
        color="#FF9800",
        label="Progress rate",
    )
    # Add trend line if enough points.
    if n_steps >= 2:
        x_numeric = np.array([(t - step_ts[0]).total_seconds() for t in step_ts])
        coeffs = np.polyfit(x_numeric, report.progress_rate, 1)
        trend = np.polyval(coeffs, x_numeric)
        ax3.plot(step_ts, trend, "--", color="#E65100", alpha=0.7, label="Trend")
    ax3.set_ylabel("New Points / Hour")
    ax3.set_xlabel("Time")
    ax3.set_title("Progress Rate Trend")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("4D Construction Monitoring", fontsize=14, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

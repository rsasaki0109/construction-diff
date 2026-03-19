"""Tests for progress chart generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from construction_diff.timeline import ProgressReport
from construction_diff.progress_chart import plot_progress


def _sample_report() -> ProgressReport:
    return ProgressReport(
        timestamps=[
            "2024-01-15T00:00:00",
            "2024-02-15T00:00:00",
            "2024-03-15T00:00:00",
        ],
        new_points_per_step=[1000, 2000],
        removed_per_step=[100, 200],
        cumulative_new=[1000, 3000],
        progress_rate=[1.34, 2.76],
    )


def test_plot_progress_creates_png(tmp_path: Path) -> None:
    report = _sample_report()
    out = tmp_path / "progress.png"
    plot_progress(report, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_progress_creates_pdf(tmp_path: Path) -> None:
    report = _sample_report()
    out = tmp_path / "progress.pdf"
    plot_progress(report, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_progress_single_step(tmp_path: Path) -> None:
    """Chart should work with a single step (2 timestamps)."""
    report = ProgressReport(
        timestamps=["2024-01-01T00:00:00", "2024-02-01T00:00:00"],
        new_points_per_step=[500],
        removed_per_step=[50],
        cumulative_new=[500],
        progress_rate=[0.67],
    )
    out = tmp_path / "single.png"
    plot_progress(report, out)
    assert out.exists()


def test_plot_progress_many_steps(tmp_path: Path) -> None:
    """Chart should work with many steps."""
    n = 10
    report = ProgressReport(
        timestamps=[f"2024-{i+1:02d}-01T00:00:00" for i in range(n + 1)],
        new_points_per_step=[100 * (i + 1) for i in range(n)],
        removed_per_step=[10 * (i + 1) for i in range(n)],
        cumulative_new=[sum(100 * (j + 1) for j in range(i + 1)) for i in range(n)],
        progress_rate=[0.5 + 0.1 * i for i in range(n)],
    )
    out = tmp_path / "many.png"
    plot_progress(report, out)
    assert out.exists()

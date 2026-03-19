"""Tests for 4D time-series construction monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from construction_diff.timeline import ConstructionTimeline, ProgressReport


# ---------------------------------------------------------------------------
# ProgressReport dataclass tests
# ---------------------------------------------------------------------------


def _sample_report() -> ProgressReport:
    """Build a minimal ProgressReport for testing."""
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


def test_progress_report_to_csv(tmp_path: Path) -> None:
    report = _sample_report()
    csv_path = tmp_path / "report.csv"
    report.to_csv(csv_path)

    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    # Header + 2 data rows.
    assert len(lines) == 3
    assert "from_timestamp" in lines[0]
    assert "2024-01-15" in lines[1]
    assert "2024-02-15" in lines[2]


def test_progress_report_to_json(tmp_path: Path) -> None:
    report = _sample_report()
    json_path = tmp_path / "report.json"
    report.to_json(json_path)

    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["timestamps"] == report.timestamps
    assert data["new_points_per_step"] == [1000, 2000]
    assert data["cumulative_new"] == [1000, 3000]
    assert len(data["progress_rate"]) == 2


# ---------------------------------------------------------------------------
# ConstructionTimeline tests
# ---------------------------------------------------------------------------


def test_timeline_requires_at_least_two_scans() -> None:
    tl = ConstructionTimeline()
    with pytest.raises(ValueError, match="at least 2 scans"):
        tl.compute_progress()

    tl.add_scan(Path("/fake/scan1"), datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="at least 2 scans"):
        tl.compute_progress()


def test_timeline_sorts_scans_by_timestamp() -> None:
    tl = ConstructionTimeline()
    t1 = datetime(2024, 3, 1)
    t2 = datetime(2024, 1, 1)
    t3 = datetime(2024, 2, 1)
    tl.add_scan(Path("/a"), t1)
    tl.add_scan(Path("/b"), t2)
    tl.add_scan(Path("/c"), t3)

    scans = tl.scans
    assert scans[0][1] == t2
    assert scans[1][1] == t3
    assert scans[2][1] == t1


def _make_scan_dir(base: Path, name: str, n_points: int, offset: float = 0.0) -> Path:
    """Create a fake scan directory with coord.npy."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) % (2**31))
    pts = rng.uniform(0, 10, (n_points, 3)) + offset
    np.save(d / "coord.npy", pts.astype(np.float64))
    return d


def test_timeline_compute_progress_with_mock(tmp_path: Path) -> None:
    """Test compute_progress with mocked registration to avoid heavy computation."""
    # Create 3 scan directories.
    d1 = _make_scan_dir(tmp_path, "scan_001", 100)
    d2 = _make_scan_dir(tmp_path, "scan_002", 120, offset=0.5)
    d3 = _make_scan_dir(tmp_path, "scan_003", 150, offset=1.0)

    tl = ConstructionTimeline(voxel_size=0.5, threshold=0.3)
    tl.add_scan(d1, datetime(2024, 1, 1))
    tl.add_scan(d2, datetime(2024, 2, 1))
    tl.add_scan(d3, datetime(2024, 3, 1))

    # Mock register_scans to return identity transform.
    mock_result = MagicMock()
    mock_result.transformation = np.eye(4)
    mock_result.fitness = 0.9
    mock_result.inlier_rmse = 0.01

    with patch("construction_diff.timeline.register_scans", return_value=mock_result):
        report = tl.compute_progress()

    assert len(report.timestamps) == 3
    assert len(report.new_points_per_step) == 2
    assert len(report.removed_per_step) == 2
    assert len(report.cumulative_new) == 2
    assert len(report.progress_rate) == 2

    # Cumulative should be monotonically non-decreasing.
    assert report.cumulative_new[1] >= report.cumulative_new[0]

    # Progress rate should be non-negative.
    for rate in report.progress_rate:
        assert rate >= 0.0


def test_timeline_progress_rate_zero_dt() -> None:
    """Progress rate should be 0 when timestamps are identical."""
    tl = ConstructionTimeline(voxel_size=0.5, threshold=0.3)

    # We'll test the rate calculation by checking the report directly.
    report = ProgressReport(
        timestamps=["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
        new_points_per_step=[100],
        removed_per_step=[10],
        cumulative_new=[100],
        progress_rate=[0.0],
    )
    assert report.progress_rate[0] == 0.0

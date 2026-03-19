"""4D time-series construction monitoring from sequential point cloud scans.

Unlike CloudCompare M3C2 which only performs pairwise comparison, this module
tracks construction progress across an arbitrary number of epochs, computing
cumulative metrics and progress rates over time.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from construction_diff.loader import load_scan
from construction_diff.registration import register_scans
from construction_diff.diff import compute_diff

logger = logging.getLogger(__name__)


@dataclass
class ProgressReport:
    """Summary of construction progress across multiple scan epochs.

    Attributes
    ----------
    timestamps : list[str]
        ISO-format timestamp for each scan epoch.
    new_points_per_step : list[int]
        Number of new (constructed) points detected at each consecutive step.
        Length is ``len(timestamps) - 1``.
    removed_per_step : list[int]
        Number of removed (demolished/occluded) points at each step.
    cumulative_new : list[int]
        Running total of new points across all steps.
    progress_rate : list[float]
        New points per hour for each step (0.0 if timestamps are identical).
    """

    timestamps: list[str] = field(default_factory=list)
    new_points_per_step: list[int] = field(default_factory=list)
    removed_per_step: list[int] = field(default_factory=list)
    cumulative_new: list[int] = field(default_factory=list)
    progress_rate: list[float] = field(default_factory=list)

    def to_csv(self, output_path: str | Path) -> None:
        """Export the timeline report as a CSV file."""
        output_path = Path(output_path)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "from_timestamp",
                "to_timestamp",
                "new_points",
                "removed_points",
                "cumulative_new",
                "progress_rate_per_hour",
            ])
            for i in range(len(self.new_points_per_step)):
                writer.writerow([
                    i + 1,
                    self.timestamps[i],
                    self.timestamps[i + 1],
                    self.new_points_per_step[i],
                    self.removed_per_step[i],
                    self.cumulative_new[i],
                    f"{self.progress_rate[i]:.2f}",
                ])

    def to_json(self, output_path: str | Path) -> None:
        """Export the timeline report as a JSON file with all metrics."""
        output_path = Path(output_path)
        data = asdict(self)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


class ConstructionTimeline:
    """Track construction progress across multiple scan epochs.

    Usage::

        timeline = ConstructionTimeline()
        timeline.add_scan("/data/scan_001", datetime(2024, 1, 15))
        timeline.add_scan("/data/scan_002", datetime(2024, 2, 15))
        timeline.add_scan("/data/scan_003", datetime(2024, 3, 15))
        report = timeline.compute_progress()
        report.to_json("progress.json")
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.05,
        threshold: float = 0.10,
    ) -> None:
        self._scans: list[tuple[Path, datetime]] = []
        self.voxel_size = voxel_size
        self.threshold = threshold

    def add_scan(self, scan_path: str | Path, timestamp: datetime) -> None:
        """Register a scan with its capture timestamp."""
        self._scans.append((Path(scan_path), timestamp))

    @property
    def scans(self) -> list[tuple[Path, datetime]]:
        """Return scans sorted by timestamp."""
        return sorted(self._scans, key=lambda x: x[1])

    def compute_progress(self) -> ProgressReport:
        """Compute pairwise diffs for consecutive scans and build a report.

        Returns
        -------
        ProgressReport
            Aggregated progress metrics across all epochs.

        Raises
        ------
        ValueError
            If fewer than 2 scans have been registered.
        """
        sorted_scans = self.scans
        if len(sorted_scans) < 2:
            raise ValueError(
                f"Need at least 2 scans to compute progress, got {len(sorted_scans)}"
            )

        timestamps = [s[1].isoformat() for s in sorted_scans]
        new_points_per_step: list[int] = []
        removed_per_step: list[int] = []
        cumulative_new: list[int] = []
        progress_rate: list[float] = []

        running_total = 0

        for i in range(len(sorted_scans) - 1):
            src_path, src_time = sorted_scans[i]
            tgt_path, tgt_time = sorted_scans[i + 1]

            logger.info(
                "Step %d/%d: %s -> %s",
                i + 1,
                len(sorted_scans) - 1,
                src_path.name,
                tgt_path.name,
            )

            src_pcd = load_scan(src_path)
            tgt_pcd = load_scan(tgt_path)

            # Register source to target.
            reg = register_scans(src_pcd, tgt_pcd, voxel_size=self.voxel_size)
            logger.info(
                "  Registration fitness=%.4f rmse=%.6f",
                reg.fitness,
                reg.inlier_rmse,
            )

            # Compute diff.
            diff = compute_diff(
                src_pcd, tgt_pcd, reg.transformation, threshold=self.threshold
            )

            n_new = diff["n_new"]
            n_removed = diff["n_removed"]
            new_points_per_step.append(n_new)
            removed_per_step.append(n_removed)

            running_total += n_new
            cumulative_new.append(running_total)

            # Progress rate: new points per hour.
            dt_hours = (tgt_time - src_time).total_seconds() / 3600.0
            rate = n_new / dt_hours if dt_hours > 0 else 0.0
            progress_rate.append(rate)

            logger.info(
                "  New=%d, Removed=%d, Cumulative=%d, Rate=%.1f pts/hr",
                n_new,
                n_removed,
                running_total,
                rate,
            )

        return ProgressReport(
            timestamps=timestamps,
            new_points_per_step=new_points_per_step,
            removed_per_step=removed_per_step,
            cumulative_new=cumulative_new,
            progress_rate=progress_rate,
        )

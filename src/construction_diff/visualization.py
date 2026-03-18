"""Visualize diff results with colour-coded point clouds."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# Colour scheme: RGB in [0, 1].
COLOR_NEW = np.array([0.0, 0.8, 0.0])       # green
COLOR_REMOVED = np.array([0.9, 0.0, 0.0])   # red
COLOR_UNCHANGED = np.array([0.6, 0.6, 0.6]) # gray


def build_diff_cloud(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    diff_result: dict[str, Any],
    transformation: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Build a single colour-coded point cloud from diff results.

    - Green:  new points (present in target only).
    - Red:    removed points (present in source only).
    - Grey:   unchanged points.
    """
    src_aligned = copy.deepcopy(source)
    src_aligned.transform(transformation)

    src_pts = np.asarray(src_aligned.points)
    tgt_pts = np.asarray(target.points)

    new_idx = diff_result["new_indices"]
    removed_idx = diff_result["removed_indices"]
    unchanged_tgt = diff_result["unchanged_target"]

    # Assemble arrays.
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    # New points (target only).
    if len(new_idx) > 0:
        pts = tgt_pts[new_idx]
        all_points.append(pts)
        all_colors.append(np.tile(COLOR_NEW, (len(pts), 1)))

    # Removed points (source only).
    if len(removed_idx) > 0:
        pts = src_pts[removed_idx]
        all_points.append(pts)
        all_colors.append(np.tile(COLOR_REMOVED, (len(pts), 1)))

    # Unchanged (from target).
    if len(unchanged_tgt) > 0:
        pts = tgt_pts[unchanged_tgt]
        all_points.append(pts)
        all_colors.append(np.tile(COLOR_UNCHANGED, (len(pts), 1)))

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    return merged


def visualize_diff(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    diff_result: dict[str, Any],
    transformation: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Display or save a diff visualisation.

    When *save_path* is ``None`` the Open3D interactive viewer is launched.
    When a path is given (e.g. ``report.png``) a matplotlib top-down plot is
    saved instead (works in headless environments).
    """
    diff_cloud = build_diff_cloud(source, target, diff_result, transformation)

    if save_path is None:
        o3d.visualization.draw_geometries(
            [diff_cloud],
            window_name="construction-diff",
            width=1280,
            height=720,
        )
        return

    # Matplotlib fallback for headless / file output.
    pts = np.asarray(diff_cloud.points)
    colors = np.asarray(diff_cloud.colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top-down view (XY).
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=0.1, marker=".")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-down (XY)")
    ax.set_aspect("equal")

    # Side view (XZ).
    ax = axes[1]
    ax.scatter(pts[:, 0], pts[:, 2], c=colors, s=0.1, marker=".")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Side view (XZ)")
    ax.set_aspect("equal")

    # Legend.
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NEW, label="New", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_REMOVED, label="Removed", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UNCHANGED, label="Unchanged", markersize=8),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11)

    fig.suptitle("Construction Diff Report", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

"""Compute differences between aligned point cloud scans."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def compute_diff(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Detect new, removed, and unchanged points between two scans.

    *source* is the earlier scan, *target* is the later scan.  The
    *transformation* aligns source into target's coordinate frame.

    The algorithm works by building KD-trees and checking nearest-neighbour
    distances in both directions:

    - **New points**: points in *target* that have no close neighbour in the
      transformed *source* (construction added).
    - **Removed points**: points in (transformed) *source* that have no close
      neighbour in *target* (demolition / occlusion change).
    - **Unchanged**: points in *target* that match a source point within the
      threshold.

    Parameters
    ----------
    source:
        Earlier-epoch point cloud (before construction progress).
    target:
        Later-epoch point cloud.
    transformation:
        4x4 matrix that aligns *source* into *target*'s frame.
    threshold:
        Distance (metres) below which two points are considered the same.

    Returns
    -------
    dict
        Keys: ``new_indices``, ``removed_indices``, ``unchanged_source``,
        ``unchanged_target``, ``n_new``, ``n_removed``, ``n_unchanged``,
        ``target_distances``, ``source_distances``.
    """
    # Transform source into target frame.
    src_aligned = copy.deepcopy(source)
    src_aligned.transform(transformation)

    src_pts = np.asarray(src_aligned.points)
    tgt_pts = np.asarray(target.points)

    # Build KD-trees.
    src_tree = KDTree(src_pts)
    tgt_tree = KDTree(tgt_pts)

    # For every target point, find closest source point.
    tgt_dists, _ = src_tree.query(tgt_pts)

    # For every source point, find closest target point.
    src_dists, _ = tgt_tree.query(src_pts)

    # Classify.
    new_mask = tgt_dists > threshold
    new_indices = np.nonzero(new_mask)[0]

    removed_mask = src_dists > threshold
    removed_indices = np.nonzero(removed_mask)[0]

    unchanged_target = np.nonzero(~new_mask)[0]
    unchanged_source = np.nonzero(~removed_mask)[0]

    return {
        "new_indices": new_indices,
        "removed_indices": removed_indices,
        "unchanged_source": unchanged_source,
        "unchanged_target": unchanged_target,
        "n_new": len(new_indices),
        "n_removed": len(removed_indices),
        "n_unchanged": len(unchanged_target),
        "target_distances": tgt_dists,
        "source_distances": src_dists,
    }

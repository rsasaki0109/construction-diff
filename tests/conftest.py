"""Shared fixtures for construction-diff tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture()
def simple_cube_pcd() -> o3d.geometry.PointCloud:
    """A small point cloud: 8 corners of a unit cube."""
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


@pytest.fixture()
def dense_plane_pcd() -> o3d.geometry.PointCloud:
    """A flat grid of points on the XY plane (z=0), useful for registration tests."""
    rng = np.random.default_rng(42)
    n = 500
    pts = np.column_stack(
        [
            rng.uniform(0, 10, n),
            rng.uniform(0, 10, n),
            rng.uniform(-0.01, 0.01, n),  # near-zero z
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

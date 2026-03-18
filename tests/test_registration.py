"""Tests for point cloud registration (ICP refinement)."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d
import pytest

from construction_diff.registration import _refine_icp, register_scans


def _make_overlapping_clouds(
    n: int = 2000,
    overlap: float = 0.8,
    seed: int = 0,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
    """Create two synthetic point clouds with known rigid transform.

    Returns (source, target, ground_truth_transform).
    The target is the source translated by a small offset.
    """
    rng = np.random.default_rng(seed)

    # Build a 3-D surface with some structure (hemisphere).
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 5.0 + rng.normal(0, 0.02, n)  # slight noise
    pts = np.column_stack(
        [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)]
    )

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pts)

    # Ground truth: small translation only (no rotation) to keep test fast/stable.
    gt_transform = np.eye(4)
    gt_transform[:3, 3] = [0.05, -0.03, 0.02]

    target = copy.deepcopy(source)
    target.transform(gt_transform)

    return source, target, gt_transform


class TestRefineICP:
    """Unit tests for ICP refinement with near-identity initial transform."""

    def test_identity_initial_converges(self) -> None:
        """ICP should converge when the initial transform is close to GT."""
        source, target, gt = _make_overlapping_clouds(n=1500)

        result = _refine_icp(source, target, initial_transform=np.eye(4), voxel_size=0.5)

        assert result.fitness > 0.3, f"ICP fitness too low: {result.fitness}"

    def test_transform_close_to_gt(self) -> None:
        """Recovered transform should be close to the ground-truth translation."""
        source, target, gt = _make_overlapping_clouds(n=2000)

        result = _refine_icp(source, target, initial_transform=np.eye(4), voxel_size=0.5)

        recovered_t = result.transformation[:3, 3]
        gt_t = gt[:3, 3]
        error = np.linalg.norm(recovered_t - gt_t)
        assert error < 0.5, f"Translation error {error:.4f} m too large"

    def test_identical_clouds_give_identity(self) -> None:
        """Aligning a cloud with itself should yield near-identity transform."""
        source, _, _ = _make_overlapping_clouds(n=1000)
        target = copy.deepcopy(source)

        result = _refine_icp(source, target, initial_transform=np.eye(4), voxel_size=0.5)

        np.testing.assert_allclose(
            result.transformation, np.eye(4), atol=0.1,
            err_msg="Self-alignment should yield identity transform",
        )
        assert result.fitness > 0.9


class TestRegisterScans:
    """Integration test for the full FPFH + ICP pipeline."""

    def test_register_with_small_offset(self) -> None:
        """Full pipeline should handle small rigid offset."""
        source, target, gt = _make_overlapping_clouds(n=3000, seed=7)

        result = register_scans(source, target, voxel_size=0.5)

        assert result.fitness > 0.2, f"Registration fitness too low: {result.fitness}"
        assert result.transformation.shape == (4, 4)

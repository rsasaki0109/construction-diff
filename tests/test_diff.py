"""Tests for diff computation (new / removed / unchanged classification)."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from construction_diff.diff import compute_diff


def _pcd_from_array(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


class TestComputeDiff:
    """Tests for the compute_diff function."""

    def test_identical_clouds_no_diff(self) -> None:
        """Two identical clouds should have zero new/removed points."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        src = _pcd_from_array(pts)
        tgt = _pcd_from_array(pts)

        result = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)

        assert result["n_new"] == 0
        assert result["n_removed"] == 0
        assert result["n_unchanged"] == len(pts)

    def test_all_new_points(self) -> None:
        """When target has points far from source, all target points are 'new'."""
        src = _pcd_from_array([[0, 0, 0], [1, 0, 0]])
        tgt = _pcd_from_array([[100, 100, 100], [200, 200, 200]])

        result = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)

        assert result["n_new"] == 2
        assert result["n_unchanged"] == 0

    def test_all_removed_points(self) -> None:
        """When source has points far from target, all source points are 'removed'."""
        src = _pcd_from_array([[100, 100, 100], [200, 200, 200]])
        tgt = _pcd_from_array([[0, 0, 0], [1, 0, 0]])

        result = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)

        assert result["n_removed"] == 2
        assert result["n_unchanged"] == 0

    def test_mixed_new_and_unchanged(self) -> None:
        """Target has some points matching source and some new ones."""
        src = _pcd_from_array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        tgt = _pcd_from_array([[0, 0, 0], [1, 0, 0], [10, 10, 10]])

        result = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)

        assert result["n_unchanged"] == 2
        assert result["n_new"] == 1
        # The source point [2,0,0] has no match in target -> removed.
        assert result["n_removed"] == 1

    def test_threshold_sensitivity(self) -> None:
        """Points just within/beyond threshold should be classified correctly."""
        src = _pcd_from_array([[0, 0, 0]])
        tgt = _pcd_from_array([[0.05, 0, 0]])

        # With threshold=0.1 -> match (unchanged).
        r1 = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)
        assert r1["n_unchanged"] == 1
        assert r1["n_new"] == 0

        # With threshold=0.01 -> no match (new + removed).
        r2 = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.01)
        assert r2["n_new"] == 1
        assert r2["n_removed"] == 1

    def test_transformation_applied(self) -> None:
        """The transformation should be applied to source before comparison."""
        src = _pcd_from_array([[0, 0, 0]])
        tgt = _pcd_from_array([[1, 0, 0]])

        # Identity -> they are 1 m apart -> both new and removed.
        r1 = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)
        assert r1["n_new"] == 1

        # Translation that moves source to target -> match.
        T = np.eye(4)
        T[0, 3] = 1.0  # translate x by 1
        r2 = compute_diff(src, tgt, transformation=T, threshold=0.1)
        assert r2["n_unchanged"] == 1
        assert r2["n_new"] == 0

    def test_result_keys(self) -> None:
        """Check that all expected keys are present in the result dict."""
        src = _pcd_from_array([[0, 0, 0]])
        tgt = _pcd_from_array([[0, 0, 0]])

        result = compute_diff(src, tgt, transformation=np.eye(4))

        expected_keys = {
            "new_indices",
            "removed_indices",
            "unchanged_source",
            "unchanged_target",
            "n_new",
            "n_removed",
            "n_unchanged",
            "target_distances",
            "source_distances",
        }
        assert set(result.keys()) == expected_keys

    def test_indices_are_valid(self) -> None:
        """Returned indices should be valid for the respective point clouds."""
        src = _pcd_from_array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        tgt = _pcd_from_array([[0, 0, 0], [5, 5, 5]])

        result = compute_diff(src, tgt, transformation=np.eye(4), threshold=0.1)

        assert all(i < 2 for i in result["new_indices"])
        assert all(i < 3 for i in result["removed_indices"])

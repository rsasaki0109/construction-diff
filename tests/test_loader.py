"""Tests for Rohbau3D .npy loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from construction_diff.loader import load_scan


class TestLoadScan:
    """Tests for load_scan with temporary .npy files."""

    def test_load_coord_only(self, tmp_path: Path) -> None:
        """Loading a directory with only coord.npy should work."""
        coords = np.random.rand(100, 3).astype(np.float64)
        np.save(tmp_path / "coord.npy", coords)

        pcd = load_scan(tmp_path)

        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert len(pcd.points) == 100
        np.testing.assert_allclose(np.asarray(pcd.points), coords, atol=1e-10)

    def test_load_with_color_uint8(self, tmp_path: Path) -> None:
        """Colors in [0, 255] should be normalized to [0, 1]."""
        n = 50
        np.save(tmp_path / "coord.npy", np.random.rand(n, 3))
        colors_255 = np.random.randint(0, 256, (n, 3)).astype(np.float64)
        np.save(tmp_path / "color.npy", colors_255)

        pcd = load_scan(tmp_path)

        assert pcd.has_colors()
        loaded_colors = np.asarray(pcd.colors)
        assert loaded_colors.max() <= 1.0
        np.testing.assert_allclose(loaded_colors, colors_255 / 255.0, atol=1e-10)

    def test_load_with_color_float(self, tmp_path: Path) -> None:
        """Colors already in [0, 1] should not be re-normalized."""
        n = 50
        np.save(tmp_path / "coord.npy", np.random.rand(n, 3))
        colors_f = np.random.rand(n, 3).astype(np.float64)
        np.save(tmp_path / "color.npy", colors_f)

        pcd = load_scan(tmp_path)

        loaded_colors = np.asarray(pcd.colors)
        np.testing.assert_allclose(loaded_colors, colors_f, atol=1e-10)

    def test_load_with_normals(self, tmp_path: Path) -> None:
        """Normals should be loaded when normal.npy is present."""
        n = 30
        np.save(tmp_path / "coord.npy", np.random.rand(n, 3))
        normals = np.random.rand(n, 3)
        np.save(tmp_path / "normal.npy", normals)

        pcd = load_scan(tmp_path)

        assert pcd.has_normals()
        np.testing.assert_allclose(np.asarray(pcd.normals), normals.astype(np.float64), atol=1e-10)

    def test_load_all_files(self, tmp_path: Path) -> None:
        """Loading with all optional files present."""
        n = 40
        np.save(tmp_path / "coord.npy", np.random.rand(n, 3))
        np.save(tmp_path / "color.npy", np.random.rand(n, 3))
        np.save(tmp_path / "normal.npy", np.random.rand(n, 3))

        pcd = load_scan(tmp_path)

        assert len(pcd.points) == n
        assert pcd.has_colors()
        assert pcd.has_normals()

    def test_missing_coord_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when coord.npy is missing."""
        with pytest.raises(FileNotFoundError, match="coord.npy not found"):
            load_scan(tmp_path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Should accept a string path, not just pathlib.Path."""
        np.save(tmp_path / "coord.npy", np.random.rand(10, 3))

        pcd = load_scan(str(tmp_path))

        assert len(pcd.points) == 10

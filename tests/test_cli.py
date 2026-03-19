"""Tests for the CLI using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from construction_diff.cli import cli


@pytest.fixture()
def scan_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create two minimal scan directories with coord.npy files."""
    src_dir = tmp_path / "scan_t0"
    tgt_dir = tmp_path / "scan_t1"
    src_dir.mkdir()
    tgt_dir.mkdir()

    rng = np.random.default_rng(42)
    n = 200

    # Source: a small random point cloud.
    src_pts = rng.uniform(0, 5, (n, 3))
    np.save(src_dir / "coord.npy", src_pts)

    # Target: source shifted slightly + some extra points.
    tgt_pts = np.vstack([src_pts + 0.01, rng.uniform(10, 15, (20, 3))])
    np.save(tgt_dir / "coord.npy", tgt_pts)

    return src_dir, tgt_dir


class TestCLI:
    """Test the Click CLI commands."""

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Detect construction progress" in result.output

    def test_align_saves_transform(self, scan_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        src, tgt = scan_dirs
        out = tmp_path / "transform.npy"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["align", str(src), str(tgt), "-o", str(out), "--voxel-size", "1.0"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Fitness" in result.output
        assert out.exists()
        T = np.load(out)
        assert T.shape == (4, 4)

    def test_align_prints_matrix(self, scan_dirs: tuple[Path, Path]) -> None:
        src, tgt = scan_dirs

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["align", str(src), str(tgt), "--voxel-size", "1.0"],
        )

        assert result.exit_code == 0
        assert "Transformation matrix" in result.output

    def test_diff_with_precomputed_transform(
        self, scan_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        src, tgt = scan_dirs
        transform_path = tmp_path / "T.npy"
        np.save(transform_path, np.eye(4))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["diff", str(src), str(tgt), "-t", str(transform_path)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "New points" in result.output
        assert "Removed points" in result.output
        assert "Unchanged points" in result.output

    def test_diff_saves_npz(self, scan_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        src, tgt = scan_dirs
        transform_path = tmp_path / "T.npy"
        np.save(transform_path, np.eye(4))
        out = tmp_path / "diff.npz"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["diff", str(src), str(tgt), "-t", str(transform_path), "-o", str(out)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()
        data = np.load(out)
        assert "new_indices" in data
        assert "removed_indices" in data

    def test_diff_auto_registration(self, scan_dirs: tuple[Path, Path]) -> None:
        """When no transform is provided, diff should run registration first."""
        src, tgt = scan_dirs

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["diff", str(src), str(tgt), "--voxel-size", "1.0"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "No transform provided" in result.output

    def test_report_saves_image(self, scan_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        src, tgt = scan_dirs
        transform_path = tmp_path / "T.npy"
        np.save(transform_path, np.eye(4))
        out = tmp_path / "report.png"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["report", str(src), str(tgt), "-t", str(transform_path), "-o", str(out)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()
        assert "Report saved to" in result.output

    def test_invalid_source_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["align", "/nonexistent/path", str(tmp_path)])
        assert result.exit_code != 0

    def test_timeline_command(self, tmp_path: Path) -> None:
        """Test the timeline CLI command with mocked registration."""
        from unittest.mock import MagicMock

        # Create 3 scan directories.
        rng = np.random.default_rng(42)
        for name in ["scan_001", "scan_002", "scan_003"]:
            d = tmp_path / "scans" / name
            d.mkdir(parents=True, exist_ok=True)
            pts = rng.uniform(0, 5, (100, 3))
            np.save(d / "coord.npy", pts)

        mock_result = MagicMock()
        mock_result.transformation = np.eye(4)
        mock_result.fitness = 0.9
        mock_result.inlier_rmse = 0.01

        out_json = tmp_path / "report.json"
        chart_png = tmp_path / "progress.png"

        runner = CliRunner()
        with patch(
            "construction_diff.timeline.register_scans", return_value=mock_result
        ):
            result = runner.invoke(
                cli,
                [
                    "timeline",
                    str(tmp_path / "scans"),
                    "-o",
                    str(out_json),
                    "--chart",
                    str(chart_png),
                    "--voxel-size",
                    "1.0",
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Discovered 3 scans" in result.output
        assert out_json.exists()
        assert chart_png.exists()

    def test_timeline_too_few_scans(self, tmp_path: Path) -> None:
        """Timeline should fail with fewer than 2 scans."""
        d = tmp_path / "scans" / "only_one"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "coord.npy", np.zeros((10, 3)))

        runner = CliRunner()
        result = runner.invoke(cli, ["timeline", str(tmp_path / "scans")])
        assert result.exit_code != 0
        assert "at least 2" in result.output

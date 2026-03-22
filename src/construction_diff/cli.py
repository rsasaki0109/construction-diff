"""Command-line interface for construction-diff."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from construction_diff.loader import load_scan
from construction_diff.registration import register_scans, register_scans_multiscale
from construction_diff.diff import compute_diff
from construction_diff.visualization import visualize_diff


@click.group()
@click.version_option()
def cli() -> None:
    """Detect construction progress by comparing point cloud scans."""


def _run_registration(
    src_pcd, tgt_pcd, *, multi_scale: bool, voxel_size: float, verbose: bool = False
):
    """Run registration with chosen strategy and return the result."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if multi_scale:
        return register_scans_multiscale(src_pcd, tgt_pcd)
    return register_scans(src_pcd, tgt_pcd, voxel_size=voxel_size)


@cli.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save the 4x4 transformation matrix (.npy).",
)
@click.option(
    "--voxel-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Voxel size for downsampling during registration.",
)
@click.option(
    "--multi-scale",
    is_flag=True,
    default=False,
    help="Use multi-scale registration (coarse-to-fine FPFH + ICP).",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
def align(
    source: Path,
    target: Path,
    output: Path | None,
    voxel_size: float,
    multi_scale: bool,
    verbose: bool,
) -> None:
    """Register two scans using global + ICP refinement.

    SOURCE and TARGET are directories containing Rohbau3D .npy files.
    """
    import time

    click.echo(f"Loading source: {source} ...")
    src_pcd = load_scan(source)
    click.echo(f"  {len(src_pcd.points):,} points")
    click.echo(f"Loading target: {target} ...")
    tgt_pcd = load_scan(target)
    click.echo(f"  {len(tgt_pcd.points):,} points")

    click.echo("Registering ...")
    t0 = time.monotonic()
    result = _run_registration(
        src_pcd, tgt_pcd, multi_scale=multi_scale, voxel_size=voxel_size, verbose=verbose
    )
    elapsed = time.monotonic() - t0

    click.echo(f"Fitness:  {result.fitness:.4f}")
    click.echo(f"RMSE:     {result.inlier_rmse:.6f}")
    click.echo(f"Time:     {elapsed:.1f}s")

    if output is not None:
        np.save(output, result.transformation)
        click.echo(f"Transformation saved to {output}")
    else:
        click.echo("Transformation matrix:")
        click.echo(result.transformation)


@cli.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-t",
    "--transform",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Pre-computed transformation matrix (.npy). If omitted, registration runs automatically.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.10,
    show_default=True,
    help="Distance threshold (m) for classifying new/removed points.",
)
@click.option(
    "--voxel-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Voxel size for downsampling during registration.",
)
@click.option(
    "--multi-scale",
    is_flag=True,
    default=False,
    help="Use multi-scale registration (coarse-to-fine FPFH + ICP).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save diff result as .npz.",
)
def diff(
    source: Path,
    target: Path,
    transform: Path | None,
    threshold: float,
    voxel_size: float,
    multi_scale: bool,
    output: Path | None,
) -> None:
    """Compute the difference between two scans.

    SOURCE is the earlier scan, TARGET is the later scan.
    """
    src_pcd = load_scan(source)
    tgt_pcd = load_scan(target)

    if transform is not None:
        transformation = np.load(transform)
    else:
        click.echo("No transform provided, running registration...")
        reg = _run_registration(
            src_pcd, tgt_pcd, multi_scale=multi_scale, voxel_size=voxel_size
        )
        transformation = reg.transformation
        click.echo(f"Registration fitness: {reg.fitness:.4f}")

    result = compute_diff(src_pcd, tgt_pcd, transformation, threshold=threshold)

    click.echo(f"New points:       {result['n_new']}")
    click.echo(f"Removed points:   {result['n_removed']}")
    click.echo(f"Unchanged points: {result['n_unchanged']}")

    if output is not None:
        np.savez(
            output,
            new_indices=result["new_indices"],
            removed_indices=result["removed_indices"],
            unchanged_source=result["unchanged_source"],
            transformation=transformation,
        )
        click.echo(f"Diff result saved to {output}")


@cli.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-t",
    "--transform",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Pre-computed transformation matrix (.npy).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.10,
    show_default=True,
    help="Distance threshold (m) for classifying new/removed points.",
)
@click.option(
    "--voxel-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Voxel size for registration downsampling.",
)
@click.option(
    "--multi-scale",
    is_flag=True,
    default=False,
    help="Use multi-scale registration (coarse-to-fine FPFH + ICP).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save visualization to image file instead of displaying.",
)
def report(
    source: Path,
    target: Path,
    transform: Path | None,
    threshold: float,
    voxel_size: float,
    multi_scale: bool,
    output: Path | None,
) -> None:
    """Generate a visual progress report comparing two scans.

    SOURCE is the earlier scan, TARGET is the later scan.
    """
    src_pcd = load_scan(source)
    tgt_pcd = load_scan(target)

    if transform is not None:
        transformation = np.load(transform)
    else:
        click.echo("Running registration...")
        reg = _run_registration(
            src_pcd, tgt_pcd, multi_scale=multi_scale, voxel_size=voxel_size
        )
        transformation = reg.transformation

    result = compute_diff(src_pcd, tgt_pcd, transformation, threshold=threshold)

    click.echo(f"New points:       {result['n_new']}")
    click.echo(f"Removed points:   {result['n_removed']}")
    click.echo(f"Unchanged points: {result['n_unchanged']}")

    visualize_diff(src_pcd, tgt_pcd, result, transformation, save_path=output)

    if output is not None:
        click.echo(f"Report saved to {output}")


@cli.command()
@click.argument("scan_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save progress report as JSON or CSV (determined by extension).",
)
@click.option(
    "--chart",
    type=click.Path(path_type=Path),
    default=None,
    help="Save progress chart image (e.g. progress.png).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.10,
    show_default=True,
    help="Distance threshold (m) for classifying new/removed points.",
)
@click.option(
    "--voxel-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Voxel size for registration downsampling.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output.")
def timeline(
    scan_dir: Path,
    output: Path | None,
    chart: Path | None,
    threshold: float,
    voxel_size: float,
    verbose: bool,
) -> None:
    """Run 4D time-series construction monitoring on a directory of scans.

    SCAN_DIR should contain subdirectories, each representing one scan epoch.
    Scans are sorted by directory name (alphabetical/timestamp order).
    Each subdirectory must contain Rohbau3D .npy files (coord.npy, etc.).

    This goes beyond pairwise M3C2 comparison by tracking cumulative
    construction progress across an arbitrary number of epochs.
    """
    from datetime import datetime as dt

    from construction_diff.timeline import ConstructionTimeline
    from construction_diff.progress_chart import plot_progress

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Auto-discover scan subdirectories.
    scan_dirs = sorted(
        [d for d in scan_dir.iterdir() if d.is_dir() and (d / "coord.npy").exists()]
    )

    if len(scan_dirs) < 2:
        raise click.ClickException(
            f"Need at least 2 scan directories in {scan_dir}, "
            f"found {len(scan_dirs)} with coord.npy"
        )

    click.echo(f"Discovered {len(scan_dirs)} scans in {scan_dir}")

    tl = ConstructionTimeline(voxel_size=voxel_size, threshold=threshold)

    for i, sd in enumerate(scan_dirs):
        # Use directory name as a synthetic timestamp (epoch index).
        # Try parsing the name as a date first, fall back to sequential.
        timestamp = _parse_dir_timestamp(sd.name, fallback_index=i)
        tl.add_scan(sd, timestamp)
        click.echo(f"  [{i}] {sd.name} -> {timestamp.isoformat()}")

    click.echo("Computing pairwise diffs...")
    report = tl.compute_progress()

    # Print summary.
    for i in range(len(report.new_points_per_step)):
        click.echo(
            f"  Step {i + 1}: new={report.new_points_per_step[i]}, "
            f"removed={report.removed_per_step[i]}, "
            f"cumulative={report.cumulative_new[i]}"
        )

    # Save report.
    if output is not None:
        suffix = output.suffix.lower()
        if suffix == ".csv":
            report.to_csv(output)
        else:
            report.to_json(output)
        click.echo(f"Report saved to {output}")

    # Save chart.
    if chart is not None:
        plot_progress(report, chart)
        click.echo(f"Chart saved to {chart}")


def _parse_dir_timestamp(name: str, fallback_index: int) -> datetime:
    """Try to parse a directory name as a date/datetime.

    Supported formats: YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD.
    Falls back to 2000-01-01 + fallback_index days.
    """
    from datetime import datetime as dt, timedelta

    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.strptime(name, fmt)
        except ValueError:
            continue

    # Fallback: use index-based synthetic timestamps (1 day apart).
    return dt(2000, 1, 1) + timedelta(days=fallback_index)

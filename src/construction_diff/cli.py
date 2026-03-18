"""Command-line interface for construction-diff."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from construction_diff.loader import load_scan
from construction_diff.registration import register_scans
from construction_diff.diff import compute_diff
from construction_diff.visualization import visualize_diff


@click.group()
@click.version_option()
def cli() -> None:
    """Detect construction progress by comparing point cloud scans."""


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
def align(source: Path, target: Path, output: Path | None, voxel_size: float) -> None:
    """Register two scans using global + ICP refinement.

    SOURCE and TARGET are directories containing Rohbau3D .npy files.
    """
    src_pcd = load_scan(source)
    tgt_pcd = load_scan(target)

    result = register_scans(src_pcd, tgt_pcd, voxel_size=voxel_size)

    click.echo(f"Fitness:  {result.fitness:.4f}")
    click.echo(f"RMSE:     {result.inlier_rmse:.6f}")

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
        reg = register_scans(src_pcd, tgt_pcd, voxel_size=voxel_size)
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
        reg = register_scans(src_pcd, tgt_pcd, voxel_size=voxel_size)
        transformation = reg.transformation

    result = compute_diff(src_pcd, tgt_pcd, transformation, threshold=threshold)

    click.echo(f"New points:       {result['n_new']}")
    click.echo(f"Removed points:   {result['n_removed']}")
    click.echo(f"Unchanged points: {result['n_unchanged']}")

    visualize_diff(src_pcd, tgt_pcd, result, transformation, save_path=output)

    if output is not None:
        click.echo(f"Report saved to {output}")

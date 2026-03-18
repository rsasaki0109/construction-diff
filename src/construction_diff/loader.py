"""Load Rohbau3D point cloud scans stored as .npy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_scan(scan_dir: Path) -> o3d.geometry.PointCloud:
    """Load a Rohbau3D scan directory into an Open3D PointCloud.

    Expected files in *scan_dir*:
        - ``coord.npy``  (N x 3) — XYZ coordinates (required)
        - ``color.npy``  (N x 3) — RGB values in [0, 255] (optional)
        - ``intensity.npy`` (N x 1) — intensity (optional)
        - ``normal.npy`` (N x 3) — surface normals (optional)

    Parameters
    ----------
    scan_dir:
        Path to the directory containing the .npy files.

    Returns
    -------
    open3d.geometry.PointCloud
    """
    scan_dir = Path(scan_dir)

    coord_path = scan_dir / "coord.npy"
    if not coord_path.exists():
        raise FileNotFoundError(f"coord.npy not found in {scan_dir}")

    coords = np.load(coord_path).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Colour -------------------------------------------------------------------
    color_path = scan_dir / "color.npy"
    if color_path.exists():
        colors = np.load(color_path).astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Normals ------------------------------------------------------------------
    normal_path = scan_dir / "normal.npy"
    if normal_path.exists():
        normals = np.load(normal_path).astype(np.float64)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

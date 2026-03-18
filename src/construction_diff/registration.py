"""Point cloud registration using Open3D: FPFH global + ICP refinement."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def _preprocess(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, fpfh


def _global_registration(
    src_down: o3d.geometry.PointCloud,
    tgt_down: o3d.geometry.PointCloud,
    src_fpfh: o3d.pipelines.registration.Feature,
    tgt_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """RANSAC-based global registration using FPFH feature matching."""
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100_000, confidence=0.999
        ),
    )

    return result


def _refine_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """Point-to-plane ICP refinement.

    Falls back to point-to-point ICP if normals are missing.
    """
    distance_threshold = voxel_size * 0.4

    # Ensure normals exist for point-to-plane.
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)

    radius_normal = voxel_size * 2.0
    if not src.has_normals():
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
    if not tgt.has_normals():
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    return result


def register_scans(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
) -> o3d.pipelines.registration.RegistrationResult:
    """Register *source* to *target* using FPFH global + ICP refinement.

    Parameters
    ----------
    source:
        The point cloud to align.
    target:
        The reference point cloud.
    voxel_size:
        Voxel size (metres) used for downsampling.

    Returns
    -------
    open3d.pipelines.registration.RegistrationResult
        Contains ``.transformation``, ``.fitness``, and ``.inlier_rmse``.
    """
    # 1. Preprocess both clouds.
    src_down, src_fpfh = _preprocess(source, voxel_size)
    tgt_down, tgt_fpfh = _preprocess(target, voxel_size)

    # 2. Global registration (coarse).
    global_result = _global_registration(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )

    # 3. ICP refinement (fine) on the downsampled clouds.
    refined = _refine_icp(src_down, tgt_down, global_result.transformation, voxel_size)

    return refined

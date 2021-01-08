__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    eye3x4,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    check_baseline,
    compute_reprojection_errors,
    _remove_correspondences_with_ids
)

MAX_ERROR = 2.0
MIN_ANGLE = 1.0
MIN_DEPTH = 0.1

def get_pose(frame_corners, point_cloud_builder, intrinsic_mat):
    _, corner_idx, points_idx = np.intersect1d(
        frame_corners.ids,
        point_cloud_builder.ids,
        return_indices=True
    )
    frame_corners = frame_corners.points[corner_idx]
    points = point_cloud_builder.points[points_idx]

    _, r, t, inliers = cv2.solvePnPRansac(
        points,
        frame_corners,
        intrinsic_mat,
        None,
        reprojectionError=MAX_ERROR,
        flags=cv2.SOLVEPNP_EPNP,
        confidence=0.9995,
    )

    good_corners = frame_corners[inliers]
    good_points = points[inliers]

    _, r, t = cv2.solvePnP(
        good_points,
        good_corners,
        intrinsic_mat,
        None,
        r,
        t,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return r, t


def track_frame(
        frame_id,
        corner_storage,
        point_cloud_builder,
        view_mats,
        intrinsic_mat,
        direction,
        first_frame_id
):
    print("Frame ", frame_id)

    r, t = get_pose(corner_storage[frame_id], point_cloud_builder, intrinsic_mat)

    new_view = rodrigues_and_translation_to_view_mat3x4(r, t)
    if new_view is None:
        view_mats[frame_id] = view_mats[frame_id - direction]
    else:
        view_mats[frame_id] = new_view

    for d in range(5):
        #TODO(RETRIANGULATION)
        other_frame_id = frame_id - d * direction
        if other_frame_id < 0 or other_frame_id >= len(view_mats):
            break
        if direction == 1 and other_frame_id < first_frame_id:
            break
        if not check_baseline(view_mats[other_frame_id], view_mats[frame_id], 0.1):
            continue
        correspondences = build_correspondences(
            corner_storage[other_frame_id],
            corner_storage[frame_id],
            ids_to_remove=point_cloud_builder.ids
        )
        if len(correspondences) < 3:
            continue
        new_points, ids, _ = triangulate_correspondences(
            correspondences,
            view_mats[other_frame_id],
            view_mats[frame_id],
            intrinsic_mat,
            TriangulationParameters(MAX_ERROR, MIN_ANGLE, MIN_DEPTH)
        )
        point_cloud_builder.add_points(ids, new_points)


def init_point_cloud(point_cloud_builder, corner_storage, intrinsic_mat, known_view_1, known_view_2):
    max_error = MAX_ERROR
    min_angle = MIN_ANGLE
    correspondences = build_correspondences(
        corner_storage[known_view_1[0]],
        corner_storage[known_view_2[0]]
    )
    new_points, ids = [], []
    while len(new_points) < 10:
        new_points, ids, _ = triangulate_correspondences(
            correspondences,
            pose_to_view_mat3x4(known_view_1[1]),
            pose_to_view_mat3x4(known_view_2[1]),
            intrinsic_mat,
            TriangulationParameters(max_error, min_angle, MIN_DEPTH)
        )
        max_error += 0.4
        min_angle -= 0.2
    point_cloud_builder.add_points(ids, new_points)


def get_pair_score(corners_1, corners_2, intrinsic_mat):
    correspondences = build_correspondences(corners_1, corners_2)
    if len(correspondences.ids) < 30:
        return None, 0
    points_1 = correspondences.points_1
    points_2 = correspondences.points_2

    h, h_mask = cv2.findHomography(
        points_1,
        points_2,
        method = cv2.RANSAC,
        ransacReprojThreshold = 1.0,
        confidence=0.9995
    )

    e, e_mask = cv2.findEssentialMat(
        points_1,
        points_2,
        cameraMatrix = intrinsic_mat,
        method = cv2.RANSAC,
        prob = 0.9995,
        threshold = 1.0
    )
    if e is None:
        return None, 0

    if e.shape != (3, 3):
        return None, 0
    
    if h_mask.flatten().sum() / e_mask.flatten().sum() > 0.7:
        return None, 0

    correspondences = _remove_correspondences_with_ids(correspondences, np.argwhere(e_mask == 0))
    r1, r2, T = cv2.decomposeEssentialMat(e)
    best_pose, best_score = None, 0
    for r in [r1, r2]:
        for t in [-T, T]:
            pose = Pose(r.T, r.T @ t)
            view = pose_to_view_mat3x4(pose)
            points, idx, median = triangulate_correspondences(
                correspondences,
                eye3x4(),
                view,
                intrinsic_mat,
                TriangulationParameters(MAX_ERROR, MIN_ANGLE, MIN_DEPTH)
            )
            if len(points) > best_score:
                best_score = len(points)
                best_pose = pose
    return best_pose, best_score


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [eye3x4()] * frame_count
    point_cloud_builder = PointCloudBuilder()

    if known_view_1 is None or known_view_2 is None:
        frame_1, frame_2 = 0, 0
        frame_1_pose, frame_2_pose = view_mat3x4_to_pose(eye3x4()), None
        best_result = 0
        for i in range(frame_count):
            for j in range(i + 10   , min(i + 45, frame_count)):
                pose, result = get_pair_score(corner_storage[i], corner_storage[j], intrinsic_mat)
                if result > best_result:
                    frame_1, frame_2, frame_2_pose = i, j, pose
                    best_result = result
        known_view_1 = (frame_1, frame_1_pose)
        known_view_2 = (frame_2, frame_2_pose)
    
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    init_point_cloud(point_cloud_builder, corner_storage, intrinsic_mat, known_view_1, known_view_2)

    for i in range(known_view_1[0] + 1, frame_count):
        track_frame(i, corner_storage, point_cloud_builder, view_mats, intrinsic_mat, 1, known_view_1[0])

    for i in range(known_view_1[0] - 1, -1, -1):
        if i == known_view_2[0]:
            continue
        track_frame(i, corner_storage, point_cloud_builder, view_mats, intrinsic_mat, -1, known_view_1[0])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
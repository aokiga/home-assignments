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
)

MIN_ANGLE = 1.0
MAX_ERROR = 2.0
MIN_DEPTH = 0.1

def track_frame(
        frame_num,
        corner_storage,
        point_cloud_builder,
        view_mats,
        intrinsic_mat,
        direction
):
    print("Frame ", frame_num)

    frame_corners = corner_storage[frame_num]
    _, corner_idx, points_idx = np.intersect1d(
        frame_corners.ids,
        point_cloud_builder.ids,
        return_indices=True
    )
    frame_corners = frame_corners.points[corner_idx]
    points = point_cloud_builder.points[points_idx]

    retval, r, t, inliers = cv2.solvePnPRansac(
        points,
        frame_corners,
        intrinsic_mat,
        None,
        reprojectionError=MAX_ERROR,
        flags=cv2.SOLVEPNP_EPNP
    )

    good_corners = frame_corners[inliers]
    good_points = points[inliers]

    retval, r, t = cv2.solvePnP(
        good_points,
        good_corners,
        intrinsic_mat,
        None,
        r,
        t,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    print("Points in cloud ", len(point_cloud_builder.ids))

    new_views = rodrigues_and_translation_to_view_mat3x4(r, t)
    if new_views is None:
        view_mats[frame_num] = view_mats[frame_num - direction]
    else:
        view_mats[frame_num] = new_views

    for i in range(5):
        #TODO(RETRIANGULATION)
        other_frame_num = frame_num - i * direction
        if not check_baseline(view_mats[other_frame_num], view_mats[frame_num], 0.1):
            continue
        correspondences = build_correspondences(
            corner_storage[other_frame_num],
            corner_storage[frame_num],
            ids_to_remove=point_cloud_builder.ids
        )
        if len(correspondences) == 0:
            continue
        new_points, ids, _ = triangulate_correspondences(
            correspondences,
            view_mats[other_frame_num],
            view_mats[frame_num],
            intrinsic_mat,
            TriangulationParameters(MAX_ERROR, MIN_ANGLE, MIN_DEPTH)
        )
        point_cloud_builder.add_points(ids, new_points)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [eye3x4()] * frame_count
    point_cloud_builder = PointCloudBuilder()

    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    min_angle = MIN_ANGLE
    max_error = MAX_ERROR
    new_points = []
    ids = []
    correspondences = build_correspondences(
        corner_storage[known_view_1[0]],
        corner_storage[known_view_2[0]]
    )
    while len(new_points) < 10:
        new_points, ids, _ = triangulate_correspondences(
            correspondences,
            pose_to_view_mat3x4(known_view_1[1]),
            pose_to_view_mat3x4(known_view_2[1]),
            intrinsic_mat,
            TriangulationParameters(MAX_ERROR, MIN_ANGLE, MIN_DEPTH)
        )
    point_cloud_builder.add_points(ids, new_points)

    for i in range(known_view_1[0] + 1, frame_count):
        track_frame(i, corner_storage, point_cloud_builder, view_mats, intrinsic_mat, 1)

    for i in range(known_view_1[0] - 1, -1, -1):
        track_frame(i, corner_storage, point_cloud_builder, view_mats, intrinsic_mat, -1)

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
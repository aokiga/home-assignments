#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    height, width = frame_sequence.frame_shape[:2]
    max_levels = min(np.log2(height), np.log2(width)).astype(np.int8)
    MAX_CORNERS = 2500
    WINDOW_SIZE = (15, 15)
    MIN_DIST = 8
    BLOCK_SIZE = 3
    QUALITY_LEVEL = 0.075

    pyramid = None
    prev_pyramid = None
    levels = 0
    prev_levels = 0
    pts_id = 0

    corner_pts = np.ndarray((0, 2), dtype=np.float32)
    corner_ids = np.ndarray(0, dtype=np.int32)
    corner_szs = np.ndarray(0, dtype=np.float32)

    for frame, image in enumerate(frame_sequence):
        levels, pyramid = cv2.buildOpticalFlowPyramid(
            img=(image * 255).astype(np.uint8),
            winSize=WINDOW_SIZE,
            maxLevel=max_levels,
            pyramid=None,
            withDerivatives=False
        )
        if corner_pts.size > 0:
            prev_corners = corner_pts
            cur_corners = None
            min_levels = min(levels, prev_levels)
            for level in reversed(range(min_levels)):
                cur_corners, status, err = cv2.calcOpticalFlowPyrLK(
                    prevImg=prev_pyramid[level],
                    nextImg=pyramid[level],
                    prevPts=(prev_corners / (2**level)).astype(np.float32),
                    nextPts=cur_corners * 2 if cur_corners is not None else None,
                    winSize=WINDOW_SIZE
                )
            mask = status[:, 0] == 1
            corner_pts = cur_corners[mask]
            corner_ids = corner_ids[mask]
            corner_szs = corner_szs[mask]
        
        def prohibit_circle(mask, x, y, r):
            mask = cv2.circle(
                img=mask,
                center=(np.int(x), np.int(y)),
                radius=np.int(r),
                color=0,
                thickness=-1
            )

        n = corner_szs.size

        if n >= MAX_CORNERS:
            prev_pyramid = pyramid
            prev_levels = levels
            builder.set_corners_at_frame(frame, FrameCorners(
            corner_ids, corner_pts, corner_szs))
            continue

        new_pts = []
        new_ids = []
        new_szs = []

        mask = np.full((height, width), 255, dtype=np.uint8)
        for x, y, r in np.column_stack((corner_pts, corner_szs)):
            prohibit_circle(mask, x, y, r)
        for i in range(levels):
            if i > 0:
                mask = cv2.pyrDown(mask).astype(np.uint8)
            new_corners = cv2.goodFeaturesToTrack(
                image=pyramid[i],
                maxCorners=max(0, MAX_CORNERS - n),
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DIST,
                mask=mask,
                blockSize=BLOCK_SIZE
            )
            if new_corners is None:
                continue
            for x, y in new_corners[:, 0, :]:
                if not mask[np.int(y), np.int(x)]:
                    continue
                new_pts.append((x * (2.0**i), y * (2.0**i)))
                new_ids.append(pts_id)
                new_szs.append(BLOCK_SIZE * (2.0**i))
                pts_id += 1
                prohibit_circle(mask, x, y, BLOCK_SIZE)

        if new_pts:
            corner_pts = np.concatenate((corner_pts, new_pts))
            corner_ids = np.concatenate((corner_ids, new_ids))
            corner_szs = np.concatenate((corner_szs, new_szs))

        prev_pyramid = pyramid
        prev_levels = levels
        builder.set_corners_at_frame(frame, FrameCorners(
            corner_ids, corner_pts, corner_szs))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

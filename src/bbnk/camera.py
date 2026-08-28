#!/usr/bin/env python3
"""Camera resolution modes for the OV5647 sensor (Picamera2).

The 4 members of Resolution are the sensor's native readout modes (queried
via Picamera2().sensor_modes on-device) - requesting one of these sizes from
Picamera2 gets a direct sensor readout with no ISP scaling. MODE_1296_972
and MODE_2592_1944 read out the full sensor array (2x2-binned and full-res,
respectively); MODE_640_480 and MODE_1920_1080 instead read out a smaller
sub-rectangle of the array (a crop, not a uniform scale of the full frame) -
see crop_camera_matrix.
"""

from enum import Enum

import numpy as np

# The OV5647's full pixel array, i.e. MODE_2592_1944's own crop rectangle -
# the frame every Resolution's crop_origin/crop_size is expressed in.
FULL_SENSOR_SIZE = (2592, 1944)


class Resolution(Enum):
    """Name -> (width, height, crop_x, crop_y, crop_w, crop_h), pixels.

    width/height: the mode's output size - matches bb9k_config.yml's
    camera.resolution.
    crop_x, crop_y, crop_w, crop_h: the sub-rectangle of the full
    FULL_SENSOR_SIZE array this mode reads out, before scaling that crop
    down to (width, height) - Picamera2().sensor_modes[i]['crop_limits'].
    (0, 0, *FULL_SENSOR_SIZE) for a full-FOV mode.
    """

    MODE_640_480 = (640, 480, 16, 0, 2560, 1920)
    MODE_1296_972 = (1296, 972, 0, 0, 2592, 1944)
    MODE_1920_1080 = (1920, 1080, 348, 434, 1928, 1080)
    MODE_2592_1944 = (2592, 1944, 0, 0, 2592, 1944)

    @property
    def width(self) -> int:
        return self.value[0]

    @property
    def height(self) -> int:
        return self.value[1]

    @property
    def crop_origin(self):
        return self.value[2], self.value[3]

    @property
    def crop_size(self):
        return self.value[4], self.value[5]


def crop_camera_matrix(camera_matrix, calib_size, resolution: Resolution):
    """Recompute a 3x3 intrinsics matrix, calibrated at calib_size, for resolution.

    Models what the sensor readout actually does: crop the full array down
    to resolution.crop_origin/.crop_size, then scale that crop to
    resolution.width/.height. Exact for every Resolution member - including
    the two crop modes (MODE_640_480, MODE_1920_1080), unlike a naive
    full-frame rescale, which is only exact for a full-FOV mode
    (MODE_1296_972, MODE_2592_1944: crop_origin (0,0), crop_size ==
    FULL_SENSOR_SIZE, so this reduces to that same plain scale for them).

    camera_matrix: calibrated at calib_size, a (width, height) pixel tuple
    - need not itself be FULL_SENSOR_SIZE (e.g. a calibration re-scaled to
    another Resolution already); crop_origin/crop_size are defined in
    FULL_SENSOR_SIZE terms and are rescaled into calib_size's pixel
    coordinates first. dist_coeffs are resolution-independent (OpenCV's
    model) and need no rescaling.

    Returns a new ndarray, shape (3, 3), dtype float64 - camera_matrix is
    left unmodified.
    """
    camera_matrix = np.array(camera_matrix, dtype=float)

    fsx = calib_size[0] / FULL_SENSOR_SIZE[0]
    fsy = calib_size[1] / FULL_SENSOR_SIZE[1]
    x0, y0 = resolution.crop_origin
    crop_w, crop_h = resolution.crop_size
    x0, crop_w = x0 * fsx, crop_w * fsx
    y0, crop_h = y0 * fsy, crop_h * fsy

    sx = resolution.width / crop_w
    sy = resolution.height / crop_h
    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx                        # fx
    scaled[0, 2] = (camera_matrix[0, 2] - x0) * sx  # cx
    scaled[1, 1] *= sy                        # fy
    scaled[1, 2] = (camera_matrix[1, 2] - y0) * sy  # cy
    return scaled

#!/usr/bin/env python3
"""Ground-plane geometry: map camera pixels to physical ground points.

Everything here is expressed in the camera's own coordinate frame: X=right,
Y=forward (ahead), Z=up, origin at the camera - the same convention
Blaster.aim_at() uses (modulo the pivot-vs-camera-center offset).
"""

import cv2
import numpy as np


def pixel_ray(u, v, camera_matrix, dist_coeffs=None):
    """Unnormalized camera-frame ray direction through pixel (u, v).

    camera_matrix is the usual 3x3 intrinsics matrix
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. u/v may be scalars or arrays of
    matching shape S (e.g. both floats, or both (H, W) pixel-index arrays).

    Raw pixel coordinates are distorted (barrel/pincushion etc.); the
    (u-cx)/fx pinhole inverse is only correct once that's been undone.
    Pass dist_coeffs (OpenCV's k1,k2,p1,p2[,k3...], as calibrated) to
    correct for it via cv2.undistortPoints. Leave it None only if u/v are
    already rectified/undistorted pixel coordinates.

    Returns an ndarray of shape S + (3,) and dtype float64: for each (u, v),
    the 3 components are (right, ahead, up) in the camera's own frame - a
    direction only, unitless and *not* normalized to unit length (its
    "ahead" component is always exactly 1.0). Points in the direction the
    pixel looks, but never itself indicates where along that direction
    anything lies; combine with GroundPlane.intersect for a metric point.
    """
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if dist_coeffs is not None:
        pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, camera_matrix, np.asarray(dist_coeffs, dtype=float))
        xn = undistorted[:, 0, 0].reshape(u.shape)
        yn = undistorted[:, 0, 1].reshape(u.shape)
    else:
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        xn = (u - cx) / fx  # right
        yn = (v - cy) / fy  # down (image v grows downward)

    return np.stack([xn, np.ones_like(xn), -yn], axis=-1)  # right, ahead, up


def image_rays(camera_matrix, width, height, dist_coeffs=None):
    """Camera-frame ray direction for every pixel of a width x height image.

    Returns an ndarray of shape (height, width, 3), dtype float64: rays[v, u]
    is the (right, ahead, up) ray direction for pixel (u, v), in the same
    unitless, non-unit-length form documented in pixel_ray().
    """
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    return pixel_ray(us, vs, camera_matrix, dist_coeffs)


class GroundPlane:
    """The ground plane, expressed in a (possibly tilted) camera's own frame.

    height_m: how high the camera sits above the ground.
    pitch_rad: rotation about the camera's local X (right) axis, positive
        tilting Y (ahead) down toward -Z (the ground).
    roll_rad: rotation about the camera's local Y (ahead) axis, i.e. about
        its own viewing direction.
    pitch=roll=0 means the camera is level, looking straight ahead - in
    which case its own axes are gravity-aligned by definition.

    In camera coordinates the ground plane is the set of points P
    satisfying down_cam . P == height_m.

    Attributes:
        height_m, pitch_rad, roll_rad: the constructor args, stored as-is
            (Python floats).
        down_cam: ndarray, shape (3,), dtype float64. Unit vector: the
            physical "straight down" direction, expressed in the camera's
            own (right, ahead, up) axes.
    """

    def __init__(self, height_m: float, pitch_rad: float = 0.0, roll_rad: float = 0.0):
        self.height_m = height_m
        self.pitch_rad = pitch_rad
        self.roll_rad = roll_rad

        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)

        # R_tilt maps a direction given in the camera's own axes to what
        # that same physical direction would be if the camera were level;
        # its transpose (inverse) re-expresses a gravity-aligned direction
        # in the camera's own, actually-tilted axes.
        R_pitch = np.array([[1, 0, 0],
                             [0, cp, sp],
                             [0, -sp, cp]])
        R_roll = np.array([[cr, 0, sr],
                            [0, 1, 0],
                            [-sr, 0, cr]])
        R_tilt = R_pitch @ R_roll

        self.down_cam = R_tilt.T @ np.array([0.0, 0.0, -1.0])

    def intersect(self, ray_dirs):
        """Intersect camera-frame ray direction(s) with the ground plane.

        ray_dirs: array-like, shape (..., 3): one or more unnormalized ray
        directions in camera coordinates (as returned by pixel_ray/
        image_rays - need not be unit length).

        Returns an ndarray of the same shape (..., 3) and dtype float64:
        each ray's camera-frame (X, Y, Z) ground point, in meters (X=right,
        Y=ahead, Z=up; Z is always exactly -height_m). Where a ray never
        reaches the ground (denom <= 1e-9, e.g. pointing above the horizon
        or dead parallel to it), all 3 components are float('nan').
        """
        ray_dirs = np.asarray(ray_dirs, dtype=float)
        denom = ray_dirs @ self.down_cam
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(denom > 1e-9, self.height_m / denom, np.nan)
        return ray_dirs * t[..., np.newaxis]

    def pixel_to_ground(self, u, v, camera_matrix, dist_coeffs=None):
        """Camera-frame (X, Y, Z) ground point under pixel (u, v).

        u, v: scalars or arrays of matching shape S (see pixel_ray).
        Returns an ndarray of shape S + (3,), dtype float64, in meters -
        see intersect() for exactly what the 3 components mean and when
        they come back as nan.
        """
        return self.intersect(pixel_ray(u, v, camera_matrix, dist_coeffs))

    def image_to_ground(self, camera_matrix, width, height, dist_coeffs=None):
        """Camera-frame (X, Y, Z) ground point under every pixel of a width x height image.

        Returns an ndarray of shape (height, width, 3), dtype float64, in
        meters: result[v, u] is pixel (u, v)'s camera-frame ground point
        (X=right, Y=ahead, Z=up), or (nan, nan, nan) if that pixel's ray
        never reaches the ground - see intersect() for details.
        """
        return self.intersect(image_rays(camera_matrix, width, height, dist_coeffs))

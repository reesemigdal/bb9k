#!/usr/bin/env python3
"""Ground-plane geometry: map camera pixels to physical ground points.

Everything here is expressed in the camera's own coordinate frame: X=right,
Y=forward (ahead), Z=up, origin at the camera - the same convention
Blaster.aim_at() uses (modulo the pivot-vs-camera-center offset).
"""

import numpy as np


def pixel_ray(u, v, camera_matrix):
    """Unnormalized camera-frame ray direction through pixel (u, v).

    camera_matrix is the usual 3x3 intrinsics matrix
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. u/v may be scalars or arrays.
    """
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    xn = (np.asarray(u, dtype=float) - cx) / fx  # right
    yn = (np.asarray(v, dtype=float) - cy) / fy  # down (image v grows downward)
    return np.stack([xn, np.ones_like(xn), -yn], axis=-1)  # right, ahead, up


def image_rays(camera_matrix, width, height):
    """Camera-frame ray direction for every pixel of a width x height image."""
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    return pixel_ray(us, vs, camera_matrix)


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

        ray_dirs: (..., 3) unnormalized ray directions in camera
        coordinates. Returns an array of the same shape holding each ray's
        camera-frame (X, Y, Z) ground intersection, or nan where the ray
        never reaches the ground (e.g. pointing above the horizon).
        """
        ray_dirs = np.asarray(ray_dirs, dtype=float)
        denom = ray_dirs @ self.down_cam
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(denom > 1e-9, self.height_m / denom, np.nan)
        return ray_dirs * t[..., np.newaxis]

    def pixel_to_ground(self, u, v, camera_matrix):
        """Camera-frame (X, Y, Z) ground point under pixel (u, v)."""
        return self.intersect(pixel_ray(u, v, camera_matrix))

    def image_to_ground(self, camera_matrix, width, height):
        """Camera-frame (X, Y, Z) ground point under every pixel of a width x height image."""
        return self.intersect(image_rays(camera_matrix, width, height))

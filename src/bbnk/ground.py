#!/usr/bin/env python3
"""Ground-plane geometry: map camera pixels to physical ground points.

Everything here is expressed in the camera's own coordinate frame: X=right,
Y=forward (ahead), Z=up, origin at the camera - the same convention
Blaster.aim_at() uses (modulo the pivot-vs-camera-center offset).
"""

import cv2
import numpy as np

from .utils import apply_transform, invert_transform


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


def world_to_camera_transform(height_m: float, pitch_rad: float = 0.0, roll_rad: float = 0.0):
    """Build T_w2c: the 4x4 homogeneous transform, world frame -> camera frame.

    World frame: the ground itself. Gravity-aligned, origin on the ground
    directly below the camera - X=right, Y=ahead, Z=up, same convention as
    the camera's own axes when level. Camera frame: origin at the camera,
    X=right, Y=ahead, Z=up in the camera's own (possibly tilted) axes.
    Camera (0, 0, -height_m) is world (0, 0, 0).

    height_m: how high the camera sits above the world origin, i.e. the
        camera's own position is world (0, 0, height_m).
    pitch_rad: rotation of the camera frame about its local X (right) axis
        relative to world, positive tilting Y (ahead) down toward -Z.
    roll_rad: rotation of the camera frame about its local Y (ahead) axis
        relative to world, i.e. about the camera's own viewing direction.
    pitch=roll=0 means the camera frame is world-aligned, just translated
    height_m up along Z.

    We don't specify yaw because the world frame is always yaw-aligned with
    camera, hence yaw is by definition 0.

    Returns an ndarray, shape (4, 4), dtype float64: T_w2c, such that for a
    world-frame point P_w, [P_c; 1] == T_w2c @ [P_w; 1] gives its
    camera-frame coordinates P_c (utils.apply_transform does this for you,
    including on a batch of points).
    """
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)

    # R_c2w maps a direction given in the camera's own axes to what that
    # same physical direction would be if the camera were level (i.e. to
    # world axes); its transpose is the rotation part of T_w2c.
    R_pitch = np.array([[1, 0, 0],
                         [0, cp, sp],
                         [0, -sp, cp]])
    R_roll = np.array([[cr, 0, sr],
                        [0, 1, 0],
                        [-sr, 0, cr]])
    R_c2w = R_pitch @ R_roll
    R_w2c = R_c2w.T

    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = R_w2c @ np.array([0.0, 0.0, -height_m])
    return T_w2c


class GroundPlane:
    """The physical ground - the world coordinate system - and its
    relationship to a (possibly tilted) camera sitting above it.

    World frame: gravity-aligned, origin on the ground directly below the
    camera - X=right, Y=ahead, Z=up. The ground plane is exactly the world
    Z=0 plane. Camera frame: origin at the camera, X=right, Y=ahead, Z=up
    in the camera's own axes; camera (0, 0, -height_m) is world (0, 0, 0).

    height_m: how high the camera sits above the ground.
    pitch_rad, roll_rad: rotation of the camera frame relative to world -
        see world_to_camera_transform for the exact sign conventions.
    pitch=roll=0 means the camera is level - world and camera axes then
    differ only by the height_m translation.

    In camera coordinates the ground plane is the set of points P
    satisfying down_cam . P == height_m.

    Attributes:
        height_m, pitch_rad, roll_rad: the constructor args, stored as-is
            (Python floats).
        T_w2c: ndarray, shape (4, 4), dtype float64. Homogeneous transform,
            world -> camera (see world_to_camera_transform).
        T_c2w: ndarray, shape (4, 4), dtype float64. Homogeneous transform,
            camera -> world; T_w2c's inverse.
        down_cam: ndarray, shape (3,), dtype float64. Unit vector: the
            physical "straight down" direction, expressed in the camera's
            own (right, ahead, up) axes.
    """

    def __init__(self, height_m: float, pitch_rad: float = 0.0, roll_rad: float = 0.0):
        self.height_m = height_m
        self.pitch_rad = pitch_rad
        self.roll_rad = roll_rad

        self.T_w2c = world_to_camera_transform(height_m, pitch_rad, roll_rad)
        self.T_c2w = invert_transform(self.T_w2c)
        # World "straight down" ((0,0,-1), gravity's direction) re-expressed
        # in the camera's own axes - a direction, not a point, so only the
        # rotation block applies (apply_transform would also add T_w2c's
        # translation). E.g. pitch=roll=0: camera axes == world axes, so
        # down_cam == (0,0,-1) exactly, and the ground plane sits height_m
        # below the camera along that axis, i.e. down_cam . (0,0,-height_m)
        # == height_m.
        self.down_cam = self.T_w2c[:3, :3] @ np.array([0.0, 0.0, -1.0])

    def to_camera(self, points_world):
        """World-frame point(s) -> camera-frame point(s).

        points_world: array-like, shape (..., 3). Returns an ndarray of the
        same shape (..., 3), dtype float64.
        """
        return apply_transform(self.T_w2c, points_world)

    def to_world(self, points_cam):
        """Camera-frame point(s) -> world-frame point(s).

        points_cam: array-like, shape (..., 3). Returns an ndarray of the
        same shape (..., 3), dtype float64. Ground points (from intersect/
        pixel_to_ground/image_to_ground) land on world Z == 0.
        """
        return apply_transform(self.T_c2w, points_cam)

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

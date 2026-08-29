#!/usr/bin/env python3
"""Unit conversion helpers."""

import math
import numpy as np

M_PER_IN = 0.0254
M_PER_FT = 0.3048


def in2m(inches: float) -> float:
    return inches * M_PER_IN


def ft2m(feet: float) -> float:
    return feet * M_PER_FT


def d2r(degrees: float) -> float:
    return math.radians(degrees)


def r2d(radians: float) -> float:
    return math.degrees(radians)

def np2PrettyStr(d):
    return '%s (%s): %s %s'%(d.shape, d.dtype, np.min(d), np.max(d))


def invert_transform(T_a2b):
    """Invert a 4x4 homogeneous coordinate transform.

    T_a2b maps frame-A coordinates to frame-B coordinates
    ([P_b;1] = T_a2b @ [P_a;1]); returns T_b2a. Plain matrix inversion -
    works for any invertible 4x4, not just rigid transforms.
    """
    return np.linalg.inv(np.asarray(T_a2b, dtype=float))


def compose_transform(T_a2b, T_b2c):
    """Compose two coordinate transforms into one: A -> B -> C becomes A -> C.

    Given T_a2b (A -> B) and T_b2c (B -> C), returns T_a2c such that
    [P_c;1] = T_a2c @ [P_a;1] for any point P. Since
    [P_c;1] = T_b2c @ (T_a2b @ [P_a;1]) = (T_b2c @ T_a2b) @ [P_a;1],
    this is just T_b2c @ T_a2b - mind the reversed order.
    """
    return np.asarray(T_b2c, dtype=float) @ np.asarray(T_a2b, dtype=float)


def apply_transform(T_a2b, points_a):
    """Apply a 4x4 homogeneous transform to one or more 3-vectors.

    points_a: array-like, shape (..., 3), frame-A coordinates. Returns an
    ndarray of the same shape (..., 3), dtype float64: each point
    transformed into frame B (implicit w=1, re-normalized by w on the way
    out so affine/projective T_a2b also work).
    """
    T_a2b = np.asarray(T_a2b, dtype=float)
    points_a = np.asarray(points_a, dtype=float)
    ones = np.ones(points_a.shape[:-1] + (1,))
    homogeneous_a = np.concatenate([points_a, ones], axis=-1)
    homogeneous_b = homogeneous_a @ T_a2b.T
    return homogeneous_b[..., :3] / homogeneous_b[..., 3:4]

def isp_compute_gamma_lut(gamma):
    """Precompute a 256-entry uint8->float16 gamma-correction lookup table.

    lut[i] = 255 * (i/255)**(1/gamma), for i in 0..255 (2.2 is a typical
    display gamma). Built once per gamma value so isp_apply can gamma-
    correct a whole image with a single fancy-index lookup.
    """
    i = np.arange(256, dtype=np.float16)
    return (255.0 * (i / 255.0) ** (1.0 / gamma)).astype(np.float16)


def isp_apply(img, gamma=2.2):
    """Software ISP pass: gamma correction + 1st/99th percentile auto-contrast.

    img (uint8) is gamma-corrected via isp_compute_gamma_lut(gamma) into
    float16, its 1st/99th brightness percentiles (of the grayscale
    gamma-corrected image, if img is color) are used to compute the linear
    scale/offset that stretches them to fill 0-255, and that scale/offset
    is applied to the (still-color) gamma-corrected image. Result is
    clipped to [0, 255] before converting back to uint8, so values can't
    roll over.
    """
    img = np.asarray(img, dtype=np.uint8)
    gamma_img = isp_compute_gamma_lut(gamma)[img]

    if gamma_img.ndim == 3 and gamma_img.shape[2] > 1:
        b, g, r = gamma_img[..., 0], gamma_img[..., 1], gamma_img[..., 2]
        gray = r * np.float16(0.299) + g * np.float16(0.587) + b * np.float16(0.114)
    else:
        gray = gamma_img

    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo:
        scale, offset = 1.0, 0.0
    else:
        scale = 255.0 / (hi - lo)
        offset = -lo * scale

    out = gamma_img * scale + offset
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

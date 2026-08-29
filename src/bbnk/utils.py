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
    """Precompute a 256-entry uint8->float32 gamma-correction lookup table.

    lut[i] = 255 * (i/255)**(1/gamma), for i in 0..255 (2.2 is a typical
    display gamma). Built once per gamma value so isp_apply can gamma-
    correct a whole image with a single fancy-index lookup.
    """
    i = np.arange(256, dtype=np.float32)
    return (255.0 * (i / 255.0) ** (1.0 / gamma)).astype(np.float32)


def isp_apply(img, gamma=2.2, percentile_sample_size=(320, 200)):
    """Software ISP pass: gamma correction + 1st/99th percentile auto-contrast.

    Rather than gamma-correcting the full image and then separately scaling
    it, this estimates the scale/offset first (from a cheap subsample) and
    folds gamma+scale+offset+clip into one combined 256-entry uint8->uint8
    LUT, so the only full-resolution work is a single fancy-index lookup.

    1st/99th brightness percentiles are estimated from a strided subsample
    of all channels (no separate grayscale conversion - np.percentile
    doesn't care that R/G/B values from the same pixel end up as separate
    samples), downsampled to about percentile_sample_size (width, height)
    pixels, gamma-corrected via isp_compute_gamma_lut(gamma). That scale/
    offset is baked into a combined LUT alongside the gamma curve, clipped
    to [0, 255] and cast to uint8 up front so applying it to the full image
    can't roll over.
    """
    img = np.asarray(img, dtype=np.uint8)
    gamma_lut = isp_compute_gamma_lut(gamma)

    h, w = img.shape[:2]
    target_w, target_h = percentile_sample_size
    stride_h = max(1, h // target_h)
    stride_w = max(1, w // target_w)
    sample = gamma_lut[img[::stride_h, ::stride_w]]

    lo, hi = np.percentile(sample, [1, 99])
    if hi <= lo:
        scale, offset = 1.0, 0.0
    else:
        scale = 255.0 / (hi - lo)
        offset = -lo * scale

    combined_lut = np.clip(gamma_lut * scale + offset, 0, 255).astype(np.uint8)
    return combined_lut[img]

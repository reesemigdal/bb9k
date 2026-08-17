#!/usr/bin/env python3
"""Unit conversion helpers."""

import math

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

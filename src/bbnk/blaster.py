#!/usr/bin/env python3
"""Two-axis turret + water solenoid: the product this repo builds towards."""

import math
import time
from typing import NamedTuple

from gpiozero import DigitalOutputDevice

from .servo import Servo

GRAVITY_MPS2 = 9.80665


class AimSolution(NamedTuple):
    """Resulting servo angles (degrees, in each servo's own coordinate frame)."""

    yaw_deg: float
    pitch_deg: float


class Blaster:
    """A CV-aimed, two-axis water turret: yaw servo + pitch servo + solenoid valve.

    World coordinates passed to aim_at() are centered on the blaster's pivot:
        X = right, Y = forward (horizontal), Z = up.
    Yaw 0 deg / pitch 0 deg (world frame) means "aim straight ahead, level" and
    is assumed to correspond to each servo's own center_angle_deg. If the
    horns are mounted such that increasing servo angle turns the opposite way
    from increasing world angle, set yaw_invert/pitch_invert. If center isn't
    exactly forward/level after mounting, correct it with
    yaw_zero_offset_deg/pitch_zero_offset_deg.

    Pitch is solved ballistically (projectile motion under gravity, given
    water_velocity_mps) rather than pointed straight at the target, so aim_at()
    is only as accurate as that velocity estimate.
    """

    def __init__(
        self,
        yaw_servo_params: dict,
        pitch_servo_params: dict,
        solenoid_gpio_pin: int,
        water_velocity_mps: float,
        yaw_invert: bool = False,
        pitch_invert: bool = False,
        yaw_zero_offset_deg: float = 0.0,
        pitch_zero_offset_deg: float = 0.0,
        gravity_mps2: float = GRAVITY_MPS2,
    ):
        self.yaw = Servo(**yaw_servo_params)
        self.pitch = Servo(**pitch_servo_params)
        self.solenoid = DigitalOutputDevice(solenoid_gpio_pin)

        self.water_velocity_mps = water_velocity_mps
        self.yaw_invert = yaw_invert
        self.pitch_invert = pitch_invert
        self.yaw_zero_offset_deg = yaw_zero_offset_deg
        self.pitch_zero_offset_deg = pitch_zero_offset_deg
        self.gravity_mps2 = gravity_mps2

    def _solve_pitch_rad_reese(self, horizontal_dist_m: float, height_m: float):
        """Return (low_arc_rad, high_arc_rad) launch angles, or None if unreachable."""
        x = horizontal_dist_m
        if x == 0:
            x = 0.000000000001

        y = height_m
        print(x, y)
        velo = self.water_velocity_mps
        g = self.gravity_mps2

        A = (-g*x**2)/(2*velo**2)
        B = x
        C = (-g*x**2)/(2*velo**2)-y

        disc_before_squarerooted = B**2-4*A*C
        if disc_before_squarerooted<0:
            print("messedup",disc_before_squarerooted)
            return None

        disc = math.sqrt(disc_before_squarerooted)
        ans1 = math.atan((-x + disc) /(2*A))
        ans2 = math.atan((-x - disc) /(2*A))

        return ans1,ans2

    def _solve_pitch_rad_old(self, horizontal_dist_m: float, height_m: float):
        """Return (low_arc_rad, high_arc_rad) launch angles, or None if unreachable."""
        if horizontal_dist_m < 1e-9:
            theta = math.pi / 2 if height_m >= 0 else -math.pi / 2
            return theta, theta

        g = self.gravity_mps2
        v2 = self.water_velocity_mps**2
        discriminant = v2**2 - g * (g * horizontal_dist_m**2 + 2 * height_m * v2)
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        theta_low = math.atan((v2 - sqrt_disc) / (g * horizontal_dist_m))
        theta_high = math.atan((v2 + sqrt_disc) / (g * horizontal_dist_m))
        return theta_low, theta_high

    def is_reachable(self, x: float, y: float, z: float) -> bool:
        """Whether (x, y, z) can be hit at the configured water velocity."""
        return self._solve_pitch_rad_reese(math.hypot(x, y), z) is not None

    def aim_at(self, x: float, y: float, z: float, prefer_high_arc: bool = False) -> AimSolution:
        """Point yaw/pitch servos to hit world point (x, y, z).

        Raises ValueError if the target is out of ballistic range (given
        water_velocity_mps) or outside the mechanical range of either servo.
        """
        yaw_deg = math.degrees(math.atan2(x, y))

        #solutions = self._solve_pitch_rad(math.hypot(x, y), z)
        solutions = self._solve_pitch_rad_reese(math.hypot(x, y), z)
        if solutions is None:
            raise ValueError(
                f"target ({x}, {y}, {z}) is out of range at "
                f"{self.water_velocity_mps} m/s water velocity"
            )
        theta_low, theta_high = solutions
        pitch_deg = math.degrees(theta_high if prefer_high_arc else theta_low)

        yaw_sign = -1 if self.yaw_invert else 1
        pitch_sign = -1 if self.pitch_invert else 1
        yaw_servo_deg = self.yaw.center_angle_deg + yaw_sign * yaw_deg + self.yaw_zero_offset_deg
        pitch_servo_deg = (
            self.pitch.center_angle_deg + pitch_sign * pitch_deg + self.pitch_zero_offset_deg
        )

        for name, servo, target_deg in (
            ("yaw", self.yaw, yaw_servo_deg),
            ("pitch", self.pitch, pitch_servo_deg),
        ):
            if not (servo.min_angle_deg <= target_deg <= servo.max_angle_deg):
                raise ValueError(
                    f"{name} target {target_deg:.1f} deg is outside servo range "
                    f"[{servo.min_angle_deg}, {servo.max_angle_deg}]"
                )

        self.yaw.setDegrees(yaw_servo_deg)
        self.pitch.setDegrees(pitch_servo_deg)
        return AimSolution(yaw_deg=yaw_servo_deg, pitch_deg=pitch_servo_deg)

    def water_on(self) -> None:
        self.solenoid.on()

    def water_off(self) -> None:
        self.solenoid.off()

    def fire(self, duration_s: float) -> None:
        """Open the solenoid for duration_s seconds, then close it."""
        self.water_on()
        time.sleep(duration_s)
        self.water_off()

    def center(self) -> None:
        self.water_off()
        self.yaw.center()
        self.pitch.center()

    def close(self) -> None:
        self.water_off()
        self.yaw.close()
        self.pitch.close()
        self.solenoid.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

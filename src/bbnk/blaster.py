#!/usr/bin/env python3
"""Two-axis turret + water solenoid: the product this repo builds towards."""

import math
import time
from typing import NamedTuple

from gpiozero import DigitalOutputDevice

from .servo import Servo, FrameServo
from .utils import d2r, r2d

GRAVITY_MPS2 = 9.80665


class AimSolution(NamedTuple):
    """Resulting servo angles (degrees, in each servo's own coordinate frame)."""

    yaw_deg: float
    pitch_deg: float


class Blaster:
    """A CV-aimed, two-axis water turret: yaw servo + pitch servo + solenoid valve.

    World coordinates passed to aim_at() are centered on the blaster's pivot:
        X = right, Y = forward (horizontal), Z = up.
    Yaw 0 deg / pitch 0 deg (world frame) means "aim straight ahead, level".
    yaw_zero_offset_deg/pitch_zero_offset_deg are the raw servo angles that
    correspond to that world-frame 0 (i.e. where each servo physically sits
    when aimed straight ahead/level); they default to each servo's own
    center_angle_deg. If the horns are mounted such that increasing servo
    angle turns the opposite way from increasing world angle, set
    yaw_invert/pitch_invert.

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
        yaw_zero_offset_deg: float = None,
        pitch_zero_offset_deg: float = None,
        gravity_mps2: float = GRAVITY_MPS2,
    ):
        raw_yaw = Servo(**yaw_servo_params)
        raw_pitch = Servo(**pitch_servo_params)
        if yaw_zero_offset_deg is None:
            yaw_zero_offset_deg = raw_yaw.center_angle_deg
        if pitch_zero_offset_deg is None:
            pitch_zero_offset_deg = raw_pitch.center_angle_deg
        self.yaw = FrameServo(raw_yaw, yaw_zero_offset_deg)
        self.pitch = FrameServo(raw_pitch, pitch_zero_offset_deg)
        self.solenoid = DigitalOutputDevice(solenoid_gpio_pin)

        self.water_velocity_mps = water_velocity_mps
        self.yaw_invert = yaw_invert
        self.pitch_invert = pitch_invert
        self.gravity_mps2 = gravity_mps2

    def _solve_pitch_rad_reese(self, horizontal_dist_m: float, height_m: float):
        """Return (low_arc_rad, high_arc_rad) launch angles, or None if unreachable."""
        x = horizontal_dist_m
        if x == 0:
            x = 0.000000000001

        y = height_m
        velo = self.water_velocity_mps
        g = self.gravity_mps2

        A = (-g*x**2)/(2*velo**2)
        B = x
        C = (-g*x**2)/(2*velo**2)-y

        disc_before_squarerooted = B**2-4*A*C
        if disc_before_squarerooted<0:
            return None

        disc = math.sqrt(disc_before_squarerooted)
        ans1 = math.atan((-x + disc) /(2*A))
        ans2 = math.atan((-x - disc) /(2*A))

        return ans1,ans2

    def _solve_pitch_rad_old(self, horizontal_dist_m: float, height_m: float):
        """Return (low_arc_rad, high_arc_rad) launch angles, or None if unreachable."""
        if horizontal_dist_m < 1e-9:
            theta = d2r(90) if height_m >= 0 else d2r(-90)
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

    def max_horizontal_range_m(self, target_z_m: float, search_bound_m: float = 200.0, tol_m: float = 0.01) -> float:
        """Max horizontal distance reachable at a fixed target height offset.

        target_z_m: target height relative to the blaster's pivot (world/
            turret Z, positive up) - e.g. for a ground point seen through
            GroundPlane, that's -height_m (see GroundPlane.max_range_points).

        Binary search over is_reachable(), which is monotonic in horizontal
        distance for a fixed target_z_m (reachability only ever gets harder
        as range grows, for a fixed drop). Raises ValueError if
        search_bound_m is itself reachable (too small a bound to bracket
        the true max range) - widen it.
        """
        if self.is_reachable(search_bound_m, 0.0, target_z_m):
            raise ValueError(f'search_bound_m={search_bound_m} is itself reachable; increase it')

        lo, hi = 0.0, search_bound_m
        while hi - lo > tol_m:
            mid = (lo + hi) / 2
            if self.is_reachable(mid, 0.0, target_z_m):
                lo = mid
            else:
                hi = mid
        return lo

    def aim_at(self, x: float, y: float, z: float, prefer_high_arc: bool = False) -> AimSolution:
        """Point yaw/pitch servos to hit world point (x, y, z).

        Raises ValueError if the target is out of ballistic range (given
        water_velocity_mps) or outside the mechanical range of either servo.
        """
        yaw_deg = r2d(math.atan2(x, y))

        #solutions = self._solve_pitch_rad(math.hypot(x, y), z)
        solutions = self._solve_pitch_rad_reese(math.hypot(x, y), z)
        if solutions is None:
            raise ValueError(
                f"target ({x}, {y}, {z}) is out of range at "
                f"{self.water_velocity_mps} m/s water velocity"
            )
        theta_low, theta_high = solutions
        pitch_deg = r2d(theta_high if prefer_high_arc else theta_low)

        yaw_sign = -1 if self.yaw_invert else 1
        pitch_sign = -1 if self.pitch_invert else 1
        yaw_frame_deg = yaw_sign * yaw_deg
        pitch_frame_deg = pitch_sign * pitch_deg

        for name, servo, target_deg in (
            ("yaw", self.yaw, yaw_frame_deg),
            ("pitch", self.pitch, pitch_frame_deg),
        ):
            if not (servo.min_angle_deg <= target_deg <= servo.max_angle_deg):
                raise ValueError(
                    f"{name} target {target_deg:.1f} deg is outside servo range "
                    f"[{servo.min_angle_deg}, {servo.max_angle_deg}]"
                )

        self.yaw.setDegrees(yaw_frame_deg)
        self.pitch.setDegrees(pitch_frame_deg)
        return AimSolution(yaw_deg=yaw_frame_deg, pitch_deg=pitch_frame_deg)

    def water_on(self) -> None:
        self.solenoid.on()

    def water_off(self) -> None:
        self.solenoid.off()

    def ready_aim_fire(self, x, y, z, high_arc=False, aim_dur_s=0.5, fire_dur_s=2.0):
        print('aim!')
        self.aim_at(x, y, z, high_arc)
        time.sleep(aim_dur_s)
        print('fire!')
        self.fire(fire_dur_s)

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
        self.yaw.setDegrees(0)
        self.pitch.setDegrees(0)
        # self.yaw.close()
        # self.pitch.close()
        self.solenoid.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

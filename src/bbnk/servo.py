#!/usr/bin/env python3
"""Angular servo control via hardware PWM (through gpiozero/lgpio)."""

from gpiozero import AngularServo


class Servo:
    """A single PWM-controlled servo addressed by angle in degrees."""

    def __init__(
        self,
        gpio_pin: int,
        pwm_hz: float,
        min_pulse_us: float,
        max_pulse_us: float,
        min_angle_deg: float,
        max_angle_deg: float,
    ):
        self.gpio_pin = gpio_pin
        self.pwm_hz = pwm_hz
        self.min_pulse_us = min_pulse_us
        self.max_pulse_us = max_pulse_us
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        self.center_us = (min_pulse_us + max_pulse_us) / 2
        self.center_angle_deg = (min_angle_deg + max_angle_deg) / 2

        self._servo = AngularServo(
            gpio_pin,
            initial_angle=self.center_angle_deg,
            min_angle=min_angle_deg,
            max_angle=max_angle_deg,
            min_pulse_width=min_pulse_us / 1_000_000,
            max_pulse_width=max_pulse_us / 1_000_000,
            frame_width=1 / pwm_hz,
        )

    def setDegrees(self, degrees: float) -> None:
        degrees = max(self.min_angle_deg, min(self.max_angle_deg, degrees))
        self._servo.angle = degrees

    def getDegrees(self) -> float:
        return self._servo.angle

    def center(self) -> None:
        self.setDegrees(self.center_angle_deg)

    def close(self) -> None:
        self._servo.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

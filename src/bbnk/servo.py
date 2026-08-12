#!/usr/bin/env python3
"""Angular servo control via hardware PWM (rpi_hardware_pwm)."""

from rpi_hardware_pwm import HardwarePWM

# Default GPIO -> (chip, pwm_channel) lookup for the 4 hardware PWM pins on a
# Raspberry Pi (this layout requires a config.txt overlay exposing all four
# channels on pwmchip0, e.g.
# dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4,pin3=18,func3=2,pin4=19,func2=2).
GPIO_TO_PWM_CHANNEL = {
    12: (0, 0),  # GPIO 12, Pin 32, PWM0_CHAN0, /sys/class/pwm/pwmchip0/pwm0
    13: (0, 1),  # GPIO 13, Pin 33, PWM0_CHAN1, /sys/class/pwm/pwmchip0/pwm1
    18: (0, 2),  # GPIO 18, Pin 12, PWM0_CHAN2, /sys/class/pwm/pwmchip0/pwm2
    19: (0, 3),  # GPIO 19, Pin 35, PWM0_CHAN3, /sys/class/pwm/pwmchip0/pwm3
}


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
        gpio_to_pwm_channel: dict = GPIO_TO_PWM_CHANNEL,
    ):
        if gpio_pin not in gpio_to_pwm_channel:
            raise ValueError(
                f"gpio_pin {gpio_pin} not in gpio_to_pwm_channel lookup "
                f"(known pins: {sorted(gpio_to_pwm_channel)})"
            )

        self.gpio_pin = gpio_pin
        self.pwm_hz = pwm_hz
        self.min_pulse_us = min_pulse_us
        self.max_pulse_us = max_pulse_us
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        self.center_us = (min_pulse_us + max_pulse_us) / 2
        self.center_angle_deg = (min_angle_deg + max_angle_deg) / 2

        chip, pwm_channel = gpio_to_pwm_channel[gpio_pin]
        self._pwm = HardwarePWM(pwm_channel=pwm_channel, hz=pwm_hz, chip=chip)
        self._pwm.start(0)
        self.setDegrees(self.center_angle_deg)

    def _degrees_to_duty_cycle(self, degrees: float) -> float:
        span_us = self.max_pulse_us - self.min_pulse_us
        span_deg = self.max_angle_deg - self.min_angle_deg
        pulse_us = self.min_pulse_us + (degrees - self.min_angle_deg) * span_us / span_deg
        period_us = 1_000_000 / self.pwm_hz
        return pulse_us / period_us * 100

    def setDegrees(self, degrees: float) -> None:
        degrees = max(self.min_angle_deg, min(self.max_angle_deg, degrees))
        self._pwm.change_duty_cycle(self._degrees_to_duty_cycle(degrees))
        self._degrees = degrees

    def getDegrees(self) -> float:
        return self._degrees

    def center(self) -> None:
        self.setDegrees(self.center_angle_deg)

    def close(self) -> None:
        self._pwm.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def interp1(x, v, xq):
    ''' linear interpolation between two points, as in matlab's interp1(x, v, xq).
        x and v are 2-tuples: (x0, x1), (v0, v1).
    '''
    x0, x1 = x
    v0, v1 = v
    t = (xq - x0) / (x1 - x0)
    return v0 + t * (v1 - v0)

class FrameServo:
    """A servo addressed in some other frame's coordinates, where frame degrees = 0
    corresponds to zeroPointDegrees on the underlying servo S. Same api as Servo.
    """

    def __init__(self, S: Servo, zeroPointDegrees: float):
        self.S = S
        self.zeroPointDegrees = zeroPointDegrees
        self.min_angle_deg = S.min_angle_deg - zeroPointDegrees
        self.max_angle_deg = S.max_angle_deg - zeroPointDegrees
        self.center_angle_deg = S.center_angle_deg - zeroPointDegrees

    def setDegrees(self, degrees: float) -> None:
        self.S.setDegrees(degrees + self.zeroPointDegrees)

    def getDegrees(self) -> float:
        return self.S.getDegrees() - self.zeroPointDegrees

    def center(self) -> None:
        self.setDegrees(self.center_angle_deg)

    def close(self) -> None:
        self.S.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

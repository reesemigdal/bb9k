import os,sys,time
from rpi_hardware_pwm import HardwarePWM

sys.path.append('../src')


def s1():
    ''' use hw pwm to set duty cycle of pin to light an led at duty cycle brightness
        pinctrl set 12 a0

        squiddo@raspberrypi:~> pinctrl set 12 a0
        squiddo@raspberrypi:~> pinctrl get 12
        12: a0    pd | lo // GPIO12 = PWM0_CHAN0

        /sys/class/pwm/pwmchip0/pwm3/

        pip install rpi-hardware-pwm


        GPIO 12	Pin 32	PWM0_CHAN0	/sys/class/pwm/pwmchip0/pwm0
        GPIO 13	Pin 33	PWM0_CHAN1	/sys/class/pwm/pwmchip0/pwm1
        GPIO 18	Pin 12	PWM0_CHAN2	/sys/class/pwm/pwmchip0/pwm2
        GPIO 19	Pin 35	PWM0_CHAN3	/sys/class/pwm/pwmchip0/pwm3

        pinctrl set 18 a3   # GPIO18 → PWM0_CHAN2

        pinctrl set 12 a0
        pinctrl set 13 a0
        pinctrl set 18 a3
        pinctrl set 19 a3

        pinctrl get 12
        pinctrl get 13
        pinctrl get 18
        pinctrl get 19

        sudo sh -c 'echo 0 > /sys/class/pwm/pwmchip0/export'
        sudo sh -c 'echo 1 > /sys/class/pwm/pwmchip0/export'
        sudo sh -c 'echo 2 > /sys/class/pwm/pwmchip0/export' # creates a pwm2 device
        sudo sh -c 'echo 3 > /sys/class/pwm/pwmchip0/export'
        # 1ms frequency
        sudo sh -c 'echo 1000000 > /sys/class/pwm/pwmchip0/pwm0/period'
        # duty cycle, based on freq above
        sudo sh -c 'echo 500000 > /sys/class/pwm/pwmchip0/pwm0/duty_cycle'
        sudo sh -c 'echo 1 > /sys/class/pwm/pwmchip0/pwm0/enable'
        sudo cat /sys/kernel/debug/pwm
        
    '''

    pwm = HardwarePWM(pwm_channel=0, hz=50, chip=0)
    pwm.start(0)
    for x in range(11):
        pwm.change_duty_cycle(10*x)
        time.sleep(1)

def main():
    return s1()

if __name__ == "__main__":
    main()

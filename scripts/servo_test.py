import os,sys,time
from rpi_hardware_pwm import HardwarePWM

sys.path.append('../src')
from bbnk.servo import Servo
from bbnk.blaster import Blaster


def s1():
    ''' use hw pwm to set duty cycle of pin to light an led at duty cycle brightness
        pinctrl set 12 a0

        squiddo@raspberrypi:~> pinctrl set 12 a0
        squiddo@raspberrypi:~> pinctrl get 12
        12: a0    pd | lo // GPIO12 = PWM0_CHAN0

        /sys/class/pwm/pwmchip0/pwm3/

        https://github.com/Pioreactor/rpi_hardware_pwm
        sudo nano /boot/firmware/config.txt
        added this:
        dtoverlay=pwm-2chan,pin=18,func=2,pin2=12,func2=4
        alternatives:
        dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4,pin3=14,func3=4,pin4=15,func4=4
        dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4,pin3=18,func3=2,pin4=19,func2=2

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

def s2():
    ''' pwm test using the new servo class '''

    if 1: # set servos to "0" (ie 135 degrees) positions
        s1 = Servo(12, 50, 500, 2500, 0, 270) # pitch
        s2 = Servo(18, 50, 500, 2500, 0, 270) # yaw
        s1.setDegrees(130)
        s2.setDegrees(125)
        time.sleep(5)
        s1._pwm.echo(0, os.path.join(s1._pwm.pwm_dir, "enable"))
        s2._pwm.echo(0, os.path.join(s2._pwm.pwm_dir, "enable"))
        return

    #S = Servo(12, 50, 500, 2500, 0, 270)
    S = Servo(18, 50, 500, 2500, 0, 270)

    for x in range(11):
        S.setDegrees(x/10.0*270)
        print('degrees:',S.getDegrees())
        time.sleep(1)
    S.setDegrees(0)

def s3():
    ''' dual servo test '''

    Yaw = Servo(18, 50, 500+100, 2500-100, 0, 270)
    Pitch = Servo(12, 50, 500+100, 2500-100, 0, 270)

    Yaw.setDegrees(0)
    Pitch.setDegrees(0)

def s4():
    ''' solenoid + mosfet test '''
    from gpiozero import DigitalOutputDevice

    solenoid = DigitalOutputDevice(15)

    # turn gpio pin 15 on
    print('turning on')
    solenoid.on()
    time.sleep(1)
    # turn gpio pin 15 off
    print('turning off')
    solenoid.off()
    solenoid.close()

def s5():
    ''' create an instantiation of the blaster and aim_at something intuitive, like 1 meter straight ahead
        using 15 meters per second for water speed
    '''
    blaster = Blaster(
        yaw_servo_params=dict(gpio_pin=18, pwm_hz=80, min_pulse_us=500+100, max_pulse_us=2500-100, min_angle_deg=0, max_angle_deg=270),
        pitch_servo_params=dict(gpio_pin=12, pwm_hz=80, min_pulse_us=500+100, max_pulse_us=2500-100, min_angle_deg=0, max_angle_deg=270),
        solenoid_gpio_pin=15,
        water_velocity_mps=7,
        pitch_invert=False, # servo goes up on positive angles - good
        yaw_invert=True, # servo turns left on positive angles - need invert
        yaw_zero_offset_deg=125,
        pitch_zero_offset_deg=130,
    )
    if 1: # actual aiming of blaster
        #print('aim:', blaster.aim_at(1, 1, -0.1))
        blaster.center()
        blaster.ready_aim_fire(1,1,1)
        return

    print('aim:', blaster.aim_at(0, 0, 1))
    print('aim:', blaster.aim_at(0, 0, -1))
    print('aim:', blaster.aim_at(0, 0, 0))
    print('aim:', blaster.aim_at(0, 20, 0, True))
    print('aim:', blaster.aim_at(0, 20, 0, False))
    print('aim:', blaster.aim_at(0, 20, -2, True))
    print('aim:', blaster.aim_at(0, 20, -2, False))
    blaster.aim_at(0,.1,0,False)

def main():
    #return s2() # blaster servo calibration to find 0,0 point (aim level straight ahead)
    return s5() # 1st full blaster instantiation
    #return s4()
    #return s3()
    return s1()

if __name__ == "__main__":
    main()

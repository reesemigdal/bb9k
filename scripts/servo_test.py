import os,sys,time

import numpy as np
import yaml
from rpi_hardware_pwm import HardwarePWM

sys.path.append('../src')
from bbnk.servo import Servo
from bbnk.blaster import Blaster
from bbnk.ground import GroundPlane
from bbnk.utils import d2r, ft2m, np2PrettyStr


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
        #blaster.ready_aim_fire(0.25, 2, 0.0)
        return

    print('aim:', blaster.aim_at(0, 0, 1))
    print('aim:', blaster.aim_at(0, 0, -1))
    print('aim:', blaster.aim_at(0, 0, 0))
    print('aim:', blaster.aim_at(0, 20, 0, True))
    print('aim:', blaster.aim_at(0, 20, 0, False))
    print('aim:', blaster.aim_at(0, 20, -2, True))
    print('aim:', blaster.aim_at(0, 20, -2, False))
    blaster.aim_at(0,.1,0,False)

def ground_plane1():
    ''' Map every pixel of the camera's image to a physical (X, Y, Z) ground
        point, in the camera's own frame (X=right, Y=forward, Z=up, origin
        at the camera) - the same convention aim_at() uses (modulo the
        pivot-vs-camera-center offset, left for later). The actual geometry
        lives in bbnk.ground.GroundPlane; this just loads the calibration
        and hands it plain numbers.
    '''
    h = ft2m(6)    # heigh the camera is above the ground
    pitch = d2r(0) # looking straight ahead, horiztonally
    roll = d2r(0) # camera is level and not rotated
    calibfn = '../data/camera_calib.yaml'

    with open(calibfn) as f:
        calib = yaml.safe_load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    width, img_height = calib['image_width'], calib['image_height']

    ground = GroundPlane(h, pitch, roll)
    print('down_cam (straight-down direction, in camera frame):', ground.down_cam)
    print('ground plane, in camera coords: down_cam . P = h =', h)

    ground_xyz = ground.image_to_ground(camera_matrix, width, img_height, dist_coeffs)
    print('ground_xyz',np2PrettyStr(ground_xyz))

    print('ground_xyz shape:', ground_xyz.shape)
    for name, (u, v) in {
        'center': (width // 2, img_height // 2),
        'bottom-center': (width // 2, img_height - 1),
        'top-center': (width // 2, 0),
    }.items():
        print(f'{name} pixel ({u},{v}) -> ground XYZ (camera frame, m):', ground_xyz[v, u])
        print('   -> world frame,m:   ',ground.to_world(ground_xyz[v,u]))

def ground_squirt1(height_m, pitch_rad, roll_rad, calibfn):
    ''' Live version of camera_view.py: shows the camera feed in a window,
        and on left-click, maps the clicked pixel to a ground XYZ (via
        GroundPlane, as in ground_plane1) and has the Blaster (params from
        s5) ready_aim_fire at it. Press 'q' to quit.
    '''
    import cv2
    from picamera2 import Picamera2

    with open(calibfn) as f:
        calib = yaml.safe_load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    width, img_height = calib['image_width'], calib['image_height']

    ground = GroundPlane(height_m, pitch_rad, roll_rad)

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

    def on_click(event, u, v, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ground_cam = ground.pixel_to_ground(u, v, camera_matrix, dist_coeffs)
        if np.isnan(ground_cam[0]):
            print(f'pixel ({u},{v}) is above the horizon, no ground point to shoot')
            return
        world_xyz = ground.to_world(ground_cam)
        print(f'pixel ({u},{v}) -> world XYZ (m): ({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})')
        # aim_at/ready_aim_fire want camera-frame coords (standing in for
        # turret-frame until T_c2t is defined - see GroundPlane.to_camera).
        x, y, z = ground.to_camera(world_xyz)
        try:
            blaster.ready_aim_fire(x, y, z)
        except ValueError as e:
            print(f'  cannot aim there: {e}')

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (width, img_height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    window_name = 'Ground Squirt'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_click)

    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        blaster.close()

def main():
    # return ground_squirt1(ft2m(6), d2r(0), d2r(0), '../data/camera_calib.yaml') # tie ground plane into gun, with gui
    return ground_plane1() # ground plane estimation using height and pitch, camera intrinsics
    #return s2() # blaster servo calibration to find 0,0 point (aim level straight ahead)
    return s5() # 1st full blaster instantiation
    #return s4()
    #return s3()
    return s1()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bunny Blaster 9000 - main application.

Params (camera intrinsics/position, blaster servo+solenoid params, yolo
model name) live in a YAML config, default data/bb9k_config.yml. This stage
wires up the three pieces built so far - the YOLO detector, the Blaster
(yaw/pitch servos + solenoid), and the camera (via GroundPlane, using the
calibration the config points at) - and runs the same live applet as
ground_squirt1 in scripts/servo_test.py: camera feed with horizon/max-range
overlays, left-click a ground point to ready/aim/fire the blaster at it.
Every frame is also run through the YOLO model and any detections are drawn
as bounding boxes over the feed. Of the detections whose centroid lands in
the "can fire" zone (below the horizon and within the blaster's max range),
the highest-confidence one is fired at automatically. Left-click still fires
at an arbitrary ground point too.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from bbnk.blaster import Blaster
from bbnk.ground import GroundPlane
from bbnk.utils import d2r

DEFAULT_CONFIG = REPO_ROOT / 'data' / 'bb9k_config.yml'


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_calibration(calib_path):
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    return camera_matrix, dist_coeffs, calib['image_width'], calib['image_height']


def create_ground_plane(camera_cfg):
    """Build the GroundPlane (camera position) + intrinsics, from config."""
    camera_matrix, dist_coeffs, width, height = load_calibration(REPO_ROOT / camera_cfg['calib_file'])
    ground = GroundPlane(
        camera_cfg['height_m'],
        d2r(camera_cfg['pitch_deg']),
        d2r(camera_cfg['roll_deg']),
    )
    return ground, camera_matrix, dist_coeffs, width, height


def create_blaster(blaster_cfg):
    return Blaster(
        yaw_servo_params=blaster_cfg['yaw_servo'],
        pitch_servo_params=blaster_cfg['pitch_servo'],
        solenoid_gpio_pin=blaster_cfg['solenoid_gpio_pin'],
        water_velocity_mps=blaster_cfg['water_velocity_mps'],
        yaw_invert=blaster_cfg.get('yaw_invert', False),
        pitch_invert=blaster_cfg.get('pitch_invert', False),
        yaw_zero_offset_deg=blaster_cfg.get('yaw_zero_offset_deg'),
        pitch_zero_offset_deg=blaster_cfg.get('pitch_zero_offset_deg'),
    )


def create_yolo_model(yolo_cfg):
    model_path = REPO_ROOT / yolo_cfg['model_name']
    print('loading yolo model:', model_path)
    return YOLO(str(model_path))


def draw_detections(frame, results, model):
    """Draw yolo boxes/labels (as in yolo_test.py's y1()) onto frame, in place."""
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls)]
        label = f'{cls_name} {float(box.conf):.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def fire_at_ground_point(ground, blaster, ground_cam, label):
    world_xyz = ground.to_world(ground_cam)
    print(f'{label} -> world XYZ (m): ({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})')
    # aim_at/ready_aim_fire want camera-frame coords (standing in for
    # turret-frame until T_c2t is defined - see GroundPlane.to_camera).
    x, y, z = ground.to_camera(world_xyz)
    try:
        blaster.ready_aim_fire(x, y, z)
    except ValueError as e:
        print(f'  cannot aim there: {e}')


def find_best_target(results, model, ground, camera_matrix, dist_coeffs, max_range_m):
    """Highest-confidence detection whose centroid is in the "can fire" zone.

    Returns (ground_cam, label) for the winner, or None if no detection's
    centroid maps to a ground point (below the horizon) within max_range_m.
    """
    best = None  # (conf, ground_cam, label)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        ground_cam = ground.pixel_to_ground(cx, cy, camera_matrix, dist_coeffs)
        if np.isnan(ground_cam[0]) or np.hypot(ground_cam[0], ground_cam[1]) > max_range_m:
            continue
        conf = float(box.conf)
        if best is None or conf > best[0]:
            cls_name = model.names[int(box.cls)]
            best = (conf, ground_cam, f'{cls_name} {conf:.2f} at pixel ({cx:.0f},{cy:.0f})')
    return best[1:] if best is not None else None


def run_ground_squirt(ground, camera_matrix, dist_coeffs, width, height, blaster, yolo_model):
    """Live camera feed; left-click a ground point to ready/aim/fire at it.

    Every frame is also run through yolo_model; the best in-range detection
    (see find_best_target) is auto-fired at. Same applet as ground_squirt1
    in scripts/servo_test.py, plus detection.
    """
    from picamera2 import Picamera2

    horizon_pts = ground.horizon_points(camera_matrix, width, dist_coeffs).astype(np.int32)
    max_range_m = blaster.max_horizontal_range_m(-ground.height_m)
    can_hit_pts = ground.max_range_points(camera_matrix, width, max_range_m, dist_coeffs).astype(np.int32)

    def on_click(event, u, v, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ground_cam = ground.pixel_to_ground(u, v, camera_matrix, dist_coeffs)
        if np.isnan(ground_cam[0]):
            print(f'pixel ({u},{v}) is above the horizon, no ground point to shoot')
            return
        fire_at_ground_point(ground, blaster, ground_cam, label=f'pixel ({u},{v})')

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    window_name = 'Bunny Blaster 9000'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_click)

    try:
        while True:
            frame = picam2.capture_array()
            results = yolo_model.predict(frame, verbose=False)
            draw_detections(frame, results, yolo_model)
            target = find_best_target(results, yolo_model, ground, camera_matrix, dist_coeffs, max_range_m)
            if target is not None:
                fire_at_ground_point(ground, blaster, *target)
            cv2.polylines(frame, [horizon_pts], isClosed=False, color=(0, 255, 255), thickness=2)
            cv2.polylines(frame, [can_hit_pts], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='path to bb9k YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)

    print('=== Bunny Blaster 9000 ===')
    yolo_model = create_yolo_model(cfg['yolo'])
    ground, camera_matrix, dist_coeffs, width, height = create_ground_plane(cfg['camera'])
    blaster = create_blaster(cfg['blaster'])

    try:
        run_ground_squirt(ground, camera_matrix, dist_coeffs, width, height, blaster, yolo_model)
    finally:
        blaster.close()


if __name__ == '__main__':
    main()

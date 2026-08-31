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
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from bbnk.blaster import Blaster
from bbnk.camera import Resolution, crop_camera_matrix
from bbnk.ground import GroundPlane
from bbnk.log import EventLogger, ImageLogger
from bbnk.motiondet import MotionDetector
from bbnk.utils import d2r, ft2m, isp_apply

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


def create_camera_intrinsics(camera_cfg):
    """Load calibration and resolve it to the configured capture resolution.

    camera_cfg['resolution'] is a bbnk.camera.Resolution member name (the
    actual capture size, which may differ from the calibration's) -
    camera_matrix is recomputed for it via crop_camera_matrix, exact for
    all 4 modes (including the two that are sensor crops, not just a
    uniform scale of the calibrated frame). Munged once here, at startup,
    so everything downstream (GroundPlane, Picamera2's own config, ...)
    just uses camera_matrix/resolution as-is.

    Returns (camera_matrix, dist_coeffs, resolution).
    """
    camera_matrix, dist_coeffs, calib_width, calib_height = load_calibration(camera_cfg['calib_file'])

    try:
        resolution = Resolution[camera_cfg['resolution']]
    except KeyError:
        raise ValueError(
            f"unknown camera.resolution {camera_cfg['resolution']!r}; "
            f"must be one of {[r.name for r in Resolution]}"
        )
    if (resolution.width, resolution.height) != (calib_width, calib_height):
        camera_matrix = crop_camera_matrix(camera_matrix, (calib_width, calib_height), resolution)

    return camera_matrix, dist_coeffs, resolution


def create_ground_plane(camera_cfg):
    """Build the GroundPlane (camera position - height/pitch/roll) from config.

    height_ft, if present, wins over height_m. No intrinsics/resolution
    involved - purely the camera's pose - see create_camera_intrinsics for
    those.
    """
    height_ft = camera_cfg.get('height_ft')
    height_m = ft2m(height_ft) if height_ft is not None else camera_cfg['height_m']
    return GroundPlane(
        height_m,
        d2r(camera_cfg['pitch_deg']),
        d2r(camera_cfg['roll_deg']),
    )


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


def create_motion_detector(motion_cfg):
    return MotionDetector(**motion_cfg)


def create_yolo_model(yolo_cfg):
    model_path = yolo_cfg['model_name']
    print('loading yolo model:', model_path)
    return YOLO(str(model_path))


def create_loggers(log_cfg):
    event_logger = EventLogger(
        log_cfg['event_log_file'],
        max_bytes=log_cfg['event_log_max_bytes'],
        prune_fraction=log_cfg['event_log_prune_fraction'],
    )
    image_logger = ImageLogger(
        log_cfg['image_dir'],
        max_bytes=log_cfg['image_max_bytes'],
        prune_fraction=log_cfg['image_prune_fraction'],
    )
    labeled_image_logger = ImageLogger(
        log_cfg['labeled_image_dir'],
        max_bytes=log_cfg['labeled_image_max_bytes'],
        prune_fraction=log_cfg['labeled_image_prune_fraction'],
    )
    return event_logger, image_logger, labeled_image_logger


def yolo_detection_boxes(results, model):
    """(x1, y1, x2, y2, label) for each box in a YOLO results object - for draw_detections."""
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls)]
        boxes.append((x1, y1, x2, y2, f'{cls_name} {float(box.conf):.2f}'))
    return boxes


def draw_detections(frame, boxes):
    """Draw bounding boxes with labels onto frame, in place.

    boxes: iterable of (x1, y1, x2, y2, label) - see yolo_detection_boxes.
    """
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def find_best_target(results, model, ground, camera_matrix, dist_coeffs, max_range_m):
    """Highest-confidence detection whose centroid is in the "can fire" zone.

    Returns (ground_cam, label, detection) for the winner, or None if no
    detection's centroid maps to a ground point (below the horizon) within
    max_range_m. detection is a dict of the winning box's class/confidence/
    bbox/pixel centroid, for event logging.
    """
    best = None  # (conf, ground_cam, label, detection)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        ground_cam = ground.pixel_to_ground(cx, cy, camera_matrix, dist_coeffs)
        if np.isnan(ground_cam[0]) or np.hypot(ground_cam[0], ground_cam[1]) > max_range_m:
            continue
        conf = float(box.conf)
        if best is None or conf > best[0]:
            cls_name = model.names[int(box.cls)]
            detection = {
                'class': cls_name,
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'pixel': [cx, cy],
            }
            best = (conf, ground_cam, f'{cls_name} {conf:.2f} at pixel ({cx:.0f},{cy:.0f})', detection)
    return best[1:] if best is not None else None


def fire_at_ground(ground, blaster, ground_cam):
    """Aim blaster at ground_cam (a camera-frame ground point) and fire.

    Returns (world_xyz, camera_xyz, aim, error) for the caller's own
    logging: world_xyz/camera_xyz are ground_cam converted to world and
    camera/turret frame (see GroundPlane.to_world/.to_camera); aim is the
    AimSolution actually fired at, or None if the target was out of range
    (camera_xyz is still returned in that case); error is that failure's
    message, or None on success.
    """
    world_xyz = ground.to_world(ground_cam)
    # aim_at/ready_aim_fire want camera-frame coords (standing in for
    # turret-frame until T_c2t is defined - see GroundPlane.to_camera).
    camera_xyz = ground.to_camera(world_xyz)
    aim, error = None, None
    try:
        aim = blaster.ready_aim_fire(*camera_xyz)
    except ValueError as e:
        error = str(e)
        print(f'  cannot aim there: {e}')
    return world_xyz, camera_xyz, aim, error


def run_ground_squirt(ground, camera_matrix, dist_coeffs, resolution, blaster, yolo_model, score_thresh,
                       auto_exposure=False, auto_gamma=2.2, event_logger=None, image_logger=None,
                       labeled_image_logger=None, motion_detector=None):
    """Live camera feed; left-click a ground point to ready/aim/fire at it.

    Every frame is also run through yolo_model, keeping only detections with
    confidence >= score_thresh; the best in-range one (see find_best_target)
    is auto-fired at. Same applet as ground_squirt1 in scripts/servo_test.py,
    plus detection.

    auto_exposure: if True, each frame's brightness/contrast is stretched
    via bbnk.utils.isp_apply (using auto_gamma) before detection/display.

    event_logger/image_logger: if given, every fire (manual click or
    auto-detect) is recorded - a JSONL record of the detection/aim/result,
    and the frame it fired on. labeled_image_logger, if given, additionally
    saves that same frame with detection boxes/labels drawn on it (see
    draw_detections) - a separate copy, so the plain frame saved by
    image_logger is untouched.

    motion_detector: if given, every frame is also run through it (see
    bbnk.motiondet.MotionDetector) and its per-frame diagnostics are
    printed. Not wired into firing yet - purely observational.
    """
    from picamera2 import Picamera2

    width, height = resolution.width, resolution.height
    horizon_pts = ground.horizon_points(camera_matrix, width, dist_coeffs).astype(np.int32)
    max_range_m = blaster.max_horizontal_range_m(-ground.height_m)
    can_hit_pts = ground.max_range_points(camera_matrix, width, max_range_m, dist_coeffs).astype(np.int32)
    frame = None
    labeled_frame = None

    def on_click(event, u, v, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ground_cam = ground.pixel_to_ground(u, v, camera_matrix, dist_coeffs)
        if np.isnan(ground_cam[0]):
            print(f'pixel ({u},{v}) is above the horizon, no ground point to shoot')
            return
        label = f'pixel ({u},{v})'

        world_xyz, camera_xyz, aim, error = fire_at_ground(ground, blaster, ground_cam)
        x, y, z = camera_xyz
        print(f'{label} -> world XYZ (m): ({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})')

        image_file = image_logger.save(frame) if image_logger is not None and frame is not None else None
        # labeled_frame is derivable from image_file + this record's detection
        # data, so labeled_image_logger just saves it for convenience, not logged.
        # There's no yolo detection box for a manual click - stub one in around
        # the clicked pixel, labeled the same as this event's own `label`.
        if labeled_image_logger is not None and frame is not None:
            manual_labeled_frame = frame.copy()
            box_half = 15
            draw_detections(manual_labeled_frame, [(u - box_half, v - box_half, u + box_half, v + box_half, label)])
            labeled_image_logger.save(manual_labeled_frame)
        if event_logger is not None:
            event_logger.log(
                trigger='manual',
                label=label,
                detection=None,
                pixel=[u, v],
                # bbox/pixel above are absolute pixel coords - meaningless without
                # the frame size they were measured in.
                image_size=(width, height),
                ground_cam=[float(c) for c in ground_cam],
                world_xyz=[float(c) for c in world_xyz],
                camera_xyz=[float(x), float(y), float(z)],
                aim=None if aim is None else {'yaw_deg': aim.yaw_deg, 'pitch_deg': aim.pitch_deg},
                fired=aim is not None,
                error=error,
                image_file=image_file,
            )

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
            if auto_exposure:
                frame = isp_apply(frame, gamma=auto_gamma)
            # results = [SimpleNamespace(boxes=[])]  # stub: no detections
            results = yolo_model.predict(frame, conf=score_thresh, verbose=False)
            labeled_frame = frame.copy()
            draw_detections(labeled_frame, yolo_detection_boxes(results, yolo_model))
            #cv2.polylines(frame, [horizon_pts], isClosed=False, color=(0, 255, 255), thickness=2)
            #cv2.polylines(frame, [can_hit_pts], isClosed=False, color=(0, 0, 255), thickness=2)

            if motion_detector is not None:
                motion_detector.process(frame)
                print(
                    f'motion: p90_diff={motion_detector.p90_diff} '
                    f'noise_floor={motion_detector.noise_floor} '
                    f'dynamic_thresh={motion_detector.dynamic_thresh} '
                    f'effective_thresh={motion_detector.effective_thresh} '
                    f'diff_thresh={motion_detector.diff_thresh}'
                )

            # Show this frame (with its boxes) before firing, so the window
            # holds the actual image being fired on while ready_aim_fire blocks.
            if 0:
                cv2.imshow(window_name, frame)
            if 0:
                # frame diff mask
                cv2.imshow(window_name, cv2.cvtColor(motion_detector.get_motion_mask(), cv2.COLOR_GRAY2BGR))
            if 1: # frame diff overlaid on img
                # debug view: frame with motion_detector's mask (upscaled to
                # frame's resolution) highlighted in red - flip this only
                # when motion_detector is actually enabled above.
                mask_full = cv2.resize(motion_detector.get_motion_mask(), (frame.shape[1], frame.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                overlay = frame.copy()
                overlay[mask_full > 0] = (0, 0, 255)
                cv2.imshow(window_name, cv2.addWeighted(frame, 0.5, overlay, 0.5, 0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            target = find_best_target(results, yolo_model, ground, camera_matrix, dist_coeffs, max_range_m)
            if target is None:
                continue

            ground_cam, label, detection = target

            world_xyz, camera_xyz, aim, error = fire_at_ground(ground, blaster, ground_cam)
            x, y, z = camera_xyz
            print(f'{label} -> world XYZ (m): ({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})')

            image_file = image_logger.save(frame) if image_logger is not None else None
            # labeled_frame is derivable from image_file + this record's detection
            # data, so labeled_image_logger just saves it for convenience, not logged.
            if labeled_image_logger is not None:
                labeled_image_logger.save(labeled_frame)
            if event_logger is not None:
                event_logger.log(
                    trigger='auto',
                    label=label,
                    detection=detection,
                    pixel=None,
                    # bbox/pixel above are absolute pixel coords - meaningless without
                    # the frame size they were measured in.
                    image_size=(width, height),
                    ground_cam=[float(c) for c in ground_cam],
                    world_xyz=[float(c) for c in world_xyz],
                    camera_xyz=[float(x), float(y), float(z)],
                    aim=None if aim is None else {'yaw_deg': aim.yaw_deg, 'pitch_deg': aim.pitch_deg},
                    fired=aim is not None,
                    error=error,
                    image_file=image_file,
                )
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
    camera_matrix, dist_coeffs, resolution = create_camera_intrinsics(cfg['camera'])
    ground = create_ground_plane(cfg['camera'])
    blaster = create_blaster(cfg['blaster'])
    event_logger, image_logger, labeled_image_logger = create_loggers(cfg['log'])
    motion_detector = create_motion_detector(cfg['motion'])

    try:
        run_ground_squirt(
            ground, camera_matrix, dist_coeffs, resolution, blaster, yolo_model,
            score_thresh=cfg['yolo']['score_thresh'],
            auto_exposure=cfg['camera'].get('auto_exposure', False),
            auto_gamma=cfg['camera'].get('auto_gamma', 2.2),
            event_logger=event_logger,
            image_logger=image_logger,
            labeled_image_logger=labeled_image_logger,
            motion_detector=motion_detector,
        )
    finally:
        blaster.close()


if __name__ == '__main__':
    main()

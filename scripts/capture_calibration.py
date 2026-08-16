#!/usr/bin/env python3
"""Capture still images for camera calibration via Picamera2.

Shows a live preview at the sensor's native resolution (or --size) with
auto exposure/white balance running. Press 'l' once the scene/lighting
looks right to lock exposure, gain, and white balance so every capture
afterward is consistent. Press space/'c' to save a PNG to --out-dir,
'u' to unlock exposure again, 'q' to quit.
"""

import argparse
from pathlib import Path

import cv2
from picamera2 import Picamera2

DEFAULT_OUT_DIR = "calib_out"


def parse_size(value):
    try:
        width_str, height_str = value.lower().split("x")
        return (int(width_str), int(height_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}, expected WIDTHxHEIGHT (e.g. 2592x1944)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=parse_size,
        default=None,
        help="capture resolution as WIDTHxHEIGHT (default: sensor's native resolution)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"directory to write calibration PNGs to (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    size = args.size or picam2.camera_properties["PixelArraySize"]
    print(f"Using resolution: {size[0]}x{size[1]}")

    config = picam2.create_still_configuration(main={"size": size, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    window_name = "Calibration Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    locked = False
    shot_count = len(list(out_dir.glob("*.png")))

    try:
        while True:
            frame = picam2.capture_array()

            status = f"{'LOCKED' if locked else 'AUTO'} | shots: {shot_count}"
            help_text = "l=lock  u=unlock  space/c=capture  q=quit"
            display = frame.copy()
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, help_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("l") and not locked:
                metadata = picam2.capture_metadata()
                picam2.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": metadata["ExposureTime"],
                    "AnalogueGain": metadata["AnalogueGain"],
                    "ColourGains": metadata["ColourGains"],
                })
                locked = True
                print(
                    f"Locked: ExposureTime={metadata['ExposureTime']}, "
                    f"AnalogueGain={metadata['AnalogueGain']:.2f}, "
                    f"ColourGains={metadata['ColourGains']}"
                )

            elif key == ord("u") and locked:
                picam2.set_controls({"AeEnable": True, "AwbEnable": True})
                locked = False
                print("Unlocked: AE/AWB running again")

            elif key in (ord(" "), ord("c")):
                shot_count += 1
                out_path = out_dir / f"calib_{shot_count:03d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"Saved {out_path}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

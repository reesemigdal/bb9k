#!/usr/bin/env python3
"""Open the CSI camera (OV5647 sensor) via Picamera2 and show a live feed.

Prints the camera's properties and supported sensor modes on startup.
Defaults to the sensor's native resolution unless --size is given.

Press 'q' to quit.
"""

import argparse

import cv2
from picamera2 import Picamera2


def parse_size(value):
    try:
        width_str, height_str = value.lower().split("x")
        return (int(width_str), int(height_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}, expected WIDTHxHEIGHT (e.g. 1280x720)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=parse_size,
        default=None,
        help="capture resolution as WIDTHxHEIGHT (default: sensor's native resolution)",
    )
    args = parser.parse_args()

    picam2 = Picamera2()

    print("Camera properties:")
    for key, value in picam2.camera_properties.items():
        print(f"  {key}: {value}")

    print("Sensor modes:")
    for mode in picam2.sensor_modes:
        print(f"  {mode}")

    size = args.size or picam2.camera_properties["PixelArraySize"]
    print(f"Using resolution: {size[0]}x{size[1]}")

    # create_video_configuration (vs. create_preview_configuration) uses
    # HighQuality noise reduction instead of the Fast/blocky denoiser, which
    # otherwise shows up as JPEG-like edge artifacting at full resolution.
    config = picam2.create_video_configuration(
        main={"size": size, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    window_name = "OV5647 Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

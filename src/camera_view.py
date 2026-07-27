#!/usr/bin/env python3
"""Open the CSI camera (OV5647 sensor) via Picamera2 and show a live feed.

Press 'q' to quit.
"""

import cv2
from picamera2 import Picamera2

WIDTH = 1280
HEIGHT = 720


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
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

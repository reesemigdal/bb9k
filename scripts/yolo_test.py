import time

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def y1():
    ''' load yolov8n and run detection on ultralytics' bundled sample image (bus.jpg)
        confirms the model downloads, cv2 loads, and inference produces real detections
        annotated result is saved next to this script
    '''
    model = YOLO('../data/yolov8n.pt')
    results = model.predict(ASSETS / 'bus.jpg', verbose=True)

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        print(f'{cls_name}: conf={float(box.conf):.2f} xyxy={box.xyxy.tolist()[0]}')

    out_path = 'yolo_test_output.jpg'
    results[0].save(out_path)
    print('annotated image saved to:', out_path)

def y2():
    ''' benchmark cpu inference speed on a synthetic 640x480 frame
        useful for gauging real-world fps on the pi 5
    '''
    model = YOLO('../data/yolov8n.pt')
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    model.predict(img, verbose=False)  # warmup

    n = 10
    t0 = time.time()
    for _ in range(n):
        model.predict(img, verbose=False)
    dt = time.time() - t0
    print(f'{n} frames in {dt:.2f}s -> {dt/n*1000:.1f}ms/frame -> {n/dt:.1f} fps')

def main():
    return y1() # real-image detection sanity check
    #return y2() # cpu inference speed benchmark

if __name__ == "__main__":
    main()

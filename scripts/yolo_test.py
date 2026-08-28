import time

import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt


def y1():
    ''' load a local test image manually with opencv, run detection, and get the
        raw model outputs (boxes) - no results.plot()/save(), boxes/labels are
        drawn by hand and plotted manually to confirm the full pipeline works
    '''
    modelfn = '../data/bundet.pt'
    modelfn = '../train/runs/bb9k_animal-2/weights/best.pt'
    imgfn = '../data/bus.jpg'
    imgfn = '../data/bunny2.webp'

    print('loading',modelfn)
    model = YOLO(modelfn)

    print('loading ',imgfn)
    img = cv2.imread(imgfn)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print('inferencing')
    n = 1
    for imgsz in (640, ):
        times = []
        for _ in range(n):
            t0 = time.time()
            results = model.predict(img, imgsz=imgsz, verbose=False)
            times.append(time.time() - t0)
        times.sort()
        median = times[n // 2]
        print(f'imgsz={imgsz}: {median*1000:.1f}ms/frame (median) -> {1/median:.1f} fps')

    boxes = results[0].boxes  # last pass (imgsz=320)

    for box in boxes:
        x1, y1_, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls)]
        label = f'{cls_name} {float(box.conf):.2f}'
        print(f'{label} xyxy=({x1}, {y1_}, {x2}, {y2})')

        cv2.rectangle(img, (x1, y1_), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1_ - th - 6), (x1 + tw, y1_), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1_ - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    out_path = 'yolo_test_output.jpg'
    cv2.imwrite(out_path, img)
    print('annotated image saved to:', out_path)

    plt.imshow(img[..., ::-1])  # bgr -> rgb
    plt.axis('off')
    plt.show()

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

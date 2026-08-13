#!/usr/bin/env python3
"""Export a YOLO26n COCO model to NCNN (FP16) for CPU inference on the Pi 5."""

from ultralytics import YOLO


def export_ncnn_fp16(model_name="yolo26n.pt", imgsz=640):
    model = YOLO(model_name)
    return model.export(format="ncnn", imgsz=imgsz, quantize=16)


def main():
    path = export_ncnn_fp16()
    print("exported to:", path)


if __name__ == "__main__":
    main()

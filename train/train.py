#!/usr/bin/env python3
"""Train YOLOv8n on the Caltech Camera Traps animal-detection dataset.

Run train/caltech/prepare_data.py first to (re)generate labels/, train.txt,
val.txt and data.yaml under ~/data/bb9k/caltech-camera-traps/.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path.home() / "data/bb9k/caltech-camera-traps/data.yaml"
RUNS_DIR = Path(__file__).parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="bb9k_animal")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found — run train/caltech/prepare_data.py first")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
    )


if __name__ == "__main__":
    main()

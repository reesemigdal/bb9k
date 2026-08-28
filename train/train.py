#!/usr/bin/env python3
"""Train YOLOv8n on an animal-detection dataset (see DATASETS below).

Each dataset's own prepare_data.py (train/caltech/, train/combined/, ...)
must be run first to generate its labels/, train.txt, val.txt and
data.yaml.
"""
import argparse
from pathlib import Path

DATASETS = {
    "caltech": Path.home() / "data/bb9k/caltech-camera-traps/data.yaml",
    "oregon": Path.home() / "data/bb9k/oregon-critters/data.yaml",
    "combined": Path.home() / "data/bb9k/combined-animal/data.yaml",
    "combined-full": Path.home() / "data/bb9k/combined-animal-full/data.yaml",
    "combined-full-domestic": Path.home() / "data/bb9k/combined-animal-full-domestic/data.yaml",
}
RUNS_DIR = Path(__file__).parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="combined")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="bb9k_animal")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="CHECKPOINT",
        help="resume an interrupted run. Bare --resume looks for "
        "train/runs/<name>/weights/last.pt; pass a path to resume a specific checkpoint "
        "instead. Epochs/model/data are restored from the checkpoint's own saved args — "
        "only --imgsz/--batch/--device/--workers above still take effect.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Ultralytics' TensorBoard logging (writes event files into the run's "
        "save dir; also persists as a global 'tensorboard' setting in Ultralytics' own "
        "settings.json, so it sticks for future runs too)",
    )
    args = parser.parse_args()

    data_yaml = DATASETS[args.dataset]
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found — run that dataset's prepare_data.py first")

    # must run before importing YOLO: the TensorBoard callback checks this setting once,
    # at import time, and is a no-op for the rest of the process if it was False then.
    from ultralytics import settings

    settings.update({"tensorboard": args.tensorboard})

    from ultralytics import YOLO

    if args.resume:
        checkpoint = args.resume if isinstance(args.resume, str) else str(RUNS_DIR / args.name / "weights" / "last.pt")
        if not Path(checkpoint).exists():
            raise SystemExit(f"{checkpoint} not found — nothing to resume")
        model = YOLO(checkpoint)
    else:
        model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

"""
train.py — YOLOv11 Object Detection Training Script
=====================================================
Usage:
    python train.py                        # uses defaults below
    python train.py --data data/data.yaml --model yolo11s.pt --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
#  DEFAULTS  (edit here or override via CLI args)
# ─────────────────────────────────────────────────────────────
DEFAULT_DATA    = "data/data.yaml"   # relative to project root
DEFAULT_MODEL   = "yolo11s.pt"       # pretrained backbone: n / s / m / l / x
DEFAULT_EPOCHS  = 200
DEFAULT_IMGSZ   = 640
DEFAULT_BATCH   = 8                  # lower to 8 or 4 if GPU runs out of memory
DEFAULT_DEVICE  = ""                 # "" = auto (GPU if available, else CPU)
DEFAULT_NAME    = "gestures_aug_run"  # folder name inside runs/detect/
DEFAULT_WORKERS = 4


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLOv11 detection model")
    p.add_argument("--data",    default=DEFAULT_DATA,    help="Path to data.yaml")
    p.add_argument("--model",   default=DEFAULT_MODEL,   help="Pretrained weights e.g. yolo11n.pt")
    p.add_argument("--epochs",  default=DEFAULT_EPOCHS,  type=int)
    p.add_argument("--imgsz",   default=DEFAULT_IMGSZ,   type=int, help="Input image size (pixels)")
    p.add_argument("--batch",   default=DEFAULT_BATCH,   type=int)
    p.add_argument("--device",  default=DEFAULT_DEVICE,  help="0 for GPU, 'cpu' for CPU")
    p.add_argument("--name",    default=DEFAULT_NAME,    help="Run name (under runs/detect/)")
    p.add_argument("--workers", default=DEFAULT_WORKERS, type=int)
    p.add_argument("--cls",     default=0.7,             type=float, help="Classification loss weight")
    p.add_argument("--patience",default=50,              type=int,   help="Early stopping patience")
    p.add_argument("--resume",  action="store_true",     help="Resume from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    # ── Resolve relative paths from the script directory ─────────────────────────
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    # ── Validate data.yaml ─────────────────────────────────────────────────────
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] data.yaml not found at: '{data_path.resolve()}'\n"
            f"        Make sure the dataset is placed correctly.\n"
            f"        Expected project layout relative to train.py:\n\n"
            f"          letters_project/\n"
            f"          ├── train.py\n"
            f"          ├── detect.py\n"
            f"          └── data/\n"
            f"              ├── data.yaml\n"
            f"              ├── train/\n"
            f"              │   ├── images/\n"
            f"              │   └── labels/\n"
            f"              └── valid/\n"
            f"                  ├── images/\n"
            f"                  └── labels/\n"
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    if args.resume:
        last_ckpt = project_root / f"runs/detect/{args.name}/weights/last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"[ERROR] No checkpoint to resume from: {last_ckpt}")
        print(f"[INFO] Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(str(model_path))   # downloads pretrained weights automatically on first run

    # ── Print config ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  YOLOv11 Training — Letters Dataset")
    print(f"{'='*55}")
    print(f"  Model   : {args.model}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img size: {args.imgsz}px")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device or 'auto'}")
    print(f"  Run name: runs/detect/{args.name}/")
    print(f"{'='*55}\n")

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train(
        data      = str(data_path),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        project   = str(project_root / "runs"),
        name      = args.name,
        workers   = args.workers,
        patience  = args.patience,  # stop early if mAP doesn't improve
        save      = True,      # saves best.pt and last.pt
        plots     = True,      # confusion matrix, F1 curve, loss plots
        # ── augmentation (important for small datasets like ours) ───────────
        augment   = True,
        cls       = args.cls,  # classification loss weight
        mosaic    = 1.0,       # mosaic augmentation (4 images in one)
        mixup     = 0.2,       # mixup augmentation (blend two images)
        flipud    = 0.3,
        fliplr    = 0.5,
        degrees   = 20.0,      # random rotation ±20°
        shear     = 10.0,      # random shear ±10°
        translate = 0.2,
        scale     = 0.5,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        resume    = args.resume,
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    best = project_root / f"runs/detect/{args.name}/weights/best.pt"
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best weights : {best}")
    print(f"  All results  : runs/detect/{args.name}/")
    print(f"\n  Next step — run live detection:")
    print(f"    python detect.py --weights {best}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

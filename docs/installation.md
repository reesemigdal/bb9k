# Installation

Manual setup commands for the bb9k dev environment on the Raspberry Pi 5.

## Prerequisites

- Raspberry Pi 5, Debian trixie (aarch64), Python 3.13
- `pip` is configured with `www.piwheels.org` as an extra index (`/etc/pip.conf`), which provides prebuilt aarch64 wheels for most packages

Create the venv from the project root, with access to system site-packages (needed for Pi-specific packages like `picamera2`, `gpiozero`, `rpi_hardware_pwm` that are installed via apt):

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

## YOLO (ultralytics / torch, CPU inference)

`pip install torch` on this box resolves to a build that declares CUDA/NVIDIA dependencies (`cuda-toolkit`, `nvidia-cudnn`, `nvidia-nccl`, ...) even though there's no GPU to use them. Install from PyTorch's dedicated CPU wheel index instead to skip all of that:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

`ultralytics` normally pulls in `opencv-python`, which would sit alongside `opencv-contrib-python` (a superset) and fight over the `cv2` package. Install `ultralytics` without its dependencies, then install the rest by hand, skipping `opencv-python`:

```bash
pip install --no-deps ultralytics
pip install opencv-contrib-python matplotlib polars nvidia-ml-py ultralytics-thop ultralytics-platform
```

`pip check` will report `ultralytics requires opencv-python` — that's expected; `opencv-contrib-python` provides the same `cv2` module plus extras.

### Verify

```bash
cd scripts
python3 yolo_test.py
```

Downloads `yolov8n.pt` into `../data` on first run, detects objects in ultralytics' bundled sample image, and saves an annotated copy to `yolo_test_output.jpg`.

# bb9k

## Calibration

Print `data/charuco_board.pdf` at 100% scale (verify with the printed ruler),
then capture calibration images and run calibration:

```
python3 scripts/capture_calibration.py
python3 scripts/calibrate_camera.py --square-size-mm 20 --marker-size-mm 15 --output data/camera_calib.yaml
```

Drop `--output` to print the calibration YAML to stdout instead of writing the file.

Other `calibrate_camera.py` flags: `--images-dir` (default `calib_out`), `--board-size`
(default `8x11`), `--aruco-dict` (default `DICT_5X5_250`), `--min-corners` (default `6`).

To regenerate the calibration board (e.g. a different size/dictionary), see
`scripts/generate_charuco_board.py`.

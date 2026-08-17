#!/usr/bin/env python3
"""Calibrate the camera from a set of ChArUco board images (see capture_calibration.py).

Detects ChArUco corners in each image, runs cv2.calibrateCamera to solve for
the camera matrix and distortion coefficients, and reports the reprojection
error. With --output, writes the result as YAML (e.g. data/camera_calib.yaml);
otherwise prints it to stdout.

--square-size-mm and --marker-size-mm are physical measurements of the
printed board and must be supplied explicitly (measure with calipers/ruler).
"""

import argparse
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

DEFAULT_IMAGES_DIR = "calib_out"

ARUCO_DICTS = {
    name: getattr(aruco, name) for name in dir(aruco) if name.startswith("DICT_")
}


def parse_board_size(value):
    try:
        cols_str, rows_str = value.lower().split("x")
        return (int(cols_str), int(rows_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid board size {value!r}, expected SQUARES_XxSQUARES_Y (e.g. 8x11)"
        )


def detect_charuco(image_paths, detector, min_corners):
    """Return (image_size, per_image_obj_points, per_image_img_points, used_paths)."""
    obj_points_list = []
    img_points_list = []
    used_paths = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skipping {path.name}: could not read image", file=sys.stderr)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < min_corners:
            found = 0 if charuco_ids is None else len(charuco_ids)
            print(
                f"  skipping {path.name}: only {found} charuco corners found "
                f"(need >= {min_corners})",
                file=sys.stderr,
            )
            continue

        obj_points, img_points = detector.getBoard().matchImagePoints(
            charuco_corners, charuco_ids
        )
        obj_points_list.append(obj_points)
        img_points_list.append(img_points)
        used_paths.append(path)
        print(f"  found {len(charuco_ids)} corners in {path.name}")

    return image_size, obj_points_list, img_points_list, used_paths


def per_image_reprojection_errors(obj_points_list, img_points_list, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points_list, img_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected.reshape(-1, 2), cv2.NORM_L2) / np.sqrt(len(projected))
        errors.append(error)
    return errors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"directory of calibration images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--board-size",
        type=parse_board_size,
        default=(8, 11),
        help="ChArUco board size as SQUARES_XxSQUARES_Y (default: 8x11)",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        required=True,
        help="chessboard square side length in mm (measure the printed board)",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        required=True,
        help="ArUco marker side length in mm (measure the printed board; always < square-size-mm)",
    )
    parser.add_argument(
        "--aruco-dict",
        default="DICT_5X5_250",
        choices=sorted(ARUCO_DICTS),
        metavar="DICT_NAME",
        help="ArUco dictionary used to generate the board (default: DICT_5X5_250; "
        "try other DICT_5X5_* sizes if detection fails)",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=6,
        help="minimum charuco corners required to use an image (default: 6)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write calibration YAML here (e.g. data/camera_calib.yaml); "
        "if omitted, print the calibration to stdout",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTS[args.aruco_dict])
    board = aruco.CharucoBoard(
        args.board_size, args.square_size_mm, args.marker_size_mm, dictionary
    )
    detector = aruco.CharucoDetector(board)

    print(
        f"Searching for a {args.board_size[0]}x{args.board_size[1]} ChArUco board "
        f"({args.aruco_dict}) in {len(image_paths)} images..."
    )
    image_size, obj_points_list, img_points_list, used_paths = detect_charuco(
        image_paths, detector, args.min_corners
    )

    if len(obj_points_list) < 3:
        print(
            f"Only found usable corners in {len(obj_points_list)} image(s); need at least "
            "a few (ideally 15+) covering varied angles/positions. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Calibrating from {len(obj_points_list)} usable images...")
    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, image_size, None, None
    )

    per_image_errors = per_image_reprojection_errors(
        obj_points_list, img_points_list, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "board_squares_x": args.board_size[0],
        "board_squares_y": args.board_size[1],
        "square_size_mm": args.square_size_mm,
        "marker_size_mm": args.marker_size_mm,
        "aruco_dict": args.aruco_dict,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "rms_reprojection_error": float(rms_error),
        "num_images_used": len(obj_points_list),
        "images_used": [p.name for p in used_paths],
    }

    print("\nPer-image reprojection error (px):")
    for path, error in zip(used_paths, per_image_errors):
        print(f"  {path.name}: {error:.4f}")
    print(f"\nRMS reprojection error: {rms_error:.4f} px")
    print(f"Mean per-image reprojection error: {np.mean(per_image_errors):.4f} px")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.safe_dump(result, f, default_flow_style=False, sort_keys=False)
        print(f"\nWrote calibration to {output_path}")
    else:
        print("\n--- calibration (yaml) ---")
        yaml.safe_dump(result, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()

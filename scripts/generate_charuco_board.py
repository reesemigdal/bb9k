#!/usr/bin/env python3
"""Generate a printable ChArUco calibration board as a PDF (or PNG).

Uses cv2.aruco.CharucoBoard directly, so the printed board is guaranteed
compatible with CharucoDetector (unlike some third-party board generators,
whose marker-to-cell layout can differ from OpenCV's convention and make
corner detection silently fail).

PDF output (the default) places the board on a real page (Letter/A4) sized
in physical points, so "print at actual size" is unambiguous, and prints a
100mm ruler on the page you can check with a tape measure afterward to
confirm nothing got rescaled by the print pipeline. This is more foolproof
than a PNG, whose physical size on paper depends on DPI metadata that some
printers/viewers ignore or reinterpret.
"""

import argparse
import io
from pathlib import Path

import cv2
import cv2.aruco as aruco
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ARUCO_DICTS = {
    name: getattr(aruco, name) for name in dir(aruco) if name.startswith("DICT_")
}

PAGE_SIZES = {"letter": letter, "a4": A4}

MM_PER_INCH = 25.4
RASTER_DPI = 300  # resolution of the embedded marker artwork, independent of print scale


def parse_board_size(value):
    try:
        cols_str, rows_str = value.lower().split("x")
        return (int(cols_str), int(rows_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid board size {value!r}, expected SQUARES_XxSQUARES_Y (e.g. 8x11)"
        )


def render_board_image(board, board_width_mm, board_height_mm):
    px_per_mm = RASTER_DPI / MM_PER_INCH
    width_px = round(board_width_mm * px_per_mm)
    height_px = round(board_height_mm * px_per_mm)
    return board.generateImage((width_px, height_px), marginSize=0, borderBits=1)


def write_pdf(output_path, board_img, board_width_mm, board_height_mm, page_size_name, caption):
    page_w, page_h = PAGE_SIZES[page_size_name]
    ruler_area_h = 20 * mm

    if board_width_mm * mm > page_w or board_height_mm * mm + ruler_area_h > page_h:
        raise SystemExit(
            f"Board ({board_width_mm:.0f}x{board_height_mm:.0f}mm) doesn't fit on a "
            f"{page_size_name} page ({page_w/mm:.0f}x{page_h/mm:.0f}mm). "
            "Use a smaller --square-size-mm, fewer squares, or --page-size a4."
        )

    png_buf = io.BytesIO()
    Image.fromarray(board_img).save(png_buf, format="PNG")
    png_buf.seek(0)
    image_reader = ImageReader(png_buf)

    x = (page_w - board_width_mm * mm) / 2
    y = page_h - (page_h - board_height_mm * mm - ruler_area_h) / 2 - board_height_mm * mm

    c = canvas.Canvas(str(output_path), pagesize=(page_w, page_h))
    c.drawImage(
        image_reader, x, y,
        width=board_width_mm * mm, height=board_height_mm * mm,
    )

    ruler_x0 = x
    ruler_y = y - 12 * mm
    c.setLineWidth(0.75)
    c.line(ruler_x0, ruler_y, ruler_x0 + 100 * mm, ruler_y)
    for tick_mm in range(0, 101, 10):
        tick_h = 2.5 * mm if tick_mm % 50 == 0 else 1.5 * mm
        tx = ruler_x0 + tick_mm * mm
        c.line(tx, ruler_y - tick_h / 2, tx, ruler_y + tick_h / 2)

    c.setFont("Helvetica", 8)
    c.drawString(
        ruler_x0, ruler_y - 6 * mm,
        "^ measure this line: must be exactly 100.0mm. If not, your printer rescaled the "
        "page — disable 'fit to page' / 'scale to fit' and print at 100% / actual size.",
    )
    c.setFont("Helvetica", 8)
    c.drawString(ruler_x0, y - 4 * mm, caption)

    c.showPage()
    c.save()


def write_png(output_path, board_img, board_width_mm, board_height_mm, margin_mm, caption):
    px_per_mm = RASTER_DPI / MM_PER_INCH
    margin_px = round(margin_mm * px_per_mm)

    padded = cv2.copyMakeBorder(
        board_img, margin_px, margin_px, margin_px, margin_px,
        cv2.BORDER_CONSTANT, value=255,
    )
    caption_height_px = round(6 * px_per_mm)
    padded = cv2.copyMakeBorder(
        padded, 0, caption_height_px, 0, 0, cv2.BORDER_CONSTANT, value=255
    )
    cv2.putText(
        padded, caption, (margin_px, padded.shape[0] - margin_px // 2),
        cv2.FONT_HERSHEY_SIMPLEX, px_per_mm * 0.12, 0, 1, cv2.LINE_AA,
    )
    Image.fromarray(padded).save(output_path, dpi=(RASTER_DPI, RASTER_DPI))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--board-size",
        type=parse_board_size,
        default=(8, 11),
        help="board size as SQUARES_XxSQUARES_Y (default: 8x11)",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=20.0,
        help="chessboard square side length in mm (default: 20.0)",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=15.0,
        help="ArUco marker side length in mm (default: 15.0, must be < square-size-mm)",
    )
    parser.add_argument(
        "--aruco-dict",
        default="DICT_5X5_250",
        choices=sorted(ARUCO_DICTS),
        metavar="DICT_NAME",
        help="ArUco dictionary to draw markers from (default: DICT_5X5_250)",
    )
    parser.add_argument(
        "--page-size",
        default="letter",
        choices=sorted(PAGE_SIZES),
        help="page size for PDF output (default: letter)",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=10.0,
        help="white margin around the board for PNG output (default: 10.0; unused for PDF)",
    )
    parser.add_argument(
        "--output",
        default="charuco_board.pdf",
        help="output path; .pdf (default, recommended) or .png",
    )
    args = parser.parse_args()

    if args.marker_size_mm >= args.square_size_mm:
        parser.error("--marker-size-mm must be smaller than --square-size-mm")

    output_path = Path(args.output)
    if output_path.suffix.lower() not in (".pdf", ".png"):
        parser.error("--output must end in .pdf or .png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    squares_x, squares_y = args.board_size
    board_width_mm = squares_x * args.square_size_mm
    board_height_mm = squares_y * args.square_size_mm

    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTS[args.aruco_dict])
    board = aruco.CharucoBoard(
        args.board_size, args.square_size_mm, args.marker_size_mm, dictionary
    )
    board_img = render_board_image(board, board_width_mm, board_height_mm)

    caption = (
        f"OpenCV CharucoBoard | {squares_x}x{squares_y} | square={args.square_size_mm}mm "
        f"| marker={args.marker_size_mm}mm | {args.aruco_dict}"
    )

    if output_path.suffix.lower() == ".pdf":
        write_pdf(output_path, board_img, board_width_mm, board_height_mm, args.page_size, caption)
        print(f"Wrote {output_path} on {args.page_size} paper")
        print(f"Board size: {board_width_mm:.1f}mm x {board_height_mm:.1f}mm")
        print(
            "Print at 100% / actual size (NOT 'fit to page'), then measure the printed "
            "ruler line on the page — it must read exactly 100.0mm."
        )
    else:
        write_png(output_path, board_img, board_width_mm, board_height_mm, args.margin_mm, caption)
        print(f"Wrote {output_path} @ {RASTER_DPI} DPI")
        print(f"Board size: {board_width_mm:.1f}mm x {board_height_mm:.1f}mm")
        print(
            "IMPORTANT: print at actual size / 100% scale, not 'fit to page' — PDF "
            "output is more reliable for this if your print pipeline supports it."
        )

    print(
        f"\nUse with calibrate_camera.py:\n"
        f"  --board-size {squares_x}x{squares_y} --square-size-mm {args.square_size_mm} "
        f"--marker-size-mm {args.marker_size_mm} --aruco-dict {args.aruco_dict}"
    )


if __name__ == "__main__":
    main()

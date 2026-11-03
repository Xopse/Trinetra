"""
test_barcode_checksum.py — Scans a document photo for its Aadhaar
CODE128 barcode (the short linear barcode encoding just the 12-digit
number), and validates it against the Verhoeff checksum — WITHOUT ever
printing the actual number to the terminal.

Usage: python3 test_barcode_checksum.py <path_to_image>
"""

import sys
from PIL import Image, ImageOps
import numpy as np
import cv2
from pyzbar.pyzbar import decode

from verhoeff import validate

if len(sys.argv) != 2:
    print("Usage: python3 test_barcode_checksum.py <path_to_image>")
    sys.exit(1)

pil_img = Image.open(sys.argv[1])
pil_img = ImageOps.exif_transpose(pil_img)
img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

results = decode(img_cv)
barcode_results = [r for r in results if r.type == "CODE128"]

if barcode_results:
    number = barcode_results[0].data.decode("utf-8")
    is_valid = validate(number)
    print(f"Barcode found. Length: {len(number)} digits")
    print(f"Verhoeff checksum valid: {is_valid}")
else:
    print("No CODE128 barcode found in this image.")
    print(f"Other codes found: {[r.type for r in results]}")

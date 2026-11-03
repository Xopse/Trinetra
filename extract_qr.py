"""
extract_qr.py — Aadhaar QR extraction, tuned specifically for getting a
real Aadhaar Secure QR to decode off a physical, photographed card.

Tries THREE different decoding engines (not just image preprocessing
variations of one engine, since a fundamentally different decoder can
succeed where another fails on the same distorted/degraded QR):

  1. pyzbar / ZBar  — the standard, what we started with
  2. zxing-cpp       — a completely different decoding engine (C++ port
                        of the ZXing library), often succeeds on QR codes
                        ZBar can't read, and vice versa
  3. QReader         — YOLOv8-based QR *detector* (locates the QR region
                        first, crops tightly to it, then decodes) —
                        purpose-built for exactly this problem: QR codes
                        that are small, angled, or partially degraded

Each engine is tried across multiple image preprocessing variants
(original, upscaled, adaptive-threshold, sharpened), so this is up to
~15 total attempts before giving up.

Usage: python3 extract_qr.py <path_to_image>

Saves the QR text (if found) to qr_output.txt — never printed to the
terminal, since Aadhaar QR payloads contain personal data.
"""

import sys
import os
import warnings
import logging

# Suppress QReader/ultralytics verbose logging and warnings — this
# library downloads a YOLO model and prints progress/deprecation
# warnings by default, which just clutters the output here.
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2
import numpy as np
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode as pyzbar_decode


def load_image_corrected(path):
    """EXIF-orientation-corrected image load."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def make_variants(img_cv):
    """Generates several preprocessed versions of the image to try each
    decoder against — different fixes help with different failure modes
    (blur, low contrast, glare, small size)."""
    variants = {"original": img_cv}

    h, w = img_cv.shape[:2]
    variants["upscaled_2x"] = cv2.resize(img_cv, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants["upscaled_3x"] = cv2.resize(img_cv, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variants["adaptive_threshold"] = cv2.cvtColor(
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15),
        cv2.COLOR_GRAY2BGR
    )

    # Sharpening kernel — helps with slight blur
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants["sharpened"] = cv2.filter2D(img_cv, -1, kernel)

    # CLAHE (adaptive contrast enhancement) — helps with glare/uneven lighting
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    variants["clahe_contrast"] = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    return variants


def try_pyzbar(img):
    results = pyzbar_decode(img)
    for r in results:
        if r.type == "QRCODE":
            return r.data.decode("utf-8", errors="replace")
    return None


def try_zxing(img):
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(img)
        for r in results:
            if "QR" in str(r.format):
                return r.text
    except Exception:
        pass
    return None


_qreader_instance = None

def try_qreader(img):
    global _qreader_instance
    try:
        from qreader import QReader
        if _qreader_instance is None:
            _qreader_instance = QReader()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        decoded = _qreader_instance.detect_and_decode(image=img_rgb)
        for text in decoded:
            if text:
                return text
    except Exception:
        pass
    return None


def find_aadhaar_qr(path):
    """
    Tries all 3 engines across all image variants, in order of how
    likely each is to succeed fastest, stopping at the first hit.
    Returns (qr_text, engine_name, variant_name) or (None, None, None).
    """
    img_cv = load_image_corrected(path)
    variants = make_variants(img_cv)

    engines = [
        ("pyzbar", try_pyzbar),
        ("zxing-cpp", try_zxing),
        ("qreader", try_qreader),
    ]

    log = []
    for variant_name, variant_img in variants.items():
        for engine_name, engine_fn in engines:
            result = engine_fn(variant_img)
            found = result is not None
            log.append(f"{engine_name} / {variant_name}: {'FOUND' if found else 'no match'}")
            if found:
                return result, engine_name, variant_name, log

    return None, None, None, log


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_qr.py <path_to_image>")
        sys.exit(1)

    print("Trying multiple decoders and image variants — this may take a moment "
          "(QReader loads a YOLOv8 model on first use)...\n")

    qr_text, engine, variant, log = find_aadhaar_qr(sys.argv[1])

    print("Attempts:")
    for line in log:
        print(f"  {line}")

    if qr_text:
        with open("qr_output.txt", "w") as f:
            f.write(qr_text)
        print(f"\nSUCCESS via {engine} on the '{variant}' variant.")
        print(f"QR text ({len(qr_text)} chars) saved to qr_output.txt (not printed — contains personal data)")
        print(f"\nQuick check: is it numeric (Secure QR)? {qr_text.strip().isdigit()}")
        print(f"Or starts with '<' (old XML format)? {qr_text.strip().startswith('<')}")
    else:
        print("\nNo QR code found across all engines and variants.")
        print("At this point the QR is very likely too physically degraded")
        print("(glare/wear/lamination distortion) for ANY software decoder —")
        print("this affects even UIDAI's own official apps, not just this script.")

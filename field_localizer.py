"""
field_localizer.py — Module E (Field Localisation) of the TRINETRA/FDD-XAI
pipeline.

Wraps a YOLOv8n model (trained on the Kaggle "aadhar-card-entity-detection"
corpus; 5 field classes: aadhaar_number, dob_gender_line, photo_or_qr, name,
rare_field) to localise the principal fields on an Aadhaar card image.

Unlike the other modules, this one is NOT itself a real-vs-fake classifier —
it only draws bounding boxes. It is wired into the pipeline for two concrete
purposes:

  1. Field-level cross-check: the Aadhaar number is independently re-read via
     OCR from just the localised aadhaar_number crop (digit-only PSM-7 pass)
     and compared against the number already extracted from whole-page OCR
     (used for the Verhoeff check in pipeline.py). A genuine card's number
     should read identically both ways; a disagreement is corroborating
     evidence of either an OCR failure worth flagging or, in the worst case,
     a tampered/obscured number region.
  2. Presence check: if the document has already been identified as an
     Aadhaar card (via QR or keyword heuristic) but one or more of the
     expected fields cannot be localised at all, that is itself a weak,
     info-level signal worth recording (e.g. a field was cropped out of
     the image, obscured, or the image quality is poor).

This module intentionally does NOT output a real/fake verdict on its own —
no deterministic-failure override is attached to anything it raises, and its
flags are capped at "medium" severity, consistent with it being a
corroborating/diagnostic signal rather than a primary check.
"""

import os
import re

import cv2
import pytesseract

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n_field_localizer.pt")

CLASS_NAMES = ["aadhaar_number", "dob_gender_line", "photo_or_qr", "name", "rare_field"]
REQUIRED_FIELDS = ["aadhaar_number", "dob_gender_line", "photo_or_qr", "name"]
CONF_THRESHOLD = 0.40
INPUT_SIZES = (640, 1280)

_model = None  # lazy singleton — loading the checkpoint is expensive


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


def _best_box_per_class(result, class_names):
    """Given one ultralytics Results object, keep only the highest-confidence
    detection per class (the card has exactly one of each field)."""
    best = {}
    boxes = result.boxes
    if boxes is None:
        return best
    for box, conf, cls_idx in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        cls_idx = int(cls_idx)
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        if conf < CONF_THRESHOLD:
            continue
        if name not in best or conf > best[name]["conf"]:
            best[name] = {"bbox": [round(c) for c in box], "conf": round(float(conf), 4)}
    return best


def _crop(img, bbox, pad=6):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]


def _ocr_digits_only(crop_img):
    if crop_img is None or crop_img.size == 0:
        return ""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(gray, config=config)
    return re.sub(r"\D", "", text)


def localize_fields(images, whole_page_aadhaar_number: str = None) -> dict:
    """
    Runs the YOLOv8n field localizer on EVERY uploaded image (BGR numpy
    arrays, as returned by cv2.imread) and keeps the most confident
    detection of each field across all of them. A card is often uploaded
    as front + back, or as an e-Aadhaar PDF page, so the front's fields may
    not be in the first image; a field is only reported missing if no
    uploaded image contains it.

    `images` may be a single image or a list. `whole_page_aadhaar_number`
    is the digit-only Aadhaar number from the full OCR pass in pipeline.py
    (the one used for the Verhoeff check); it is cross-checked against an
    independent crop-only OCR read of the best localised aadhaar_number box.

    Returns a dict in the same shape as the other analysis modules
    (module name, detections, flags list) so it can be passed straight into
    evidence_aggregation.build_evidence_bundle.
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    flags = []
    detections = {}   # field -> {"bbox", "conf", "image_index"}
    try:
        model = _get_model()
        for idx, img in enumerate(images):
            # Two input sizes: 640 suits close-up card photos (what the model
            # was trained on); 1280 is needed when the card is small inside a
            # bigger image (e-Aadhaar PDF page, phone photo with background),
            # where at 640 the small field text simply disappears.
            for size in INPUT_SIZES:
                result = model.predict(img, conf=CONF_THRESHOLD, imgsz=size, verbose=False)[0]
                for field, det in _best_box_per_class(result, CLASS_NAMES).items():
                    if field not in detections or det["conf"] > detections[field]["conf"]:
                        detections[field] = {**det, "image_index": idx}
    except Exception as e:
        return {
            "module": "field_localization",
            "detections": {},
            "crop_ocr_number": None,
            "flags": [{
                "code": "field_localizer_unavailable",
                "severity": "info",
                "description": f"Field localizer could not run: {e}",
            }],
        }

    # Fields YOLO couldn't find are kept as data only (not reported as
    # flags): a missed detection usually reflects framing, glare or a card
    # layout the model wasn't trained on, not tampering, and listing it
    # added noise to otherwise clean reports.
    missing_fields = [f for f in REQUIRED_FIELDS if f not in detections]

    crop_ocr_number = None
    if "aadhaar_number" in detections:
        det = detections["aadhaar_number"]
        crop_ocr_number = _ocr_digits_only(_crop(images[det["image_index"]], det["bbox"]))

        if whole_page_aadhaar_number and crop_ocr_number:
            if len(crop_ocr_number) == 12 and crop_ocr_number != whole_page_aadhaar_number:
                flags.append({
                    "code": "field_localizer_number_mismatch",
                    "severity": "medium",
                    "description": (
                        f"Independent crop-only OCR of the localised Aadhaar-number field "
                        f"read '{crop_ocr_number}', which disagrees with the number "
                        f"'{whole_page_aadhaar_number}' extracted from whole-page OCR. "
                        f"Two independent OCR passes over the same document disagreeing on "
                        f"the Aadhaar number is a corroborating signal worth investigating "
                        f"alongside the other modules, rather than being decisive alone."
                    ),
                })

    return {
        "module": "field_localization",
        "detections": detections,
        "crop_ocr_number": crop_ocr_number,
        "missing_fields": missing_fields,
        "flags": flags,
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 field_localizer.py <image_path> [more image paths]")
        sys.exit(1)

    images = [cv2.imread(p) for p in sys.argv[1:]]
    if any(im is None for im in images):
        raise ValueError("Could not read one of the images")

    print(json.dumps(localize_fields(images), indent=2, default=str))

# TRINETRA (FDD-XAI) — Full Pipeline + Professional Web Frontend

## Module E: YOLOv8n Field Localizer (now wired in)
`field_localizer.py` loads the trained YOLOv8n checkpoint from
`models/yolov8n_field_localizer.pt` (5 classes: aadhaar_number,
dob_gender_line, photo_or_qr, name, rare_field) and is now called from
`pipeline.py` for every Aadhaar-typed document. It does two real things,
not just decoration:
1. Crops the localised `aadhaar_number` box and independently re-OCRs just
   that region (digit-only), then cross-checks it against the number
   already read from whole-page OCR (used for the Verhoeff check). A
   disagreement raises a `medium`-severity, non-deterministic-override flag
   via `evidence_aggregation.py`.
2. Flags (at `info` severity) any of the four required fields that could
   not be localised at all on an Aadhaar-typed document.
It is **not** a real/fake classifier — MobileNetV2 was evaluated separately
during the project but was not integrated, since its fake-recall (48%) was
judged too weak to trust as a production signal; TRINETRA's real/fake
verdict is still produced entirely by the five deterministic modules.
Requires `pip install ultralytics` in addition to the existing requirements.

## What's new in this drop
- **Unified upload**: one drag-and-drop field accepts multiple files at once —
  images (front/back/any angle) AND PDFs (e-Aadhaar downloads) mixed together.
  PDFs are auto-converted to images (one per page) via PyMuPDF before the
  pipeline runs, so e-Aadhaar downloads work exactly like photographed cards.
- **Professional UI redesign** — branded header, verdict badge, score bar,
  side-by-side QR photo comparison, categorized evidence flags.
- **All real-world bugs found through actual card testing are fixed**:
  - OCR now tries all 4 rotations and picks the one Tesseract itself is most
    confident about (fixes upside-down photo captures producing garbled text)
  - QR V2 format photo-boundary calculation fixed (V2 drops the old binary
    mobile_hash block in favor of a text field — was truncating photo data)
  - DOB cross-check now normalizes date separators (06-02-2005 vs 06/02/2005
    are the same date, no longer falsely flagged as a mismatch)
  - Copy-move and ELA forensics now exclude detected TEXT regions and QR code
    regions — printed glyphs and dense QR patterns were causing false
    "tampering" flags on completely genuine documents
  - metadata_analysis.py KeyError fixed (was crashing on real camera EXIF data)
  - Windows Tesseract path auto-configured

## Setup (Windows)
```
venv\Scripts\Activate.ps1
python -m pip install flask opencv-python-headless qrcode pyzbar cryptography pytesseract pymupdf ultralytics
```
Plus exiftool, tesseract-ocr, and zbar as covered in earlier setup if not
already installed.

## Running
```
python app.py
```
Open **http://127.0.0.1:5000** — drag/drop or browse to select any combination
of image files and/or a PDF, then click Analyze.

## Enabling real QR signature verification
In `app.py`, set:
```python
UIDAI_CERT_PATH = r"C:\Trinetra\Certificates\UIDAI1.cer"
```
(Use whichever certificate you found actually verifies against your real
card — confirmed working via a positive test AND two negative controls
during development.)

## Known limitations (still honest, still true)
- Image forensics thresholds are empirically tuned against real-world false
  positives found during testing, not against a large statistically
  representative dataset — expect to keep finding edge cases.
- The Kaggle-trained YOLO localizer and MobileNetV2 classifier are not yet
  wired into this local pipeline as additional evidence sources.
- Live camera-feed scanning (in progress, next milestone) is not yet built —
  current flow is upload-based only.

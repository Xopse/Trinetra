"""
pipeline.py — Full TRINETRA/FDD-XAI pipeline orchestrator.
"""

import re
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import pytesseract
from pyzbar.pyzbar import decode as qr_decode

_TESSERACT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_TESSERACT_WINDOWS_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WINDOWS_PATH

from verhoeff import validate as verhoeff_validate
from mrz import parse_td3
from metadata_analysis import analyze as analyze_metadata
from image_forensics import analyze as analyze_forensics, _get_text_regions as get_text_regions
from qr_signature_verification import analyze as analyze_qr
from qr_photo_extraction import qr_photo_data_uri, PhotoDecodeError
from field_localizer import localize_fields
from evidence_aggregation import build_evidence_bundle
from llm_reasoning import generate_report


MAX_OCR_SIDE = 2000   # Tesseract reads card text as well or better at this size, and much faster
OSD_MIN_CONF = 1.5    # Tesseract orientation-detection confidence we trust on its own
UPRIGHT_OK_CONF = 50  # upright read at least this confident -> don't try other rotations
_ROTATE_CODE = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _ocr_pass(gray):
    """One Tesseract call gives both the words and their confidences
    (the old code called Tesseract twice per rotation for the same thing)."""
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data["conf"] if float(c) >= 0]
    mean_conf = sum(confs) / len(confs) if confs else -1
    lines = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(w) for w in lines.values()), mean_conf


def _detect_rotation(gray):
    """Tesseract's orientation detection on a small copy (~1s). Returns
    (clockwise degrees needed to make the text upright, confidence), or
    (None, 0) if it can't tell (e.g. too little text)."""
    h, w = gray.shape
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    try:
        osd = pytesseract.image_to_osd(small, output_type=pytesseract.Output.DICT)
        return int(osd["rotate"]) % 360, float(osd["orientation_conf"])
    except Exception:
        return None, 0.0


def run_ocr(img):
    """Reads the card's text. Most uploads are upright (cv2.imread already
    applies the phone's EXIF rotation), so: if Tesseract's orientation check
    is confident, read once in that orientation; otherwise read upright and
    only try the other rotations if that read is weak, keeping whichever
    Tesseract is most confident about. The common case is ONE OCR pass;
    the old code always did eight."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_OCR_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def read(angle):
        code = _ROTATE_CODE[angle]
        return _ocr_pass(gray if code is None else cv2.rotate(gray, code))

    osd_angle, osd_conf = _detect_rotation(gray)
    if osd_angle is not None and osd_conf >= OSD_MIN_CONF:
        return read(osd_angle)[0]

    best_text, best_conf = read(0)
    if best_conf >= UPRIGHT_OK_CONF:
        return best_text
    for angle in (180, 90, 270):
        text, conf = read(angle)
        if conf > best_conf:
            best_text, best_conf = text, conf
        if conf >= UPRIGHT_OK_CONF:
            break
    return best_text


_qreader = None  # QReader loads a detection model; build it once, not per call


def try_decode_qr_raw(img):
    """
    Tries multiple QR decoding engines in order (pyzbar, then zxing-cpp,
    then QReader), since dense/high-resolution QR codes — like those on
    e-Aadhaar PDFs or newer Secure QR v2 cards — can defeat pyzbar alone
    even when the QR is perfectly clean and scannable. Falling back
    through stronger engines here (rather than only in the separate
    extract_qr.py CLI script) is what makes the actual pipeline reliable
    on real-world documents, not just synthetic test images.
    """
    results = qr_decode(img)
    for r in results:
        if r.type == "QRCODE":
            return r.data.decode("utf-8", errors="ignore")

    try:
        import zxingcpp
        zresults = zxingcpp.read_barcodes(img)
        for r in zresults:
            if "QR" in str(r.format):
                return r.text
    except Exception:
        pass

    try:
        global _qreader
        if _qreader is None:
            from qreader import QReader
            _qreader = QReader()
        qreader = _qreader
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        decoded = qreader.detect_and_decode(image=img_rgb)
        for text in decoded:
            if text:
                return text
    except Exception:
        pass

    # Camera captures: the Secure QR is very dense, and in a webcam/phone frame
    # each QR module is only 2-4 pixels wide and slightly blurred, so the
    # decoders above often fail on the raw frame. Locate the QR, crop it,
    # enlarge it and clean it up, then try the fast decoders again.
    return _decode_qr_enhanced(img)


def _qr_crops(img):
    """Candidate QR regions (with a margin), from QReader's detector and OpenCV's."""
    h, w = img.shape[:2]
    boxes = []
    try:
        if _qreader is not None:
            for det in _qreader.detect(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)):
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                boxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    try:
        ok, pts = cv2.QRCodeDetector().detect(img)
        if ok and pts is not None:
            xs, ys = pts.reshape(-1, 2)[:, 0], pts.reshape(-1, 2)[:, 1]
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    except Exception:
        pass
    crops = []
    for x1, y1, x2, y2 in boxes:
        m = int(0.12 * max(x2 - x1, y2 - y1))  # keep the quiet zone
        x1, y1, x2, y2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
        if x2 - x1 > 20 and y2 - y1 > 20:
            crops.append(img[y1:y2, x1:x2])
    return crops


def _decode_qr_enhanced(img):
    import numpy as np
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    for region in _qr_crops(img) + [img]:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        side = max(gray.shape)
        for target in (1200, 1800):  # enlarge so each module is several pixels wide
            scale = target / side
            if region is img and scale < 1.5:
                continue  # the full frame is only worth retrying when enlarging a lot
            big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            candidates = [
                big,
                cv2.filter2D(clahe.apply(big), -1, sharpen),
                cv2.threshold(cv2.GaussianBlur(big, (3, 3), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 41, 7),
            ]
            for cand in candidates:
                try:
                    import zxingcpp
                    for r in zxingcpp.read_barcodes(cand):
                        if "QR" in str(r.format) and r.text:
                            return r.text
                except Exception:
                    pass
                for r in qr_decode(cand):
                    if r.type == "QRCODE":
                        return r.data.decode("utf-8", errors="ignore")
    return None


def find_mrz_lines(ocr_text):
    candidate_lines = []
    for raw_line in ocr_text.splitlines():
        line = raw_line.strip().replace(" ", "")
        cleaned = re.sub(r"[^A-Z0-9<]", "", line.upper())
        if 35 <= len(cleaned) <= 46:
            candidate_lines.append(cleaned)

    for i in range(len(candidate_lines) - 1):
        l1, l2 = candidate_lines[i], candidate_lines[i + 1]
        if l1.startswith("P<") and any(ch.isdigit() for ch in l2):
            return l1.ljust(44, "<")[:44], l2.ljust(44, "<")[:44]
    return None


def run_pipeline(image_paths, uidai_cert_path=None, force_template_llm: bool = False, extra_flags: list = None,
                 progress=None) -> dict:
    """progress, if given, is called with a stage key ("ocr", "qr",
    "checksum", "localizer", "forensics", "aggregate", "report") as each
    stage starts, so the web UI can show live progress."""
    def step(key):
        if progress:
            progress(key)

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if not image_paths:
        raise ValueError("At least one image path is required.")

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image at {path}")
        images.append((path, img))

    # Copy-move's text-exclusion boxes come from a full-resolution OCR pass,
    # exactly as image_forensics always computed them (it relies on that
    # pass's low-confidence "words" over background patterns too). It is
    # started now in background threads, so it runs at the same time as
    # the OCR, QR, checksum and YOLO stages instead of after them.
    box_pool = ThreadPoolExecutor(max_workers=max(1, len(images)))
    text_box_futures = [box_pool.submit(get_text_regions, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        for _, img in images]

    step("ocr")
    ocr_texts = []
    for path, img in images:
        ocr_texts.append(run_ocr(img))
    combined_ocr_text = "\n".join(ocr_texts)

    doc_type = "unknown"
    qr_payload = None
    mrz_lines = None
    qr_missing_but_expected = False

    step("qr")
    for path, img in images:
        qr_payload = try_decode_qr_raw(img)
        if qr_payload:
            doc_type = "aadhaar"
            break

    if doc_type == "unknown":
        mrz_lines = find_mrz_lines(combined_ocr_text)
        if mrz_lines:
            doc_type = "passport"

    if doc_type == "unknown":
        # No QR and no MRZ found — but before giving up, check whether
        # the document LOOKS like it's meant to be an Aadhaar card
        # (keywords, or a 12-digit Aadhaar-shaped number in the OCR
        # text). If so, a completely missing/unscannable QR on a
        # document that otherwise presents as Aadhaar is itself a real
        # red flag worth surfacing explicitly — not something to
        # silently fall through to "unknown" and barely check at all.
        ocr_upper = combined_ocr_text.upper()
        aadhaar_keywords = ("AADHAAR", "आधार", "GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION")
        looks_like_aadhaar = (
            any(kw in ocr_upper for kw in aadhaar_keywords)
            or re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", combined_ocr_text)
        )
        if looks_like_aadhaar:
            doc_type = "aadhaar"
            qr_missing_but_expected = True

    verhoeff_result = None
    mrz_result = None
    qr_result = None
    localizer_result = None

    if doc_type == "aadhaar":
        if qr_missing_but_expected:
            qr_result = {
                "module": "qr_signature_verification",
                "decode_success": False,
                "flags": [{
                    "code": "qr_missing_on_aadhaar_document",
                    "severity": "high",
                    "description": (
                        "This document presents as an Aadhaar card (matching keywords "
                        "or number format found in the printed text), but NO QR code "
                        "could be detected anywhere in the image, even after trying "
                        "multiple decoding engines. A genuine, undamaged Aadhaar card "
                        "always has a scannable QR code — a completely missing or "
                        "undetectable QR is itself a meaningful red flag, not merely "
                        "an inconvenience. This could indicate: the QR region was "
                        "deliberately removed/covered, the document is a low-quality "
                        "fabrication that never had a real QR, or (less likely, since "
                        "multiple engines were tried) a severe image quality issue. "
                        "Cross-check against other modules before concluding tampering, "
                        "but do not treat this as a benign gap."
                    ),
                }],
            }
        else:
            try:
                qr_result = analyze_qr(qr_payload, ocr_text=combined_ocr_text, cert_path=uidai_cert_path)
            except Exception as e:
                qr_result = {
                    "module": "qr_signature_verification",
                    "decode_success": False,
                    "flags": [{"code": "qr_decode_failed", "severity": "medium",
                               "description": f"Could not decode as a Secure QR Code: {e}"}],
                }

        if qr_result.get("decode_success") and qr_result.get("photo_bytes"):
            try:
                # kept in memory only, never written to disk
                qr_result["extracted_photo_data_uri"] = qr_photo_data_uri(qr_result["photo_bytes"])
            except PhotoDecodeError as e:
                qr_result.setdefault("flags", []).append({
                    "code": "photo_decode_failed", "severity": "info",
                    "description": f"Could not decode embedded photo: {e}",
                })

        step("checksum")
        match = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", combined_ocr_text)
        if match:
            candidate_number = match.group(1).replace(" ", "")
            is_valid = verhoeff_validate(candidate_number)
            verhoeff_result = {
                "valid": is_valid,
                "number": candidate_number,
                "description": (
                    f"Aadhaar number '{candidate_number}' fails the Verhoeff checksum — "
                    f"internally inconsistent, indicating a digit was altered or the "
                    f"number is fabricated." if not is_valid else
                    f"Aadhaar number '{candidate_number}' passes the Verhoeff checksum."
                ),
            }

        # Module E: YOLOv8n field localization, run on every uploaded image.
        # Crops the aadhaar_number field and independently re-OCRs just
        # that region as a cross-check against the whole-page OCR number
        # used above for the Verhoeff check (see field_localizer.py).
        step("localizer")
        whole_page_number = verhoeff_result["number"] if verhoeff_result else None
        localizer_result = localize_fields([im for _, im in images], whole_page_aadhaar_number=whole_page_number)

    elif doc_type == "passport":
        line1, line2 = mrz_lines
        mrz_result = parse_td3(line1, line2)

    step("forensics")
    per_image_results = []
    metadata_results_combined = {"flags": []}
    forensics_results_combined = {"flags": []}

    for idx, (path, img) in enumerate(images):
        label = f"image_{idx + 1}" if len(images) > 1 else "image"

        meta_result = analyze_metadata(path)
        for flag in meta_result.get("flags", []):
            flag["description"] = f"[{label}] " + flag["description"]
            metadata_results_combined["flags"].append(flag)

        # Pixel forensics (ELA, copy-move) needs a photographed or scanned image.
        # A page rendered from a PDF is a lossless drawing of vector text, so it
        # has no compression history for ELA, and an e-Aadhaar prints the same
        # name, number and logos several times, which copy-move mistakes for
        # tampering. These checks are therefore not run on PDF pages; the PDF is
        # still verified by its UIDAI QR signature, OCR cross-check, checksum,
        # password check and metadata.
        if re.search(r"_page\d+\.png$", path):
            forensics_result = {"module": "image_forensics", "flags": [], "skipped": "pdf_page"}
        else:
            forensics_result = analyze_forensics(path, img=img, text_boxes=text_box_futures[idx].result())
            for flag in forensics_result.get("flags", []):
                flag["description"] = f"[{label}] " + flag["description"]

        forensics_results_combined["flags"].extend(forensics_result["flags"])

        per_image_results.append({
            "label": label,
            "path": path,
            "metadata": meta_result,
            "forensics": forensics_result,
        })

    box_pool.shutdown(wait=False)
    step("aggregate")
    evidence_bundle = build_evidence_bundle(
        document_type=doc_type,
        verhoeff_result=verhoeff_result,
        mrz_result=mrz_result,
        qr_result=qr_result,
        metadata_result=metadata_results_combined,
        forensics_result=forensics_results_combined,
        localizer_result=localizer_result,
        extra_flags=extra_flags,
    )

    step("report")
    report_result = generate_report(evidence_bundle, force_template=force_template_llm)

    return {
        "document_type": doc_type,
        "num_images": len(images),
        "combined_ocr_text": combined_ocr_text,
        "per_image_results": per_image_results,
        "qr_result": qr_result,
        "localizer_result": localizer_result,
        "evidence_bundle": evidence_bundle,
        "report": report_result["report"],
        "generation_method": report_result["generation_method"],
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 pipeline.py <front_image_path> [back_image_path] [uidai_cert_path]")
        sys.exit(1)

    args = sys.argv[1:]
    cert_path = None
    image_paths = []
    for a in args:
        if a.lower().endswith((".cer", ".pem", ".crt")):
            cert_path = a
        else:
            image_paths.append(a)

    result = run_pipeline(image_paths, uidai_cert_path=cert_path)

    print(f"Document type: {result['document_type']}")
    print(f"Images analyzed: {result['num_images']}")
    print(f"Report generated via: {result['generation_method']}")
    print("\n" + "=" * 60)
    print(result["report"])
    print("=" * 60)

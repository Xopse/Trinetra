"""
image_forensics.py — Module 3 (Image Forensics) of the TRINETRA/FDD-XAI pipeline.

Two classical, well-established forensic techniques, neither requiring
any training data:

1. Error Level Analysis (ELA): re-compresses the image at a known JPEG
   quality and diffs it against the original. Regions that were edited
   and re-saved after the rest of the image was compressed show a
   different error signature than the untouched surrounding area.

2. Copy-move detection: uses ORB keypoint matching + RANSAC-style
   filtering to find regions of the SAME image that closely match
   another region — a classic signature of someone copying a patch to
   cover/duplicate content.

Both exclude detected TEXT regions (OCR bounding boxes) and QR CODE
regions from their anomaly checks — printed glyphs and dense QR
patterns both produce strong false positives (matching letter corners
against each other, or flagging naturally high-frequency QR compression
as "anomalous") that have nothing to do with genuine tampering. This
was discovered and fixed through real-card testing, not assumed upfront.

Produces STRUCTURED evidence only (anomaly scores + flagged regions),
no verdict — feeds into Evidence Aggregation / the LLM Reasoning Layer.
"""

import cv2
import numpy as np


def _get_qr_region_boxes(img):
    """
    Detects QR code bounding boxes so both ELA and copy-move can exclude
    them. QR codes are extremely high-frequency, dense black/white
    patterns that naturally show different compression error AND
    self-similar corner geometry than smooth photo/text regions even
    when completely untouched.

    Tries pyzbar first, then falls back to zxing-cpp if pyzbar finds
    nothing. This matters in practice: plain pyzbar can fail to detect
    dense, high-resolution QR codes (e.g. on e-Aadhaar PDF pages) even
    though the QR is perfectly valid — when that happens, the exclusion
    box list would silently come back EMPTY, leaving the QR pattern
    fully exposed to copy-move matching against everything else in the
    image (this was confirmed via real e-Aadhaar testing: text letters
    were matching against the QR region because it was never excluded).
    """
    boxes = []
    try:
        from pyzbar.pyzbar import decode as qr_decode
        results = qr_decode(img)
        for r in results:
            if r.type == "QRCODE":
                x, y, w, h = r.rect
                pad = 10
                boxes.append((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    except Exception:
        pass

    if not boxes:
        try:
            import zxingcpp
            zresults = zxingcpp.read_barcodes(img)
            for r in zresults:
                if "QR" in str(r.format):
                    pos = r.position
                    xs = [pos.top_left.x, pos.top_right.x, pos.bottom_left.x, pos.bottom_right.x]
                    ys = [pos.top_left.y, pos.top_right.y, pos.bottom_left.y, pos.bottom_right.y]
                    pad = 10
                    boxes.append((
                        max(0, min(xs) - pad), max(0, min(ys) - pad),
                        max(xs) + pad, max(ys) + pad,
                    ))
        except Exception:
            pass

    return boxes


def _box_overlaps_region(cell_box, exclude_boxes, overlap_threshold=0.3):
    """Returns True if cell_box overlaps any exclude_box by more than
    overlap_threshold fraction of the cell's own area."""
    cx1, cy1, cx2, cy2 = cell_box
    cell_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    for ex1, ey1, ex2, ey2 in exclude_boxes:
        ix1, iy1 = max(cx1, ex1), max(cy1, ey1)
        ix2, iy2 = min(cx2, ex2), min(cy2, ey2)
        if ix2 > ix1 and iy2 > iy1:
            overlap_area = (ix2 - ix1) * (iy2 - iy1)
            if overlap_area / cell_area > overlap_threshold:
                return True
    return False


def _get_text_regions(gray_img):
    """
    Uses Tesseract to find bounding boxes of detected text words. Text
    glyphs (letters like E, R, digits like 2, 7, etc.) share very
    similar small-scale corner/edge geometry across different
    occurrences of the same or similar characters — this causes ORB
    keypoint matching to falsely pair them up, entirely unrelated to
    any genuine copy-move tampering.

    Deliberately LOW confidence threshold (>0, not the usual >30 or
    higher): for our purpose here (excluding regions), MISSING a real
    text region costs us a false tamper flag (expensive), while
    excluding a region that wasn't actually text costs almost nothing
    (cheap) — so we bias strongly toward excluding more rather than
    less. This matters in practice: real photographed ID cards (angle,
    lighting, print quality) often score lower OCR confidence than
    clean synthetic test images, and a stricter threshold let some real
    text regions slip through uncaught during testing.
    """
    try:
        import pytesseract
        data = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    boxes = []
    for i in range(len(data["text"])):
        if data["text"][i].strip() and int(data["conf"][i]) > 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            pad = 4
            boxes.append((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    return boxes


def _point_in_any_box(x, y, boxes):
    for x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def error_level_analysis(original, qr_boxes, quality: int = 90, amplify: int = 15):
    """
    Returns (ela_image, mean_error, std_error, region_flags).
    region_flags excludes any region overlapping a detected QR code.
    """

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", original, encode_param)
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(original, recompressed)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ela_amplified = np.clip(diff_gray.astype(np.int32) * amplify, 0, 255).astype(np.uint8)

    mean_error = float(np.mean(diff_gray))
    std_error = float(np.std(diff_gray))

    h, w = diff_gray.shape
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    outlier_threshold = 1.0 * std_error

    region_flags = []
    if cell_h > 0 and cell_w > 0:
        for gy in range(grid_size):
            for gx in range(grid_size):
                y1, y2 = gy * cell_h, min((gy + 1) * cell_h, h)
                x1, x2 = gx * cell_w, min((gx + 1) * cell_w, w)

                if qr_boxes and _box_overlaps_region((x1, y1, x2, y2), qr_boxes):
                    continue

                cell_mean = float(np.mean(diff_gray[y1:y2, x1:x2]))
                deviation = cell_mean - mean_error
                if abs(deviation) > outlier_threshold and abs(deviation) > 1.0:
                    region_flags.append({
                        "box": [x1, y1, x2, y2],
                        "error_level": round(cell_mean, 2),
                        "direction": "higher" if deviation > 0 else "lower",
                    })

    return ela_amplified, mean_error, std_error, region_flags


def copy_move_detection(img, qr_boxes, text_boxes=None, min_matches: int = 10, distance_threshold: float = 60.0):
    """
    Detects regions within the SAME image that closely match each other,
    excluding detected TEXT and QR regions (see module docstring).
    text_boxes can be passed in from the pipeline's own OCR pass so
    Tesseract isn't run a second time on the same image.
    Returns a list of matched region pairs.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if text_boxes is None:
        text_boxes = _get_text_regions(gray)
    exclude_boxes = list(text_boxes) + qr_boxes

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < min_matches * 2:
        return []

    patch_radius = 8
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    texture_floor = 15.0

    textured_mask = []
    text_mask = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        x1, x2 = max(0, x - patch_radius), min(w, x + patch_radius)
        y1, y2 = max(0, y - patch_radius), min(h, y + patch_radius)
        patch_var = float(np.var(laplacian[y1:y2, x1:x2])) if (y2 > y1 and x2 > x1) else 0.0
        textured_mask.append(patch_var > texture_floor)
        text_mask.append(_point_in_any_box(x, y, exclude_boxes))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    good_matches = []
    for match_group in matches:
        if len(match_group) < 3:
            continue
        m, n = match_group[1], match_group[2]
        if not (textured_mask[m.queryIdx] and textured_mask[m.trainIdx]):
            continue
        if text_mask[m.queryIdx] or text_mask[m.trainIdx]:
            continue
        if m.distance < 0.65 * n.distance:
            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt
            spatial_dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if spatial_dist > distance_threshold:
                good_matches.append((pt1, pt2, spatial_dist))

    if len(good_matches) < min_matches:
        return []

    region_pairs = []
    for pt1, pt2, dist in good_matches[:20]:
        region_pairs.append({
            "point_a": [round(pt1[0], 1), round(pt1[1], 1)],
            "point_b": [round(pt2[0], 1), round(pt2[1], 1)],
            "spatial_distance": round(float(dist), 1),
        })

    return region_pairs


def analyze(image_path: str, save_ela_to: str = None, img=None, text_boxes=None) -> dict:
    flags = []
    if img is None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
    qr_boxes = _get_qr_region_boxes(img)  # detected once, shared by ELA and copy-move

    ela_image, mean_error, std_error, ela_regions = error_level_analysis(img, qr_boxes)

    if save_ela_to:
        cv2.imwrite(save_ela_to, ela_image)

    if ela_regions:
        directions = set(r["direction"] for r in ela_regions)
        flags.append({
            "code": "ela_anomaly_regions",
            "severity": "high" if len(ela_regions) <= 3 else "medium",
            "description": (
                f"Error Level Analysis found {len(ela_regions)} region(s) with a "
                f"compression error signature statistically inconsistent with the "
                f"rest of the image (error level > 1.0 std. dev. from the image "
                f"mean of {mean_error:.2f}; direction(s): {', '.join(directions)}). "
                f"Higher-than-normal error suggests a freshly-edited, less-compressed "
                f"region; lower-than-normal error suggests a region that was already "
                f"compressed once before (at tampering time) and has settled — both "
                f"are consistent with that region being edited and re-saved "
                f"separately from the rest of the document. "
                f"Flagged regions: {[r['box'] for r in ela_regions[:5]]}"
            ),
        })

    copy_move_matches = copy_move_detection(img, qr_boxes, text_boxes)
    if copy_move_matches:
        flags.append({
            "code": "copy_move_detected",
            "severity": "high" if len(copy_move_matches) >= 12 else "medium",
            "description": (
                f"Found {len(copy_move_matches)} pairs of spatially-separated but "
                f"visually near-identical keypoint regions within the same image — "
                f"a signature of copy-move tampering (duplicating a patch to hide "
                f"or replace content elsewhere in the document). Note: some "
                f"legitimate documents have repetitive design elements (e.g. "
                f"border patterns, emblems) that can trigger this signal too — "
                f"cross-check against the field locations from the OCR/template "
                f"module before treating this as conclusive."
            ),
        })

    result = {
        "module": "image_forensics",
        "ela": {
            "mean_error": round(mean_error, 3),
            "std_error": round(std_error, 3),
            "flagged_regions": ela_regions,
        },
        "copy_move": {
            "match_count": len(copy_move_matches),
            "matches": copy_move_matches,
        },
        "flags": flags,
    }
    return result


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python3 image_forensics.py <image_path> [ela_output_path]")
        sys.exit(1)

    ela_out = sys.argv[2] if len(sys.argv) > 2 else None
    result = analyze(sys.argv[1], save_ela_to=ela_out)
    print(f"ELA mean error: {result['ela']['mean_error']}")
    print(f"ELA flagged regions: {len(result['ela']['flagged_regions'])}")
    print(f"Copy-move matches: {result['copy_move']['match_count']}")
    print(f"\nFlags raised: {len(result['flags'])}")
    for f in result["flags"]:
        print(f"  [{f['severity'].upper()}] {f['code']}: {f['description'][:200]}...")

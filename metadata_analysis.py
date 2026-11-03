"""metadata_analysis.py — Module 4: EXIF/file metadata forensics."""

import os
import json
import subprocess
import shutil
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))

EXIFTOOL_INSTALL_HELP = (
    "ExifTool is required but was not found.\n"
    "  1. Download the Windows zip from https://exiftool.org and extract it.\n"
    "  2. Rename 'exiftool(-k).exe' to 'exiftool.exe'.\n"
    f"  3. Put exiftool.exe AND its 'exiftool_files' folder inside {_HERE}\n"
    "     (or anywhere on your PATH), then start the app again."
)

EDITING_SOFTWARE_SIGNATURES = [
    "photoshop", "gimp", "canva", "paint.net", "pixlr", "affinity photo",
    "lightroom", "snapseed", "picsart", "corel", "illustrator", "inkscape",
]
CAMERA_CAPTURE_FIELDS = [
    "Make", "Model", "LensModel", "FNumber", "ExposureTime", "ISO",
    "FocalLength", "GPSLatitude", "GPSLongitude",
]


def exiftool_path():
    """ExifTool on the PATH, or exiftool.exe / exiftool placed in the app folder."""
    found = shutil.which("exiftool")
    if found:
        return found
    for name in ("exiftool.exe", "exiftool"):
        local = os.path.join(_HERE, name)
        if os.path.isfile(local):
            return local
    return None


def extract_metadata_exiftool(image_path):
    result = subprocess.run([exiftool_path(), "-json", "-G", image_path],
                             capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed: {result.stderr}")
    data = json.loads(result.stdout)
    return data[0] if data else {}


def extract_metadata_pil(image_path):
    """Fallback for a single file ExifTool fails to read. Uses Pillow's getexif()
    (works for JPEG, PNG, WebP and TIFF, unlike the older _getexif(), which
    was JPEG-only), the camera sub-IFD (exposure, lens, ISO), and any XMP
    or text chunks, which is where editors like Photoshop or Canva often
    leave their "edited with" signature."""
    from PIL import Image
    from PIL.ExifTags import TAGS
    img = Image.open(image_path)
    result = {}
    try:
        exif = img.getexif()
        for tag_id, value in exif.items():
            result[str(TAGS.get(tag_id, tag_id))] = str(value)
        for tag_id, value in exif.get_ifd(0x8769).items():   # Exif sub-IFD: camera settings
            result[str(TAGS.get(tag_id, tag_id))] = str(value)
    except Exception:
        pass
    for key, value in (img.info or {}).items():               # XMP / PNG text chunks
        if key in ("exif", "icc_profile", "dpi", "transparency"):
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str) and value.strip():
            result[f"Info:{key}"] = value[:2000]
    result["Format"] = img.format
    result["ImageSize"] = f"{img.width}x{img.height}"
    return result


def _find_field_ci(metadata: dict, keyword: str):
    hits = {}
    for k, v in metadata.items():
        if keyword.lower() in str(k).lower() or keyword.lower() in str(v).lower():
            hits[k] = v
    return hits


def _try_parse_datetime(value):
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S%z", "%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(value)[:19], fmt.replace("%z", ""))
        except ValueError:
            continue
    return None


def analyze(image_path: str) -> dict:
    flags = []
    if exiftool_path() is None:
        raise RuntimeError(EXIFTOOL_INSTALL_HELP)
    try:
        metadata = extract_metadata_exiftool(image_path)
        source = "exiftool"
    except Exception:
        # ExifTool is installed but choked on this one file: read what we can.
        metadata = extract_metadata_pil(image_path)
        source = "PIL_fallback"

    editing_hits = {}
    for sig in EDITING_SOFTWARE_SIGNATURES:
        hits = _find_field_ci(metadata, sig)
        if hits:
            editing_hits.update(hits)

    if editing_hits:
        flags.append({
            "code": "editing_software_signature", "severity": "high",
            "description": f"File metadata contains an image-editing software signature: {editing_hits}.",
        })

    camera_fields_present = {}
    for f in CAMERA_CAPTURE_FIELDS:
        for k in metadata.keys():
            if f.lower() in k.lower():
                camera_fields_present[f] = metadata[k]
                break

    non_trivial_keys = [
        k for k in metadata.keys()
        if k.split(":")[-1] not in ("Format", "ImageSize", "FileSize", "FileType",
                                     "FileName", "Directory", "FileModifyDate",
                                     "FileAccessDate", "FileInodeChangeDate", "FilePermissions")
    ]
    metadata_present = len(non_trivial_keys) > 0
    if not metadata_present:
        flags.append({
            # info, not medium: genuine uploads very often have no camera
            # metadata (screenshots, scans, WhatsApp-forwarded photos,
            # e-Aadhaar PDF pages), so its absence is not evidence of
            # tampering. Real editing traces are caught separately by
            # the editing_software_signature check.
            "code": "metadata_absent", "severity": "info",
            "description": (
                "No camera metadata found in this image. This is normal for screenshots, "
                "scans, PDF pages and images sent through messaging apps, so it is not "
                "evidence of tampering on its own; recorded for completeness."
            ),
        })

    def _first_matching(*suffixes):
        for k, v in metadata.items():
            if k.split(":")[-1] in suffixes:
                return v
        return None

    create_date = _first_matching("CreateDate", "DateTimeOriginal")
    modify_date = _first_matching("ModifyDate")
    dt_create = _try_parse_datetime(create_date)
    dt_modify = _try_parse_datetime(modify_date)

    timestamp_inconsistency = False
    if dt_create and dt_modify and dt_modify > dt_create:
        gap = dt_modify - dt_create
        timestamp_inconsistency = True
        flags.append({
            "code": "modify_after_create", "severity": "medium",
            "description": f"ModifyDate ({modify_date}) is later than CreateDate ({create_date}) by {gap}.",
        })

    return {
        "module": "metadata_analysis",
        "metadata_source": source,
        "raw_metadata": metadata,
        "signals": {
            "editing_software_found": bool(editing_hits),
            "editing_software_hits": editing_hits,
            "has_camera_capture_fields": len(camera_fields_present) > 0,
            "camera_capture_fields": camera_fields_present,
            "metadata_present": metadata_present,
            "timestamp_inconsistency": timestamp_inconsistency,
        },
        "flags": flags,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 metadata_analysis.py <image_path>")
        sys.exit(1)
    result = analyze(sys.argv[1])
    print(f"Metadata source: {result['metadata_source']}")
    print(f"Flags raised: {len(result['flags'])}")
    for f in result["flags"]:
        print(f"  [{f['severity'].upper()}] {f['code']}: {f['description']}")

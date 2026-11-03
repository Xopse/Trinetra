"""
qr_photo_extraction.py — Decodes the embedded photo from an Aadhaar
Secure QR Code's photo_bytes field (JPEG2000 format, per UIDAI spec)
into a standard, viewable image (PNG).
"""

import io
from PIL import Image


class PhotoDecodeError(Exception):
    pass


def decode_qr_photo(photo_bytes: bytes) -> Image.Image:
    if not photo_bytes or len(photo_bytes) < 10:
        raise PhotoDecodeError("No photo data present in this QR code (photo_bytes is empty or too short).")

    try:
        img = Image.open(io.BytesIO(photo_bytes))
        img.load()
        return img.convert("RGB")
    except Exception as e:
        raise PhotoDecodeError(f"Failed to decode photo bytes as JPEG2000: {e}")


def qr_photo_data_uri(photo_bytes: bytes) -> str:
    """The QR's embedded photo as a PNG data URI, built in memory so it can
    be shown on the result page without ever being written to disk."""
    import io, base64
    buf = io.BytesIO()
    decode_qr_photo(photo_bytes).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_qr_photo(photo_bytes: bytes, output_path: str) -> dict:
    img = decode_qr_photo(photo_bytes)
    img.save(output_path, format="PNG")
    return {
        "width": img.width,
        "height": img.height,
        "output_path": output_path,
    }


if __name__ == "__main__":
    from qr_signature_verification import decode_secure_qr
    import re

    with open("qr_signature_verification.py") as f:
        content = f.read()
    m = re.search(r'SAMPLE_QR_V1 = \(\n(.*?)\n    \)', content, re.DOTALL)
    lines = re.findall(r'"([0-9]+)"', m.group(1))
    sample_qr = "".join(lines)

    decoded = decode_secure_qr(sample_qr)
    result = save_qr_photo(decoded["photo_bytes"], "test_extracted_photo.png")
    print(f"Decoded photo: {result['width']}x{result['height']}, saved to {result['output_path']}")

"""
pdf_utils.py — Converts PDF documents (e.g. e-Aadhaar downloads from
UIDAI) into images so the rest of the TRINETRA pipeline (which operates
on images) can process them unchanged.

e-Aadhaar PDFs are typically single-page, high-resolution, and contain
a large, cleanly-rendered QR code — actually easier to process reliably
than a photographed physical card.
"""

import os
import pymupdf  # PyMuPDF


class PDFConversionError(Exception):
    pass


def is_encrypted(pdf_path: str) -> bool:
    """
    Checks whether a PDF requires a password to open, without needing
    to know what the password actually is. Genuine e-Aadhaar downloads
    from UIDAI are ALWAYS password-protected — a PDF presenting itself
    as an e-Aadhaar that opens with no password at all is itself an
    unusual, worth-flagging signal (not proof of tampering on its own,
    but inconsistent with how UIDAI actually issues these documents).
    """
    try:
        doc = pymupdf.open(pdf_path)
        needs_pass = doc.needs_pass
        doc.close()
        return needs_pass
    except Exception:
        return False


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300, password: str = None) -> list:
    """
    Converts every page of a PDF into a separate PNG image at the given
    DPI. Supports password-protected PDFs (e-Aadhaar downloads from
    UIDAI are encrypted by default — the password is typically the
    first 4 letters of the person's name in CAPS followed by their
    birth year, e.g. "RAHU1990").

    Returns a list of output image paths, one per page, in order.
    Raises PDFConversionError with a clear message if the PDF is
    encrypted and no password (or the wrong password) was given.
    """
    if not os.path.exists(pdf_path):
        raise PDFConversionError(f"PDF not found: {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise PDFConversionError(f"Could not open PDF: {e}")

    if doc.needs_pass:
        if not password:
            doc.close()
            raise PDFConversionError(
                "This PDF is password-protected (common for e-Aadhaar downloads — "
                "try the first 4 letters of the name in CAPS followed by the "
                "birth year, e.g. 'RAHU1990'). Please provide the password."
            )
        if not doc.authenticate(password):
            doc.close()
            raise PDFConversionError("Incorrect password for this PDF.")

    if doc.page_count == 0:
        doc.close()
        raise PDFConversionError("PDF has no pages.")

    zoom = dpi / 72  # PDF base unit is 72 DPI
    matrix = pymupdf.Matrix(zoom, zoom)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_paths = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        out_path = os.path.join(output_dir, f"{base_name}_page{page_num + 1}.png")
        pix.save(out_path)
        output_paths.append(out_path)


    doc.close()
    return output_paths


def is_pdf(filepath: str) -> bool:
    """Checks the file extension AND the actual file header (more
    reliable than trusting a possibly-wrong extension from an upload)."""
    if filepath.lower().endswith(".pdf"):
        return True
    try:
        with open(filepath, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pdf_utils.py <path_to_pdf>")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(sys.argv[1])) or "."
    paths = pdf_to_images(sys.argv[1], out_dir)
    print(f"Converted {len(paths)} page(s):")
    for p in paths:
        print(f"  {p}")

"""
app.py — TRINETRA/FDD-XAI web frontend.

Single unified upload: accepts multiple files in one field, images
and/or PDFs (e.g. e-Aadhaar downloads) mixed together. PDFs are
automatically converted to images (one per page) before the pipeline
processes them.

Run with: python3 app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import time
import uuid
import base64
import shutil
import tempfile
import threading
import cv2
from flask import Flask, request, render_template, redirect, url_for, jsonify

from pipeline import run_pipeline
from pdf_utils import pdf_to_images, is_pdf, is_encrypted, PDFConversionError
from metadata_analysis import exiftool_path, EXIFTOOL_INSTALL_HELP

app = Flask(__name__)
# PRIVACY: uploaded documents are never kept. Each scan works inside its own
# private temporary folder, which is deleted as soon as that scan finishes
# (including when it fails). Images shown on the result page are embedded
# in the page from memory, and a result is forgotten once it has been viewed.
JOB_TTL_SECONDS = 10 * 60   # unviewed results are discarded after this

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pdf"}

# Set these to your UIDAI certificate paths to enable real QR signature
# verification. Different Aadhaar document types use DIFFERENT
# certificate chains (confirmed via testing): physical Secure QR cards
# verify against one certificate, e-Aadhaar PDF downloads verify
# against a different "Offline e-KYC" certificate. List all available
# certificates here — each is tried automatically, whichever matches.
UIDAI_CERT_PATHS = [
     r"C:\Codes\Trinetra\Certificates\UIDAI1.cer",                        # physical Secure QR cards
     r"C:\Codes\Trinetra\Certificates\uidai_offline_publickey_2026.cer",  # e-Aadhaar PDF downloads
]


# Pipeline stages in the order they run, shown on the live progress page.
STAGES = [
    ("upload", "Upload & PDF conversion", "Files received, PDF pages turned into images"),
    ("ocr", "Reading the document (OCR)", "Detecting orientation and reading the printed text"),
    ("qr", "Secure QR verification", "Decoding the QR and checking UIDAI's digital signature"),
    ("checksum", "Aadhaar number checksum", "Verhoeff check on the 12-digit number"),
    ("localizer", "Field localisation (YOLOv8n)", "Finding name, DOB, number and photo regions"),
    ("forensics", "Image & metadata forensics", "Error Level Analysis, copy-move search, EXIF"),
    ("aggregate", "Evidence aggregation", "Weighing every flag into one authenticity score"),
    ("report", "Writing the forensic report", "Explaining which checks passed and why"),
]

# In-memory job store: fine for a single local server process.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _thumbnail_data_uri(path, max_side=900):
    """Small JPEG of the first page, embedded in the result page from memory."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        for jid in [j for j, job in JOBS.items() if job["created"] < cutoff]:
            del JOBS[jid]


def _run_job(job_id, work_dir, image_paths, extra_flags, conversion_error, original_filenames):
    def on_progress(key):
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["current"] = key
            job["started"][key] = time.time()

    front_image = None
    try:
        result = run_pipeline(image_paths, uidai_cert_path=UIDAI_CERT_PATHS,
                              extra_flags=extra_flags, progress=on_progress)
        error = conversion_error
        front_image = _thumbnail_data_uri(image_paths[0])
    except Exception as e:
        result, error = None, str(e)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)   # nothing from the upload stays on disk

    extracted_photo_url = (result.get("qr_result") or {}).get("extracted_photo_data_uri") if result else None

    with JOBS_LOCK:
        JOBS[job_id].update({
            "done": True,
            "finished": time.time(),
            "context": dict(
                result=result, error=error, filename=", ".join(original_filenames),
                front_image_url=front_image,
                extracted_photo_url=extracted_photo_url,
            ),
        })


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/camera", methods=["GET"])
def camera():
    """Live webcam scan: captures go through the same /analyze pipeline."""
    return render_template("camera.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_files = request.files.getlist("documents")
    uploaded_files = [f for f in uploaded_files if f and f.filename]
    pdf_password = request.form.get("pdf_password", "").strip() or None

    if not uploaded_files:
        return redirect(url_for("index"))

    _purge_old_jobs()
    work_dir = tempfile.mkdtemp(prefix="trinetra_")   # private, deleted after the scan
    image_paths = []
    original_filenames = []
    conversion_error = None
    extra_flags = []

    for file in uploaded_files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        save_path = os.path.join(work_dir, f"{uuid.uuid4().hex}{ext}")
        file.save(save_path)
        original_filenames.append(file.filename)

        if is_pdf(save_path):
            if not is_encrypted(save_path):
                extra_flags.append({
                    "source": "PDF Security Check",
                    "code": "pdf_not_password_protected",
                    "severity": "high",
                    "description": (
                        f"'{file.filename}' opened with NO password required. Genuine "
                        f"e-Aadhaar PDFs downloaded from UIDAI's own site are ALWAYS "
                        f"password-protected — a PDF presenting itself as an e-Aadhaar "
                        f"with no password protection at all is inconsistent with how "
                        f"UIDAI actually issues these documents. This could indicate the "
                        f"PDF was re-saved/re-exported by different software after the "
                        f"original download (which would strip the password), or that "
                        f"it was never a genuine UIDAI-issued PDF to begin with."
                    ),
                })
            try:
                page_images = pdf_to_images(save_path, work_dir, password=pdf_password)
                image_paths.extend(page_images)
            except PDFConversionError as e:
                conversion_error = f"Could not convert PDF '{file.filename}': {e}"
        else:
            image_paths.append(save_path)

    if not image_paths:
        shutil.rmtree(work_dir, ignore_errors=True)
        return render_template(
            "result.html",
            result=None,
            error=conversion_error or "No valid image or PDF pages found in the uploaded files.",
            filename=", ".join(original_filenames),
            front_image_url=None,
            extracted_photo_url=None,
        )

    job_id = uuid.uuid4().hex
    now = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = {"current": "upload", "started": {"upload": now}, "created": now,
                        "done": False, "context": None}
    threading.Thread(
        target=_run_job,
        args=(job_id, work_dir, image_paths, extra_flags, conversion_error, original_filenames),
        daemon=True,
    ).start()
    return redirect(url_for("progress_page", job_id=job_id))


@app.route("/progress/<job_id>")
def progress_page(job_id):
    if job_id not in JOBS:
        return redirect(url_for("index"))
    return render_template("progress.html", job_id=job_id, stages=STAGES)


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        now = job.get("finished") or time.time()
        return jsonify({
            "current": job["current"],
            "done": job["done"],
            "elapsed": round(now - job["created"], 1),
            "started": {k: round(v - job["created"], 2) for k, v in job["started"].items()},
        })


@app.route("/result/<job_id>")
def result_page(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return redirect(url_for("index"))
        if not job["done"]:
            return redirect(url_for("progress_page", job_id=job_id))
        del JOBS[job_id]   # shown once, then forgotten
    resp = app.make_response(render_template("result.html", **job["context"]))
    resp.headers["Cache-Control"] = "no-store"   # don't let the browser cache the report either
    return resp


if __name__ == "__main__":
    if exiftool_path() is None:
        print(EXIFTOOL_INSTALL_HELP)
        sys.exit(1)
    # Load the YOLOv8n model once at startup, so the first scan isn't
    # slowed down by model loading.
    from field_localizer import _get_model
    _get_model()
    # Two servers in one process, so uploads, the PC webcam AND phones all work
    # with a plain `python app.py`:
    #   http://127.0.0.1:5000        this PC (uploads + webcam)
    #   https://<this PC's IP>:5443  phones on the same Wi-Fi. Browsers only
    #                                allow camera access over HTTPS on other
    #                                devices; the certificate is self-signed,
    #                                so the phone shows a warning once.
    import socket
    from werkzeug.serving import make_server

    def lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))   # no traffic is sent; just picks the Wi-Fi interface
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None

    try:
        phone_server = make_server("0.0.0.0", 5443, app, threaded=True, ssl_context="adhoc")
        threading.Thread(target=phone_server.serve_forever, daemon=True).start()
        ip = lan_ip()
        print("\n  On this PC:       http://127.0.0.1:5000")
        print(f"  On your phone:    https://{ip or '<this-PC-IP>'}:5443/camera   (same Wi-Fi; accept the certificate warning)\n")
    except (Exception, SystemExit) as e:   # werkzeug calls sys.exit() if the port is taken
        print(f"\n  Phone access unavailable ({e}). This PC still works at http://127.0.0.1:5000\n")

    app.run(debug=False, host="127.0.0.1", port=5000)

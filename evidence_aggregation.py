"""
evidence_aggregation.py — Module 5 (Evidence Aggregation) of the
TRINETRA/FDD-XAI pipeline.
"""

import json
from datetime import datetime, timezone


SEVERITY_WEIGHTS = {
    "critical": 40,
    "high": 20,
    "medium": 10,
    "info": 0,
}

MODULE_SOURCE_LABELS = {
    "verhoeff_checksum": "Aadhaar Number Checksum (Verhoeff)",
    "mrz_checksum": "Passport MRZ Checksum (ICAO 9303)",
    "qr_signature_verification": "Aadhaar QR Signature Verification",
    "metadata_analysis": "File Metadata Analysis (EXIF)",
    "image_forensics": "Image Forensics (ELA / Copy-Move)",
    "field_localization": "Field Localization (YOLOv8n)",
}


def build_evidence_bundle(
    document_type: str,
    verhoeff_result: dict = None,
    mrz_result: dict = None,
    qr_result: dict = None,
    metadata_result: dict = None,
    forensics_result: dict = None,
    localizer_result: dict = None,
    extra_flags: list = None,
) -> dict:
    all_flags = []

    if extra_flags:
        for flag in extra_flags:
            all_flags.append({
                "source": flag.get("source", "Document-Level Check"),
                "code": flag["code"],
                "severity": flag["severity"],
                "description": flag["description"],
            })

    if verhoeff_result is not None:
        source = MODULE_SOURCE_LABELS["verhoeff_checksum"]
        if verhoeff_result.get("valid") is False:
            all_flags.append({
                "source": source,
                "code": "checksum_invalid",
                "severity": "high",
                "description": verhoeff_result.get(
                    "description",
                    "Aadhaar number fails the Verhoeff checksum — internally inconsistent number.",
                ),
            })

    if mrz_result is not None:
        source = MODULE_SOURCE_LABELS["mrz_checksum"]
        for reason in mrz_result.get("reasons", []):
            all_flags.append({
                "source": source,
                "code": "mrz_checksum_invalid",
                "severity": "high",
                "description": reason,
            })

    if qr_result is not None:
        source = MODULE_SOURCE_LABELS["qr_signature_verification"]
        for flag in qr_result.get("flags", []):
            all_flags.append({
                "source": source,
                "code": flag["code"],
                "severity": flag["severity"],
                "description": flag["description"],
            })

    if metadata_result is not None:
        source = MODULE_SOURCE_LABELS["metadata_analysis"]
        for flag in metadata_result.get("flags", []):
            all_flags.append({
                "source": source,
                "code": flag["code"],
                "severity": flag["severity"],
                "description": flag["description"],
            })

    if forensics_result is not None:
        source = MODULE_SOURCE_LABELS["image_forensics"]
        for flag in forensics_result.get("flags", []):
            all_flags.append({
                "source": source,
                "code": flag["code"],
                "severity": flag["severity"],
                "description": flag["description"],
            })

    if localizer_result is not None:
        source = MODULE_SOURCE_LABELS["field_localization"]
        for flag in localizer_result.get("flags", []):
            all_flags.append({
                "source": source,
                "code": flag["code"],
                "severity": flag["severity"],
                "description": flag["description"],
            })

    contradictions = _find_contradictions(all_flags)

    score = 100
    for flag in all_flags:
        score -= SEVERITY_WEIGHTS.get(flag["severity"], 0)
    score = max(0, score)

    # If QR signature verification was never even attempted (no
    # certificate configured), the document has NOT been cryptographically
    # verified — this must never look the same as a document that WAS
    # checked and came back clean. We cap the score so an unverified
    # Secure QR document can never present as a high-confidence "GENUINE"
    # result purely because nothing was actually checked.
    signature_verification_skipped = any(
        f["code"] == "signature_not_checked" for f in all_flags
    )
    if signature_verification_skipped:
        score = min(score, 70)

    has_critical = any(f["severity"] == "critical" for f in all_flags)
    has_deterministic_failure = any(
        f["code"] in ("checksum_invalid", "mrz_checksum_invalid", "signature_invalid")
        for f in all_flags
    )
    # Note: field_localizer_number_mismatch is deliberately NOT included
    # above — it is a probabilistic corroborating signal (two OCR passes
    # disagreeing), not a mathematically/cryptographically unambiguous
    # failure, so it must never force the deterministic-failure override.

    bundle = {
        "document_type": document_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flags": all_flags,
        "flag_count": len(all_flags),
        "flag_count_by_severity": {
            sev: sum(1 for f in all_flags if f["severity"] == sev)
            for sev in ("critical", "high", "medium", "info")
        },
        "contradictions": contradictions,
        "pre_llm_authenticity_score": score,
        "has_critical_flag": has_critical,
        "has_deterministic_failure": has_deterministic_failure,
        "signature_verification_skipped": signature_verification_skipped,
        "modules_run": [
            name for name, result in [
                ("verhoeff_checksum", verhoeff_result),
                ("mrz_checksum", mrz_result),
                ("qr_signature_verification", qr_result),
                ("metadata_analysis", metadata_result),
                ("image_forensics", forensics_result),
                ("field_localization", localizer_result),
            ] if result is not None
        ],
    }
    return bundle


def _find_contradictions(all_flags: list) -> list:
    contradictions = []
    codes_present = {f["code"] for f in all_flags}

    tamper_signal_codes = {
        "checksum_invalid", "mrz_checksum_invalid", "signature_invalid",
        "editing_software_signature", "ela_anomaly_regions", "copy_move_detected",
        "ocr_mismatch_name", "ocr_mismatch_dob",
    }
    matching = codes_present & tamper_signal_codes
    if len(matching) >= 2:
        contradictions.append({
            "type": "corroboration",
            "description": (
                f"{len(matching)} independent modules ({', '.join(sorted(matching))}) "
                f"raised tamper-consistent flags — independent corroboration across "
                f"evidence sources significantly strengthens confidence that this "
                f"document has been altered, rather than relying on a single signal."
            ),
        })

    return contradictions


if __name__ == "__main__":
    mock_verhoeff = {"valid": False, "description": "Aadhaar number fails Verhoeff checksum."}
    bundle = build_evidence_bundle(document_type="aadhaar", verhoeff_result=mock_verhoeff)
    print(json.dumps(bundle, indent=2, default=str))

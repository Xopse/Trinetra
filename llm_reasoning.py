"""
llm_reasoning.py — Module 6 (LLM Reasoning Layer) of the TRINETRA/FDD-XAI
pipeline. The LLM does NOT decide independently — it reasons over
structured evidence that deterministic/forensic modules already produced.
"""

import os
import json


SYSTEM_PROMPT = """You are the reasoning layer of TRINETRA, a fake-document detection system.
You will be given a structured JSON evidence bundle produced by independent forensic modules
(checksum validators, QR/MRZ signature verification, EXIF metadata analysis, image forensics).

Your job is strictly to CORRELATE and EXPLAIN this evidence in a clear, professional,
human-readable forensic report — you must NOT invent findings, speculate about anything
not present in the evidence, or make claims the evidence doesn't support.

Structure your report with:
1. VERDICT: one of GENUINE, SUSPICIOUS, or FAKE, with a one-line justification
2. FORGERY TYPE (only if not GENUINE)
3. CONFIDENCE
4. EXPLANATION: a few sentences per flag

Do not use markdown headers with # symbols, just clear section labels in plain text."""


def _try_anthropic_llm(evidence_bundle: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "Here is the structured evidence bundle for a document that was "
                    "just analyzed. Write the forensic report.\n\n"
                    f"{json.dumps(evidence_bundle, indent=2, default=str)}"
                ),
            }],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks) if text_blocks else None
    except Exception:
        return None


def _template_based_report(evidence_bundle: dict) -> str:
    flags = evidence_bundle.get("flags", [])
    score = evidence_bundle.get("pre_llm_authenticity_score", 100)
    has_critical = evidence_bundle.get("has_critical_flag", False)
    has_deterministic_failure = evidence_bundle.get("has_deterministic_failure", False)
    doc_type = evidence_bundle.get("document_type", "document")
    contradictions = evidence_bundle.get("contradictions", [])
    signature_skipped = evidence_bundle.get("signature_verification_skipped", False)

    if has_critical or has_deterministic_failure:
        verdict = "FAKE"
        justification = (
            "A deterministic, cryptographic or checksum-based check failed — "
            "this is conclusive evidence of tampering, not a probabilistic signal."
        )
    elif score < 60:
        verdict = "FAKE"
        justification = f"Multiple corroborating forensic signals (authenticity score {score}/100)."
    elif score < 85:
        verdict = "SUSPICIOUS"
        if signature_skipped and len(flags) == 1:
            justification = (
                f"No forensic anomalies were found, BUT the QR's cryptographic "
                f"signature was never verified (no UIDAI certificate was "
                f"configured) — this is NOT the same as a verified-clean result. "
                f"Score is capped at {score}/100 specifically because the "
                f"strongest available check was skipped, not because anomalies "
                f"were detected."
            )
        else:
            justification = f"Some forensic anomalies detected (authenticity score {score}/100) that warrant manual review."
    else:
        verdict = "GENUINE"
        justification = f"No significant tampering signals detected (authenticity score {score}/100)."

    codes = {f["code"] for f in flags}
    forgery_type = None
    is_only_skip_flag = signature_skipped and len(flags) == 1
    if verdict != "GENUINE" and not is_only_skip_flag:
        if "signature_invalid" in codes:
            forgery_type = "Digitally-signed data tampering (QR/MRZ signature does not verify)"
        elif codes & {"checksum_invalid", "mrz_checksum_invalid"}:
            forgery_type = "Fabricated or digit-edited identifier (checksum-inconsistent number)"
        elif "copy_move_detected" in codes:
            forgery_type = "Copy-move tampering (content duplicated/moved within the same image)"
        elif "ela_anomaly_regions" in codes:
            forgery_type = "Localized digital editing (region re-compressed separately from rest of document)"
        elif codes & {"ocr_mismatch_name", "ocr_mismatch_dob"}:
            forgery_type = "Re-used genuine QR/MRZ with altered printed data"
        elif "editing_software_signature" in codes:
            forgery_type = "Digital text/image tampering (editing software signature in file metadata)"
        else:
            forgery_type = "Unclassified anomaly (see individual flags below)"

    if has_deterministic_failure or has_critical:
        confidence = "High — based on deterministic/cryptographic evidence, not a probabilistic estimate."
    elif len(contradictions) > 0:
        confidence = (
            f"Elevated — {len(contradictions)} independent module(s) corroborate each "
            f"other, which meaningfully increases confidence over any single signal alone."
        )
    else:
        scored = [f for f in flags if f["severity"] != "info"]
        if verdict == "GENUINE" and not scored:
            confidence = "Standard — no tampering signals found; any notes below are informational only."
        elif len(scored) == 1:
            confidence = "Low-to-moderate — based on a single forensic signal; recommend cross-checking with additional modules or manual review before final action."
        else:
            confidence = "Low-to-moderate — several weak signals with no deterministic failure; recommend manual review."

    lines = []
    lines.append(f"VERDICT: {verdict}")
    lines.append(justification)
    lines.append("")

    if forgery_type:
        lines.append(f"FORGERY TYPE: {forgery_type}")
        lines.append("")

    lines.append(f"CONFIDENCE: {confidence}")
    lines.append("")

    lines.append("EXPLANATION:")
    if not flags:
        lines.append(
            f"No forensic modules raised any flags on this {doc_type}. All checksum, "
            f"signature, metadata, and image-forensic checks passed cleanly."
        )
    else:
        for i, flag in enumerate(flags, 1):
            lines.append(f"{i}. [{flag['source']}] {flag['description']}")

    if contradictions:
        lines.append("")
        lines.append("CROSS-MODULE CORROBORATION:")
        for c in contradictions:
            lines.append(f"- {c['description']}")

    return "\n".join(lines)


def generate_report(evidence_bundle: dict, force_template: bool = False) -> dict:
    if not force_template:
        llm_report = _try_anthropic_llm(evidence_bundle)
        if llm_report:
            return {"report": llm_report, "generation_method": "llm"}

    template_report = _template_based_report(evidence_bundle)
    return {"report": template_report, "generation_method": "template"}


if __name__ == "__main__":
    from evidence_aggregation import build_evidence_bundle

    mock_verhoeff = {"valid": False, "description": "Test."}
    bundle = build_evidence_bundle(document_type="aadhaar", verhoeff_result=mock_verhoeff)
    result = generate_report(bundle)
    print(f"[Generated via: {result['generation_method']}]\n")
    print(result["report"])

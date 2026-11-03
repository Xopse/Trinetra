"""
qr_signature_verification.py — Module 2 (QR/Barcode & MRZ Validation) of the
TRINETRA/FDD-XAI pipeline — Aadhaar QR Code decode + signature verification.

Supports THREE real-world Aadhaar QR variants, auto-detected:
  1. Secure QR v1 (2018-2022)
  2. Secure QR v2 (2022 onward) — adds "version" + "last_4_digits_mobile_no"
     text fields, and drops the old binary mobile_hash block (replaced by
     the text field) — confirmed via real V2 card testing.
  3. Old format (pre-2018): plain, UNSIGNED XML.
"""

import gzip
import json
import re

TEXT_FIELD_NAMES_V1 = [
    "email_mobile_indicator", "reference_id", "name", "dob", "gender",
    "care_of", "district", "landmark", "house", "location", "pincode",
    "post_office", "state", "street", "sub_district", "vtc",
]
TEXT_FIELD_NAMES_V2 = (
    ["version"] + TEXT_FIELD_NAMES_V1 + ["last_4_digits_mobile_no"]
)

SIGNATURE_LENGTH = 256
HASH_LENGTH = 32
DELIMITER = 0xFF


class QRDecodeError(Exception):
    pass


def is_secure_qr(qr_text: str) -> bool:
    return qr_text.strip().isdigit()


def decode_old_format_qr(qr_text: str) -> dict:
    """Decodes the OLD (pre-2018) Aadhaar QR format: plain unsigned XML.
    NO digital signature exists in this format at all."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(qr_text.strip())
        return dict(root.attrib)
    except ET.ParseError as e:
        raise QRDecodeError(f"Failed to parse as old-format Aadhaar QR XML: {e}")


def qr_payload_to_bytes(qr_text: str) -> bytes:
    qr_text = qr_text.strip()
    if not qr_text.isdigit():
        raise QRDecodeError("QR payload is not numeric — not a Secure QR Code.")
    n = int(qr_text)
    byte_len = (n.bit_length() + 7) // 8
    return n.to_bytes(byte_len, "big")


def decompress_payload(raw_bytes: bytes) -> bytes:
    try:
        return gzip.decompress(raw_bytes)
    except OSError as e:
        raise QRDecodeError(f"Failed to gzip-decompress QR payload: {e}")


def split_text_fields(data: bytes, field_names: list):
    fields = []
    start = 0
    for i, b in enumerate(data):
        if b == DELIMITER:
            fields.append(data[start:i])
            start = i + 1
            if len(fields) == len(field_names):
                return fields, start
    raise QRDecodeError(
        f"Only found {len(fields)} of {len(field_names)} expected delimited text fields."
    )


def decode_secure_qr(qr_text: str) -> dict:
    """
    Full decode, auto-detecting the extended field format (version
    marker + last_4_digits_mobile_no) vs the older fixed 16-field
    format. UIDAI uses multiple version markers over time (confirmed
    "V2" and "V5" via real card/e-Aadhaar testing so far) — we detect
    ANY marker matching the pattern "V" followed by a digit, rather
    than hardcoding "V2" specifically, since hardcoding one literal
    version string caused a real field-offset bug on a V5-format
    e-Aadhaar (every field shifted by one position, producing garbled
    output) that a broader pattern match avoids.
    """
    raw = qr_payload_to_bytes(qr_text)
    data = decompress_payload(raw)

    first_two = data[:2].decode("ISO-8859-1", errors="replace")
    has_version_marker = (
        len(first_two) == 2 and first_two[0] == "V" and first_two[1].isdigit()
    )
    field_names = TEXT_FIELD_NAMES_V2 if has_version_marker else TEXT_FIELD_NAMES_V1

    text_fields_raw, photo_start = split_text_fields(data, field_names)
    fields = {}
    for name, raw_val in zip(field_names, text_fields_raw):
        try:
            fields[name] = raw_val.decode("ISO-8859-1")
        except UnicodeDecodeError:
            fields[name] = raw_val.decode("ISO-8859-1", errors="replace")

    indicator = fields.get("email_mobile_indicator", "0").strip()
    has_email = indicator in ("1", "3")
    has_mobile = indicator in ("2", "3")

    signature = data[-SIGNATURE_LENGTH:]
    cursor = len(data) - SIGNATURE_LENGTH

    mobile_hash = None
    email_hash = None
    if has_version_marker:
        # The extended format adds a plain-text "last_4_digits_mobile_no"
        # field (already captured above), which replaces the older
        # 32-byte binary mobile_hash block entirely rather than
        # coexisting with it — confirmed empirically against real cards.
        if has_email:
            cursor -= HASH_LENGTH
            email_hash = data[cursor:cursor + HASH_LENGTH]
    else:
        if has_mobile:
            cursor -= HASH_LENGTH
            mobile_hash = data[cursor:cursor + HASH_LENGTH]
        if has_email:
            cursor -= HASH_LENGTH
            email_hash = data[cursor:cursor + HASH_LENGTH]

    photo_end = cursor
    photo_bytes = data[photo_start:photo_end]
    signed_data = data[:-SIGNATURE_LENGTH]

    return {
        "qr_version": fields.get("version", "v1_no_marker") if has_version_marker else "v1",
        "fields": fields,
        "has_email": has_email,
        "has_mobile": has_mobile,
        "mobile_hash": mobile_hash.hex() if mobile_hash else None,
        "email_hash": email_hash.hex() if email_hash else None,
        "photo_bytes_length": len(photo_bytes),
        "photo_bytes": photo_bytes,
        "signed_data": signed_data,
        "signature": signature,
        "total_payload_bytes": len(data),
    }


def verify_signature(signed_data: bytes, signature: bytes, cert_path: str) -> dict:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature

    try:
        with open(cert_path, "rb") as f:
            cert_data = f.read()
        try:
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        except ValueError:
            cert = x509.load_der_x509_certificate(cert_data, default_backend())
        public_key = cert.public_key()
    except Exception as e:
        return {"valid": False, "error": f"Failed to load certificate: {e}"}

    try:
        public_key.verify(signature, signed_data, padding.PKCS1v15(), hashes.SHA256())
        return {"valid": True, "error": None}
    except InvalidSignature:
        return {"valid": False, "error": "Signature does not match — data was altered after signing, or wrong certificate."}
    except Exception as e:
        return {"valid": False, "error": f"Verification error: {e}"}


def analyze(qr_text: str, ocr_text: str = "", cert_path=None) -> dict:
    """
    Main entry point. Auto-detects Secure QR (v1/v2/v5/etc.) vs
    old-format XML, decodes, verifies signature if possible, cross-
    checks fields against OCR'd printed text.

    cert_path can be a single path, a list of paths, or None. Different
    Aadhaar document types use DIFFERENT UIDAI certificate chains —
    confirmed via real testing: physical Secure QR cards verify against
    one certificate, while e-Aadhaar PDF downloads verify against a
    different "Offline e-KYC" certificate. When given a list, each is
    tried in turn and the first one that verifies successfully is used;
    if none verify, the failure from the LAST attempted certificate is
    reported (arbitrary but consistent).
    """
    flags = []
    result = {"module": "qr_signature_verification"}

    if not qr_text or not qr_text.strip():
        result["decode_success"] = False
        result["flags"] = [{"code": "qr_empty", "severity": "medium",
                             "description": "No QR payload provided."}]
        return result

    if isinstance(cert_path, str):
        cert_paths = [cert_path]
    elif cert_path is None:
        cert_paths = []
    else:
        cert_paths = list(cert_path)

    if is_secure_qr(qr_text):
        try:
            decoded = decode_secure_qr(qr_text)
        except QRDecodeError as e:
            result["decode_success"] = False
            result["flags"] = [{"code": "qr_decode_failed", "severity": "high", "description": str(e)}]
            return result

        result["decode_success"] = True
        result["qr_format"] = f"secure_{decoded['qr_version']}"
        result["fields"] = decoded["fields"]
        result["has_email"] = decoded["has_email"]
        result["has_mobile"] = decoded["has_mobile"]
        result["photo_bytes"] = decoded["photo_bytes"]

        if cert_paths:
            sig_result = None
            matched_cert = None
            for cp in cert_paths:
                sig_result = verify_signature(decoded["signed_data"], decoded["signature"], cp)
                if sig_result["valid"]:
                    matched_cert = cp
                    break

            result["signature_valid"] = sig_result["valid"] if sig_result else False
            result["matched_certificate"] = matched_cert
            if not result["signature_valid"]:
                flags.append({
                    "code": "signature_invalid", "severity": "critical",
                    "description": (
                        f"The QR code's digital signature does not verify against "
                        f"any of the {len(cert_paths)} configured UIDAI certificate(s) "
                        f"({sig_result['error'] if sig_result else 'no certificates provided'}). "
                        f"This is the strongest possible signal of tampering — though note "
                        f"that different Aadhaar document types (physical card vs. "
                        f"e-Aadhaar PDF) use different certificate chains, so this can "
                        f"also indicate the wrong certificate was configured rather than "
                        f"genuine tampering."
                    ),
                })
        else:
            result["signature_valid"] = None
            result["matched_certificate"] = None
            flags.append({
                "code": "signature_not_checked", "severity": "info",
                "description": "No UIDAI public certificate path was provided — signature verification was skipped.",
            })

        name_key, dob_key = "name", "dob"

        # --- Secure QR should always carry certain essential fields ---
        # A genuine, correctly-generated Secure QR always has a non-empty
        # name and dob (they're mandatory per spec), and always carries
        # embedded photo bytes (UIDAI always includes the photo in this
        # format, unlike the old format which never did). Missing any
        # of these on a document claiming to be Secure QR format is
        # itself a red flag — either the QR was hand-crafted/fabricated
        # rather than genuinely issued, or something was stripped out
        # after issuance.
        if not result["fields"].get("name", "").strip():
            flags.append({
                "code": "essential_field_missing_name", "severity": "high",
                "description": (
                    "The QR claims to be a genuine Secure QR Code, but its 'name' "
                    "field is empty. UIDAI always populates this field on "
                    "genuinely-issued Secure QR codes — an empty value here is "
                    "consistent with a hand-crafted or incompletely fabricated QR "
                    "rather than one genuinely issued by UIDAI."
                ),
            })
        if not result["fields"].get("dob", "").strip():
            flags.append({
                "code": "essential_field_missing_dob", "severity": "high",
                "description": (
                    "The QR claims to be a genuine Secure QR Code, but its 'dob' "
                    "field is empty. This is mandatory on genuinely-issued Secure "
                    "QR codes and its absence is consistent with a fabricated QR."
                ),
            })
        if not decoded.get("photo_bytes") or len(decoded["photo_bytes"]) < 10:
            flags.append({
                "code": "essential_field_missing_photo", "severity": "high",
                "description": (
                    "The QR claims to be a genuine Secure QR Code, but no "
                    "embedded photo data is present. UIDAI always embeds a "
                    "photo in this format — its absence is consistent with a "
                    "fabricated QR that was never genuinely issued by UIDAI, "
                    "rather than a scanning or decoding failure on our end."
                ),
            })

    else:
        try:
            fields = decode_old_format_qr(qr_text)
        except QRDecodeError as e:
            result["decode_success"] = False
            result["flags"] = [{"code": "qr_decode_failed", "severity": "high", "description": str(e)}]
            return result

        result["decode_success"] = True
        result["qr_format"] = "old_unsigned"
        result["fields"] = fields
        result["has_email"] = None
        result["has_mobile"] = None
        result["signature_valid"] = None

        flags.append({
            "code": "old_format_no_signature", "severity": "medium",
            "description": (
                "This QR code uses the OLD (pre-2018) Aadhaar format, which "
                "carries NO digital signature at all. Treat this as a much "
                "weaker signal than a verified Secure QR signature."
            ),
        })

        # Old-format Aadhaar QR field names vary by card generation —
        # confirmed via real card testing that some use "dateOfBirth"
        # (a full date, e.g. "09-11-2002") rather than "yob" (year-only)
        # as older documentation suggested. We check for either, since
        # assuming only one caused a real false "missing field" flag on
        # a genuine card during testing.
        if fields.get("dateOfBirth", "").strip():
            dob_key = "dateOfBirth"
        elif fields.get("yob", "").strip():
            dob_key = "yob"
        else:
            dob_key = "dateOfBirth"  # neither present; used below for the missing-field check
        name_key = "name"

        if not fields.get("name", "").strip():
            flags.append({
                "code": "essential_field_missing_name", "severity": "high",
                "description": (
                    "This old-format QR has an empty or missing 'name' field. "
                    "Even the unsigned old format always carries a name on "
                    "genuinely-issued Aadhaar cards — its absence suggests "
                    "fabricated or corrupted QR content."
                ),
            })
        if not fields.get("dateOfBirth", "").strip() and not fields.get("yob", "").strip():
            flags.append({
                "code": "essential_field_missing_dob", "severity": "high",
                "description": (
                    "This old-format QR has an empty or missing date-of-birth "
                    "field (checked both 'dateOfBirth' and 'yob' variants, since "
                    "different card generations use different field names). "
                    "Genuinely-issued old-format Aadhaar QR codes always carry "
                    "this field — its absence suggests fabricated or corrupted "
                    "QR content."
                ),
            })
        if not fields.get("uid", "").strip():
            flags.append({
                "code": "essential_field_missing_uid", "severity": "high",
                "description": (
                    "This old-format QR has an empty or missing 'uid' (Aadhaar "
                    "number) field — a mandatory field on any genuinely-issued "
                    "Aadhaar QR code, old or new format."
                ),
            })
        # Note: old-format QR never embeds a photo at all (that's a
        # Secure QR-only feature per spec) — so we do NOT flag a missing
        # photo here; doing so would be a false positive on every single
        # genuine old-format card.

    # --- Cross-check decoded fields against OCR'd printed text ---
    if ocr_text:
        ocr_clean = ocr_text.replace("\n", " ").upper()

        name_val = result["fields"].get(name_key, "").strip()
        if name_val and name_val.upper() not in ocr_clean:
            flags.append({
                "code": f"ocr_mismatch_{name_key}", "severity": "high",
                "description": (
                    f"QR code states {name_key}='{name_val}', but this value was not "
                    f"found in the visible printed text on the card."
                ),
            })

        # DOB: normalize separators before comparing — cards commonly
        # print dates with '/' while the QR uses '-' (or vice versa);
        # these represent the same date and should NOT be flagged.
        dob_val = result["fields"].get(dob_key, "").strip()
        if dob_val:
            dob_digits = re.sub(r"[^0-9]", "", dob_val)
            if len(dob_digits) >= 6:
                # Full date (e.g. Secure QR's "dob"): require an exact
                # match against a normalized full-length date run in the
                # OCR text — this is what our earlier real-card testing
                # validated (06-02-2005 vs 06/02/2005 style separator
                # differences).
                ocr_digit_runs = re.findall(r"[0-9/\-.]{6,}", ocr_clean)
                ocr_digits_normalized = ["".join(re.sub(r"[^0-9]", "", run)) for run in ocr_digit_runs]
                found = dob_digits in ocr_digits_normalized
            else:
                # Year-only (old-format's "yob"): a 4-digit year would
                # never exactly equal a full 8-digit date run, so we
                # instead check whether the year appears as a substring
                # anywhere in the OCR'd digits — e.g. yob='1990' should
                # match a printed "Year of Birth: 1990" or a full DOB
                # like "01-01-1990" that CONTAINS that year.
                ocr_digits_all = re.sub(r"[^0-9]", "", ocr_clean)
                found = dob_digits in ocr_digits_all

            if not found:
                flags.append({
                    "code": f"ocr_mismatch_{dob_key}", "severity": "high",
                    "description": (
                        f"QR code states {dob_key}='{dob_val}', but this value was not "
                        f"found in the visible printed text on the card."
                    ),
                })

    result["flags"] = flags
    return result


if __name__ == "__main__":
    SAMPLE_QR_V1 = (
        "6979414848205548481619299442879901900893978332594614407044767717485407280104077714658698163325"
        "4016592128309207342330475784547018105670320152702236829179158252347037547125048879213091817896"
        "0780916888458384839645665300702247935633624019813036393088163236712473854151749949445813964737"
        "8808680614169273221404741476596583953169248831376224396335169577064812987140578144885819479190"
        "1735376449702321251422539637849791380113187983854424360999016219982836248160700805048307125945"
        "2576059693434157675562679159040363687813986166559938331942922836443418391319795873869700141049"
        "3839281298692342829951566712530309758759364649701153639921979798429707566199261950037418171329"
        "2832073720480149486691606667761984140406333846771047176975075217175867767090842003649561788636"
        "3610598886726092988757709295557040780378302139789734199991461679044102983722912974666922509563"
        "3201097644321593502503404440714110515167034889128258965583435965030225845348564582051521348800"
        "7425744428770877741946689835166296310733412027054533827806137754273369492833880848916544842254"
        "4694094166094244063778474429325991647984140708818946296448967023186648190423733849487281309889"
        "0875845640034370370387108798950180220865436012752487216677041817312930119747601017807577565413"
        "9775456933754801313242406960998794794367225765664479395931955906845912618090380231221781720061"
        "5049956918521883874933723828159703728892446400999753093833679817602359729232832096508699018453"
        "1426188862965408313308973495924965144113396593829090645266653313774582036138982013368561474719"
        "1544471348944666115605897582518290632263703002821758234795698472614393484045582514022737308650"
        "5348221458918002830204382143835758330281837414397399700274504752640575576040704500669442350133"
        "7081780299815080324840337828812644300041900356816429114261098230198976752026002079876882796597"
        "2356150155944861820577814761529181707464031570052168962394285217060334660615876080650361331530"
        "7443219595213136856423416800544777019034577702491762987963917116171992985207826530916075926098"
        "9590618158889891835294735614366674503961584445497685736312628248483551986529867423016255476553"
        "6919220542416862309689752295117009281712815499026823653023336774129517888398068697960405122358"
        "9931173433785868453115672141628011447336882646309848525239426007579038641587529092257056868643"
        "9586036262465414002334117870088922801660529414759784318799843806130096998190881240404138869293"
        "3097823353052967206662202433041750863582782113557899579980148012093322934589404638591065919864"
        "3452043381058356930922492926422826384147737894932931244395821593929443266946426021653407456088"
        "2723006838459792812340253078330291135526952675203790833430237852831740601433198364243363569730"
        "2053510773934416911412400559008190912299316051468655201830018102397084643225883899560362917601"
        "7555884381910541823458023961017432363660609526272294014370606369884649967328537762118057053778"
        "8160304936809915237889489342387891057012783726694920184573202789672963922380028271124448024265"
        "6443966863415084478303513802421275423938494102838304095949885032467995444446876069548815105975"
        "1568641099382890758897969914118016089306260333810485790323984585678313027593541356927543990878"
        "9983311663211937449259444259898972766208"
    )
    result = analyze(SAMPLE_QR_V1)
    print("=== Secure QR v1 sample ===")
    print("Decode success:", result["decode_success"], "| format:", result.get("qr_format"))
    print(json.dumps(result["fields"], indent=2, ensure_ascii=False))

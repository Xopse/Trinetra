"""ICAO 9303 MRZ parser + checksum validator for TD3-format passports."""

_WEIGHTS = [7, 3, 1]
_CHAR_VALUES = {**{str(i): i for i in range(10)}, "<": 0}
for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    _CHAR_VALUES[ch] = i + 10


def _char_value(ch: str) -> int:
    return _CHAR_VALUES.get(ch.upper(), 0)


def compute_check_digit(data: str) -> str:
    total = 0
    for i, ch in enumerate(data):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return str(total % 10)


def parse_td3(line1: str, line2: str) -> dict:
    line1 = line1.strip().upper().ljust(44, "<")[:44]
    line2 = line2.strip().upper().ljust(44, "<")[:44]
    result = {"raw_line1": line1, "raw_line2": line2, "checks": {}, "fields": {}, "reasons": []}

    doc_type = line1[0:2]
    issuing_country = line1[2:5]
    names_field = line1[5:44]
    surname, _, given_names = names_field.partition("<<")
    surname = surname.replace("<", " ").strip()
    given_names = given_names.replace("<", " ").strip()

    result["fields"]["document_type"] = doc_type.replace("<", "")
    result["fields"]["issuing_country"] = issuing_country
    result["fields"]["surname"] = surname
    result["fields"]["given_names"] = given_names

    passport_no = line2[0:9]
    passport_no_check = line2[9]
    nationality = line2[10:13]
    dob = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_check = line2[27]
    personal_no = line2[28:42]
    personal_no_check = line2[42]
    composite_check = line2[43]

    result["fields"]["passport_number"] = passport_no.replace("<", "")
    result["fields"]["nationality"] = nationality
    result["fields"]["date_of_birth"] = dob
    result["fields"]["sex"] = sex
    result["fields"]["expiry_date"] = expiry
    result["fields"]["personal_number"] = personal_no.replace("<", "")

    def check(name, data, given_check_digit):
        computed = compute_check_digit(data)
        ok = computed == given_check_digit
        result["checks"][name] = {"expected": given_check_digit, "computed": computed, "valid": ok}
        if not ok:
            result["reasons"].append(
                f"{name} checksum mismatch: MRZ states check digit '{given_check_digit}' "
                f"but computed '{computed}' from the field data — possible tampering."
            )
        return ok

    check("passport_number", passport_no, passport_no_check)
    check("date_of_birth", dob, dob_check)
    check("expiry_date", expiry, expiry_check)
    if personal_no.strip("<"):
        check("personal_number", personal_no, personal_no_check)

    composite_data = (passport_no + passport_no_check + dob + dob_check +
                       expiry + expiry_check + personal_no + personal_no_check)
    check("composite", composite_data, composite_check)

    result["all_valid"] = all(c["valid"] for c in result["checks"].values())
    return result


if __name__ == "__main__":
    line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    res = parse_td3(line1, line2)
    print("Fields:", res["fields"])
    print("All valid:", res["all_valid"])

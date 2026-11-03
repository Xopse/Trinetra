"""Verhoeff checksum algorithm for Aadhaar number validation."""

_MULT = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_PERM = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _string_to_digit_array(s):
    return [int(ch) for ch in s]


def validate(number: str) -> bool:
    number = number.strip().replace(" ", "")
    if not number.isdigit():
        return False
    digits = _string_to_digit_array(number)[::-1]
    c = 0
    for i, digit in enumerate(digits):
        c = _MULT[c][_PERM[i % 8][digit]]
    return c == 0


def generate_check_digit(number_without_check_digit: str) -> str:
    digits = _string_to_digit_array(number_without_check_digit)[::-1]
    c = 0
    for i, digit in enumerate(digits):
        c = _MULT[c][_PERM[(i + 1) % 8][digit]]
    return str(_INV[c])


if __name__ == "__main__":
    base = "234123412341"
    check_digit = generate_check_digit(base[:-1])
    valid_number = base[:-1] + check_digit
    print(f"Generated valid test number: {valid_number} -> valid: {validate(valid_number)}")
    tampered = valid_number[:-1] + str((int(valid_number[-1]) + 1) % 10)
    print(f"Tampered number: {tampered} -> valid: {validate(tampered)}")

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_BN_PUNCT = {
    "，": ",",
    "।": ".",
    "ঃ": ":",
    "—": "-",
    "–": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}
_MONTHS = {
    1: "জানুয়ারি",
    2: "ফেব্রুয়ারি",
    3: "মার্চ",
    4: "এপ্রিল",
    5: "মে",
    6: "জুন",
    7: "জুলাই",
    8: "আগস্ট",
    9: "সেপ্টেম্বর",
    10: "অক্টোবর",
    11: "নভেম্বর",
    12: "ডিসেম্বর",
}
_ONES = ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
_TEENS = {
    10: "দশ",
    11: "এগারো",
    12: "বারো",
    13: "তেরো",
    14: "চৌদ্দ",
    15: "পনেরো",
    16: "ষোলো",
    17: "সতেরো",
    18: "আঠারো",
    19: "উনিশ",
}
_TWENTIES = {
    20: "বিশ",
    21: "একুশ",
    22: "বাইশ",
    23: "তেইশ",
    24: "চব্বিশ",
    25: "পঁচিশ",
    26: "ছাব্বিশ",
    27: "সাতাশ",
    28: "আটাশ",
    29: "ঊনত্রিশ",
}
_TENS = {
    2: "বিশ",
    3: "ত্রিশ",
    4: "চল্লিশ",
    5: "পঞ্চাশ",
    6: "ষাট",
    7: "সত্তর",
    8: "আশি",
    9: "নব্বই",
}
_HUNDREDS = {
    1: "একশ",
    2: "দুইশ",
    3: "তিনশ",
    4: "চারশ",
    5: "পাঁচশ",
    6: "ছয়শ",
    7: "সাতশ",
    8: "আটশ",
    9: "নয়শ",
}


@dataclass(slots=True)
class NormalizationConfig:
    expand_abbreviations: bool = True
    normalize_unicode: bool = True
    normalize_punctuation: bool = True
    normalize_dates: bool = True
    normalize_currency: bool = True
    normalize_numbers: bool = True


_ABBREVIATIONS = {
    "ডা.": "ডাক্তার",
    "প্রা.": "প্রায়",
    "মি.": "মিস্টার",
    "মিসেস.": "মিসেস",
    "শ্রী.": "শ্রী",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_punctuation(text: str) -> str:
    for source, target in _BN_PUNCT.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _two_digit_to_words(number: int) -> str:
    if number < 10:
        return _ONES[number]
    if number in _TEENS:
        return _TEENS[number]
    if number in _TWENTIES:
        return _TWENTIES[number]
    tens, ones = divmod(number, 10)
    if ones == 0:
        return _TENS.get(tens, str(number))
    compound = {
        3: {1: "একত্রিশ", 2: "বত্রিশ", 3: "তেত্রিশ", 4: "চৌত্রিশ", 5: "পঁয়ত্রিশ", 6: "ছত্রিশ", 7: "সাঁইত্রিশ", 8: "আটত্রিশ", 9: "ঊনচল্লিশ"},
        4: {1: "একচল্লিশ", 2: "বিয়াল্লিশ", 3: "তেতাল্লিশ", 4: "চুয়াল্লিশ", 5: "পঁয়তাল্লিশ", 6: "ছেচল্লিশ", 7: "সাতচল্লিশ", 8: "আটচল্লিশ", 9: "ঊনপঞ্চাশ"},
        5: {1: "একান্ন", 2: "বায়ান্ন", 3: "তেপান্ন", 4: "চুয়ান্ন", 5: "পঞ্চান্ন", 6: "ছাপ্পান্ন", 7: "সাতান্ন", 8: "আটান্ন", 9: "ঊনষাট"},
        6: {1: "একষট্টি", 2: "বাষট্টি", 3: "তেষট্টি", 4: "চৌষট্টি", 5: "পঁয়ষট্টি", 6: "ছেষট্টি", 7: "সাতষট্টি", 8: "আটষট্টি", 9: "ঊনসত্তর"},
        7: {1: "একাত্তর", 2: "বাহাত্তর", 3: "তিয়াত্তর", 4: "চুয়াত্তর", 5: "পঁচাত্তর", 6: "ছিয়াত্তর", 7: "সাতাত্তর", 8: "আটাত্তর", 9: "ঊনআশি"},
        8: {1: "একাশি", 2: "বিরাশি", 3: "তিরাশি", 4: "চুরাশি", 5: "পঁচাশি", 6: "ছিয়াশি", 7: "সাতাশি", 8: "আটাশি", 9: "ঊননব্বই"},
        9: {1: "একানব্বই", 2: "বিরানব্বই", 3: "তিরানব্বই", 4: "চুরানব্বই", 5: "পঁচানব্বই", 6: "ছিয়ানব্বই", 7: "সাতানব্বই", 8: "আটানব্বই", 9: "নিরানব্বই"},
    }
    return compound.get(tens, {}).get(ones, f"{_TENS.get(tens, str(tens * 10))} {_ONES[ones]}")


def number_to_bn_words(number: int) -> str:
    if number < 0:
        return f"ঋণাত্মক {number_to_bn_words(abs(number))}"
    if number < 10:
        return _ONES[number]
    if number < 100:
        return _two_digit_to_words(number)
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        prefix = _HUNDREDS.get(hundreds, f"{number // 100}শ")
        if remainder == 0:
            return prefix
        return f"{prefix} {_two_digit_to_words(remainder)}"

    scales = [
        (10**7, "কোটি"),
        (10**5, "লক্ষ"),
        (10**3, "হাজার"),
    ]
    for scale_value, scale_name in scales:
        if number >= scale_value:
            left, remainder = divmod(number, scale_value)
            if remainder == 0:
                return f"{number_to_bn_words(left)} {scale_name}"
            return f"{number_to_bn_words(left)} {scale_name} {number_to_bn_words(remainder)}"
    return str(number)


def expand_numbers(text: str) -> str:
    def replace_decimal(match: re.Match[str]) -> str:
        value = match.group(0)
        integer_part = int(Decimal(value))
        return number_to_bn_words(integer_part)

    def replace_integer(match: re.Match[str]) -> str:
        return number_to_bn_words(int(match.group(0)))

    text = re.sub(r"\b\d+\.\d+\b", replace_decimal, text)
    text = re.sub(r"\b\d+\b", replace_integer, text)
    return text


def expand_currency(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        amount = int(match.group(1).translate(_BN_DIGITS))
        return f"{number_to_bn_words(amount)} টাকা"

    return re.sub(r"(?:৳|Tk\.?|টাকা\s*)([০-৯0-9]+)", repl, text)


def expand_dates(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        day = int(match.group(1).translate(_BN_DIGITS))
        month = int(match.group(2).translate(_BN_DIGITS))
        year = int(match.group(3).translate(_BN_DIGITS))
        return f"{number_to_bn_words(day)} {_MONTHS.get(month, str(month))} {number_to_bn_words(year)}"

    return re.sub(r"\b([০-৯0-9]{1,2})/([০-৯0-9]{1,2})/([০-৯0-9]{2,4})\b", repl, text)


def expand_abbreviations(text: str) -> str:
    for source, target in _ABBREVIATIONS.items():
        text = text.replace(source, target)
    return text


def normalize_bangla_text(text: str, config: NormalizationConfig | None = None) -> str:
    config = config or NormalizationConfig()
    normalized = text
    if config.normalize_unicode:
        normalized = normalize_unicode(normalized)
    if config.normalize_punctuation:
        normalized = normalize_punctuation(normalized)
    if config.expand_abbreviations:
        normalized = expand_abbreviations(normalized)
    if config.normalize_dates:
        normalized = expand_dates(normalized)
    if config.normalize_currency:
        normalized = expand_currency(normalized)
    if config.normalize_numbers:
        normalized = normalized.translate(_BN_DIGITS)
        normalized = expand_numbers(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Normalize Bangla text")
    parser.add_argument("text", nargs="*", help="Text to normalize")
    args = parser.parse_args()
    input_text = " ".join(args.text) if args.text else sys.stdin.read()
    print(normalize_bangla_text(input_text))

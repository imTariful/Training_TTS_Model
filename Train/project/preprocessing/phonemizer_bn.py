from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize_bn import normalize_bangla_text

_BN_LETTER_MAP = {
    "অ": "a",
    "আ": "aa",
    "ই": "i",
    "ঈ": "ii",
    "উ": "u",
    "ঊ": "uu",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}

_ARABIC_LOAN_MAP = {
    "আল্লাহ": "allah",
    "ইনশাআল্লাহ": "inshallah",
    "মাশাআল্লাহ": "mashallah",
}


@dataclass(slots=True)
class PhonemizerConfig:
    preserve_unknown_tokens: bool = True
    use_normalization: bool = True


class BanglaPhonemizer:
    def __init__(self, config: PhonemizerConfig | None = None) -> None:
        self.config = config or PhonemizerConfig()

    def phonemize(self, text: str) -> str:
        source = normalize_bangla_text(text) if self.config.use_normalization else text
        tokens = re.findall(r"[\w\u0980-\u09FF]+|[^\w\s]", source, flags=re.UNICODE)
        phonemes = [self._token_to_phonemes(token) for token in tokens if token.strip()]
        return " ".join(part for part in phonemes if part)

    def _token_to_phonemes(self, token: str) -> str:
        if token in _ARABIC_LOAN_MAP:
            return _ARABIC_LOAN_MAP[token]
        if re.fullmatch(r"[A-Za-z0-9]+", token):
            return " ".join(list(token.lower()))
        if re.fullmatch(r"[\u0980-\u09FF]+", token):
            return self._bangla_to_phonemes(token)
        return token

    def _bangla_to_phonemes(self, token: str) -> str:
        return " ".join(_BN_LETTER_MAP.get(char, char) for char in token)


def phonemize_bangla(text: str, use_normalization: bool = True) -> str:
    return BanglaPhonemizer(PhonemizerConfig(use_normalization=use_normalization)).phonemize(text)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Phonemize Bangla text")
    parser.add_argument("text", nargs="*", help="Text to phonemize")
    args = parser.parse_args()
    input_text = " ".join(args.text) if args.text else sys.stdin.read()
    print(phonemize_bangla(input_text))

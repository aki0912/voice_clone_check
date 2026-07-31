from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from pykakasi import kakasi


_NON_READING = re.compile(r"[\s、。！？!?「」『』（）()\[\]【】・,.:;…—ー\-_'\"`]+")


@lru_cache(maxsize=1)
def _converter():
    return kakasi()


def normalize_japanese_reading(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    converted = _converter().convert(normalized)
    reading = "".join(item.get("hira") or item["orig"] for item in converted)
    return _NON_READING.sub("", reading)


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_japanese_reading(reference)
    hyp = normalize_japanese_reading(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


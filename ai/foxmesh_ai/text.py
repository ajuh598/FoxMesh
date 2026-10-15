"""Text tokenizer for prompts in Korean and English, with no vocabulary file to ship.

Each word yields itself plus character n-grams (with word boundaries), hashed
into a fixed number of buckets. Character n-grams let inflected Korean words
share features with their stems ("검을" shares "<검" with "검"), and unseen words
still map to useful buckets.
"""

from __future__ import annotations

import re
import unicodedata
import zlib

VOCAB_SIZE = 1 << 15
PAD_ID = 0
MAX_TOKENS = 96
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().strip()


def _bucket(token: str) -> int:
    return 1 + zlib.crc32(token.encode("utf-8")) % (VOCAB_SIZE - 1)


def tokenize(text: str, max_tokens: int = MAX_TOKENS) -> list[int]:
    ids: list[int] = []
    for word in _WORD.findall(normalize(text)):
        ids.append(_bucket("w:" + word))
        padded = f"<{word}>"
        for n in (2, 3):
            ids.extend(_bucket(f"{n}:{padded[i:i + n]}") for i in range(len(padded) - n + 1))
    return ids[:max_tokens]


def batch_tokenize(texts: list[str], max_tokens: int = MAX_TOKENS) -> tuple[list[list[int]], int]:
    """Tokenize and right-pad with PAD_ID; returns (rows, length)."""
    rows = [tokenize(t, max_tokens) or [PAD_ID] for t in texts]
    length = max(len(r) for r in rows)
    return [r + [PAD_ID] * (length - len(r)) for r in rows], length

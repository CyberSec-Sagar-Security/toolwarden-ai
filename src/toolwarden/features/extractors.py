"""Engineered features for the Phase 4 LightGBM layer. Pure functions over
text — source-agnostic, no dependency on which dataset a string came from.

Each extractor targets a specific injection tell named in the build spec:
imperative phrasing in non-command content, invisible/zero-width Unicode,
base64 blobs, mid-document language/topic shift, known jailbreak
signatures. None of these alone should be trusted as a verdict — they're
inputs to the LightGBM ensemble trained in Phase 4, not a classifier here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_IMPERATIVE_PATTERNS = [
    r"\bignore (all|the|any)? ?(previous|prior|above)\b",
    r"\bdisregard (all|the|any)? ?(previous|prior|above)\b",
    r"\bforget (all|the|any)? ?(previous|prior|above)\b",
    r"\bnew instructions?:",
    r"\bsystem prompt:",
    r"\byou (are|must|should) now\b",
    r"\bact as\b",
    r"\bdo not (follow|comply with)\b",
    r"\boverride (the|any|all)?\b",
    r"\bstop (following|obeying)\b",
]
_IMPERATIVE_RE = re.compile("|".join(_IMPERATIVE_PATTERNS), re.IGNORECASE)

_ZERO_WIDTH_CHARS = "​‌‍⁠﻿᠎"
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")

# Conservative floor (40 chars) to avoid flagging short incidental
# alphanumeric runs (e.g. hex ids) as base64.
_BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")

_JAILBREAK_SIGNATURES = [
    "do anything now",
    "dan mode",
    "developer mode",
    "jailbreak",
    "no restrictions",
    "no longer bound by",
    "ignore your programming",
    "pretend you are an ai without",
    "hypothetically, if you had no rules",
    "you have been freed",
]
_JAILBREAK_RE = re.compile("|".join(re.escape(s) for s in _JAILBREAK_SIGNATURES), re.IGNORECASE)


def imperative_phrasing_score(text: str) -> float:
    """Count of imperative/override-style phrases per 100 characters."""
    if not text:
        return 0.0
    hits = len(_IMPERATIVE_RE.findall(text))
    return hits / max(len(text), 1) * 100


def zero_width_char_count(text: str) -> int:
    return len(_ZERO_WIDTH_RE.findall(text))


def base64_blob_count(text: str) -> int:
    return len(_BASE64_RE.findall(text))


def base64_blob_max_length(text: str) -> int:
    matches = _BASE64_RE.findall(text)
    return max((len(m) for m in matches), default=0)


def script_mix_count(text: str) -> int:
    """Number of distinct Unicode scripts (via character name prefix) used
    in the text, ignoring common/whitespace/punctuation. A jump from 1 to 2+
    scripts mid-document is a cheap, dependency-free proxy for a language
    shift smuggled into otherwise-Latin content.
    """
    scripts: set[str] = set()
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        scripts.add(name.split(" ")[0])
    return len(scripts)


def topic_shift_score(text: str) -> float:
    """Jaccard distance between the word sets of the first and second half
    of the text. High distance = vocabulary changes abruptly partway
    through, one cheap signal for injected content spliced into otherwise
    unrelated document text.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return 0.0
    mid = len(words) // 2
    first, second = set(words[:mid]), set(words[mid:])
    union = first | second
    if not union:
        return 0.0
    jaccard_similarity = len(first & second) / len(union)
    return 1.0 - jaccard_similarity


def jailbreak_signature_count(text: str) -> int:
    return len(_JAILBREAK_RE.findall(text))


def extract_all(text: str) -> dict[str, Any]:
    return {
        "imperative_phrasing_score": imperative_phrasing_score(text),
        "zero_width_char_count": zero_width_char_count(text),
        "base64_blob_count": base64_blob_count(text),
        "base64_blob_max_length": base64_blob_max_length(text),
        "script_mix_count": script_mix_count(text),
        "topic_shift_score": topic_shift_score(text),
        "jailbreak_signature_count": jailbreak_signature_count(text),
        "text_length": len(text),
    }

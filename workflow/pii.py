"""PII scrubbing for outbound LLM calls.

GLBA §501 prohibits passing nonpublic personal information to a third-party
service without contractual safeguards. This module guarantees the LLM only
sees anonymised feature vectors.

Two modes:
  * ``MODE_FEATURE_ONLY`` (default) — strip every field on the PII denylist
    *and* every value that pattern-matches a PII signature, before prompt
    construction.
  * ``MODE_REDACTION`` — for free-text fields, replace PII spans with tokens
    (``[NAME_1]``, ``[SSN_REDACTED]``, ``[ADDRESS_1]``) and remember the
    mapping so re-insertion is possible after the LLM responds.

Public API:
    scrubber = PIIScrubber()
    cleaned, scrubbed_field_names = scrubber.scrub(applicant_dict)
    redacted_text, mapping = scrubber.redact_text(some_free_text)
    restored = scrubber.restore_text(redacted_text, mapping)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Pattern, Tuple

# Field-name denylist: any key that contains one of these tokens (case-
# insensitive, substring match) is dropped in feature-only mode.
PII_FIELD_TOKENS: Tuple[str, ...] = (
    "name", "ssn", "social_security",
    "phone", "email", "address",
    "dob", "birth", "birthdate",
    "account_number", "account_no", "acct",
    "tax_id", "tin", "ein",
    "passport", "license", "drivers_license",
    "ip_address",
)

# Field-name allowlist: even if the key contains a denylist token, keep it if
# it matches one of these — they are model features, not PII.
PII_FIELD_ALLOWLIST: Tuple[str, ...] = (
    "AGE",  # an integer feature, not a date of birth
)

SSN_RE: Pattern[str] = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
PHONE_RE: Pattern[str] = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
EMAIL_RE: Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# Loose US street-address heuristic: number, words, suffix.
ADDRESS_RE: Pattern[str] = re.compile(
    r"\b\d{1,6}\s+([A-Z][A-Za-z]+\s){1,4}"
    r"(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court)\b"
)
# DOB as a standalone date (YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY).
DOB_RE: Pattern[str] = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)


MODE_FEATURE_ONLY = "feature_only"
MODE_REDACTION = "redaction"


@dataclass
class RedactionMap:
    """Reversible mapping from token to original PII span.

    Lives only in process memory — never persisted. The audit log stores
    *which fields* were scrubbed, not their values.
    """

    forward: Dict[str, str] = field(default_factory=dict)  # token -> original
    counts: Dict[str, int] = field(default_factory=dict)   # tag -> next index

    def assign(self, tag: str, original: str) -> str:
        """Return a stable token for ``original`` under ``tag``."""
        for token, value in self.forward.items():
            if value == original and token.startswith(f"[{tag}_"):
                return token
        idx = self.counts.get(tag, 0) + 1
        self.counts[tag] = idx
        token = f"[{tag}_{idx}]"
        self.forward[token] = original
        return token


def _field_is_pii(name: str) -> bool:
    """Return True if a feature key looks like PII."""
    if name in PII_FIELD_ALLOWLIST:
        return False
    lower = name.lower()
    return any(tok in lower for tok in PII_FIELD_TOKENS)


def _value_contains_pii(value: Any) -> bool:
    """Return True if a string value pattern-matches PII."""
    if not isinstance(value, str):
        return False
    return bool(
        SSN_RE.search(value)
        or PHONE_RE.search(value)
        or EMAIL_RE.search(value)
        or ADDRESS_RE.search(value)
    )


class PIIScrubber:
    """Remove or redact PII before LLM calls.

    Args:
        mode: ``MODE_FEATURE_ONLY`` (default) or ``MODE_REDACTION``.
        extra_field_tokens: Add domain-specific PII keys (e.g. internal IDs).
        extra_allowlist: Keep these keys even if they look like PII.
    """

    def __init__(
        self,
        mode: str = MODE_FEATURE_ONLY,
        extra_field_tokens: Tuple[str, ...] = (),
        extra_allowlist: Tuple[str, ...] = (),
    ) -> None:
        if mode not in (MODE_FEATURE_ONLY, MODE_REDACTION):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.field_tokens = PII_FIELD_TOKENS + tuple(t.lower() for t in extra_field_tokens)
        self.field_allowlist = PII_FIELD_ALLOWLIST + tuple(extra_allowlist)

    def _is_pii_field(self, name: str) -> bool:
        if name in self.field_allowlist:
            return False
        lower = name.lower()
        return any(tok in lower for tok in self.field_tokens)

    def scrub(self, applicant: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Drop PII fields from an applicant dict.

        Returns:
            (cleaned_dict, list_of_field_names_that_were_scrubbed)
        """
        cleaned: Dict[str, Any] = {}
        scrubbed: List[str] = []
        for k, v in applicant.items():
            if self._is_pii_field(k):
                scrubbed.append(k)
                continue
            if _value_contains_pii(v):
                scrubbed.append(k)
                continue
            cleaned[k] = v
        return cleaned, scrubbed

    def redact_text(self, text: str) -> Tuple[str, RedactionMap]:
        """Replace PII spans in free text with reversible tokens."""
        if self.mode != MODE_REDACTION:
            raise RuntimeError("redact_text requires MODE_REDACTION")
        mapping = RedactionMap()

        def _sub(pattern: Pattern[str], tag: str, body: str) -> str:
            return pattern.sub(lambda m: mapping.assign(tag, m.group(0)), body)

        out = text
        out = _sub(SSN_RE, "SSN_REDACTED", out)
        out = _sub(EMAIL_RE, "EMAIL", out)
        out = _sub(PHONE_RE, "PHONE", out)
        out = _sub(ADDRESS_RE, "ADDRESS", out)
        out = _sub(DOB_RE, "DOB", out)
        return out, mapping

    def restore_text(self, text: str, mapping: RedactionMap) -> str:
        """Re-insert PII into redacted text using the saved mapping."""
        out = text
        for token, original in mapping.forward.items():
            out = out.replace(token, original)
        return out

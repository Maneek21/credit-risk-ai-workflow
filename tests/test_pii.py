"""Tests for workflow.pii."""
from __future__ import annotations

import pytest

from workflow.pii import (
    MODE_REDACTION,
    PIIScrubber,
)


def test_drops_obvious_pii_fields() -> None:
    s = PIIScrubber()
    cleaned, scrubbed = s.scrub({
        "name": "Jane Doe",
        "ssn": "123-45-6789",
        "phone": "555-867-5309",
        "email": "j@example.com",
        "address": "742 Evergreen Ter",
        "LIMIT_BAL": 200000,
        "PAY_0": -1,
    })
    assert "name" not in cleaned
    assert "ssn" not in cleaned
    assert "phone" not in cleaned
    assert "email" not in cleaned
    assert "address" not in cleaned
    assert cleaned["LIMIT_BAL"] == 200000
    assert cleaned["PAY_0"] == -1
    assert set(scrubbed) >= {"name", "ssn", "phone", "email", "address"}


def test_age_is_kept_despite_birth_substring_pattern() -> None:
    """AGE is a numeric feature — must survive the scrub even though the
    denylist uses substring matching."""
    s = PIIScrubber()
    cleaned, _ = s.scrub({"AGE": 35, "LIMIT_BAL": 100000})
    assert cleaned["AGE"] == 35


def test_drops_field_whose_value_pattern_matches_ssn() -> None:
    s = PIIScrubber()
    cleaned, scrubbed = s.scrub({
        "internal_id": "AGENT-7",
        "comment": "borrower SSN 999-12-3456 verified",
    })
    assert "comment" not in cleaned, "SSN-bearing free text must be dropped"
    assert "comment" in scrubbed
    assert cleaned["internal_id"] == "AGENT-7"


def test_handles_phone_with_various_formats() -> None:
    s = PIIScrubber()
    for phone in ("555-867-5309", "(555) 867-5309", "555.867.5309", "+1 555 867 5309"):
        cleaned, _ = s.scrub({"note": f"call back at {phone}"})
        assert "note" not in cleaned, f"missed format: {phone}"


def test_redact_then_restore_roundtrips_text() -> None:
    s = PIIScrubber(mode=MODE_REDACTION)
    text = "Contact Jane at jane@x.com or 555-867-5309. SSN 123-45-6789."
    redacted, mapping = s.redact_text(text)
    assert "jane@x.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "555-867-5309" not in redacted
    assert "[SSN_REDACTED_1]" in redacted or "[EMAIL_1]" in redacted
    restored = s.restore_text(redacted, mapping)
    assert "jane@x.com" in restored
    assert "123-45-6789" in restored


def test_redact_text_requires_redaction_mode() -> None:
    with pytest.raises(RuntimeError):
        PIIScrubber().redact_text("123-45-6789")


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        PIIScrubber(mode="bogus")


def test_extra_field_tokens_extend_denylist() -> None:
    s = PIIScrubber(extra_field_tokens=("internal_customer_id",))
    cleaned, scrubbed = s.scrub({"internal_customer_id": 42, "PAY_0": 0})
    assert "internal_customer_id" not in cleaned
    assert "internal_customer_id" in scrubbed


def test_extra_allowlist_keeps_field() -> None:
    s = PIIScrubber(extra_allowlist=("CUSTOMER_AGE",))
    cleaned, _ = s.scrub({"CUSTOMER_AGE": 30})
    assert cleaned["CUSTOMER_AGE"] == 30


def test_feature_dict_with_only_model_features_passes_through() -> None:
    """The 23-feature UCI applicant dict must round-trip with no losses."""
    applicant = {
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 45000,
        "BILL_AMT4": 43000, "BILL_AMT5": 40000, "BILL_AMT6": 38000,
        "PAY_AMT1": 5000, "PAY_AMT2": 5000, "PAY_AMT3": 4500,
        "PAY_AMT4": 4500, "PAY_AMT5": 4000, "PAY_AMT6": 4000,
    }
    cleaned, scrubbed = PIIScrubber().scrub(applicant)
    assert cleaned == applicant
    assert scrubbed == []

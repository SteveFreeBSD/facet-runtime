from __future__ import annotations

import pytest

from facet_runtime.adapters.image_contract import parse_transcription
from facet_runtime.errors import FacetRuntimeError


def test_parse_transcription_accepts_fenced_runtime_json() -> None:
    transcription, uncertainties = parse_transcription(
        '```json\n{"transcription":"x^2 + sqrt(17)","uncertainties":[]}\n```'
    )
    assert transcription == "x^2 + sqrt(17)"
    assert uncertainties == ()


def test_parse_transcription_preserves_uncertainties() -> None:
    transcription, uncertainties = parse_transcription(
        '{"transcription":"code: B7","uncertainties":["B might be 8"]}'
    )
    assert transcription == "code: B7"
    assert uncertainties == ("B might be 8",)


def test_parse_transcription_rejects_unstructured_output() -> None:
    with pytest.raises(FacetRuntimeError, match="invalid transcription JSON"):
        parse_transcription("plain prose")

"""Shared normalization at the image-adapter boundary."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from facet_runtime.errors import FacetRuntimeError

TRANSCRIPTION_PROMPT = (
    "Transcribe every line of visible text exactly. Preserve capitalization, "
    "punctuation, the exponent, fraction slash, and radical. Do not solve or "
    "explain. Return uncertainties only for characters you cannot read."
)

TRANSCRIPTION_JSON_PROMPT = (
    f"{TRANSCRIPTION_PROMPT} Return ONLY JSON with keys transcription (string) "
    "and uncertainties (array of strings)."
)

TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "transcription": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["transcription", "uncertainties"],
}


def encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_types.get(path.suffix.lower())
    if media_type is None:
        raise FacetRuntimeError("image must be a PNG or JPEG file")
    return base64.b64encode(path.read_bytes()).decode("ascii"), media_type


def parse_transcription(raw: str) -> tuple[str, tuple[str, ...]]:
    text = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise FacetRuntimeError(
            "vision runtime returned invalid transcription JSON"
        ) from error
    transcription = payload.get("transcription")
    uncertainties = payload.get("uncertainties")
    if not isinstance(transcription, str):
        raise FacetRuntimeError("vision runtime returned no transcription string")
    if not isinstance(uncertainties, list) or not all(
        isinstance(item, str) for item in uncertainties
    ):
        raise FacetRuntimeError("vision runtime returned invalid uncertainties")
    return transcription.strip(), tuple(uncertainties)

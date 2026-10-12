"""Image-only extraction: descriptors and visible text, without claim context."""
from __future__ import annotations

import json

from .text_extractor import parse_json

IMAGE_PROMPT = (
    'Describe only visible content. Return ONLY a JSON object with this exact structure: '
    '{"objects": [], "persons": [], "landmarks": [], "ocr": []}. '
    'All four values must be arrays of strings; use an empty array when nothing is found. '
    'Use double-quoted JSON strings and escape quotes inside them. '
    'Return compact JSON on one line, with no markdown, comments, or explanation. '
    'List each distinct description only once, even when several similar objects or people are visible. '
    'Close all arrays and the JSON object as soon as the distinct items are listed. '
    'Use objects for non-person physical items; put written words and map labels in ocr. '
    'Persons must use visible descriptions such as man in a suit, never guessed names or identities. '
    'Landmarks are recognizable physical places, not furniture, podiums, or generic backgrounds; '
    'omit uncertain ones. OCR must contain each distinct visible line only once, including text '
    'on maps and money; preserve readable characters and omit unreadable text instead of guessing. '
    'Do not infer dates or locations not shown, assess truthfulness, or produce a verdict. '
    'Example for a scene with a bicycle, a person in a red coat, and a STOP sign: '
    '{"objects": ["bicycle", "road sign"], "persons": ["person in a red coat"], '
    '"landmarks": [], "ocr": ["STOP"]}. '
    'Extract the supplied image only; do not copy the example.'
)


class ImageExtractor:
    def __init__(self, backend):
        self.backend = backend

    def extract(self, image_path):
        retry_note = ""
        for attempt in range(2):
            prompt = retry_note + IMAGE_PROMPT
            response = ""
            try:
                response = self.backend.generate_image(image_path, prompt)
                payload = parse_json(response)
                result = {}
                for key in ("objects", "persons", "landmarks", "ocr"):
                    values = payload.get(key)
                    if not isinstance(values, list):
                        raise ValueError("Invalid " + key + " array")
                    cleaned_values = []
                    for value in values:
                        if isinstance(value, dict):
                            field = "text" if key == "ocr" else "description"
                            value = value.get(field)
                        if not isinstance(value, str):
                            raise ValueError("Invalid " + key + " item: expected a string")
                        value = value.strip()
                        if value and value not in cleaned_values:
                            cleaned_values.append(value)
                    result[key] = cleaned_values
                result["errors"] = []
                result["retries"] = attempt
                return result
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                retry_note = (
                    f"Your previous response was invalid: {exc}. "
                    "Return a compact, complete JSON object matching the schema below. "
                    "List each distinct item once and close all arrays.\n"
                )
                if attempt:
                    return {
                        "objects": [],
                        "persons": [],
                        "landmarks": [],
                        "ocr": [],
                        "errors": [{"error": str(exc), "response": response}],
                        "retries": 1,
                    }

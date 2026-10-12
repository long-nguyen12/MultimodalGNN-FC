"""Image-only extraction: descriptors and visible text, without claim context."""
from __future__ import annotations

import json

from .text_extractor import parse_json

IMAGE_PROMPT = (
    'Describe only visible content. Return ONLY a JSON object with this exact structure: '
    '{"objects": [], "persons": [], "landmarks": [], "ocr": []}. '
    'All four values must be arrays of strings; use an empty array when nothing is found. '
    'Use double-quoted JSON strings and escape quotes inside them. '
    'Return each string once, with no markdown, comments, or explanation. '
    'Persons must use visible descriptions such as man in a suit, never guessed names or identities. '
    'Landmarks must be identifiable from the image; omit uncertain ones. OCR must be exact visible text. '
    'Do not infer dates or locations not shown, assess truthfulness, or produce a verdict.'
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
                        if not isinstance(value, str):
                            raise ValueError("Invalid " + key + " array")
                        value = value.strip()
                        if value and value not in cleaned_values:
                            cleaned_values.append(value)
                    result[key] = cleaned_values
                result["errors"] = []
                return result
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                retry_note = (
                    f"Your previous response was invalid: {exc}. "
                    "Return a complete JSON object matching the schema below.\n"
                )
                if attempt:
                    return {
                        "objects": [],
                        "persons": [],
                        "landmarks": [],
                        "ocr": [],
                        "errors": [{"error": str(exc), "response": response}],
                    }

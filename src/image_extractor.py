"""Image-only extraction: descriptors and visible text, without claim context."""
from __future__ import annotations

import json

from .text_extractor import parse_json

IMAGE_PROMPT = (
    'Extract only visible content from the supplied image. Return one compact JSON object '
    'with exactly these four arrays of strings: '
    '{"objects": [], "persons": [], "landmarks": [], "ocr": []}.\n'
    'Use these definitions:\n'
    '- objects: non-person physical items, such as furniture, vehicles, clothing, signs, '
    'animals, and ordinary buildings. Use short descriptions of what is visible.\n'
    '- persons: visible people described by clothing, posture, or appearance. '
    'Put people only here, never in objects. Do not guess names or identities.\n'
    '- landmarks: unmistakably recognizable named monuments, buildings, or natural features '
    'actually visible in the image. Generic rooms, furniture, podiums, streets, '
    'and inferred cities are not landmarks. If unsure, use []. '
    'An identified landmark belongs here, not in objects. '
    'A printed place name alone belongs to ocr and does not identify a visible landmark.\n'
    '- ocr: readable written words, including captions, signs, map labels, and text on money. '
    'Copy each distinct readable line once, preserving characters, case, accents, and punctuation. '
    'Do not paraphrase or guess unreadable words.\n'
    'Give each physical item or person one category. A physical sign and its written text '
    'may appear separately in objects and ocr. Use [] when a category has no valid items; '
    'do not fill it just to make it nonempty. List each distinct description only once.\n'
    'Example scene: a person in a blue apron beside a chair and a sign reading OPEN.\n'
    'Output: {"objects": ["chair", "sign"], "persons": ["person in a blue apron"], '
    '"landmarks": [], "ocr": ["OPEN"]}\n'
    'Example scene: an unmistakable view of the Eiffel Tower, with no people or readable text.\n'
    'Output: {"objects": [], "persons": [], "landmarks": ["Eiffel Tower"], "ocr": []}\n'
    'Extract the supplied image, not the examples. Use double-quoted strings and escape embedded quotes. '
    'Close all arrays and the JSON object, then stop. '
    'Return no markdown, comments, reasoning, dates or locations not shown, or truthfulness verdict.'
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

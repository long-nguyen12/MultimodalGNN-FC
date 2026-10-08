"""Qwen inference and strictly source-anchored text entity extraction."""

from __future__ import annotations

import json
import re

from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

TEXT_PROMPT = (
    "Extract literal named entities from the text. Return ONLY JSON with an entities array. "
    "Each item has text (an exact substring) and type: PERSON, LOCATION, DATE, or ORGANIZATION. "
    "Do not infer missing facts, use outside knowledge, assess truthfulness, or return a verdict. "
    "Do not resolve relative dates. Return an empty entities array when there are no entities.\nText:\n"
)


def parse_json(response):
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.S).strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
    payload = json.loads(response)
    return payload


class QwenBackend:
    def __init__(
        self,
        model_id="Qwen/Qwen3.5-4B",
        device="cuda",
        cache_dir=None,
        max_new_tokens=1536,
        max_image_side=1024,
        seed=42,
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.seed = seed
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float32
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = (
            AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)
            .to(self.device)
            .eval()
        )

    @property
    def tokenizer(self):
        self.load()
        return self.processor.tokenizer

    def generate(self, messages):
        self.load()
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            result = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = result[0, input_length:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def generate_text(self, prompt):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.generate(messages)

    def generate_image(self, image_path, prompt):
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((self.max_image_side, self.max_image_side))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.generate(messages)

    def unload(self):
        self.model = None
        self.processor = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TextExtractor:
    def __init__(self, backend, window=2048, overlap=128):
        self.backend = backend
        self.window = window
        self.overlap = overlap

    def extract(self, text):
        if not text.strip():
            return {"entities": [], "errors": []}
        tokenizer = self.backend.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, len(tokens), self.window - self.overlap):
            chunk_tokens = tokens[start : start + self.window]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

        entities = []
        errors = []
        seen = set()
        aliases = {"GPE": "LOCATION", "LOC": "LOCATION", "ORG": "ORGANIZATION"}
        allowed_types = ("PERSON", "LOCATION", "DATE", "ORGANIZATION")
        for chunk_index, chunk in enumerate(chunks):
            for attempt in range(2):
                prompt = TEXT_PROMPT + chunk
                if attempt:
                    prompt += "\nReturn a valid JSON object only."
                try:
                    response = self.backend.generate_text(prompt)
                    payload = parse_json(response)
                    if not isinstance(payload.get("entities"), list):
                        raise ValueError("Missing entities array")
                    # Commit entities only after the whole response is valid.
                    pending = []
                    pending_keys = set()
                    for item in payload["entities"]:
                        literal = item.get("text", "")
                        raw_type = str(item.get("type", "")).upper()
                        entity_type = aliases.get(raw_type, raw_type)
                        if not isinstance(literal, str) or not literal.strip():
                            continue
                        if entity_type not in allowed_types:
                            continue
                        if literal not in chunk:
                            continue
                        spans = []
                        for match in re.finditer(re.escape(literal), text):
                            spans.append({"start": match.start(), "end": match.end()})
                        key = (literal, entity_type)
                        if spans and key not in seen and key not in pending_keys:
                            pending.append(
                                {"text": literal, "type": entity_type, "spans": spans}
                            )
                            pending_keys.add(key)
                    entities.extend(pending)
                    seen.update(pending_keys)
                    break
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    if attempt:
                        errors.append({"chunk": chunk_index, "error": str(exc)})
        return {"entities": entities, "errors": errors}

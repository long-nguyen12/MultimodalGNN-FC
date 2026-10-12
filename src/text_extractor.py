"""Qwen inference and strictly source-anchored text entity extraction."""

from __future__ import annotations

import json
import re
import time

from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

TEXT_PROMPT = (
    'Extract every explicit named entity from Text, including short sentences. '
    'This is name tagging; do not assess whether the statement is true.\n'
    'Use only these types:\n'
    '- PERSON: names of individual people; omit job titles from the name.\n'
    '- LOCATION: named towns, cities, countries, regions, or other places.\n'
    '- DATE: dates or years explicitly written; do not resolve relative dates.\n'
    '- ORGANIZATION: named companies, stores, agencies, institutions, and other organizations. '
    'Include company names used for websites, platforms, or services, even without the word company.\n'
    'Recognize the names written in Text and copy each one exactly, preserving spelling and case. '
    'Include names mentioned only once, in negated statements, or in disputed claims. '
    'Do not add absent names, expand abbreviations, or infer identities or dates. '
    'Generic occupations, unnamed people or places, money amounts, diseases, viruses, '
    'and standalone product/model names do not belong to these types.\n'
    'Return one complete JSON object: {"entities": [{"text": "...", "type": "..."}]}. '
    'Use each text/type pair once, with double-quoted strings and escaped embedded quotes. '
    'Return no spans, markdown, comments, reasoning, or verdict.\n'
    'Example Text: Asha Rao visited Harbor Foods in Riverton in 2024.\n'
    'Output: {"entities": [{"text": "Asha Rao", "type": "PERSON"}, '
    '{"text": "Harbor Foods", "type": "ORGANIZATION"}, '
    '{"text": "Riverton", "type": "LOCATION"}, {"text": "2024", "type": "DATE"}]}\n'
    'Example Text: Loopline users reported a problem with Pinecrest Analytics.\n'
    'Output: {"entities": [{"text": "Loopline", "type": "ORGANIZATION"}, '
    '{"text": "Pinecrest Analytics", "type": "ORGANIZATION"}]}\n'
    'Example Text: The package is red and costs $20.\n'
    'Output: {"entities": []}\n'
    'Before returning an empty array, check Text again for all four types. '
    'Return {"entities": []} only when none are present. Extract the Text below, not the examples.\n'
    'Text:\n'
)


def parse_json(response):
    response = response.strip()
    # A thinking block can start in the prompt, leaving only its closing tag here.
    if not response.startswith(("{", "[", "```")) and "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    if response.startswith("<think>"):
        raise ValueError("Incomplete Qwen thinking output")
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


class QwenBackend:
    def __init__(
        self,
        model_id="Qwen/Qwen3.5-4B",
        device="cuda",
        max_new_tokens=1536,
        max_image_side=1024,
        seed=42,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.seed = seed
        self.model = None
        self.processor = None
        self.generation_parameters = {
            "text": {"do_sample": False},
            "image": {
                "do_sample": True, "temperature": 0.7, "top_p": 0.8,
                "top_k": 20, "repetition_penalty": 1.1,
            },
        }
        self.generation_stats = {
            kind: {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "seconds": 0.0, "at_limit": 0,
            }
            for kind in ("text", "image")
        }

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
        print(f"[qwen] loaded {self.model_id}, device={self.device}, dtype={dtype}", flush=True)
        print(
            f"[qwen] thinking=False, seed={self.seed}, max_new_tokens={self.max_new_tokens}, "
            f"generation={json.dumps(self.generation_parameters)}",
            flush=True,
        )

    @property
    def tokenizer(self):
        self.load()
        return self.processor.tokenizer

    def generate(self, messages, image=None):
        self.load()
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            chat_template=self.processor.chat_template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=prompt,
            images=image,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        kind = "image" if image is not None else "text"
        generation_kwargs = self.generation_parameters[kind]
        if image is not None:
            torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                **generation_kwargs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_token_id,
            )
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = result[0, input_length:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        count = generated_tokens.numel()
        stats = self.generation_stats[kind]
        stats["calls"] += 1
        stats["input_tokens"] += input_length
        stats["output_tokens"] += count
        stats["seconds"] += elapsed
        stats["at_limit"] += int(count >= self.max_new_tokens)
        print(
            f"[qwen:{kind}] "
            f"input_tokens={input_length}, output_tokens={count}, "
            f"time={elapsed:.2f}s, tokens/s={count / elapsed:.1f}, "
            f"at_limit={count >= self.max_new_tokens}",
            flush=True,
        )
        return response

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
            return self.generate(messages, image=image)

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
            return {"entities": [], "errors": [], "diagnostics": [], "retries": 0}
        tokenizer = self.backend.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, len(tokens), self.window - self.overlap):
            chunk_tokens = tokens[start : start + self.window]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
            if start + self.window >= len(tokens):
                break

        entities = []
        errors = []
        diagnostics = []
        retries = 0
        seen = set()
        aliases = {"GPE": "LOCATION", "LOC": "LOCATION", "ORG": "ORGANIZATION"}
        allowed_types = ("PERSON", "LOCATION", "DATE", "ORGANIZATION")
        for chunk_index, chunk in enumerate(chunks):
            retry_note = ""
            for attempt in range(2):
                retries += attempt
                prompt = retry_note + TEXT_PROMPT + chunk
                response = ""
                try:
                    response = self.backend.generate_text(prompt)
                    payload = parse_json(response)
                    if not isinstance(payload.get("entities"), list):
                        raise ValueError("Missing entities array")
                    # Commit entities only after the whole response is valid.
                    pending = []
                    pending_keys = set()
                    discarded = {
                        "invalid_text": 0, "unsupported_type": 0,
                        "not_in_chunk": 0, "not_in_source": 0, "duplicate": 0,
                    }
                    for item in payload["entities"]:
                        if not isinstance(item, dict):
                            raise ValueError("Invalid entity object")
                        literal = item.get("text", "")
                        raw_type = str(item.get("type", "")).upper()
                        entity_type = aliases.get(raw_type, raw_type)
                        if not isinstance(literal, str) or not literal.strip():
                            discarded["invalid_text"] += 1
                            continue
                        if entity_type not in allowed_types:
                            discarded["unsupported_type"] += 1
                            continue
                        if literal not in chunk:
                            discarded["not_in_chunk"] += 1
                            continue
                        spans = []
                        for match in re.finditer(re.escape(literal), text):
                            spans.append({"start": match.start(), "end": match.end()})
                        key = (literal, entity_type)
                        if not spans:
                            discarded["not_in_source"] += 1
                            continue
                        if key in seen or key in pending_keys:
                            discarded["duplicate"] += 1
                            continue
                        pending.append(
                            {"text": literal, "type": entity_type, "spans": spans}
                        )
                        pending_keys.add(key)
                    if not pending:
                        diagnostics.append({
                            "chunk": chunk_index, "attempt": attempt + 1,
                            "candidates": len(payload["entities"]),
                            "discarded": discarded, "response": response,
                        })
                    entities.extend(pending)
                    seen.update(pending_keys)
                    break
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    retry_note = (
                        f"Your previous response was invalid: {exc}. "
                        "Return a complete JSON object matching the schema below.\n"
                    )
                    if attempt:
                        errors.append(
                            {"chunk": chunk_index, "error": str(exc), "response": response}
                        )
        return {
            "entities": entities, "errors": errors,
            "diagnostics": diagnostics, "retries": retries,
        }

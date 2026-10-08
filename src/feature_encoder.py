"""Frozen CLIP embeddings with complete long-text chunk pooling."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


class FeatureEncoder:
    def __init__(
        self, model_id="openai/clip-vit-base-patch32", device="cuda", cache_dir=None
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _features(result):
        # Transformers 5 returns an output object; older releases return a tensor.
        if hasattr(result, "detach"):
            return result
        return result.pooler_output

    def encode_text(self, text):
        self.load()

        max_tokens = self.model.config.text_config.max_position_embeddings
        inputs = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            stride=11,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        chunk_features = []
        with torch.inference_mode():
            for start in range(0, len(inputs["input_ids"]), 32):
                chunk_inputs = {}
                for key in ("input_ids", "attention_mask"):
                    if key in inputs:
                        chunk_inputs[key] = inputs[key][start : start + 32].to(
                            self.device
                        )
                result = self.model.get_text_features(**chunk_inputs)
                features = self._features(result).float()
                chunk_features.append(F.normalize(features, dim=-1))

        # Every chunk contributes, including the tail of a long evidence text.
        mean_features = torch.cat(chunk_features).mean(dim=0)
        return F.normalize(mean_features, dim=0).cpu().clone()

    def encode_image(self, image_path):
        self.load()
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            result = self.model.get_image_features(**inputs)
            features = self._features(result).float()[0]
        return F.normalize(features, dim=0).cpu().clone()

    def unload(self):
        self.model = None
        self.processor = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

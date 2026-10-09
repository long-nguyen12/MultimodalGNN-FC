"""Named caches with atomic writes and explicit compatibility checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def fingerprint(value) -> str:
    serialized = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ContentCache:
    def __init__(self, root, metadata, tensor=False):
        self.metadata = metadata
        self.fingerprint = fingerprint(metadata)
        self.root = Path(root)
        self.tensor = tensor
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "metadata.json"
        if manifest.exists():
            saved_metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if saved_metadata != metadata:
                raise ValueError(
                    f"Incompatible cache metadata: {manifest}. Choose another --cache-root."
                )
        else:
            atomic_json(manifest, metadata)

    def get_or_compute(self, name, input_key, compute):
        key = fingerprint(input_key)
        suffix = ".pt" if self.tensor else ".json"
        path = self.root / (str(name) + suffix)

        if path.exists():
            if self.tensor:
                import torch
                result = torch.load(path, map_location="cpu", weights_only=True)
            else:
                result = json.loads(path.read_text(encoding="utf-8"))
            if result["key"] != key or result["fingerprint"] != self.fingerprint:
                raise ValueError("Incompatible cache entry: " + str(path))
            return result["value"]

        value = compute()
        result = {"key": key, "fingerprint": self.fingerprint, "value": value}
        if self.tensor:
            atomic_torch(path, result)
        else:
            atomic_json(path, result)
        return value

"""Named caches with atomic writes."""
from __future__ import annotations

import json
import os
from pathlib import Path


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
    def __init__(self, root, tensor=False):
        self.root = Path(root)
        self.tensor = tensor
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_compute(self, name, compute):
        suffix = ".pt" if self.tensor else ".json"
        path = self.root / (str(name) + suffix)

        if path.exists():
            if self.tensor:
                import torch
                result = torch.load(path, map_location="cpu", weights_only=True)
            else:
                result = json.loads(path.read_text(encoding="utf-8"))
            return result["value"]

        value = compute()
        result = {"value": value}
        if self.tensor:
            atomic_torch(path, result)
        else:
            atomic_json(path, result)
        return value

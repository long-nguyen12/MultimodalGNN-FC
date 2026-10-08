"""Record dependency, device and checkpoint readiness before a server build."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
import sys

from .cache import atomic_json
from .config import BuildConfig


def preflight(device="cuda", cache_dir=None, load_clip=False):
    report = {
        "python": platform.python_version(),
        "python_recommended": "3.11",
        "packages": {},
        "device": device,
        "models": {},
        "errors": [],
    }
    packages = (
        "torch",
        "torchvision",
        "torch-geometric",
        "transformers",
        "accelerate",
        "Pillow",
        "scikit-learn",
        "networkx",
    )
    for package in packages:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["errors"].append("Missing dependency: " + package)
    if sys.version_info < (3, 10):
        report["errors"].append("Python >=3.10 required; Python3.11 recommended")
    try:
        import torch

        report["cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                report["errors"].append("CUDA is unavailable")
            else:
                properties = torch.cuda.get_device_properties(device)
                report["gpu"] = {
                    "name": properties.name,
                    "memory_gib": properties.total_memory / 2**30,
                }
                report["server_24gb_profile"] = properties.total_memory >= 22 * 2**30
    except ImportError:
        pass
    try:
        from huggingface_hub import try_to_load_from_cache

        config = BuildConfig()
        models = (("encoder", config.encoder_id), ("extractor", config.extractor_id))
        for key, model_id in models:
            status = {"model_id": model_id, "ready": False}
            try:
                filenames = [
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors.index.json",
                ]
                cached_files = {}
                for filename in filenames:
                    cached_files[filename] = try_to_load_from_cache(
                        model_id, filename, cache_dir=cache_dir
                    )
                weights_ready = any(
                    isinstance(path, str) for path in cached_files.values()
                )
                index_path = cached_files["model.safetensors.index.json"]
                if isinstance(index_path, str):
                    from pathlib import Path

                    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
                    shard_names = set(index["weight_map"].values())
                    weights_ready = True
                    for shard_name in shard_names:
                        shard_path = try_to_load_from_cache(
                            model_id, shard_name, cache_dir=cache_dir
                        )
                        if not isinstance(shard_path, str):
                            weights_ready = False
                            break
                status["ready"] = weights_ready
                if not weights_ready:
                    status["error"] = (
                        "Checkpoint weights are not cached; download on the GPU server before extraction"
                    )
            except (OSError, ValueError) as exc:
                status["error"] = str(exc)
            report["models"][key] = status
    except ImportError as exc:
        report["errors"].append(str(exc))
    encoder_ready = report["models"].get("encoder", {}).get("ready")
    if load_clip and encoder_ready:
        from .feature_encoder import FeatureEncoder

        encoder = FeatureEncoder(device=device, cache_dir=cache_dir)
        vector = encoder.encode_text("A photograph of a tower in Paris in 2017.")
        report["clip_smoke"] = {
            "dimension": vector.numel(),
            "norm": float(vector.norm()),
            "finite": bool(vector.isfinite().all()),
        }
        encoder.unload()
    models_ready = all(
        report["models"].get(key, {}).get("ready", False)
        for key in ("encoder", "extractor")
    )
    device_ready = not device.startswith("cuda") or report.get(
        "server_24gb_profile", False
    )
    report["full_pilot_ready"] = not report["errors"] and models_ready and device_ready
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--load-clip", action="store_true")
    parser.add_argument("--output", default="outputs/preflight.json")
    args = parser.parse_args(argv)
    report = preflight(
        device=args.device,
        cache_dir=args.hf_cache_dir,
        load_clip=args.load_clip,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["full_pilot_ready"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

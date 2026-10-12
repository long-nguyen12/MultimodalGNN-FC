"""Cache extraction, then CLIP features, then reproducible evidence graphs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .cache import ContentCache, atomic_json, atomic_torch, file_hash, fingerprint
from .config import BuildConfig
from .dataset import MochegDataset
from .graph_builder import GraphBuilder


def build_graphs(
    data_root="mocheg",
    cache_root="cache",
    graph_root="graphs",
    config=None,
    splits=("train", "val", "test"),
    stage="full",
    limit=None,
    encoder=None,
    backend=None,
):
    config = config or BuildConfig()
    data_root = Path(data_root)

    # Include every claim in each requested split.
    selected = []
    datasets = {}
    for split in splits:
        if split not in datasets:
            datasets[split] = MochegDataset(data_root, split)
        dataset = datasets[split]
        records = list(zip(dataset.samples, dataset.targets))
        if limit:
            records = records[:limit]
        selected.extend(records)

    from .feature_encoder import FeatureEncoder
    from .text_extractor import QwenBackend, TextExtractor, TEXT_PROMPT
    from .image_extractor import ImageExtractor, IMAGE_PROMPT

    if encoder is None:
        encoder = FeatureEncoder(
            model_id=config.encoder_id,
            device=config.device,
        )
    if stage == "full":
        backend = QwenBackend(
            model_id=config.extractor_id,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            max_image_side=config.max_image_side,
            seed=config.seed,
        )
    # Include every gold image so compatibility is stable across split
    # selection and changes when any underlying pixels change.
    all_image_hashes = {}
    for split in ("train", "val", "test"):
        if split not in datasets:
            datasets[split] = MochegDataset(data_root, split)
        for sample in datasets[split].samples:
            for image in sample.image_evidence:
                if image.relative_path in all_image_hashes:
                    continue
                image_path = data_root / image.relative_path
                image_hash = (
                    file_hash(image_path) if image_path.is_file() else "missing"
                )
                all_image_hashes[image.relative_path] = image_hash
    # Targets and retrieval-only columns do not affect graph compatibility.
    source_hashes = {}
    for split, dataset in sorted(datasets.items()):
        source_hashes[split] = fingerprint(
            [asdict(sample) for sample in dataset.samples]
        )
    code_root = Path(__file__).parent
    extraction_metadata = {
        "model": config.extractor_id,
        "prompts": fingerprint([TEXT_PROMPT, IMAGE_PROMPT]),
        "window": config.text_window,
        "overlap": config.text_overlap,
        "max_new_tokens": config.max_new_tokens,
        "max_image_side": config.max_image_side,
        "seed": config.seed,
        "code": [
            file_hash(code_root / name)
            for name in ("text_extractor.py", "image_extractor.py")
        ],
    }
    feature_metadata = {
        "model": config.encoder_id,
        "chunking": "all-chunks-77-stride11-normalized-mean-v1",
        "code": file_hash(code_root / "feature_encoder.py"),
    }
    graph_metadata = {
        "graph_stage": stage,
        "features": feature_metadata,
        "extraction": extraction_metadata if stage == "full" else None,
        "top_k": config.top_k,
        "sources": source_hashes,
        "image_sources": fingerprint(all_image_hashes),
        "code": [
            file_hash(code_root / name)
            for name in (
                "graph_builder.py",
                "schema.py",
                "dataset.py",
                "build_graphs.py",
            )
        ],
    }
    graph_fingerprint = fingerprint(graph_metadata)
    output = Path(graph_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != graph_fingerprint:
            raise ValueError(
                f"Incompatible graph cache: {manifest_path}. Choose another --graph-root."
            )
    else:
        manifest = {
            "fingerprint": graph_fingerprint,
            "metadata": graph_metadata,
            "records": [],
        }

    # Target labels are never supplied to any extractor or encoder.
    image_hashes_by_claim = {}
    for sample, _ in selected:
        image_hashes = {
            image.image_id: all_image_hashes[image.relative_path]
            for image in sample.image_evidence
        }
        image_hashes_by_claim[(sample.split, sample.claim_id)] = image_hashes
    timings = {
        "extraction_seconds": 0.0,
        "encoding_graph_seconds": 0.0,
        "graphs_reused": 0,
        "graphs_built": 0,
    }
    import torch

    monitor_cuda = config.device.startswith("cuda") and torch.cuda.is_available()
    memory = {}

    def start_memory():
        if monitor_cuda:
            torch.cuda.reset_peak_memory_stats(config.device)

    def finish_memory(phase):
        if monitor_cuda:
            memory[phase] = {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(config.device)
                / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(config.device)
                / 2**30,
            }
        else:
            memory[phase] = None

    # Finish Qwen extraction and release its weights before loading CLIP.
    extraction_cache = None
    if stage == "full":
        extraction_cache = ContentCache(
            Path(cache_root) / "extraction", extraction_metadata
        )
        text_extractor = TextExtractor(backend, config.text_window, config.text_overlap)
        image_extractor = ImageExtractor(backend)
        start_memory()
        started = time.perf_counter()
        for number, (sample, _) in enumerate(selected, 1):
            claim_dir = Path(sample.split) / sample.claim_id
            texts = [("claim", sample.claim_text)] + [
                (f"text_{evidence.evidence_id}", evidence.text)
                for evidence in sample.text_evidence
            ]
            for name, text in texts:
                extraction_cache.get_or_compute(
                    claim_dir / name,
                    {"kind": "text", "text": text},
                    lambda text=text: text_extractor.extract(text),
                )
            for image in sample.image_evidence:
                image_hash = image_hashes_by_claim[(sample.split, sample.claim_id)][
                    image.image_id
                ]
                key = {"kind": "image", "content_hash": image_hash}
                if image_hash == "missing":
                    key["missing_path"] = image.relative_path

                def extract_image(image=image):
                    try:
                        return image_extractor.extract(data_root / image.relative_path)
                    except OSError as exc:
                        return {
                            "objects": [],
                            "persons": [],
                            "landmarks": [],
                            "ocr": [],
                            "errors": [{"error": str(exc)}],
                        }

                extraction_cache.get_or_compute(
                    claim_dir / f"image_{image.image_id}", key, extract_image
                )
            print(
                f"extraction {number}/{len(selected)} {sample.split}/{sample.claim_id}",
                flush=True,
            )
        timings["extraction_seconds"] = time.perf_counter() - started
        finish_memory("extraction")
        backend.unload()

    # Encode the original inputs and extracted concepts, then build each graph.
    feature_cache = ContentCache(
        Path(cache_root) / "features", feature_metadata, tensor=True
    )
    records_by_claim = {}
    for record in manifest["records"]:
        records_by_claim[(record["split"], record["claim_id"])] = record
    builder = GraphBuilder(config.top_k)
    start_memory()
    started = time.perf_counter()

    def encoded_text(name, text):
        return feature_cache.get_or_compute(
            name, {"kind": "text", "text": text}, lambda: encoder.encode_text(text)
        )

    def require_extracted():
        raise ValueError("Extraction cache incomplete; rerun the extraction stage")

    for number, (sample, target) in enumerate(selected, 1):
        claim_key = (sample.split, sample.claim_id)
        claim_dir = Path(sample.split) / sample.claim_id
        image_hashes = image_hashes_by_claim[claim_key]
        input_hash = fingerprint({"sample": asdict(sample), "images": image_hashes})
        record = records_by_claim.get(claim_key)
        if (
            record
            and record.get("input_hash") == input_hash
            and record["label"] == target.label
        ):
            graph_exists = (output / record["path"]).exists()
            if graph_exists and (output / record["provenance_path"]).exists():
                timings["graphs_reused"] += 1
                continue

        cached_features = {
            "claim": encoded_text(claim_dir / "claim", sample.claim_text),
            "texts": {},
            "images": {},
            "text_extractions": {},
            "image_extractions": {},
            "entity_features": {},
            "concept_features": {},
            "errors": [],
        }
        owners = [("claim:0", "claim", sample.claim_text)]
        for index, evidence in enumerate(sample.text_evidence):
            name = f"text_{evidence.evidence_id}"
            cached_features["texts"][evidence.evidence_id] = encoded_text(
                claim_dir / name, evidence.text
            )
            owners.append((f"evidence_text:{index}", name, evidence.text))

        if extraction_cache:
            for owner, name, text in owners:
                extraction = extraction_cache.get_or_compute(
                    claim_dir / name, {"kind": "text", "text": text}, require_extracted
                )
                cached_features["text_extractions"][owner] = extraction
                for error in extraction.get("errors", []):
                    cached_features["errors"].append({"owner": owner, **error})
                for index, entity in enumerate(extraction["entities"]):
                    cached_features["entity_features"][entity["text"]] = encoded_text(
                        claim_dir / f"entity_{name}_{index}", entity["text"]
                    )

        for image in sample.image_evidence:
            try:
                image_features = feature_cache.get_or_compute(
                    claim_dir / f"image_{image.image_id}",
                    {"kind": "image", "content_hash": image_hashes[image.image_id]},
                    lambda image=image: encoder.encode_image(
                        data_root / image.relative_path
                    ),
                )
                cached_features["images"][image.image_id] = image_features
            except OSError as exc:
                cached_features["errors"].append(
                    {
                        "owner": image.image_id,
                        "phase": "image_decode",
                        "error": str(exc),
                    }
                )
                continue

            if extraction_cache:
                extraction_key = {
                    "kind": "image",
                    "content_hash": image_hashes[image.image_id],
                }
                if image_hashes[image.image_id] == "missing":
                    extraction_key["missing_path"] = image.relative_path
                extraction = extraction_cache.get_or_compute(
                    claim_dir / f"image_{image.image_id}",
                    extraction_key,
                    require_extracted,
                )
                cached_features["image_extractions"][image.image_id] = extraction
                for error in extraction.get("errors", []):
                    cached_features["errors"].append({"owner": image.image_id, **error})
                for group in ("objects", "persons", "landmarks", "ocr"):
                    for index, concept in enumerate(extraction[group]):
                        cached_features["concept_features"][concept] = encoded_text(
                            claim_dir / f"visual_{image.image_id}_{group}_{index}",
                            concept,
                        )

        graph = builder.build(sample, cached_features)
        provenance = {
            **builder.provenance,
            "fingerprint": graph_fingerprint,
            "input_hash": input_hash,
        }
        path = f"{sample.split}/{sample.claim_id}.pt"
        provenance_path = f"{sample.split}/{sample.claim_id}.json"
        graph_payload = {
            "graph": graph,
            "label": target.label,
            "fingerprint": graph_fingerprint,
            "input_hash": input_hash,
        }
        atomic_torch(output / path, graph_payload)
        atomic_json(output / provenance_path, provenance)
        records_by_claim[claim_key] = {
            "claim_id": sample.claim_id,
            "split": sample.split,
            "label": target.label,
            "path": path,
            "provenance_path": provenance_path,
            "input_hash": input_hash,
            "modality": provenance["modality"],
            "extraction_errors": len(cached_features["errors"]),
        }
        manifest["records"] = list(records_by_claim.values())
        atomic_json(manifest_path, manifest)
        timings["graphs_built"] += 1
        print(
            f"graph {number}/{len(selected)} {sample.split}/{sample.claim_id}",
            flush=True,
        )
    timings["encoding_graph_seconds"] = time.perf_counter() - started
    finish_memory("encoding_graph")
    encoder.unload()

    selected_records = [
        records_by_claim[(sample.split, sample.claim_id)] for sample, _ in selected
    ]
    failed_claims = sum(
        bool(record["extraction_errors"]) for record in selected_records
    )
    build_report = {
        "timings": timings,
        "selected_claims": len(selected),
        "records": len(records_by_claim),
        "stage": stage,
        "device": config.device,
        "vram": memory,
        "claim_error_count": failed_claims,
        "claim_error_rate": (
            failed_claims / len(selected_records) if selected_records else 0.0
        ),
        "extraction_decode_errors": sum(
            record["extraction_errors"] for record in selected_records
        ),
    }
    atomic_json(output / "build_report.json", build_report)
    print(f"graph_root={output.as_posix()}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="mocheg")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--graph-root", default="graphs")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--stage", choices=("features", "full"), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = vars(parser.parse_args(argv))
    config = BuildConfig(device=args.pop("device"))
    build_graphs(config=config, **args)


if __name__ == "__main__":
    main()

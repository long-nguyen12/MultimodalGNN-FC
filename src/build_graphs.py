"""Cache extraction, then CLIP features, then reproducible evidence graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .cache import ContentCache, atomic_json, atomic_torch
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
        print(
            f"[dataset] {split}: loaded {len(dataset.samples)} claims, selected {len(records)}",
            flush=True,
        )

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
    output = Path(graph_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    records = []
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        fields = (
            "claim_id", "split", "label", "path", "provenance_path",
            "modality", "extraction_errors",
        )
        records = [{field: record[field] for field in fields} for record in saved["records"]]
    manifest = {
        "metadata": {"graph_stage": stage, "config": config.to_dict()},
        "records": records,
    }
    print(f"[manifest] ready: {len(records)} cached graph records", flush=True)
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
    extraction_counts = {
        kind: {"new": 0, "cached": 0, "retries": 0}
        for kind in ("text", "image")
    }
    qwen_generation = None
    if stage == "full":
        qwen_generation = {
            "model_id": backend.model_id,
            "enable_thinking": False,
            "seed": backend.seed,
            "max_new_tokens": backend.max_new_tokens,
            "parameters": backend.generation_parameters,
            "stats": backend.generation_stats,
            "prompts": {"text": TEXT_PROMPT, "image": IMAGE_PROMPT},
        }
        extraction_cache = ContentCache(Path(cache_root) / "extraction")
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
                cache_hit = (extraction_cache.root / (str(claim_dir / name) + ".json")).exists()
                extraction = extraction_cache.get_or_compute(
                    claim_dir / name,
                    lambda text=text: text_extractor.extract(text),
                )
                extraction_counts["text"]["cached" if cache_hit else "new"] += 1
                if not cache_hit:
                    extraction_counts["text"]["retries"] += extraction.get("retries", 0)
                errors = len(extraction.get("errors", []))
                print(
                    f"[extract:text] {claim_dir}/{name}: {'WARN' if errors else 'OK'} "
                    f"({'cache' if cache_hit else 'new'}), "
                    f"entities={len(extraction['entities'])}, errors={errors}",
                    flush=True,
                )
                for diagnostic in extraction.get("diagnostics", []):
                    print(
                        f"[extract:text:empty] {claim_dir}/{name}: chunk={diagnostic['chunk']}, "
                        f"candidates={diagnostic['candidates']}, discarded={diagnostic['discarded']}",
                        flush=True,
                    )
            for image in sample.image_evidence:
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

                name = claim_dir / f"image_{image.image_id}"
                cache_hit = (extraction_cache.root / (str(name) + ".json")).exists()
                extraction = extraction_cache.get_or_compute(
                    name, extract_image
                )
                extraction_counts["image"]["cached" if cache_hit else "new"] += 1
                if not cache_hit:
                    extraction_counts["image"]["retries"] += extraction.get("retries", 0)
                errors = len(extraction.get("errors", []))
                concepts = sum(len(extraction[group]) for group in ("objects", "persons", "landmarks", "ocr"))
                print(
                    f"[extract:image] {name}: {'WARN' if errors else 'OK'} "
                    f"({'cache' if cache_hit else 'new'}), concepts={concepts}, errors={errors}",
                    flush=True,
                )
            print(
                f"extraction {number}/{len(selected)} {sample.split}/{sample.claim_id}",
                flush=True,
            )
        timings["extraction_seconds"] = time.perf_counter() - started
        finish_memory("extraction")
        backend.unload()
        print(
            f"[extraction] complete: {timings['extraction_seconds']:.1f}s; Qwen released",
            flush=True,
        )

    # Encode the original inputs and extracted concepts, then build each graph.
    feature_cache = ContentCache(Path(cache_root) / "features", tensor=True)
    records_by_claim = {}
    for record in manifest["records"]:
        records_by_claim[(record["split"], record["claim_id"])] = record
    builder = GraphBuilder(config.top_k)
    start_memory()
    started = time.perf_counter()

    def encoded_text(name, text):
        return feature_cache.get_or_compute(
            name, lambda: encoder.encode_text(text)
        )

    def require_extracted():
        raise ValueError("Extraction cache incomplete; rerun the extraction stage")

    for number, (sample, target) in enumerate(selected, 1):
        claim_key = (sample.split, sample.claim_id)
        claim_dir = Path(sample.split) / sample.claim_id
        record = records_by_claim.get(claim_key)
        if record and record["label"] == target.label:
            graph_exists = (output / record["path"]).exists()
            if graph_exists and (output / record["provenance_path"]).exists():
                timings["graphs_reused"] += 1
                print(
                    f"[graph] {number}/{len(selected)} {claim_dir}: reused cached graph",
                    flush=True,
                )
                continue

        cache_hit = (feature_cache.root / claim_dir / "claim.pt").exists()
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
        print(f"[encode:text] {claim_dir}/claim: OK ({'cache' if cache_hit else 'new'})", flush=True)
        owners = [("claim:0", "claim", sample.claim_text)]
        for index, evidence in enumerate(sample.text_evidence):
            name = f"text_{evidence.evidence_id}"
            cache_hit = (feature_cache.root / claim_dir / (name + ".pt")).exists()
            cached_features["texts"][evidence.evidence_id] = encoded_text(
                claim_dir / name, evidence.text
            )
            print(f"[encode:text] {claim_dir}/{name}: OK ({'cache' if cache_hit else 'new'})", flush=True)
            owners.append((f"evidence_text:{index}", name, evidence.text))

        if extraction_cache:
            for owner, name, text in owners:
                extraction = extraction_cache.get_or_compute(
                    claim_dir / name, require_extracted
                )
                cached_features["text_extractions"][owner] = extraction
                for error in extraction.get("errors", []):
                    cached_features["errors"].append({"owner": owner, **error})
                entity_cache_hits = 0
                for index, entity in enumerate(extraction["entities"]):
                    feature_name = claim_dir / f"entity_{name}_{index}"
                    entity_cache_hits += (feature_cache.root / (str(feature_name) + ".pt")).exists()
                    cached_features["entity_features"][entity["text"]] = encoded_text(
                        feature_name, entity["text"]
                    )
                print(
                    f"[encode:entities] {claim_dir}/{name}: OK, "
                    f"cache={entity_cache_hits}, new={len(extraction['entities']) - entity_cache_hits}",
                    flush=True,
                )

        for image in sample.image_evidence:
            name = claim_dir / f"image_{image.image_id}"
            cache_hit = (feature_cache.root / (str(name) + ".pt")).exists()
            try:
                image_features = feature_cache.get_or_compute(
                    name,
                    lambda image=image: encoder.encode_image(
                        data_root / image.relative_path
                    ),
                )
                cached_features["images"][image.image_id] = image_features
                print(f"[encode:image] {name}: OK ({'cache' if cache_hit else 'new'})", flush=True)
            except OSError as exc:
                cached_features["errors"].append(
                    {
                        "owner": image.image_id,
                        "phase": "image_decode",
                        "error": str(exc),
                    }
                )
                print(f"[encode:image] {name}: ERROR ({exc}); image skipped", flush=True)
                continue

            if extraction_cache:
                extraction = extraction_cache.get_or_compute(
                    claim_dir / f"image_{image.image_id}",
                    require_extracted,
                )
                cached_features["image_extractions"][image.image_id] = extraction
                for error in extraction.get("errors", []):
                    cached_features["errors"].append({"owner": image.image_id, **error})
                concept_count = 0
                concept_cache_hits = 0
                for group in ("objects", "persons", "landmarks", "ocr"):
                    for index, concept in enumerate(extraction[group]):
                        feature_name = claim_dir / f"visual_{image.image_id}_{group}_{index}"
                        concept_count += 1
                        concept_cache_hits += (feature_cache.root / (str(feature_name) + ".pt")).exists()
                        cached_features["concept_features"][concept] = encoded_text(
                            feature_name, concept,
                        )
                print(
                    f"[encode:concepts] {name}: OK, "
                    f"cache={concept_cache_hits}, new={concept_count - concept_cache_hits}",
                    flush=True,
                )

        graph = builder.build(sample, cached_features)
        provenance = builder.provenance
        provenance["text_extraction_diagnostics"] = [
            {"owner": owner, **diagnostic}
            for owner, extraction in cached_features["text_extractions"].items()
            for diagnostic in extraction.get("diagnostics", [])
        ]
        print(
            f"[graph:build] {claim_dir}: OK, "
            f"nodes={len(provenance['nodes'])}, edges={len(provenance['edges'])}",
            flush=True,
        )
        path = f"{sample.split}/{sample.claim_id}.pt"
        provenance_path = f"{sample.split}/{sample.claim_id}.json"
        graph_payload = {
            "graph": graph,
            "label": target.label,
        }
        atomic_torch(output / path, graph_payload)
        print(f"[save:graph] {output / path}: OK", flush=True)
        atomic_json(output / provenance_path, provenance)
        print(f"[save:provenance] {output / provenance_path}: OK", flush=True)
        records_by_claim[claim_key] = {
            "claim_id": sample.claim_id,
            "split": sample.split,
            "label": target.label,
            "path": path,
            "provenance_path": provenance_path,
            "modality": provenance["modality"],
            "extraction_errors": len(cached_features["errors"]),
        }
        manifest["records"] = list(records_by_claim.values())
        atomic_json(manifest_path, manifest)
        print(f"[save:manifest] {manifest_path}: OK", flush=True)
        timings["graphs_built"] += 1
        print(
            f"graph {number}/{len(selected)} {sample.split}/{sample.claim_id}",
            flush=True,
        )
    timings["encoding_graph_seconds"] = time.perf_counter() - started
    finish_memory("encoding_graph")
    encoder.unload()
    print(
        f"[encoding/graphs] complete: {timings['encoding_graph_seconds']:.1f}s, "
        f"built={timings['graphs_built']}, reused={timings['graphs_reused']}; CLIP released",
        flush=True,
    )

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
        "extraction_counts": extraction_counts,
        "qwen_generation": qwen_generation,
        "claim_error_count": failed_claims,
        "claim_error_rate": (
            failed_claims / len(selected_records) if selected_records else 0.0
        ),
        "extraction_decode_errors": sum(
            record["extraction_errors"] for record in selected_records
        ),
    }
    atomic_json(manifest_path, manifest)
    print(f"[save:manifest] {manifest_path}: OK", flush=True)
    atomic_json(output / "build_report.json", build_report)
    print(f"[save:report] {output / 'build_report.json'}: OK", flush=True)
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
    parser.add_argument("--extractor-id", default=BuildConfig.extractor_id)
    args = vars(parser.parse_args(argv))
    config = BuildConfig(
        device=args.pop("device"), extractor_id=args.pop("extractor_id")
    )
    build_graphs(config=config, **args)


if __name__ == "__main__":
    main()

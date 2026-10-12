"""Benchmark a small sample and estimate fresh-cache graph build time."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import time

import torch
from tqdm.auto import tqdm

from src.cache import atomic_json, file_hash
from src.config import BuildConfig
from src.dataset import MochegDataset, SPLITS, modality_group
from src.feature_encoder import FeatureEncoder
from src.graph_builder import GraphBuilder
from src.image_extractor import ImageExtractor
from src.text_extractor import QwenBackend, TextExtractor


def estimate_runtime(data_root, samples=5, device="cuda", stage="full", output="outputs/runtime_estimate.json"):
    config = BuildConfig(device=device)
    data_root = Path(data_root)

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    groups = defaultdict(list)
    split_counts = {}
    print("Reading datasets and hashing gold images...", flush=True)
    for split in SPLITS:
        dataset = MochegDataset(data_root, split)
        split_counts[split] = len(dataset)
        for sample in dataset.samples:
            groups[modality_group(sample)].append(sample)
    image_hashes = {}
    for group in groups.values():
        for sample in group:
            for image in sample.image_evidence:
                if image.relative_path not in image_hashes:
                    path = data_root / image.relative_path
                    image_hashes[image.relative_path] = file_hash(path) if path.is_file() else "missing"
    dataset_hash_seconds = time.perf_counter() - started

    rng = random.Random(config.seed)
    rows = []
    for group, claims in sorted(groups.items()):
        for sample in rng.sample(claims, min(samples, len(claims))):
            rows.append({
                "sample": sample,
                "modality": group,
                "qwen_seconds": 0.0,
                "features": {
                    "texts": {}, "images": {},
                    "text_extractions": {}, "image_extractions": {},
                    "entity_features": {}, "concept_features": {}, "errors": [],
                },
            })
    if not rows:
        raise ValueError("No claims found in the dataset")
    first_image = next((
        data_root / image.relative_path
        for row in rows for image in row["sample"].image_evidence
        if (data_root / image.relative_path).is_file()
    ), None)
    print(f"Benchmarking {len(rows)} claims; {sum(split_counts.values())} claims in full dataset.", flush=True)

    model_startup_seconds = 0.0
    if stage == "full":
        print("Loading and warming up Qwen...", flush=True)
        started = time.perf_counter()
        backend = QwenBackend(
            model_id=config.extractor_id, device=device,
            max_new_tokens=config.max_new_tokens,
            max_image_side=config.max_image_side, seed=config.seed,
        )
        backend.load()
        text_extractor = TextExtractor(backend, config.text_window, config.text_overlap)
        image_extractor = ImageExtractor(backend)
        text_extractor.extract("Paris is in France.")
        if first_image is not None:
            try:
                image_extractor.extract(first_image)
            except OSError:
                pass
        synchronize()
        model_startup_seconds += time.perf_counter() - started

        for row in tqdm(rows, desc="Qwen extraction"):
            sample, features = row["sample"], row["features"]
            synchronize()
            started = time.perf_counter()
            owners = [("claim:0", sample.claim_text)] + [
                (f"evidence_text:{index}", evidence.text)
                for index, evidence in enumerate(sample.text_evidence)
            ]
            for owner, text in owners:
                extraction = text_extractor.extract(text)
                features["text_extractions"][owner] = extraction
                features["errors"].extend({"owner": owner, **error} for error in extraction["errors"])
            for image in sample.image_evidence:
                try:
                    extraction = image_extractor.extract(data_root / image.relative_path)
                    features["image_extractions"][image.image_id] = extraction
                    features["errors"].extend({"owner": image.image_id, **error} for error in extraction["errors"])
                except OSError as error:
                    features["errors"].append({"owner": image.image_id, "error": str(error)})
            synchronize()
            row["qwen_seconds"] = time.perf_counter() - started
        backend.unload()

    print("Loading and warming up CLIP...", flush=True)
    started = time.perf_counter()
    encoder = FeatureEncoder(model_id=config.encoder_id, device=device)
    encoder.load()
    encoder.encode_text("Paris is in France.")
    if first_image is not None:
        try:
            encoder.encode_image(first_image)
        except OSError:
            pass
    synchronize()
    model_startup_seconds += time.perf_counter() - started
    builder = GraphBuilder(config.top_k)
    for row in tqdm(rows, desc="CLIP and graph"):
        sample, features = row["sample"], row["features"]
        synchronize()
        started = time.perf_counter()
        features["claim"] = encoder.encode_text(sample.claim_text)
        for evidence in sample.text_evidence:
            features["texts"][evidence.evidence_id] = encoder.encode_text(evidence.text)
        for extraction in features["text_extractions"].values():
            for entity in extraction["entities"]:
                features["entity_features"][entity["text"]] = encoder.encode_text(entity["text"])
        for image in sample.image_evidence:
            try:
                features["images"][image.image_id] = encoder.encode_image(data_root / image.relative_path)
            except OSError as error:
                features["errors"].append({"owner": image.image_id, "phase": "image_decode", "error": str(error)})
                continue
            extraction = features["image_extractions"].get(image.image_id, {})
            for group in ("objects", "persons", "landmarks", "ocr"):
                for concept in extraction.get(group, []):
                    features["concept_features"][concept] = encoder.encode_text(concept)
        synchronize()
        row["clip_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        builder.build(sample, features)
        row["graph_seconds"] = time.perf_counter() - started
    encoder.unload()

    phases = ("qwen_seconds", "clip_seconds", "graph_seconds")
    estimated = {phase: 0.0 for phase in phases}
    group_report = {}
    for group, claims in sorted(groups.items()):
        measured = [row for row in rows if row["modality"] == group]
        means = {phase: sum(row[phase] for row in measured) / len(measured) for phase in phases}
        group_report[group] = {
            "total_claims": len(claims), "sampled_claims": len(measured),
            "mean_seconds_per_claim": means,
            "sample_claims_with_errors": sum(bool(row["features"]["errors"]) for row in measured),
        }
        for phase in phases:
            estimated[phase] += len(claims) * means[phase]
    total_seconds = dataset_hash_seconds + model_startup_seconds + sum(estimated.values())
    report = {
        "scope": "Fresh-cache processing estimate; excludes cache, graph and manifest writes.",
        "stage": stage, "config": asdict(config), "split_claims": split_counts,
        "sample_size_per_modality": samples, "sampled_claims": len(rows),
        "dataset_and_hash_seconds": dataset_hash_seconds,
        "model_load_and_warmup_seconds": model_startup_seconds,
        "estimated_processing_seconds": estimated,
        "estimated_total_seconds": total_seconds, "estimated_total_hours": total_seconds / 3600,
        "groups": group_report,
        "samples": [
            {"claim_id": row["sample"].claim_id, "split": row["sample"].split,
             "modality": row["modality"], **{phase: row[phase] for phase in phases},
             "errors": row["features"]["errors"]}
            for row in rows
        ],
    }
    atomic_json(output, report)
    for phase, seconds in estimated.items():
        print(f"Estimated {phase.removesuffix('_seconds')}: {seconds / 3600:.2f} hours")
    print(f"Estimated total: {total_seconds / 3600:.2f} hours ({total_seconds / 86400:.2f} days)")
    print("Cache/graph/manifest writes are excluded; JSON errors can reduce downstream work.")
    print(f"Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="mocheg")
    parser.add_argument("--samples", type=int, default=5, help="Random claims per modality group (seed 42)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("full", "features"), default="full")
    parser.add_argument("--output", default="outputs/runtime_estimate.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    estimate_runtime(**vars(args))


if __name__ == "__main__":
    main()

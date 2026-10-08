"""Audit all claims in the original MOCHEG splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import LABELS, SPLITS, MochegDataset, modality_group


EXPECTED_COUNTS = {"train": 11669, "val": 1490, "test": 2442}


def audit_dataset(dataset: MochegDataset, check_images: bool = False) -> dict:
    if check_images:
        from PIL import Image

    label_counts = Counter(LABELS[target.label] for target in dataset.targets)
    modality_counts = Counter(modality_group(sample) for sample in dataset.samples)
    modalities_by_label = {label: Counter() for label in LABELS}
    text_owners = defaultdict(set)
    gold_image_paths = set()
    text_evidence_count = 0
    image_association_count = 0
    for sample, target in dataset:
        label_name = LABELS[target.label]
        modalities_by_label[label_name][modality_group(sample)] += 1
        text_evidence_count += len(sample.text_evidence)
        image_association_count += len(sample.image_evidence)
        for evidence in sample.text_evidence:
            text_owners[evidence.evidence_id].add(sample.claim_id)
        for evidence in sample.image_evidence:
            gold_image_paths.add(evidence.relative_path)

    missing_images = []
    unreadable_images = []
    readable_image_count = 0
    for relative_path in sorted(gold_image_paths):
        path = dataset.root / relative_path
        if not path.is_file():
            missing_images.append(relative_path)
        elif check_images:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.convert("RGB").load()
                readable_image_count += 1
            except Exception as error:
                unreadable_images.append({
                    "relative_path": relative_path,
                    "error": f"{type(error).__name__}: {error}",
                })

    shared_text_ids = {}
    for evidence_id, owners in text_owners.items():
        if len(owners) > 1:
            shared_text_ids[evidence_id] = sorted(owners)
    modality_report = {}
    for label_name, counts in modalities_by_label.items():
        modality_report[label_name] = dict(sorted(counts.items()))

    return {
        "split": dataset.split,
        "claims": len(dataset),
        "expected_claims": EXPECTED_COUNTS[dataset.split],
        "matches_expected_claim_count": len(dataset) == EXPECTED_COUNTS[dataset.split],
        "corpus_rows": dataset.corpus_rows,
        "qrels_rows": dataset.qrels_rows,
        "labels": {label: label_counts[label] for label in LABELS},
        "original_truthfulness": dict(sorted(dataset.original_truthfulness.items())),
        "modality_groups": dict(sorted(modality_counts.items())),
        "modalities_by_label": modality_report,
        "text_evidence": text_evidence_count,
        "unique_gold_images": len(gold_image_paths),
        "gold_image_associations": image_association_count,
        "images": {
            "decode_checked": check_images,
            "existing": len(gold_image_paths) - len(missing_images),
            "readable": readable_image_count if check_images else None,
            "missing": missing_images,
            "unreadable": unreadable_images,
        },
        "shared_text_evidence_ids": shared_text_ids,
        "warnings": dataset.warnings,
    }


def write_audit(
    data_root: str | Path,
    output_dir: str | Path,
    check_images: bool = False,
) -> dict:
    # Audit every claim in each original split.
    datasets = {split: MochegDataset(data_root, split) for split in SPLITS}
    split_audits = {split: audit_dataset(dataset, check_images) for split, dataset in datasets.items()}
    overlaps = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        first_claim_ids = set(datasets[first].claim_ids)
        second_claim_ids = set(datasets[second].claim_ids)
        overlaps[f"{first}/{second}"] = sorted(first_claim_ids & second_claim_ids)
    result = {
        "splits": split_audits,
        "split_overlap": overlaps,
    }

    # Save the full dataset audit.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Format the same audit results as a readable Markdown report.
    lines = [
        "# MOCHEG dataset audit",
        "",
        "Prediction unit: one claim, after grouping Corpus2 rows by claim_id.",
        "",
        "| Split | Claims | Corpus rows | Gold text | Unique gold images | Supported | Refuted | NEI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, audit in split_audits.items():
        lines.append(
            f"| {split} | {audit['claims']} | {audit['corpus_rows']} | "
            f"{audit['text_evidence']} | {audit['unique_gold_images']} | "
            f"{audit['labels']['supported']} | {audit['labels']['refuted']} | "
            f"{audit['labels']['NEI']} |"
        )
    lines.extend([
        "", "Modality groups:", "",
        "| Split | No evidence | Text only | Image only | Multimodal |",
        "|---|---:|---:|---:|---:|",
    ])
    for split, audit in split_audits.items():
        groups = audit["modality_groups"]
        lines.append(
            f"| {split} | {groups.get('no_evidence', 0)} | "
            f"{groups.get('text_only', 0)} | {groups.get('image_only', 0)} | "
            f"{groups.get('multimodal', 0)} |"
        )
    overlap_summary = ", ".join(f"{pair}: {len(ids)}" for pair, ids in overlaps.items())
    lines.extend([
        "",
        "Gold image mapping uses TOPIC and distinct evidence_id, including rows "
        "with RELEVANCY=0. DOCUMENT# is a retrieval candidate. Filenames retain zero padding.",
        "",
        "Labels and ruling_outline are stored in Target only; no fact-check URL, "
        "Headline, Truthfulness, or retrieval relevance enters SampleInput.",
        "",
        "Missing evidence does not change the original cleaned_truthfulness label. "
        "Original truthfulness categories and image failures are listed in audit.json.",
        "",
        f"Split overlap: {overlap_summary}.",
    ])
    for split, audit in split_audits.items():
        image_audit = audit["images"]
        lines.append(
            f"\n{split}: missing gold images={len(image_audit['missing'])}; "
            f"decode checked={check_images}; unreadable={len(image_audit['unreadable'])}; "
            f"mapping warnings={len(audit['warnings'])}."
        )
    lines.append("")
    (output_dir / "dataset_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="mocheg")
    parser.add_argument("--output-dir", default="outputs/audit")
    parser.add_argument("--check-images", action="store_true", help="Decode every referenced gold image (requires Pillow)")
    args = parser.parse_args()
    report = write_audit(args.data_root, args.output_dir, args.check_images)
    for split, audit in report["splits"].items():
        print(f"{split}: {audit['claims']} claims, {audit['unique_gold_images']} gold images, modalities={audit['modality_groups']}")
    print(f"Audit written to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()

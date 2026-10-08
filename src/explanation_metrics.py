"""Evaluate generated language against gold rulings after explanation generation."""

import argparse
import json
from pathlib import Path

from .dataset import MochegDataset, normalize_id
from .explain import explanation_quality


def read_explanations(path):
    path = Path(path)
    if path.is_dir():
        records = []
        for file_path in sorted(path.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "claim_id" in payload and "explanation" in payload:
                records.append(payload)
        return records
    if path.suffix.casefold() == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]


def evaluate_explanations(records, dataset, bertscore=False):
    """Gold references join here; neither model inference nor prompts are used."""
    references = {sample.claim_id: target.ruling_outline for sample, target in dataset}
    predictions = []
    rulings = []
    claim_ids = []
    seen = set()
    missing_generated = []
    missing_ruling = []
    for record in records:
        claim_id = normalize_id(str(record["claim_id"]))
        seen.add(claim_id)
        prediction = record["explanation"]
        if not isinstance(prediction, str) or not prediction.strip():
            missing_generated.append(claim_id)
            continue
        reference = references[claim_id]
        if not reference.strip():
            missing_ruling.append(claim_id)
            continue
        predictions.append(prediction.strip())
        rulings.append(reference)
        claim_ids.append(claim_id)
    metrics = {}
    if predictions:
        metrics = explanation_quality(predictions, rulings, bertscore=bertscore)
    return {
        "split": dataset.split,
        "input_count": len(records),
        "evaluated_count": len(predictions),
        "claim_ids": claim_ids,
        "missing_generated": missing_generated,
        "missing_ruling": missing_ruling,
        "metrics": metrics,
        "metric_scope": "Language quality only; faithfulness is evaluated through graph perturbation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-input", required=True,
        help="Explanation JSONL, JSON file, or directory of per-claim explanation JSON",
    )
    parser.add_argument("--data-root", default="mocheg")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--bertscore", action="store_true",
        help="Also compute BERTScore; requires its optional dependency/model",
    )
    args = parser.parse_args(argv)
    dataset = MochegDataset(args.data_root, args.split)
    records = read_explanations(args.quality_input)
    result = evaluate_explanations(records, dataset, bertscore=args.bertscore)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

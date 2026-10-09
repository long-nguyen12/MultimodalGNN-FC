"""Run selected model/ablation comparisons using the shared cached trainer."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .ablations import ABLATIONS
from .baseline import BaselineModel
from .evaluate import evaluate_model, load_model, write_evaluation
from .graph_dataset import GraphDataset, collate_graphs
from .train import TrainConfig, train_run, validate_training_cache

MODELS = tuple(BaselineModel.MODALITIES) + (
    "gcn", "rgcn", "attention", "learned", "claim_conditioned"
)


def applicable(model, ablation):
    if ablation == "none":
        return True
    graph_model = model not in BaselineModel.MODALITIES
    if ablation == "claim_conditioning":
        return model == "claim_conditioned"
    if ablation == "relation_types":
        return model in ("rgcn", "attention", "learned", "claim_conditioned")
    if ablation == "edge_scores":
        return model in ("gcn", "attention", "learned", "claim_conditioned")
    if ablation in ("semantic", "cross_modal"):
        return graph_model
    if ablation in ("entity", "visual_concept"):
        return graph_model or model == "B4-flat"
    if ablation == "text":
        return graph_model or model in ("B1", "B3", "B4-flat")
    if ablation == "image":
        return graph_model or model in ("B2", "B3", "B4-flat")
    return False


def experiment_matrix(
    base, models, ablations=("none",), include_evidence_only=False
):
    """Omit combinations whose ablation changes no input or model behavior."""
    for values, choices, name in ((models, MODELS, "model"), (ablations, ABLATIONS, "ablation")):
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in choices for value in values)
        ):
            raise ValueError(f"Invalid or duplicate {name} selection")
    configs = []
    training_controls = (False, True) if include_evidence_only else (False,)
    for model in models:
        for ablation in ablations:
            if not applicable(model, ablation):
                continue
            for evidence_only in training_controls:
                group = "evidence_only" if evidence_only else "all_train"
                output = Path(base.output_dir) / model / ablation / group / f"seed_{base.seed}"
                config = replace(
                    base,
                    model=model,
                    ablation=ablation,
                    evidence_only=evidence_only,
                    output_dir=str(output),
                )
                configs.append(config)
    if not configs:
        raise ValueError("No applicable model/ablation combinations were selected")
    return configs


def prevalidate(configs, evaluate_test=False):
    """Reject invalid cache/split settings before the first training run."""
    fingerprints = {}
    splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    for config in configs:
        manifest_path = Path(config.graph_root) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_training_cache(config, manifest)
        if (
            config.batch_size < 1
            or config.epochs < 1
            or config.patience < 1
            or config.hidden_dim < 1
        ):
            raise ValueError("Batch size, epochs, patience and hidden dimension must be positive")
        if not 0 <= config.dropout < 1 or config.learning_rate <= 0 or config.weight_decay < 0:
            raise ValueError("Invalid dropout or optimizer parameters")
        if config.model in ("attention", "learned", "claim_conditioned") and config.hidden_dim % 4:
            raise ValueError("Attention hidden dimension must be divisible by four")
        for split in splits:
            filter_missing_evidence = config.evidence_only and split == "train"
            dataset = GraphDataset(
                config.graph_root,
                split,
                ablation=config.ablation,
                evidence_only=filter_missing_evidence,
            )
            if not len(dataset):
                raise ValueError(f"Empty {split} cache for {config.model}/{config.ablation}")
        fingerprints[config.output_dir] = manifest["fingerprint"]
    return fingerprints


def aggregate_results(rows):
    baselines = {
        (row["model"], row["evidence_only"], row["split"]): row["macro_f1"]
        for row in rows if row["ablation"] == "none"
    }
    summary = []
    for row in rows:
        result = {
            key: row[key]
            for key in ("model", "ablation", "evidence_only", "seed", "split", "accuracy", "macro_f1")
        }
        baseline = baselines.get((row["model"], row["evidence_only"], row["split"]))
        result["macro_f1_delta"] = row["macro_f1"] - baseline if baseline is not None else ""
        summary.append(result)
    return summary


def write_table(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_matrix(configs, output_dir, evaluate_test=False):
    fingerprints = prevalidate(configs, evaluate_test)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_manifest = {
        "evaluate_test": evaluate_test,
        "configs": [asdict(config) for config in configs],
    }
    (output / "experiment_matrix.json").write_text(
        json.dumps(matrix_manifest, indent=2), encoding="utf-8"
    )
    results = []
    for config in configs:
        metrics = {"val": train_run(config)}
        checkpoint_path = Path(config.output_dir) / "best.pt"
        if evaluate_test:
            model, payload = load_model(checkpoint_path, config.device)
            test_data = GraphDataset(config.graph_root, "test", ablation=config.ablation)
            test_loader = DataLoader(
                test_data, batch_size=config.batch_size, collate_fn=collate_graphs
            )
            test_metrics, test_predictions = evaluate_model(
                model, test_loader, config.device
            )
            write_evaluation(
                Path(config.output_dir) / "test", test_metrics, test_predictions
            )
            metrics["test"] = test_metrics
        for split, values in metrics.items():
            results.append({
                "model": config.model,
                "ablation": config.ablation,
                "evidence_only": config.evidence_only,
                "seed": config.seed,
                "split": split,
                "count": values["count"],
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "cache_fingerprint": fingerprints[config.output_dir],
                "checkpoint": str(checkpoint_path),
            })
        write_table(output / "experiment_results.csv", results)
        write_table(output / "experiment_summary.csv", aggregate_results(results))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=["none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-evidence-only", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test only after the caller has frozen the experimental design",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = vars(parser.parse_args(argv))
    models = args.pop("models")
    ablations = args.pop("ablations")
    include_evidence_only = args.pop("include_evidence_only")
    evaluate_test = args.pop("evaluate_test")
    base_config = TrainConfig(**args)
    configs = experiment_matrix(
        base_config, models, ablations, include_evidence_only
    )
    run_matrix(configs, base_config.output_dir, evaluate_test)


if __name__ == "__main__":
    main()

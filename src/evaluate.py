"""Evaluation from cached graphs; no extractor or encoder is imported."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .baseline import BaselineModel
from .hetero_gnn import GraphModel

LABEL_NAMES = ("supported", "refuted", "NEI")


def build_model(config):
    kind = config["kind"]
    model_options = {}
    for name in ("hidden_dim", "dropout"):
        if name in config:
            model_options[name] = config[name]
    if kind in BaselineModel.MODALITIES:
        return BaselineModel(kind, **model_options)
    return GraphModel(
        kind, relation_types=config.get("relation_types", True), **model_options
    )


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def classification_metrics(targets, predictions):
    if not targets:
        return {"count": 0}
    report = classification_report(
        targets,
        predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(
        targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "count": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(macro_f1),
        "per_class": {name: report[name] for name in LABEL_NAMES},
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
    }


def graph_modalities(batch):
    num_graphs = batch["claim"].x.size(0)
    has_text = torch.bincount(batch["evidence_text"].batch, minlength=num_graphs) > 0
    has_image = torch.bincount(batch["evidence_image"].batch, minlength=num_graphs) > 0
    modalities = []
    for text_present, image_present in zip(has_text.tolist(), has_image.tolist()):
        if text_present and image_present:
            modality = "multimodal"
        elif text_present:
            modality = "text_only"
        elif image_present:
            modality = "image_only"
        else:
            modality = "no_evidence"
        modalities.append(modality)
    return modalities


@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    model.eval()
    rows = []
    total_loss = 0.0
    for batch, labels in loader:
        modalities = graph_modalities(batch)
        claim_ids = batch.claim_id
        batch = batch.to(device)
        labels = labels.to(device)
        logits = model(batch)
        batch_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(batch_loss)
        probabilities = logits.softmax(-1).cpu()
        sample_predictions = zip(
            claim_ids, labels.cpu().tolist(), probabilities.tolist(), modalities
        )
        for claim_id, gold_label, class_probabilities, modality in sample_predictions:
            prediction = int(np.argmax(class_probabilities))
            rows.append({
                "claim_id": str(claim_id),
                "label": gold_label,
                "prediction": prediction,
                "label_name": LABEL_NAMES[gold_label],
                "prediction_name": LABEL_NAMES[prediction],
                "probabilities": class_probabilities,
                "modality": modality,
            })
    targets = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(rows) if rows else None
    metrics["by_modality"] = {}
    for modality in ("no_evidence", "text_only", "image_only", "multimodal"):
        group_rows = [row for row in rows if row["modality"] == modality]
        group_targets = [row["label"] for row in group_rows]
        group_predictions = [row["prediction"] for row in group_rows]
        metrics["by_modality"][modality] = classification_metrics(
            group_targets, group_predictions
        )
    return metrics, rows


def write_evaluation(output_dir, metrics, rows):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = [row for row in rows if row["label"] != row["prediction"]]
    (output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")


def summarize_seeds(metrics):
    summary = {"runs": len(metrics)}
    for name in ("accuracy", "macro_f1"):
        values = [result[name] for result in metrics]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph-root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    from .graph_dataset import GraphDataset, collate_graphs
    model, checkpoint = load_model(args.checkpoint, args.device)
    train_config = checkpoint["train_config"]
    root = Path(args.graph_root or train_config["graph_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["fingerprint"] != checkpoint["cache_fingerprint"]:
        raise ValueError("Checkpoint and graph cache fingerprints differ")
    dataset = GraphDataset(
        root, args.split, ablation=train_config.get("ablation", "none")
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_graphs)
    metrics, rows = evaluate_model(model, loader, args.device)
    metrics.update({
        "split": args.split,
        "cache_fingerprint": manifest["fingerprint"],
        "checkpoint": str(args.checkpoint),
    })
    write_evaluation(args.output_dir, metrics, rows)
    print(json.dumps({
        "split": args.split,
        "count": metrics["count"],
        "macro_f1": metrics.get("macro_f1"),
    }))


if __name__ == "__main__":
    main()

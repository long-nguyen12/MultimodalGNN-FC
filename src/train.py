"""Train only from feature/graph caches and select checkpoints using validation."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .ablations import ABLATIONS
from .baseline import BaselineModel
from .evaluate import build_model, evaluate_model, write_evaluation


@dataclass
class TrainConfig:
    graph_root: str
    output_dir: str
    model: str = "B3"
    hidden_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42
    device: str = "cpu"
    ablation: str = "none"
    evidence_only: bool = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_training_cache(config, manifest):
    metadata = manifest.get("metadata", {})
    is_baseline = config.model in BaselineModel.MODALITIES
    if config.model == "B4-flat" or not is_baseline:
        if metadata.get("graph_stage") != "full":
            raise ValueError("B4-flat and GNN training require a full extraction graph cache")


def train_run(config):
    from .graph_dataset import GraphDataset, collate_graphs
    root = Path(config.graph_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_training_cache(config, manifest)
    seed_everything(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    train_data = GraphDataset(
        root, "train", ablation=config.ablation, evidence_only=config.evidence_only
    )
    val_data = GraphDataset(root, "val", ablation=config.ablation)
    
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, collate_fn=collate_graphs
    )
    kind = config.model
    if kind == "claim_conditioned" and config.ablation == "claim_conditioning":
        kind = "learned"
    model_config = {
        "kind": kind,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "relation_types": config.ablation != "relation_types",
    }
    model = build_model(model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = output / "best.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}")
        for batch, labels in progress:
            batch = batch.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            sample_count += labels.numel()
            progress.set_postfix(loss=total_loss / sample_count)
        val_metrics, val_predictions = evaluate_model(
            model, val_loader, config.device, desc=f"Val {epoch}/{config.epochs}"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        print(json.dumps(history[-1]), flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            checkpoint = {
                "model_config": model_config,
                "state_dict": model.state_dict(),
                "train_config": asdict(config),
                "cache_fingerprint": manifest["fingerprint"],
                "best_epoch": epoch,
                "val_metrics": val_metrics,
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)
            write_evaluation(output / "val", val_metrics, val_predictions)
        else:
            epochs_without_improvement += 1
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epochs_without_improvement >= config.patience:
            break
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return best["val_metrics"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-dir", required=True)
    model_choices = tuple(BaselineModel.MODALITIES) + (
        "gcn", "rgcn", "attention", "learned", "claim_conditioned"
    )
    parser.add_argument("--model", choices=model_choices, default="B3")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--evidence-only", action="store_true")
    args = vars(parser.parse_args(argv))
    args["output_dir"] = str(Path(args["output_dir"]) / f"seed_{args['seed']}")
    train_run(TrainConfig(**args))


if __name__ == "__main__":
    main()

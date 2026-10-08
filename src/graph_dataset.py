from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch


class GraphDataset:
    def __init__(self, root, split, ablation=None, evidence_only=False):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records = []
        for record in self.manifest["records"]:
            if record["split"] != split:
                continue
            if evidence_only and record["modality"] in ("none", "no_evidence"):
                continue
            self.records.append(record)
        self.ablation = ablation or "none"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        graph_path = self.root / record["path"]
        payload = torch.load(graph_path, map_location="cpu", weights_only=False)
        graph = payload["graph"]
        if self.ablation != "none":
            from .ablations import apply_ablation

            graph = apply_ablation(graph, self.ablation)
        label = int(payload["label"])
        return graph, label


def collate_graphs(samples):
    graphs, labels = zip(*samples)
    batch = Batch.from_data_list(graphs)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return batch, label_tensor

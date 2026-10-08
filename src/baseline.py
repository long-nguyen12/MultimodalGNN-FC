"""Frozen-feature baselines and the shared trainable input representation."""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from .schema import EDGE_TYPES, NODE_TYPES


class InputProjector(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.projection = nn.Linear(512, hidden_dim)
        self.node_type = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.subtype = nn.Embedding(16, hidden_dim)

    def forward(self, graph):
        projected_features = []
        for type_id, node_type in enumerate(NODE_TYPES):
            nodes = graph[node_type]
            default_subtypes = torch.zeros(
                nodes.x.size(0), dtype=torch.long, device=nodes.x.device
            )
            subtype_ids = nodes.get("subtype", default_subtypes)
            features = self.projection(nodes.x)
            features = features + self.node_type.weight[type_id]
            features = features + self.subtype(subtype_ids)
            projected_features.append(features)
        return torch.cat(projected_features, dim=0)


@dataclass
class HomogeneousView:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_score: torch.Tensor
    batch: torch.Tensor
    claim_index: torch.Tensor
    offsets: dict
    num_graphs: int


def homogeneous_view(graph):
    """Flatten using the schema registry, including empty node and edge stores."""
    device = graph["claim"].x.device
    offsets = {}
    node_batches = []
    node_count = 0
    for node_type in NODE_TYPES:
        nodes = graph[node_type]
        offsets[node_type] = node_count
        default_batch = torch.zeros(nodes.x.size(0), dtype=torch.long, device=device)
        node_batches.append(nodes.get("batch", default_batch))
        node_count += nodes.x.size(0)

    edge_indices = []
    relation_ids = []
    edge_scores = []
    for relation_id, edge_type in enumerate(EDGE_TYPES):
        source_type, _, destination_type = edge_type
        edges = graph[edge_type]
        edge_index = edges.edge_index.clone()
        edge_index[0] += offsets[source_type]
        edge_index[1] += offsets[destination_type]
        edge_indices.append(edge_index)
        relation_ids.append(torch.full(
            (edge_index.size(1),), relation_id, dtype=torch.long, device=device
        ))
        edge_scores.append(edges.edge_attr.reshape(-1))

    num_graphs = graph["claim"].x.size(0)
    claim_index = torch.arange(num_graphs, device=device) + offsets["claim"]
    return HomogeneousView(
        edge_index=torch.cat(edge_indices, dim=1),
        edge_type=torch.cat(relation_ids),
        edge_score=torch.cat(edge_scores),
        batch=torch.cat(node_batches),
        claim_index=claim_index,
        offsets=offsets,
        num_graphs=num_graphs,
    )


class BaselineModel(nn.Module):
    MODALITIES = {
        "B0": ("claim",),
        "B1": ("claim", "evidence_text"),
        "B2": ("claim", "evidence_image"),
        "B3": ("claim", "evidence_text", "evidence_image"),
        "B4-flat": NODE_TYPES,
    }

    def __init__(self, kind="B3", hidden_dim=256, dropout=0.2):
        super().__init__()
        
        self.kind = kind
        self.projector = InputProjector(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.MODALITIES[kind]), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, graph, return_attention=False):
        hidden = self.projector(graph)
        node_counts = [graph[node_type].x.size(0) for node_type in NODE_TYPES]
        features_by_type = dict(zip(NODE_TYPES, hidden.split(node_counts)))
        num_graphs = graph["claim"].x.size(0)

        pooled_features = []
        for node_type in self.MODALITIES[self.kind]:
            features = features_by_type[node_type]
            default_batch = torch.zeros(
                features.size(0), dtype=torch.long, device=features.device
            )
            node_batch = graph[node_type].get("batch", default_batch)
            pooled = scatter(
                features, node_batch, dim=0, dim_size=num_graphs, reduce="mean"
            )
            pooled_features.append(pooled)

        logits = self.classifier(torch.cat(pooled_features, dim=-1))
        if return_attention:
            return logits, []
        return logits

"""Homogeneous GCN and relation-only R-GCN controls."""

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, RGCNConv

from .baseline import InputProjector, homogeneous_view
from .schema import EDGE_TYPES


class GCNModel(nn.Module):
    def __init__(
        self, hidden_dim=256, dropout=0.2, relation_types=False, collapse_relations=False
    ):
        super().__init__()
        self.kind = "rgcn" if relation_types else "gcn"
        self.relation_types = relation_types
        self.collapse_relations = collapse_relations
        self.dropout = dropout
        self.projector = InputProjector(hidden_dim)
        layers = []
        for _ in range(2):
            if relation_types:
                layer = RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    len(EDGE_TYPES),
                    num_bases=min(8, len(EDGE_TYPES)),
                )
            else:
                layer = GCNConv(hidden_dim, hidden_dim)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, graph, return_attention=False):
        hidden = self.projector(graph)
        view = homogeneous_view(graph)
        for layer in self.layers:
            if self.relation_types:
                relation_ids = view.edge_type
                if self.collapse_relations:
                    relation_ids = relation_ids * 0
                hidden = layer(hidden, view.edge_index, relation_ids)
            else:
                hidden = layer(hidden, view.edge_index, view.edge_score)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.classifier(hidden[view.claim_index])
        if return_attention:
            return logits, []
        return logits


class RGCNModel(GCNModel):
    def __init__(self, hidden_dim=256, dropout=0.2, relation_types=True):
        super().__init__(
            hidden_dim,
            dropout,
            relation_types=True,
            collapse_relations=not relation_types,
        )

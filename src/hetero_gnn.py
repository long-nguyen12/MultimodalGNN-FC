"""Typed attention, learned relations, and claim-conditioned reasoning."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, softmax

from .baseline import InputProjector, homogeneous_view
from .gcn import GCNModel, RGCNModel
from .schema import EDGE_TYPES, SIMILARITY_RELATIONS


class TypedAttentionLayer(nn.Module):
    def __init__(
        self, hidden_dim, heads=4, learned=False, conditioned=False, relation_types=True
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.learned = learned
        self.conditioned = conditioned
        self.relation_types = relation_types
        self.relation = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in EDGE_TYPES
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.gate_network = None
        if learned:
            self.gate_network = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.claim_query = None
        if conditioned:
            self.claim_query = nn.Linear(hidden_dim, hidden_dim, bias=False)

        similarity_ids = []
        for relation_id, (_, relation_name, _) in enumerate(EDGE_TYPES):
            forward_name = relation_name.removeprefix("rev_")
            if relation_name in SIMILARITY_RELATIONS or forward_name in SIMILARITY_RELATIONS:
                similarity_ids.append(relation_id)
        self.register_buffer("similarity_ids", torch.tensor(similarity_ids, dtype=torch.long))

    def forward(self, hidden, view, claim_hidden):
        source_nodes, destination_nodes = view.edge_index
        edge_count = source_nodes.numel()
        messages = hidden.new_zeros((edge_count, self.hidden_dim))
        if self.relation_types:
            for relation_id, transform in enumerate(self.relation):
                relation_mask = view.edge_type == relation_id
                relation_sources = source_nodes[relation_mask]
                messages[relation_mask] = transform(hidden[relation_sources])
        else:
            messages = self.relation[0](hidden[source_nodes])

        edge_keys = self.key(messages)
        head_dim = self.hidden_dim // self.heads
        head_keys = edge_keys.reshape(-1, self.heads, head_dim)
        destination_queries = self.query(hidden[destination_nodes])
        head_queries = destination_queries.reshape(-1, self.heads, head_dim)
        attention_logits = (head_queries * head_keys).sum(-1) / math.sqrt(head_dim)
        if self.conditioned:
            claim_queries = self.claim_query(claim_hidden)
            edge_graph_ids = view.batch[destination_nodes]
            edge_claim_queries = claim_queries[edge_graph_ids]
            claim_scores = (edge_claim_queries * edge_keys).sum(-1, keepdim=True)
            attention_logits = attention_logits + claim_scores / math.sqrt(self.hidden_dim)

        score_prior = view.edge_score.clamp_min(1e-8).log().unsqueeze(-1)
        attention_logits = attention_logits + score_prior
        edge_gates = hidden.new_ones(edge_count)
        if self.learned:
            similarity_mask = torch.isin(view.edge_type, self.similarity_ids)
            source_hidden = hidden[source_nodes[similarity_mask]]
            destination_hidden = hidden[destination_nodes[similarity_mask]]
            gate_inputs = torch.cat(
                (source_hidden, destination_hidden, source_hidden * destination_hidden),
                dim=-1,
            )
            edge_gates[similarity_mask] = self.gate_network(gate_inputs).sigmoid().squeeze(-1)
            gate_prior = edge_gates.clamp_min(1e-8).log().unsqueeze(-1)
            attention_logits = attention_logits + gate_prior

        attention_weights = softmax(
            attention_logits, destination_nodes, num_nodes=hidden.size(0)
        )
        head_messages = messages.reshape(-1, self.heads, head_dim)
        weighted_messages = head_messages * attention_weights.unsqueeze(-1)
        weighted_messages = weighted_messages.reshape(-1, self.hidden_dim)
        aggregated_messages = scatter(
            weighted_messages,
            destination_nodes,
            dim=0,
            dim_size=hidden.size(0),
            reduce="sum",
        )
        output = self.self_projection(hidden) + self.output(aggregated_messages)
        attention = {
            "edge_index": view.edge_index,
            "edge_type": view.edge_type,
            "alpha": attention_weights,
            "gate": edge_gates,
        }
        return output, attention


class TypedAttentionModel(nn.Module):
    def __init__(
        self, hidden_dim=256, dropout=0.2, learned=False, conditioned=False, relation_types=True
    ):
        super().__init__()
        self.dropout = dropout
        self.projector = InputProjector(hidden_dim)
        self.layers = nn.ModuleList(
            TypedAttentionLayer(
                hidden_dim,
                learned=learned,
                conditioned=conditioned,
                relation_types=relation_types,
            )
            for _ in range(2)
        )
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, graph, return_attention=False):
        hidden = self.projector(graph)
        view = homogeneous_view(graph)
        claim_hidden = hidden[view.claim_index]
        attention = []
        for layer in self.layers:
            hidden, weights = layer(hidden, view, claim_hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            attention.append(weights)
        logits = self.classifier(hidden[view.claim_index])
        if return_attention:
            return logits, attention
        return logits


class GraphModel(nn.Module):
    """Common public classifier; model kind is checkpointed by the trainer."""

    def __init__(self, kind="gcn", hidden_dim=256, dropout=0.2, relation_types=True):
        super().__init__()
        self.kind = kind
        if kind == "gcn":
            self.network = GCNModel(hidden_dim, dropout)
        elif kind == "rgcn":
            self.network = RGCNModel(hidden_dim, dropout, relation_types=relation_types)
        elif kind in ("attention", "learned", "claim_conditioned"):
            learned_relations = kind in ("learned", "claim_conditioned")
            claim_conditioning = kind == "claim_conditioned"
            self.network = TypedAttentionModel(
                hidden_dim,
                dropout,
                learned=learned_relations,
                conditioned=claim_conditioning,
                relation_types=relation_types,
            )
        else:
            raise ValueError(f"Unknown graph model: {kind}")

    def forward(self, graph, return_attention=False):
        return self.network(graph, return_attention=return_attention)

"""Ablations preserve the schema and remove evidence-owned descendants."""

import torch

from .schema import CROSS_MODAL_RELATIONS, EDGE_TYPES, NODE_TYPES, SEMANTIC_RELATIONS

ABLATIONS = (
    "none",
    "entity",
    "visual_concept",
    "text",
    "image",
    "semantic",
    "cross_modal",
    "relation_types",
    "edge_scores",
    "claim_conditioning",
)


def select_nodes(graph, masks, preserve_claim=True):
    """Slice a cached graph and remap its edges and ownership indices."""
    keep_masks = {}
    for node_type in NODE_TYPES:
        nodes = graph[node_type]
        default_mask = torch.ones(
            nodes.x.size(0), dtype=torch.bool, device=nodes.x.device
        )
        keep_masks[node_type] = masks.get(node_type, default_mask).clone()
    if preserve_claim:
        keep_masks["claim"].fill_(True)

    # Removing an owner also removes every node extracted from that owner.
    changed = True
    while changed:
        changed = False
        for node_type in ("entity", "visual_concept"):
            nodes = graph[node_type]
            if "owner_type" not in nodes:
                continue
            valid_nodes = keep_masks[node_type].clone()
            for owner_type_id, owner_type in enumerate(NODE_TYPES):
                owned_nodes = nodes.owner_type == owner_type_id
                if owned_nodes.any():
                    owner_indices = nodes.owner_index[owned_nodes]
                    valid_nodes[owned_nodes] &= keep_masks[owner_type][owner_indices]
            if not torch.equal(valid_nodes, keep_masks[node_type]):
                keep_masks[node_type] = valid_nodes
                changed = True

    node_index_maps = {}
    result = graph.clone()
    for node_type in NODE_TYPES:
        keep_mask = keep_masks[node_type]
        kept_count = int(keep_mask.sum())
        index_map = torch.full(
            (keep_mask.numel(),), -1, dtype=torch.long, device=keep_mask.device
        )
        index_map[keep_mask] = torch.arange(kept_count, device=keep_mask.device)
        node_index_maps[node_type] = index_map
        for field, value in graph[node_type].items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim
                and value.size(0) == keep_mask.numel()
            ):
                result[node_type][field] = value[keep_mask]
        result[node_type].num_nodes = kept_count

    for node_type in ("entity", "visual_concept"):
        nodes = result[node_type]
        if "owner_type" in nodes:
            for owner_type_id, owner_type in enumerate(NODE_TYPES):
                owned_nodes = nodes.owner_type == owner_type_id
                old_owner_indices = nodes.owner_index[owned_nodes]
                nodes.owner_index[owned_nodes] = node_index_maps[owner_type][old_owner_indices]

    for edge_type in EDGE_TYPES:
        source_type, _, destination_type = edge_type
        edge_index = graph[edge_type].edge_index
        source_kept = keep_masks[source_type][edge_index[0]]
        destination_kept = keep_masks[destination_type][edge_index[1]]
        edge_keep = source_kept & destination_kept
        source_indices = node_index_maps[source_type][edge_index[0, edge_keep]]
        destination_indices = node_index_maps[destination_type][edge_index[1, edge_keep]]
        result[edge_type].edge_index = torch.stack((source_indices, destination_indices))
        result[edge_type].edge_attr = graph[edge_type].edge_attr[edge_keep]
    return result


def apply_ablation(graph, name="none"):
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    node_type = {
        "entity": "entity",
        "visual_concept": "visual_concept",
        "text": "evidence_text",
        "image": "evidence_image",
    }.get(name)
    if node_type:
        nodes = graph[node_type]
        remove_all_nodes = torch.zeros(
            nodes.x.size(0), dtype=torch.bool, device=nodes.x.device
        )
        return select_nodes(graph, {node_type: remove_all_nodes})
    result = graph.clone()
    if name == "semantic":
        removed_relations = SEMANTIC_RELATIONS
    elif name == "cross_modal":
        removed_relations = CROSS_MODAL_RELATIONS
    else:
        removed_relations = ()
    for edge_type in EDGE_TYPES:
        if edge_type[1] in removed_relations:
            result[edge_type].edge_index = result[edge_type].edge_index[:, :0]
            result[edge_type].edge_attr = result[edge_type].edge_attr[:0]
        elif name == "edge_scores":
            result[edge_type].edge_attr = torch.ones_like(result[edge_type].edge_attr)
    return result

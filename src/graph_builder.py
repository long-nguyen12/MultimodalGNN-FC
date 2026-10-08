"""Build scored, claim-rooted graphs exclusively from allowed sample inputs."""
from __future__ import annotations

from datetime import datetime
import re
import unicodedata

import torch
from torch_geometric.data import HeteroData

from .schema import (CONCEPT_TYPES, EDGE_TYPES, ENTITY_TYPES, FEATURE_DIM,
                     NODE_TYPES)


def normalize_name(text):
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized_text.split())


def explicit_date(text):
    text = text.strip()
    if re.fullmatch(r"\d{4}", text):
        return ("year", text)
    formats = (
        ("%Y-%m-%d", "day"), ("%d %B %Y", "day"), ("%d %b %Y", "day"),
        ("%B %d, %Y", "day"), ("%b %d, %Y", "day"), ("%B %d %Y", "day"),
        ("%Y-%m", "month"), ("%B %Y", "month"), ("%b %Y", "month"),
    )
    for date_format, resolution in formats:
        try:
            date = datetime.strptime(text, date_format)
            output_format = "%Y-%m-%d" if resolution == "day" else "%Y-%m"
            return resolution, date.strftime(output_format)
        except ValueError:
            continue
    return None


class GraphBuilder:
    def __init__(self, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.provenance = None

    def build(self, sample, cached_features):
        graph = HeteroData()
        nodes = {name: [] for name in NODE_TYPES}
        features = {name: [] for name in NODE_TYPES}
        edges = {edge: {} for edge in EDGE_TYPES}

        def add_node(kind, text, feature, source=None, owner=None, subtype=0, **attributes):
            feature = torch.as_tensor(feature, dtype=torch.float32).detach().cpu().reshape(-1)
            if feature.shape != (FEATURE_DIM,) or not torch.isfinite(feature).all():
                raise ValueError("Invalid cached feature for " + kind)
            index = len(nodes[kind])
            node_id = f"{kind}:{index}"
            nodes[kind].append({
                "id": node_id,
                "type": kind,
                "index": index,
                "text": text,
                "source": source,
                "owner": owner,
                "subtype": subtype,
                **attributes,
            })
            features[kind].append(feature)
            return index

        def add_edge(source_type, relation, target_type, source_index, target_index, score=1.0):
            forward_pair = (source_index, target_index)
            reverse_pair = (target_index, source_index)
            score = min(1.0, max(0.0, float(score)))
            forward_type = (source_type, relation, target_type)
            reverse_type = (target_type, "rev_" + relation, source_type)
            edges[forward_type][forward_pair] = max(
                score, edges[forward_type].get(forward_pair, 0.0),
            )
            edges[reverse_type][reverse_pair] = max(
                score, edges[reverse_type].get(reverse_pair, 0.0),
            )

        def add_entities(owner_id, owner_text, source):
            extraction = cached_features.get("text_extractions", {}).get(owner_id, {})
            seen_entities = set()
            for entity in extraction.get("entities", []):
                literal = entity["text"]
                entity_type = entity["type"]
                entity_key = (normalize_name(literal), entity_type)
                if entity_key in seen_entities or entity_type not in ENTITY_TYPES or literal not in owner_text:
                    continue
                seen_entities.add(entity_key)
                spans = []
                for match in re.finditer(re.escape(literal), owner_text):
                    spans.append({"start": match.start(), "end": match.end()})
                entity_index = add_node(
                    "entity", literal, cached_features["entity_features"][literal],
                    source=source, owner=owner_id, subtype=ENTITY_TYPES[entity_type],
                    entity_type=entity_type, spans=spans,
                )
                owner_type, owner_index = owner_id.split(":")
                add_edge(owner_type, "mentions", "entity", int(owner_index), entity_index)

        # First create nodes and their structural ownership edges.
        add_node("claim", sample.claim_text, cached_features["claim"], source=sample.claim_id)
        add_entities("claim:0", sample.claim_text, sample.claim_id)
        for evidence in sample.text_evidence:
            text_index = add_node(
                "evidence_text", evidence.text, cached_features["texts"][evidence.evidence_id],
                source=evidence.evidence_id, evidence_id=evidence.evidence_id,
            )
            add_edge("claim", "has_text", "evidence_text", 0, text_index)
            add_entities(f"evidence_text:{text_index}", evidence.text, evidence.evidence_id)
        for evidence in sample.image_evidence:
            if evidence.image_id not in cached_features.get("images", {}):
                continue  # A decode error is recorded separately; the claim is retained.
            image_index = add_node(
                "evidence_image", evidence.image_id, cached_features["images"][evidence.image_id],
                source=evidence.relative_path, image_id=evidence.image_id,
                relative_path=evidence.relative_path,
            )
            add_edge("claim", "has_image", "evidence_image", 0, image_index)
            extraction = cached_features.get("image_extractions", {}).get(evidence.image_id, {})
            concept_groups = (
                ("objects", "OBJECT"), ("persons", "PERSON"),
                ("landmarks", "LANDMARK"), ("ocr", "OCR"),
            )
            for group, concept_type in concept_groups:
                for literal in dict.fromkeys(extraction.get(group, [])):
                    concept_index = add_node(
                        "visual_concept", literal, cached_features["concept_features"][literal],
                        source=evidence.relative_path, owner=f"evidence_image:{image_index}",
                        subtype=CONCEPT_TYPES[concept_type], concept_type=concept_type,
                    )
                    add_edge("evidence_image", "contains", "visual_concept", image_index, concept_index)

        # Match exact entity names and explicitly parseable dates across owners.
        for source_index, source in enumerate(nodes["entity"]):
            for target_index in range(source_index + 1, len(nodes["entity"])):
                target = nodes["entity"][target_index]
                if source["owner"] == target["owner"] or source["entity_type"] != target["entity_type"]:
                    continue
                if normalize_name(source["text"]) == normalize_name(target["text"]):
                    add_edge("entity", "same_entity", "entity", source_index, target_index)
                if source["entity_type"] == "DATE":
                    source_date = explicit_date(source["text"])
                    target_date = explicit_date(target["text"])
                    if source_date and target_date and source_date[0] == target_date[0]:
                        relation = "same_date" if source_date == target_date else "different_date"
                        add_edge("entity", relation, "entity", source_index, target_index)

        def add_similarity_edges(source_type, relation, target_type,
                                 source_ids=None, target_ids=None, eligible=None):
            if source_ids is None:
                source_ids = range(len(nodes[source_type]))
            if target_ids is None:
                target_ids = range(len(nodes[target_type]))
            for source_index in source_ids:
                candidates = []
                for target_index in target_ids:
                    if source_type == target_type and source_index == target_index:
                        continue
                    if eligible is not None and not eligible(source_index, target_index):
                        continue
                    score = float(torch.dot(
                        features[source_type][source_index], features[target_type][target_index],
                    ))
                    if score > 0:
                        candidates.append((score, target_index))
                ranked_candidates = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))
                for score, target_index in ranked_candidates[:self.top_k]:
                    add_edge(source_type, relation, target_type, source_index, target_index, score)

        # Similarity uses the original cached CLIP vectors, with top-k per source.
        add_similarity_edges(
            "entity", "entity_similarity", "entity",
            eligible=lambda source, target: (
                nodes["entity"][source]["entity_type"] == nodes["entity"][target]["entity_type"]
                and nodes["entity"][source]["owner"] != nodes["entity"][target]["owner"]
            ),
        )
        add_similarity_edges("evidence_image", "visual_similarity", "evidence_image")
        add_similarity_edges("entity", "image_similarity", "evidence_image")
        add_similarity_edges("entity", "concept_similarity", "visual_concept")

        # Store tensors in registry order so batching and relation IDs stay stable.
        for kind in NODE_TYPES:
            if features[kind]:
                graph[kind].x = torch.stack(features[kind])
            else:
                graph[kind].x = torch.empty((0, FEATURE_DIM))
            subtypes = []
            owners = []
            for node in nodes[kind]:
                subtypes.append(node["subtype"])
                if node["owner"]:
                    owners.append(node["owner"].split(":"))
                else:
                    owners.append((kind, node["index"]))
            graph[kind].subtype = torch.tensor(subtypes, dtype=torch.long)
            graph[kind].owner_type = torch.tensor([NODE_TYPES.index(owner[0]) for owner in owners], dtype=torch.long)
            graph[kind].owner_index = torch.tensor([int(owner[1]) for owner in owners], dtype=torch.long)
            graph[kind].num_nodes = len(nodes[kind])
        provenance_edges = []
        for relation_id, edge_type in enumerate(EDGE_TYPES):
            pairs = list(edges[edge_type])
            scores = list(edges[edge_type].values())
            if pairs:
                graph[edge_type].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            else:
                graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph[edge_type].edge_attr = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
            for index, (pair, score) in enumerate(zip(pairs, scores)):
                source_index, target_index = pair
                provenance_edges.append({
                    "id": f"edge:{relation_id}:{index}",
                    "source": f"{edge_type[0]}:{source_index}",
                    "target": f"{edge_type[2]}:{target_index}",
                    "relation": edge_type[1],
                    "score": score,
                    "edge_type": list(edge_type),
                    "index": index,
                })
        graph.claim_id = sample.claim_id
        graph.split = sample.split
        has_text = bool(nodes["evidence_text"])
        has_image = bool(nodes["evidence_image"])
        if has_text and has_image:
            modality = "both"
        elif has_text:
            modality = "text"
        elif has_image:
            modality = "image"
        else:
            modality = "none"
        provenance_nodes = []
        for kind in NODE_TYPES:
            provenance_nodes.extend(nodes[kind])
        self.provenance = {
            "claim_id": sample.claim_id,
            "split": sample.split,
            "claim_text": sample.claim_text,
            "modality": modality,
            "nodes": provenance_nodes,
            "edges": provenance_edges,
            "extraction_errors": cached_features.get("errors", []),
        }
        # validate_graph(graph)
        return graph


def validate_graph(graph):
    for kind in NODE_TYPES:
        store = graph[kind]
        expected_shape = (store.num_nodes, FEATURE_DIM)
        if store.x.shape != expected_shape or not torch.isfinite(store.x).all():
            raise ValueError("Invalid node features for " + kind)
    for source_type, relation, target_type in EDGE_TYPES:
        store = graph[source_type, relation, target_type]
        indices = store.edge_index
        if indices.shape[0] != 2 or store.edge_attr.shape != (indices.shape[1], 1):
            raise ValueError("Invalid edge shape for " + relation)
        if not torch.isfinite(store.edge_attr).all() or (store.edge_attr < 0).any() or (store.edge_attr > 1).any():
            raise ValueError("Invalid edge score for " + relation)
        if indices.numel():
            negative_index = (indices < 0).any()
            invalid_source = indices[0].max() >= graph[source_type].num_nodes
            invalid_target = indices[1].max() >= graph[target_type].num_nodes
            if negative_index or invalid_source or invalid_target:
                raise ValueError("Out-of-range edge for " + relation)

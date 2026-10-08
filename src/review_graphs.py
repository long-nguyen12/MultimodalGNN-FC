"""Export graphs for optional human review and graph-audit annotations."""

import argparse
import csv
import json
from pathlib import Path

from .visualize_graph import render_graph


REVIEW_COLUMNS = (
    "claim_id", "fingerprint", "input_hash", "status", "reviewer", "notes",
    "graph_path", "provenance_path",
)
REVIEW_STATUSES = frozenset(("pending", "approved", "needs_changes"))


def _read_review(path, allow_legacy=False):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = set(REVIEW_COLUMNS)
        if allow_legacy:
            required_columns.remove("input_hash")
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV is missing required columns: {}".format(", ".join(REVIEW_COLUMNS)))
        rows = list(reader)
    for row in rows:
        row.setdefault("input_hash", None)
    claim_ids = {row["claim_id"] for row in rows}
    if len(claim_ids) != len(rows):
        raise ValueError("Review CSV contains duplicate claim IDs")
    return rows


def validate_review(review_csv, claim_ids, fingerprint=None, expected_count=None, input_hashes=None):
    """Validate human approval of the requested graphs, regardless of their count."""
    expected = {str(claim_id) for claim_id in claim_ids}
    if expected_count is not None and len(expected) != expected_count:
        raise ValueError(
            "Expected {} distinct claims, found {}".format(expected_count, len(expected))
        )
    rows = _read_review(review_csv)
    by_id = {row["claim_id"]: row for row in rows}
    if set(by_id) != expected:
        raise ValueError("Review claim IDs do not match the requested graphs")
    pending = []
    for claim_id, row in by_id.items():
        if row["status"] != "approved" or not row["reviewer"].strip():
            pending.append(claim_id)
    if pending:
        raise ValueError(
            "{} graphs require human approval with a reviewer name".format(len(pending))
        )
    if fingerprint is not None and any(row["fingerprint"] != fingerprint for row in rows):
        raise ValueError("Review fingerprint does not match the current graph cache")
    if input_hashes is not None:
        hashes = {str(claim_id): value for claim_id, value in input_hashes.items()}
        if not expected.issubset(hashes):
            raise ValueError("Expected input hashes are missing")
        if any(row["input_hash"] != hashes[row["claim_id"]] for row in rows):
            raise ValueError(
                "Review input hash does not match the current claim inputs; "
                "export and review the changed graphs"
            )
    reviewers = sorted({row["reviewer"].strip() for row in rows})
    return {"approved": len(rows), "fingerprint": fingerprint, "reviewers": reviewers}


def build_review(graph_root, split, output_dir, claim_ids=None):
    graph_root = Path(graph_root)
    output_dir = Path(output_dir)
    manifest = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest["fingerprint"]
    selected_ids = None
    if claim_ids is not None:
        selected_ids = {str(claim_id) for claim_id in claim_ids}
    records = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        if selected_ids is not None and str(record["claim_id"]) not in selected_ids:
            continue
        records.append(record)
    if selected_ids is not None:
        found_ids = {str(record["claim_id"]) for record in records}
        if found_ids != selected_ids:
            raise ValueError("Requested claims are absent from this graph manifest/split")
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "review.csv"
    previous = {}
    if review_path.exists():
        for row in _read_review(review_path, allow_legacy=True):
            previous[row["claim_id"]] = row
    rows = []
    for record in records:
        claim_id = str(record["claim_id"])
        provenance_path = graph_root / record["provenance_path"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        render_graph(provenance, output_dir / "graphs" / (claim_id + ".html"))
        row = {
            "claim_id": claim_id,
            "fingerprint": fingerprint,
            "input_hash": record.get("input_hash", ""),
            "status": "pending",
            "reviewer": "",
            "notes": "",
            "graph_path": record["path"],
            "provenance_path": record["provenance_path"],
        }
        old = previous.get(claim_id)
        if old:
            same_fingerprint = old["fingerprint"] == fingerprint
            same_input = old["input_hash"] is not None and old["input_hash"] == row["input_hash"]
            if same_fingerprint and same_input:
                for key in ("status", "reviewer", "notes"):
                    row[key] = old[key]
        rows.append(row)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    instructions = {
        "fingerprint": fingerprint,
        "claim_ids": [row["claim_id"] for row in rows],
        "graph_stage": manifest.get("metadata", {}).get("graph_stage"),
        "stage_note": (
            "Feature-stage exports contain CLIP evidence nodes only; "
            "full-stage exports also contain extracted entities and visual concepts."
        ),
        "instructions": (
            "Open graphs/<claim_id>.html. Check extraction, source ownership, relations and leakage. "
            "Set status=approved or needs_changes and fill reviewer/notes. "
            "Approval is manual and binds both graph fingerprint and per-claim input hash; "
            "changed inputs require a new review. This export never auto-approves."
        ),
        "annotation_schema": {
            "claim_id": "ID from manifest",
            "entities_gold": [{
                "owner": "claim:0 or evidence_text:N",
                "text": "exact entity text",
                "entity_type": "PERSON|LOCATION|DATE|ORGANIZATION",
            }],
            "concepts_review": [{"node_id": "visual_concept:N", "correct": True}],
            "relations_review": [{"edge_id": "edge:R:E", "correct": True}],
        },
        "annotation_note": (
            "entities_gold must be exhaustive per annotated graph to calculate recall. "
            "Concept/relation metrics reflect only manually reviewed items, with coverage counts reported."
        ),
    }
    instructions_json = json.dumps(instructions, ensure_ascii=False, indent=2)
    (output_dir / "review_instructions.json").write_text(instructions_json, encoding="utf-8")
    return {
        "count": len(rows),
        "review_csv": str(review_path),
        "fingerprint": fingerprint,
        "graph_stage": manifest.get("metadata", {}).get("graph_stage"),
    }


def graph_audit_metrics(provenances, annotations):
    """Compare human gold entities and reviewed concepts/edges; never invent gold."""
    from .schema import SEMANTIC_RELATIONS

    by_id = {str(item["claim_id"]): item for item in provenances}
    counts = {
        "entity_tp": 0,
        "entity_predicted": 0,
        "entity_gold": 0,
        "concept_correct": 0,
        "concept_reviewed": 0,
        "concept_total": 0,
        "matching_correct": 0,
        "matching_reviewed": 0,
        "matching_total": 0,
    }
    seen = set()

    def entity_key(item):
        return (
            item.get("owner"),
            " ".join(item["text"].casefold().split()),
            item["entity_type"].upper(),
        )

    def is_forward_semantic(edge):
        relation = edge["relation"]
        return relation in SEMANTIC_RELATIONS and not relation.startswith("rev_")

    for annotation in annotations:
        claim_id = str(annotation["claim_id"])
        if claim_id in seen:
            raise ValueError("Duplicate annotation for claim " + claim_id)
        seen.add(claim_id)
        provenance = by_id[claim_id]
        nodes = {node["id"]: node for node in provenance["nodes"]}
        edges = {edge["id"]: edge for edge in provenance["edges"]}
        if "entities_gold" in annotation:
            predicted = {
                entity_key(node)
                for node in nodes.values()
                if node["type"] == "entity"
            }
            gold = {entity_key(node) for node in annotation["entities_gold"]}
            counts["entity_tp"] += len(predicted & gold)
            counts["entity_predicted"] += len(predicted)
            counts["entity_gold"] += len(gold)
        counts["concept_total"] += sum(node["type"] == "visual_concept" for node in nodes.values())
        counts["matching_total"] += sum(is_forward_semantic(edge) for edge in edges.values())

        reviewed_concepts = set()
        for item in annotation.get("concepts_review", []):
            node_id = item["node_id"]
            if (
                node_id in reviewed_concepts
                or node_id not in nodes
                or nodes[node_id]["type"] != "visual_concept"
                or type(item["correct"]) is not bool
            ):
                raise ValueError("Invalid or duplicate manual review item " + node_id)
            reviewed_concepts.add(node_id)
            counts["concept_reviewed"] += 1
            counts["concept_correct"] += int(item["correct"])

        reviewed_relations = set()
        for item in annotation.get("relations_review", []):
            edge_id = item["edge_id"]
            if (
                edge_id in reviewed_relations
                or edge_id not in edges
                or not is_forward_semantic(edges[edge_id])
                or type(item["correct"]) is not bool
            ):
                raise ValueError("Invalid or duplicate manual review item " + edge_id)
            reviewed_relations.add(edge_id)
            counts["matching_reviewed"] += 1
            counts["matching_correct"] += int(item["correct"])

    def ratio(numerator, denominator):
        if not denominator:
            return None
        return numerator / denominator

    return {
        "entity_precision": ratio(counts["entity_tp"], counts["entity_predicted"]),
        "entity_recall": ratio(counts["entity_tp"], counts["entity_gold"]),
        "visual_concept_accuracy": ratio(counts["concept_correct"], counts["concept_reviewed"]),
        "matching_precision": ratio(counts["matching_correct"], counts["matching_reviewed"]),
        "concept_review_coverage": ratio(counts["concept_reviewed"], counts["concept_total"]),
        "matching_review_coverage": ratio(counts["matching_reviewed"], counts["matching_total"]),
        "counts": counts,
        "annotated_claims": len(seen),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--annotations", help="Human annotation JSON list for graph-audit metrics")
    args = parser.parse_args(argv)
    graph_root = Path(args.graph_root)
    manifest = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == args.split]
    if args.validate:
        input_hashes = {
            str(record["claim_id"]): record["input_hash"]
            for record in records
        }
        result = validate_review(
            Path(args.output_dir) / "review.csv",
            [record["claim_id"] for record in records],
            manifest["fingerprint"],
            input_hashes=input_hashes,
        )
    else:
        result = build_review(graph_root, args.split, args.output_dir)
    if args.annotations:
        annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
        provenances = []
        for record in records:
            provenance_path = graph_root / record["provenance_path"]
            provenances.append(json.loads(provenance_path.read_text(encoding="utf-8")))
        metrics = graph_audit_metrics(provenances, annotations)
        metrics_path = Path(args.output_dir) / "graph_audit_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        result["audit_metrics"] = metrics
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

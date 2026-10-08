"""The fixed V1 graph registry shared by builders, models and explanations."""

FEATURE_DIM = 512
NODE_TYPES = ("claim", "evidence_text", "evidence_image", "entity", "visual_concept")
ENTITY_TYPES = {"UNKNOWN": 0, "PERSON": 1, "LOCATION": 2, "DATE": 3, "ORGANIZATION": 4}
CONCEPT_TYPES = {"UNKNOWN": 0, "OBJECT": 5, "PERSON": 6, "LANDMARK": 7, "OCR": 8}
FORWARD_EDGE_TYPES = (
    ("claim", "has_text", "evidence_text"),
    ("claim", "has_image", "evidence_image"),
    ("claim", "mentions", "entity"),
    ("evidence_text", "mentions", "entity"),
    ("evidence_image", "contains", "visual_concept"),
    ("entity", "same_entity", "entity"),
    ("entity", "entity_similarity", "entity"),
    ("entity", "same_date", "entity"),
    ("entity", "different_date", "entity"),
    ("evidence_image", "visual_similarity", "evidence_image"),
    ("entity", "image_similarity", "evidence_image"),
    ("entity", "concept_similarity", "visual_concept"),
)
# Relation IDs depend on this order: each forward edge precedes its reverse.
_edge_types = []
for _source_type, _relation, _target_type in FORWARD_EDGE_TYPES:
    _edge_types.append((_source_type, _relation, _target_type))
    _edge_types.append((_target_type, "rev_" + _relation, _source_type))
EDGE_TYPES = tuple(_edge_types)

STRUCTURAL_RELATIONS = frozenset(
    ("has_text", "rev_has_text", "has_image", "rev_has_image",
     "mentions", "rev_mentions", "contains", "rev_contains")
)
SIMILARITY_RELATIONS = frozenset(
    ("entity_similarity", "rev_entity_similarity",
     "visual_similarity", "rev_visual_similarity",
     "image_similarity", "rev_image_similarity",
     "concept_similarity", "rev_concept_similarity")
)
CROSS_MODAL_RELATIONS = frozenset(
    ("image_similarity", "rev_image_similarity", "concept_similarity", "rev_concept_similarity")
)
SEMANTIC_RELATIONS = frozenset(rel for _, rel, _ in EDGE_TYPES) - STRUCTURAL_RELATIONS

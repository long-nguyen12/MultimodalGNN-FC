"""Claim-level MOCHEG inputs with targets kept outside the feature pipeline."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


LABELS = ("supported", "refuted", "NEI")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TextEvidence:
    evidence_id: str
    text: str


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    relative_path: str


@dataclass(frozen=True)
class SampleInput:
    claim_id: str
    split: str
    claim_text: str
    text_evidence: tuple[TextEvidence, ...]
    image_evidence: tuple[ImageEvidence, ...]


@dataclass(frozen=True)
class Target:
    label: int
    ruling_outline: str


class _EvidenceHTMLParser(HTMLParser):
    _blocks = {"p", "div", "br", "li", "blockquote", "section", "tr", "h1", "h2", "h3"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if tag in self._blocks and not self.hidden_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        if tag in self._blocks and not self.hidden_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden_depth:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """Remove markup while preserving quoted text, dates, and paragraph breaks."""
    parser = _EvidenceHTMLParser()
    parser.feed(text)
    parser.close()
    plain_text = "".join(parser.parts)
    paragraphs = []
    for line in plain_text.splitlines():
        paragraph = re.sub(r"\s+", " ", line).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs)


def normalize_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Missing identifier")
    if re.fullmatch(r"\d+(?:\.0+)?", value):
        return str(int(value.split(".")[0]))
    return value


def modality_group(sample: SampleInput) -> str:
    if sample.text_evidence and sample.image_evidence:
        return "multimodal"
    if sample.text_evidence:
        return "text_only"
    if sample.image_evidence:
        return "image_only"
    return "no_evidence"


def _read_csv(path: Path, required_fields: set[str]):
    # Several MOCHEG explanation fields exceed csv's default 128 KiB limit.
    csv.field_size_limit(128 * 1024 * 1024)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in required_fields):
                raise ValueError(f"{path}:{line_number}: malformed CSV row")
            yield line_number, row


class MochegDataset:
    """Group Corpus2 rows by claim and attach gold images from qrels evidence_id.

    Image files are not opened by the loader. Missing or corrupt images remain
    associated with their claim so the audit and extraction can report them.
    """

    def __init__(self, root: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        self.root = Path(root)
        self.split = split
        self.warnings: list[str] = []
        self.corpus_rows = 0
        self.qrels_rows = 0
        self.original_truthfulness = Counter()
        # Collect claim fields and gold text, keeping the original claim order.
        claims_by_id = {}
        texts_by_claim = defaultdict(dict)
        corpus_path = self.root / split / "Corpus2.csv"
        required = {
            "claim_id", "evidence_id", "Claim", "Evidence",
            "cleaned_truthfulness", "ruling_outline",
        }
        label_index = {name.casefold(): index for index, name in enumerate(LABELS)}
        for line_number, row in _read_csv(corpus_path, required):
            self.corpus_rows += 1
            claim_id = normalize_id(row["claim_id"])
            claim_text = row["Claim"].strip()
            
            label_name = row["cleaned_truthfulness"].strip().casefold()
            
            label = label_index[label_name]
            ruling_outline = row["ruling_outline"].strip()
            claim_fields = (claim_text, label, ruling_outline)
            if claim_id in claims_by_id and claims_by_id[claim_id] != claim_fields:
                field_names = ("Claim", "cleaned_truthfulness", "ruling_outline")
                previous_fields = claims_by_id[claim_id]
                differing = []
                for field_name, previous, current in zip(field_names, previous_fields, claim_fields):
                    if previous != current:
                        differing.append(field_name)
                raise ValueError(f"{corpus_path}:{line_number}: inconsistent {', '.join(differing)} for claim {claim_id}")
            if claim_id not in claims_by_id:
                self.original_truthfulness[row.get("Truthfulness", "").strip()] += 1
                claims_by_id[claim_id] = claim_fields
            evidence_id = normalize_id(row["evidence_id"])
            text = strip_html(row["Evidence"])
            claim_texts = texts_by_claim[claim_id]
            
            if text:
                claim_texts[evidence_id] = text

        # Attach distinct gold filenames from qrels to their owning claims.
        images_by_claim = defaultdict(dict)
        image_owners = defaultdict(set)
        qrels_path = self.root / split / "img_evidence_qrels.csv"
        for line_number, row in _read_csv(qrels_path, {"TOPIC", "evidence_id"}):
            self.qrels_rows += 1
            claim_id = normalize_id(row["TOPIC"])
            # The gold filename must preserve zero padding; DOCUMENT# is a
            # retrieval candidate, and RELEVANCY is not a verification feature.
            filename = row["evidence_id"].strip()
            if (
                not filename or filename in {".", ".."}
                or "/" in filename or "\\" in filename or ":" in filename
            ):
                raise ValueError(f"{qrels_path}:{line_number}: invalid gold image filename {filename!r}")
            if claim_id not in claims_by_id:
                self.warnings.append(f"qrels topic {claim_id} is absent from Corpus2 (row {line_number})")
                continue
            relative_path = (Path(split) / "images" / filename).as_posix()
            images_by_claim[claim_id][filename] = ImageEvidence(filename, relative_path)
            image_owners[filename].add(claim_id)
        for filename, owners in image_owners.items():
            if len(owners) > 1:
                self.warnings.append(f"gold image {filename} is assigned to multiple claims: {', '.join(sorted(owners))}")

        # Return one input and a separate target for each claim.
        self.claim_ids = list(claims_by_id)
        self.samples = []
        self.targets = []
        for claim_id in self.claim_ids:
            claim_text, label, ruling_outline = claims_by_id[claim_id]
            text_evidence = []
            for evidence_id, text in texts_by_claim[claim_id].items():
                text_evidence.append(TextEvidence(evidence_id, text))
            image_evidence = tuple(images_by_claim[claim_id].values())
            sample = SampleInput(
                claim_id, split, claim_text, tuple(text_evidence), image_evidence,
            )
            self.samples.append(sample)
            self.targets.append(Target(label, ruling_outline))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]



"""Portable, offline graph inspection from provenance (no model required)."""

import argparse
import html
import json
from pathlib import Path


COLORS = {
    "claim": "#bbdefb",
    "evidence_text": "#c8e6c9",
    "evidence_image": "#ffe0b2",
    "entity": "#e1bee7",
    "visual_concept": "#ffccbc",
}


def render_graph(provenance, output_path, explanation=None):
    """Write self-contained HTML; the supplied provenance is never modified."""
    nodes = provenance.get("nodes", [])
    edges = provenance.get("edges", [])
    selected_edges = set((explanation or {}).get("edge_ids", []))

    # Place each node type in a column, keeping provenance order within it.
    groups = {}
    for node_type in COLORS:
        groups[node_type] = []
        for node in nodes:
            if node["type"] == node_type:
                groups[node_type].append(node)
    largest_group = max((len(group) for group in groups.values()), default=1)
    height = max(460, largest_group * 68 + 120)
    positions = {}
    for column, (kind, group) in enumerate(groups.items()):
        for row, node in enumerate(group):
            positions[node["id"]] = (130 + column * 270, 85 + row * 68)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1370" height="{}" '
        'role="img" aria-label="Evidence graph">'.format(height)
    ]
    svg.append(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>'
    )

    # Draw edges first so node labels remain above them.
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        source_x, source_y = positions[edge["source"]]
        target_x, target_y = positions[edge["target"]]
        control_x = (source_x + target_x) / 2
        control_y = (source_y + target_y) / 2 - 18
        highlighted = edge["id"] in selected_edges
        color = "#dc2626" if highlighted else "#94a3b8"
        width = 3 if highlighted else 1
        title = "{}: {} → {} (score {:.4f})".format(
            edge["relation"], edge["source"], edge["target"],
            float(edge.get("score", 1)),
        )
        edge_svg = (
            '<path d="M{source_x},{source_y} Q{control_x},{control_y} {target_x},{target_y}" '
            'stroke="{color}" stroke-width="{width}" fill="none" marker-end="url(#arrow)">'
            '<title>{title}</title></path>'
        ).format(
            source_x=source_x, source_y=source_y,
            control_x=control_x, control_y=control_y,
            target_x=target_x, target_y=target_y,
            color=color, width=width, title=html.escape(title),
        )
        svg.append(edge_svg)
    for kind, group in groups.items():
        for node in group:
            x, y = positions[node["id"]]
            text = str(node.get("text", node["id"]))
            title = json.dumps(node, ensure_ascii=False, indent=2)
            short_text = text
            if len(text) > 26:
                short_text = text[:23] + "…"
            node_svg = (
                '<g><title>{title}</title><rect x="{rect_x}" y="{rect_y}" '
                'width="230" height="46" rx="8" fill="{color}" stroke="#475569"/>'
                '<text x="{x}" y="{label_y}" text-anchor="middle" font-size="12">'
                '{label}</text><text x="{x}" y="{id_y}" text-anchor="middle" '
                'font-size="11" fill="#475569">{node_id}</text></g>'
            ).format(
                title=html.escape(title), rect_x=x-115, rect_y=y-23,
                color=COLORS[kind], x=x, label_y=y-3,
                label=html.escape(short_text), id_y=y+13,
                node_id=html.escape(node["id"]),
            )
            svg.append(node_svg)
    svg.append("</svg>")
    legend_items = []
    for kind, color in COLORS.items():
        legend_items.append(
            '<span style="background:{};padding:5px 9px;border-radius:5px">{}</span>'
            .format(color, kind)
        )
    legend = " ".join(legend_items)

    # Escape every provenance field before inserting it into the HTML tables.
    node_rows = []
    for node in nodes:
        cells = []
        for key in ("id", "text", "owner", "source"):
            cells.append(html.escape(str(node.get(key, ""))))
        node_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(*cells)
        )
    edge_rows = []
    for edge in edges:
        edge_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.4f}</td></tr>"
            .format(
                html.escape(edge["id"]), html.escape(edge["source"]),
                html.escape(edge["relation"]), html.escape(edge["target"]),
                float(edge.get("score", 1)),
            )
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Graph review: {claim_id}</title>\n'
        '<style>body{{font:14px system-ui;margin:24px;color:#172033}}'
        '.graph{{overflow:auto;border:1px solid #ddd}}'
        'table{{border-collapse:collapse;width:100%}}'
        'td,th{{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}}'
        'pre{{white-space:pre-wrap;overflow-wrap:anywhere}}'
        'summary{{cursor:pointer;font-weight:bold;margin:20px 0}}'
        'h1{{font-size:23px}}</style>\n'
        '<h1>Claim {claim_id} ({split})</h1><p>{claim_text}</p><p>{legend}</p>'
        '<p>Hover over nodes and edges for source details. '
        'Red edges belong to the selected explanation.</p>\n'
        '<div class="graph">{svg}</div><details><summary>Nodes and ownership</summary>'
        '<table><tr><th>ID</th><th>Content</th><th>Owner</th><th>Source</th></tr>'
        '{node_rows}</table></details>\n'
        '<details><summary>Relations and scores</summary><table><tr>'
        '<th>ID</th><th>Source</th><th>Relation</th><th>Target</th><th>Score</th></tr>'
        '{edge_rows}</table></details>\n'
        '<details><summary>Full provenance</summary>'
        '<pre>{provenance}</pre></details></html>'
    ).format(
        claim_id=html.escape(str(provenance.get("claim_id", ""))),
        split=html.escape(str(provenance.get("split", ""))),
        claim_text=html.escape(str(provenance.get("claim_text", ""))),
        legend=legend,
        svg="".join(svg),
        node_rows="".join(node_rows),
        edge_rows="".join(edge_rows),
        provenance=html.escape(json.dumps(provenance, ensure_ascii=False, indent=2)),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--explanation")
    args = parser.parse_args(argv)
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    explanation = None
    if args.explanation:
        explanation = json.loads(Path(args.explanation).read_text(encoding="utf-8"))
    render_graph(provenance, args.output, explanation)


if __name__ == "__main__":
    main()

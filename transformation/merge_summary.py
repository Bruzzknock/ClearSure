import json
from pathlib import Path
from typing import Dict, Any

from kg_utils import update_kg


def tree_to_kg(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a summary tree into a minimal KG dict."""
    nodes = []
    edges = []
    counter = [0]

    def _walk(node: Dict[str, Any], parent_id: str | None = None) -> None:
        counter[0] += 1
        sid = f"s{counter[0]}"
        label = node.get("summary") or ""
        nodes.append({"id": sid, "label": label, "type": "Summary"})
        if parent_id:
            edges.append({"source": parent_id, "relation": "HAS_CHILD", "target": sid})
        for child in node.get("children", []):
            if isinstance(child, dict):
                _walk(child, sid)

    _walk(tree)
    return {"nodes": nodes, "edges": edges}


def merge_summary_into_kg(
    summary_tree_path: Path,
    kg_path: Path,
    out_path: Path | None = None,
) -> Dict[str, Any]:
    """Merge *summary_tree_path* into the KG at *kg_path*."""
    tree = json.loads(summary_tree_path.read_text(encoding="utf-8"))
    summary_kg = tree_to_kg(tree)
    merged_kg = update_kg(summary_kg, kg_path=kg_path, save=True)
    if out_path:
        out_path.write_text(json.dumps(merged_kg, indent=2), encoding="utf-8")
    return merged_kg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge sentence-level KG with summary tree KG"
    )
    parser.add_argument("summary", type=Path, help="Path to summary tree JSON")
    parser.add_argument("kg", type=Path, help="Path to sentence-level KG JSON")
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional path for merged KG"
    )
    args = parser.parse_args()

    merge_summary_into_kg(args.summary, args.kg, args.out)

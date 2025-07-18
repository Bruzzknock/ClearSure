# kg_utils.py
# --- add near the top -------------------------------------------------------
import json, re, os
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
from copy import deepcopy

_EDGE = Tuple[str, str, str]

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_ID_PREFIX_RE = re.compile(r"^([nsw])(\d+)$")
_EDGE_ID_RE = re.compile(r"^e(\d+)$")
_BRACKET_REF_RE = re.compile(r"\[(n\d+|s\d+|w\d+)\]")


def _strip_fence(text: str) -> str:
    """Remove ```json fences and surrounding blank lines."""
    return _FENCE_RE.sub("", text).strip()


# Helpers ────────────────────────────────────────────────────────────────────────
_BRACE_RE = re.compile(r"\{.*\}", re.S)


def _extract_json_block(text: str) -> str:
    """
    Return the first well-balanced {...} block in *text*.
    Raises ValueError if none is found or braces are unbalanced.
    """
    text = _strip_fence(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No opening '{' found in patch string.", text)

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced braces in patch string.")


def _rewrite_bracket_refs(text: str, id_map: dict[str, str]) -> str:
    """
    Replace every [s1]/[n3]/[w4] in *text* with the *new* ID
    according to *id_map*.
    """

    def _repl(m):
        old = m.group(1)
        return f"[{id_map.get(old, old)}]"

    return _BRACKET_REF_RE.sub(_repl, text)


def _load_patch(
    patch: Union[str, Dict[str, Any]], objectType="edges_patch"
) -> List[Dict[str, Any]]:
    """
    Accepts …
      • dict                      -> returns it directly
      • path-like str             -> reads the file
      • objectType                -> what we want to merge
      • raw JSON string (with / without ``` fences, with / without chatter)

    Returns list[edge] from the `objectType` key.
    """
    if isinstance(patch, dict):
        return patch.get(objectType, [])

    # Treat as file path if it exists
    p = Path(patch)
    patch_str = p.read_text(encoding="utf-8") if p.exists() else patch

    patch_str = _strip_fence(patch_str)

    # ① try straight JSON
    try:
        obj = json.loads(patch_str)
    except json.JSONDecodeError:
        # ② fall back to extracting the first {...} block
        obj = json.loads(_extract_json_block(patch_str))

    return obj.get(objectType, [])


def _dedupe_edges(edges: list[dict]) -> set[tuple[str, str, str]]:
    """Return a set of (source, relation, target) triples."""
    return {(e["source"], e["relation"], e["target"]) for e in edges}


def _dedupe_nodes(nodes: list[dict]) -> set[str]:
    """Return a set of node IDs."""
    return {n["id"] for n in nodes}


def clean_kg(
    patch: Union[str, Dict[str, Any]],
    kg_path: str | os.PathLike = "final_kg.json",
    save: bool = True,
    indent: int = 2,
    *,
    id_map: Optional[Dict[str, str]] = None,
    reassign_edge_ids: bool = True,
    drop_missing: bool = True,
) -> Dict[str, Any]:
    """
    Merge an *edges_patch* into the KG at *kg_path*.

    Parameters
    ----------
    patch
        Dict or JSON/string containing an `edges_patch` list.
    id_map
        Old-ID→new-ID map returned by `update_kg()` for the *same* sentence.
        Used to rewrite e["source"]/e["target"] so they match renumbered nodes.
    reassign_edge_ids
        If True, give each incoming edge a fresh sequential edgeId.
    drop_missing
        If True, skip edges whose mapped endpoints are not present in the KG.

    Returns the updated KG dict.
    """
    kg_path = Path(kg_path)
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in kg["nodes"]}

    new_edges = deepcopy(_load_patch(patch))  # defensive copy

    # Rewrite source/target using id_map (if provided)
    if id_map:
        for e in new_edges:
            e["source"] = id_map.get(e["source"], e["source"])
            e["target"] = id_map.get(e["target"], e["target"])

    # Reassign edgeIds
    if reassign_edge_ids:
        counter = [_max_edge_index(kg["edges"])]
        for e in new_edges:
            e["edgeId"] = _next_edge_id(counter)

    # Optionally drop edges pointing to unknown nodes
    if drop_missing:
        new_edges = [
            e for e in new_edges if e["source"] in node_ids and e["target"] in node_ids
        ]

    # Deduplicate + append
    seen = _dedupe_edges(kg["edges"])
    for e in new_edges:
        triple = (e["source"], e["relation"], e["target"])
        if triple not in seen:
            kg["edges"].append(e)
            seen.add(triple)

    if save:
        kg_path.write_text(json.dumps(kg, indent=indent), encoding="utf-8")
    return kg


def update_kg(
    new_kg: Union[str, Dict[str, Any]],
    kg_path: str | os.PathLike = "final_kg.json",
    save: bool = True,
    indent: int = 2,
    *,
    return_id_map: bool = False,
):
    """
    Merge *new_kg* (nodes + raw edges) into the KG at *kg_path*.

    Renumbers n#/s#/w# style IDs and rewrites edges accordingly.
    Also rewrites bracket refs in node labels.

    If `return_id_map=True`, returns (kg, id_map); else just kg.
    """
    kg_path = Path(kg_path)
    kg = json.loads(kg_path.read_text(encoding="utf-8"))

    patch_nodes = _load_patch(new_kg, "nodes")
    patch_edges = _load_patch(new_kg, "edges")

    node_counters = {p: _max_index(kg["nodes"], p) for p in ("n", "s", "w")}
    edge_counter = [_max_edge_index(kg["edges"])]

    id_map: Dict[str, str] = {}
    for node in patch_nodes:
        m = _ID_PREFIX_RE.match(node["id"])
        if m:
            prefix = m.group(1)
            node_counters[prefix] += 1
            new_id = f"{prefix}{node_counters[prefix]}"
            id_map[node["id"]] = new_id
            node["id"] = new_id
        else:
            id_map[node["id"]] = node["id"]

    # Fix bracket references in labels
    for node in patch_nodes:
        lbl = node.get("label")
        if isinstance(lbl, str):
            node["label"] = _rewrite_bracket_refs(lbl, id_map)

    # Rewrite edges + assign new edgeIds
    for edge in patch_edges:
        edge["source"] = id_map.get(edge["source"], edge["source"])
        edge["target"] = id_map.get(edge["target"], edge["target"])
        edge["edgeId"] = _next_edge_id(edge_counter)

    # Deduplicate + append
    seen_nodes = _dedupe_nodes(kg["nodes"])
    for n in patch_nodes:
        if n["id"] not in seen_nodes:
            kg["nodes"].append(n)
            seen_nodes.add(n["id"])

    seen_edges = _dedupe_edges(kg["edges"])
    for e in patch_edges:
        trip = (e["source"], e["relation"], e["target"])
        if trip not in seen_edges:
            kg["edges"].append(e)
            seen_edges.add(trip)

    if save:
        kg_path.write_text(json.dumps(kg, indent=indent), encoding="utf-8")

    return (kg, id_map) if return_id_map else kg


def _max_index(nodes: list[dict], prefix: str) -> int:
    idx = [
        int(m.group(2))
        for n in nodes
        if (m := _ID_PREFIX_RE.match(n["id"])) and m.group(1) == prefix
    ]
    return max(idx, default=0)


def _max_edge_index(edges: list[dict]) -> int:
    idx = [
        int(m.group(1))
        for e in edges
        if "edgeId" in e
        if (m := _EDGE_ID_RE.match(e["edgeId"]))
    ]
    return max(idx, default=0)


def _next_edge_id(counter: list[int]) -> str:
    """counter is a 1-item list so we can mutate it inside a loop."""
    counter[0] += 1
    return f"e{counter[0]}"


def _fresh_id(old_id: str, counters: dict[str, int]) -> str:
    """Given 'n1' etc. return next free 'nX' and increment counter."""
    prefix = old_id[0]
    counters[prefix] += 1
    return f"{prefix}{counters[prefix]}"


def merge_duplicate_nodes(kg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nodes with identical label and type, updating edges."""
    mapping: Dict[str, str] = {}
    label_map: Dict[tuple, str] = {}
    new_nodes: List[dict] = []

    for node in kg.get("nodes", []):
        key = (node.get("label"), node.get("type"))
        existing = label_map.get(key)
        if existing:
            mapping[node["id"]] = existing
        else:
            label_map[key] = node["id"]
            new_nodes.append(node)
    kg["nodes"] = new_nodes

    for edge in kg.get("edges", []):
        edge["source"] = mapping.get(edge["source"], edge["source"])
        edge["target"] = mapping.get(edge["target"], edge["target"])

    seen: set[_EDGE] = set()
    dedup_edges: List[dict] = []
    for e in kg.get("edges", []):
        trip = (e["source"], e["relation"], e["target"])
        if trip not in seen:
            dedup_edges.append(e)
            seen.add(trip)
    kg["edges"] = dedup_edges
    return kg

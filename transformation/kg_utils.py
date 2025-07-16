# kg_utils.py
# --- add near the top -------------------------------------------------------
import json, re, os
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
from copy import deepcopy

_EDGE = Tuple[str, str, str]

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_ID_PREFIX_RE  = re.compile(r"^([nsw])(\d+)$")
_EDGE_ID_RE    = re.compile(r"^e(\d+)$")

def _strip_fence(text: str) -> str:
    """Remove ```json fences and surrounding blank lines."""
    return _FENCE_RE.sub("", text).strip()


# NEW ▶────────────────────────────────────────────────────────────────────────
_BRACE_RE = re.compile(r"\{.*\}", re.S)

def _extract_json_block(text: str) -> str:
    """
    Return the first well-balanced {...} block in *text*.
    Raises ValueError if none is found or braces are unbalanced.
    """
    text = _strip_fence(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No opening '{' found in patch string.")

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced braces in patch string.")
# ◀────────────────────────────────────────────────────────────────────────────


def _load_patch(patch: Union[str, Dict[str, Any]], objectType = "edges_patch") -> List[Dict[str, Any]]:
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
) -> Dict[str, Any]:
    """
    Merge *patch* into the KG stored at *kg_path*.

    Parameters
    ----------
    patch : str | dict
        • dict with `"edges_patch"` or
        • path / JSON string (optionally wrapped in ``` fences)
    kg_path : str | PathLike
        Location of the KG JSON file (read & optionally overwritten).
    save : bool
        Persist the updated KG back to *kg_path* (default True).
    indent : int
        JSON indentation level for pretty saving.

    Returns
    -------
    dict
        The updated knowledge-graph object.
    """
    kg_path = Path(kg_path)
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    new_edges = _load_patch(patch)
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
) -> Dict[str, Any]:
    """
    Merge *new_kg* into the KG at *kg_path*.

    * Every node that starts with n#/s#/w# gets a fresh sequential ID
      (continuing from whatever is already in the KG).
    * All edges in the patch are rewritten to use those new node IDs.
    * Each edge receives a new sequential `edgeId` ("e5", "e6", …).
    * Duplicate nodes / edges are ignored.
    """
    kg_path = Path(kg_path)
    kg      = json.loads(kg_path.read_text(encoding="utf-8"))

    # ── pull lists from the patch ----------------------------------------
    patch_nodes = _load_patch(new_kg, "nodes")
    patch_edges = _load_patch(new_kg, "edges")

    # ── counters for next IDs -------------------------------------------
    node_counters = {p: _max_index(kg["nodes"], p) for p in ("n", "s", "w")}
    edge_counter  = [_max_edge_index(kg["edges"])]          # list = mutable int

    # ── build mapping old_id ➜ new_id for *all* incoming nodes ----------
    id_map: Dict[str, str] = {}
    for node in patch_nodes:
        m = _ID_PREFIX_RE.match(node["id"])
        if m:                                              # n#/s#/w# → renumber
            prefix = m.group(1)
            node_counters[prefix] += 1
            new_id = f"{prefix}{node_counters[prefix]}"
            id_map[node["id"]] = new_id
            node["id"] = new_id
        else:                                              # custom/external ID
            id_map[node["id"]] = node["id"]

    # ── rewrite *all* edges and give new edgeIds ------------------------
    for edge in patch_edges:
        edge["source"]  = id_map.get(edge["source"], edge["source"])
        edge["target"]  = id_map.get(edge["target"], edge["target"])
        edge["edgeId"]  = _next_edge_id(edge_counter)

    # ── dedupe & append --------------------------------------------------
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

    # ── save & return ----------------------------------------------------
    if save:
        kg_path.write_text(json.dumps(kg, indent=indent), encoding="utf-8")
    return kg

def _max_index(nodes: list[dict], prefix: str) -> int:
    idx = [int(m.group(2))
           for n in nodes
           if (m := _ID_PREFIX_RE.match(n["id"])) and m.group(1) == prefix]
    return max(idx, default=0)

def _max_edge_index(edges: list[dict]) -> int:
    idx = [int(m.group(1))
           for e in edges if "edgeId" in e
           if (m := _EDGE_ID_RE.match(e["edgeId"]))]
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
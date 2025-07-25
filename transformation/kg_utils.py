# kg_utils.py
# --- add near the top -------------------------------------------------------
import json, re, os
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
from copy import deepcopy

_EDGE = Tuple[str, str, str]

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_ID_PREFIX_RE = re.compile(r"^([nswrt])(\d+)$")
_EDGE_ID_RE = re.compile(r"^e(\d+)$")
_BRACKET_REF_RE = re.compile(r"\[(n\d+|s\d+|w\d+|r\d+|t\d+)\]")


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
    Merge an *edges_patch* (and optional ``nodes`` list) into the KG at
    *kg_path*.

    Parameters
    ----------
    patch
        Dict or JSON/string containing an ``edges_patch`` list and optionally a
        ``nodes`` list.
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
    new_nodes = deepcopy(_load_patch(patch, "nodes"))

    # Deduplicate and append nodes first so edges don't get dropped
    if new_nodes:
        seen_nodes = _dedupe_nodes(kg["nodes"])
        for n in new_nodes:
            if n["id"] not in seen_nodes:
                kg["nodes"].append(n)
                node_ids.add(n["id"])
                seen_nodes.add(n["id"])

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

    node_counters = {p: _max_index(kg["nodes"], p) for p in ("n", "s", "w", "r", "t")}
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


def causal_edges_to_rules(
    kg: Dict[str, Any], *, remove_causal_edges: bool = False, source_doc: str | None = None
) -> Dict[str, Any]:
    """Convert ``CAUSES`` edges into explicit Rule nodes.

    Each ``CAUSES`` edge ``sA -> sB`` becomes a new ``Rule`` node ``rK`` with
    ``HAS_CONDITION`` and ``HAS_CONCLUSION`` links to the respective statements.

    Parameters
    ----------
    kg : dict
        Knowledge graph in the internal JSON format.
    remove_causal_edges : bool, optional
        If ``True``, drop the original ``CAUSES`` edges after conversion.
    source_doc : str, optional
        Optional source document identifier stored on the Rule node.

    Returns
    -------
    dict
        The updated KG.
    """

    rule_idx = _max_index(kg.get("nodes", []), "r")
    new_nodes: list[dict] = []
    new_edges: list[dict] = []
    keep_edges: list[dict] = []

    for e in kg.get("edges", []):
        if e.get("relation") == "CAUSES":
            rule_idx += 1
            rid = f"r{rule_idx}"
            label = f"IF [{e['source']}] THEN [{e['target']}]"
            node = {"id": rid, "label": label, "type": "Rule"}
            if source_doc is not None:
                node["sourceDoc"] = source_doc
            new_nodes.append(node)
            new_edges.append({"source": rid, "relation": "HAS_CONDITION", "target": e["source"]})
            new_edges.append({"source": rid, "relation": "HAS_CONCLUSION", "target": e["target"]})
            if not remove_causal_edges:
                keep_edges.append(e)
        else:
            keep_edges.append(e)

    kg.setdefault("nodes", []).extend(new_nodes)
    kg["edges"] = keep_edges + new_edges
    return kg


def consolidate_rules_to_topics(kg: Dict[str, Any]) -> Dict[str, Any]:
    """Connect Rule nodes to their common Topic and remove redundant edges.

    If all statements referenced by a Rule belong to the same Topic via
    ``HAS_STATEMENT`` edges, the Rule itself is linked to that Topic and the
    individual ``HAS_STATEMENT`` edges from the Topic to those statements are
    removed.  This keeps the topic tree compact while preserving the logical
    structure between statements and rules.
    """

    # Map statement -> list of topic ids
    stmt_topics: Dict[str, set[str]] = {}
    for e in kg.get("edges", []):
        if e.get("relation") == "HAS_STATEMENT":
            stmt_topics.setdefault(e["target"], set()).add(e["source"])

    # Collect rule information
    rule_nodes = {n["id"] for n in kg.get("nodes", []) if n.get("type") == "Rule"}
    rule_refs: Dict[str, set[str]] = {rid: set() for rid in rule_nodes}

    for e in kg.get("edges", []):
        if e.get("source") in rule_nodes and e.get("relation") in {
            "HAS_CONDITION",
            "HAS_CONCLUSION",
        }:
            rule_refs[e["source"]].add(e["target"])

    # Prepare edge lookup for removals
    existing_edges = {(e["source"], e["relation"], e["target"]): e for e in kg.get("edges", [])}

    next_edge_idx = _max_edge_index(kg.get("edges", []))
    new_edges: list[dict] = []
    remove_keys: set[tuple[str, str, str]] = set()

    for rid, stmts in rule_refs.items():
        topics: set[str] = set()
        for sid in stmts:
            topics.update(stmt_topics.get(sid, set()))
        if len(topics) == 1:
            topic = next(iter(topics))
            next_edge_idx += 1
            new_edges.append(
                {"source": topic, "relation": "HAS_STATEMENT", "target": rid, "edgeId": f"e{next_edge_idx}"}
            )
            for sid in stmts:
                key = (topic, "HAS_STATEMENT", sid)
                if key in existing_edges:
                    remove_keys.add(key)

    if remove_keys or new_edges:
        kg["edges"] = [
            e for e in kg.get("edges", []) if (e["source"], e["relation"], e["target"]) not in remove_keys
        ] + new_edges

    return kg

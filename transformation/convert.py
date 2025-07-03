import json
from pathlib import Path

def clean_relation(s):
    return s.upper().replace(" ", "_").replace("-", "_")

BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "structured"
input = file_path / "final_kg_o3.json"
output = file_path / "import_kg.cypher"

# Load your merged knowledge graph
with open(input, "r", encoding="utf-8") as f:
    kg = json.load(f)

# Helper to clean labels
def escape(s):
    return s.replace('"', '\\"')

cypher_nodes = []
cypher_edges = []

# Use a set to track all created nodes (avoid duplicate CREATEs)
node_ids = set()

# Here: NO "for graph in kg"
for node in kg["nodes"]:
    nid = node["id"]
    label = escape(node["label"])

    # Avoid duplicating the same node
    if nid in node_ids:
        continue
    node_ids.add(nid)

    props = {k: v for k, v in node.items() if k not in ["id", "label"]}
    # Also handle nested attributes
    if "attributes" in props:
        attrs = props.pop("attributes")
        props.update(attrs)

    prop_str = ""
    if props:
        prop_pairs = [f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' for k, v in props.items()]
        prop_str = ", " + ", ".join(prop_pairs)

    cypher_nodes.append(
        f'CREATE (:Entity {{id: "{nid}", label: "{label}"{prop_str}}});'
    )

for edge in kg["edges"]:
    src = edge["source"]
    tgt = edge["target"]
    rel = clean_relation(escape(edge["relation"]))
    attr_str = ""
    if "attributes" in edge and edge["attributes"]:
        attr_pairs = [f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' for k, v in edge["attributes"].items()]
        attr_str = " { " + ", ".join(attr_pairs) + " }"

    cypher_edges.append(
        f'''
MATCH (a {{id: "{src}"}}), (b {{id: "{tgt}"}})
CREATE (a)-[:{rel}{attr_str}]->(b);
'''
    )

# Write Cypher script to file
with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(cypher_nodes + cypher_edges))
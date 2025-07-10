# run_pipeline.py
import json, pathlib, itertools
from neo4j import GraphDatabase
from convert import clean_relation, escape               # reuse your helpers

KG_PATH = pathlib.Path("structured/final_kg.json")
BOLT_URI = "bolt://localhost:7687"
driver   = GraphDatabase.driver(BOLT_URI, auth=("neo4j", "your_password"))

def kg_to_statements(kg):
    for n in kg["nodes"]:
        props = {k: v for k, v in n.items() if k not in ("id", "label")}
        if "attributes" in props:
            props.update(props.pop("attributes"))
        prop_str = ", ".join(f'{k}: {json.dumps(v)}' for k, v in props.items())
        yield f'CREATE (:Entity {{id: "{n["id"]}", label: "{escape(n["label"])}"{", " + prop_str if prop_str else ""}}});'

    for e in kg["edges"]:
        attr = e.get("attributes") or {}
        a_str = (" { " + ", ".join(f'{k}: {json.dumps(v)}' for k, v in attr.items()) + " }") if attr else ""
        yield (
            f'MATCH (a {{id: "{e["source"]}"}}), (b {{id: "{e["target"]}"}}) '
            f'CREATE (a)-[:{clean_relation(escape(e["relation"]))}{a_str}]->(b);'
        )

def load_and_push():
    kg = json.loads(pathlib.Path(KG_PATH).read_text())
    stmts = list(kg_to_statements(kg))          # or stream if huge
    with driver.session() as sess:
        for chunk in _chunk(stmts, 1000):       # 1000-statement batches
            tx = "\n".join(chunk)
            sess.run(tx)

def _chunk(iterable, size):
    it = iter(iterable)
    for first in it:
        yield list(itertools.chain([first], itertools.islice(it, size-1)))

if __name__ == "__main__":
    load_and_push()
    print("✅ Graph ingested!")

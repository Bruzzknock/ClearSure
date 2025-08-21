# run_pipeline.py
import json, itertools
from neo4j import GraphDatabase
from contextlib import ExitStack
from pathlib import Path
from convert import edge_to_cypher, node_to_cypher
import pathlib
import env  # noqa: F401

BASE_DIR = Path(__file__).resolve().parents[1]
KG_PATH   = BASE_DIR / "structured" / "final_kg.json"
OUT_PATH  = BASE_DIR / "structured" / "import_kg.cypher"
BOLT_URI  = "bolt://localhost:7687"
driver    = GraphDatabase.driver(BOLT_URI, auth=("neo4j", "12345678"))

def kg_to_statements(kg):
    node_ids = set()
    for node in kg["nodes"]:
        if node["id"] in node_ids:
            continue
        node_ids.add(node["id"])
        yield node_to_cypher(node)

    for edge in kg["edges"]:
        yield edge_to_cypher(edge)

def load_and_push(save_to: Path | None = None) -> None:
    kg     = json.loads(KG_PATH.read_text(encoding="utf-8"))
    stmts  = kg_to_statements(kg)

    with ExitStack() as stack:
        sess   = stack.enter_context(driver.session())
        tx     = stack.enter_context(sess.begin_transaction())   # ← open TX
        writer = stack.enter_context(save_to.open("w", encoding="utf-8")) if save_to else None

        for i, stmt in enumerate(stmts, 1):
            if writer:
                writer.write(stmt + "\n")
            tx.run(stmt)               # ← run one statement
            # optional progress log every 1 000 rows
            if i % 1000 == 0:
                print(f"{i} statements sent…")

        tx.commit()


def push_cypher_file(path: Path) -> None:
    """Push pre-generated Cypher statements to Neo4j.

    Previously every *line* in ``path`` was treated as a complete statement.
    Multi-line queries such as a ``MATCH`` on one line followed by ``CREATE`` on
    the next were therefore split and Neo4j received a bare ``MATCH`` which
    results in ``CypherSyntaxError``.  To support multi-line statements we
    accumulate lines until a terminating semicolon is encountered.
    """

    stmts: list[str] = []
    buffer: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        buffer.append(stripped)
        if stripped.endswith(";"):
            stmts.append(" ".join(buffer))
            buffer.clear()
    if buffer:
        stmts.append(" ".join(buffer))

    with ExitStack() as stack:
        sess = stack.enter_context(driver.session())
        tx = stack.enter_context(sess.begin_transaction())

        for i, stmt in enumerate(stmts, 1):
            tx.run(stmt)
            if i % 1000 == 0:
                print(f"{i} statements sent…")

        tx.commit()

def clear_database(drop_meta: bool = False) -> None:
    with driver.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")
        if drop_meta:
            for rec in sess.run("SHOW CONSTRAINTS"):
                name = rec["name"]
                if name:
                    # Use backticks so constraint names with special characters
                    # (e.g. hyphens) are parsed correctly by Neo4j
                    sess.run(f"DROP CONSTRAINT `{name}` IF EXISTS")
            for rec in sess.run("SHOW INDEXES"):
                name = rec["name"]
                if name:
                    # Same treatment for index names; backticks escape hyphens
                    sess.run(f"DROP INDEX `{name}` IF EXISTS")

def _chunk(iterable, size):
    it = iter(iterable)
    for first in it:
        yield list(itertools.chain([first], itertools.islice(it, size-1)))

if __name__ == "__main__":
    clear_database(drop_meta=True)           # wipe
    load_and_push(save_to=OUT_PATH)          # reload + save copy
    print("✅ Graph ingested and written to", OUT_PATH)

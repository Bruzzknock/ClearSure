from neo4j import GraphDatabase


def clear_database(uri: str, user: str, password: str, drop_meta: bool = False) -> None:
    """Delete all nodes and optionally indexes/constraints from Neo4j."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")
        if drop_meta:
            for rec in sess.run("SHOW CONSTRAINTS"):
                name = rec["name"]
                if name:
                    sess.run(f"DROP CONSTRAINT `{name}` IF EXISTS")
            for rec in sess.run("SHOW INDEXES"):
                name = rec["name"]
                if name:
                    sess.run(f"DROP INDEX `{name}` IF EXISTS")
    driver.close()

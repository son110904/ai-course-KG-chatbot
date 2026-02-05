from neo4j import GraphDatabase
from config import *

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def save_graph(graph):
    with driver.session() as session:
        for e in graph.get("entities", []):
            session.run(
                f"MERGE (n:{e['type']} {{id: $id}})",
                id=e["id"]
            )

        for r in graph.get("relations", []):
            session.run(
                f"""
                MATCH (a {{id: $src}}), (b {{id: $tgt}})
                MERGE (a)-[:{r['type']}]->(b)
                """,
                src=r["source"],
                tgt=r["target"]
            )

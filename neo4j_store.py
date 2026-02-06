import re
from typing import Dict

from neo4j import GraphDatabase

from config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _safe_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", label.strip())
    return clean or "Entity"


def _safe_rel_type(rel_type: str) -> str:
    clean = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper().strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "RELATED_TO"


def ensure_schema() -> None:
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run(
            "CREATE FULLTEXT INDEX entity_id_ft IF NOT EXISTS FOR (n:Entity) ON EACH [n.id, n.description]"
        )


def save_graph(graph: Dict, chunk: Dict) -> None:
    chunk_id = chunk.get("id")
    chunk_text = chunk.get("content", "")
    chunk_source = chunk.get("source", "")

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text, c.source = $source
            """,
            id=chunk_id,
            text=chunk_text,
            source=chunk_source,
        )

        for entity in graph.get("entities", []):
            label = _safe_label(entity.get("type", "Entity"))
            session.run(
                f"""
                MERGE (e:Entity:{label} {{id: $id}})
                ON CREATE SET e.description = $description
                ON MATCH SET e.description = CASE
                    WHEN e.description IS NULL OR e.description = '' THEN $description
                    WHEN $description IS NULL OR $description = '' THEN e.description
                    WHEN e.description CONTAINS $description THEN e.description
                    ELSE e.description + ' | ' + $description
                END
                """,
                id=entity.get("id"),
                description=entity.get("description", ""),
            )
            session.run(
                """
                MATCH (e:Entity {id: $entity_id}), (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                entity_id=entity.get("id"),
                chunk_id=chunk_id,
            )

        for relation in graph.get("relations", []):
            rel_type = _safe_rel_type(relation.get("type", "RELATED_TO"))
            session.run(
                f"""
                MATCH (s:Entity {{id: $source}}), (t:Entity {{id: $target}}), (c:Chunk {{id: $chunk_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                ON CREATE SET r.description = $description, r.weight = 1
                ON MATCH SET r.weight = coalesce(r.weight, 1) + 1,
                             r.description = CASE
                                WHEN r.description IS NULL OR r.description = '' THEN $description
                                WHEN $description IS NULL OR $description = '' THEN r.description
                                WHEN r.description CONTAINS $description THEN r.description
                                ELSE r.description + ' | ' + $description
                             END
                """,
                source=relation.get("source"),
                target=relation.get("target"),
                description=relation.get("description", ""),
                chunk_id=chunk_id,
            )

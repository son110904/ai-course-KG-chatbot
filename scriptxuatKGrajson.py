"""
Neo4j Graph Export to JSON
Xuất toàn bộ thông tin graph từ Neo4j ra định dạng JSON.

Yêu cầu:
    pip install neo4j

Cách dùng:
    python neo4j_graph_export.py
    python neo4j_graph_export.py --uri bolt://localhost:7687 --user neo4j --password secret --output graph.json
"""

import json
import argparse
from neo4j import GraphDatabase


# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DB_URL= "neo4j+s://aa2ceabd.databases.neo4j.io"
DB_USER= "neo4j"
DB_PASSWORD= "1TsTblk_ygKXqdp3IZn-r4RgMjLbZFFXMXr-yh0ytNY"
DB_DATABASE="neo4j"
# ─────────────────────────────────────────────────────────────────────────────


def neo4j_value_to_python(value):
    """Chuyển đổi các kiểu dữ liệu Neo4j sang Python thuần (JSON-serializable)."""
    # Neo4j Node / Relationship objects
    if hasattr(value, "_properties"):
        return dict(value._properties)
    # Temporal types (Date, DateTime, Duration, …)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    # Point
    if hasattr(value, "x") and hasattr(value, "y"):
        result = {"x": value.x, "y": value.y}
        if hasattr(value, "z"):
            result["z"] = value.z
        return result
    return value


def fetch_nodes(session):
    """Lấy tất cả nodes cùng labels và properties."""
    result = session.run(
        "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props"
    )
    nodes = []
    for record in result:
        props = {k: neo4j_value_to_python(v) for k, v in record["props"].items()}
        nodes.append({
            "id":         record["id"],
            "labels":     record["labels"],
            "properties": props,
        })
    return nodes


def fetch_relationships(session):
    """Lấy tất cả relationships cùng type và properties."""
    result = session.run(
        """
        MATCH (s)-[r]->(t)
        RETURN id(r)   AS id,
               id(s)   AS source,
               id(t)   AS target,
               type(r) AS type,
               properties(r) AS props
        """
    )
    relationships = []
    for record in result:
        props = {k: neo4j_value_to_python(v) for k, v in record["props"].items()}
        relationships.append({
            "id":         record["id"],
            "source":     record["source"],
            "target":     record["target"],
            "type":       record["type"],
            "properties": props,
        })
    return relationships


def fetch_schema(session):
    """Lấy schema: node labels, relationship types, indexes, constraints."""
    schema = {}

    # Node labels
    labels_result = session.run("CALL db.labels()")
    schema["node_labels"] = [r["label"] for r in labels_result]

    # Relationship types
    rel_result = session.run("CALL db.relationshipTypes()")
    schema["relationship_types"] = [r["relationshipType"] for r in rel_result]

    # Indexes
    try:
        idx_result = session.run("SHOW INDEXES")
        schema["indexes"] = [dict(r) for r in idx_result]
    except Exception:
        schema["indexes"] = []

    # Constraints
    try:
        con_result = session.run("SHOW CONSTRAINTS")
        schema["constraints"] = [dict(r) for r in con_result]
    except Exception:
        schema["constraints"] = []

    return schema


def export_graph(uri, user, password, output_file):
    print(f"[+] Kết nối tới Neo4j: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            print("[+] Đang lấy schema …")
            schema = fetch_schema(session)

            print("[+] Đang lấy nodes …")
            nodes = fetch_nodes(session)

            print("[+] Đang lấy relationships …")
            relationships = fetch_relationships(session)
    finally:
        driver.close()

    graph_data = {
        "metadata": {
            "uri":                uri,
            "total_nodes":        len(nodes),
            "total_relationships": len(relationships),
        },
        "schema":        schema,
        "nodes":         nodes,
        "relationships": relationships,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ Xuất thành công!")
    print(f"   Nodes         : {len(nodes)}")
    print(f"   Relationships : {len(relationships)}")
    print(f"   File output   : {output_file}")

    # In preview ra terminal
    preview = {
        "metadata": graph_data["metadata"],
        "schema":   graph_data["schema"],
        "nodes (preview)":         nodes[:3],
        "relationships (preview)": relationships[:3],
    }
    print("\n── JSON Preview ─────────────────────────────────")
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))

    return graph_data


def parse_args():
    parser = argparse.ArgumentParser(description="Export Neo4j graph to JSON")
    parser.add_argument("--uri",      default=DB_URL,      help="Neo4j URI")
    parser.add_argument("--user",     default=DB_USER,     help="Username")
    parser.add_argument("--password", default=DB_PASSWORD, help="Password")
    parser.add_argument("--output",   default=DB_DATABASE,   help="Output JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_graph(
        uri=args.uri,
        user=args.user,
        password=args.password,
        output_file=args.output,
    )
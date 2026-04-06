"""
Neo4j Knowledge Graph Export to JSON
Xuất toàn bộ Knowledge Graph từ Neo4j ra file JSON có cấu trúc đẹp, dễ đọc.

Yêu cầu:
    pip install neo4j

Cách dùng:
    python export_kg_neo4j.py
    python export_kg_neo4j.py --uri bolt://localhost:7687 --user neo4j --password secret --output my_kg.json
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime
from neo4j import GraphDatabase


# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DB_URL      = "neo4j+s://bdfc7297.databases.neo4j.io"
DB_USER     = "bdfc7297"
DB_PASSWORD = "0WWAdQtwxMMeoPqTT62bTLBb0DOVqlZs3bjNlASrPDs"
DB_DATABASE = "bdfc7297"
OUTPUT_FILE = "knowledge_graph.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def neo4j_to_python(value):
    """Chuyển kiểu dữ liệu Neo4j → Python thuần (JSON-serializable)."""
    if hasattr(value, "_properties"):
        return {k: neo4j_to_python(v) for k, v in value._properties.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "x") and hasattr(value, "y"):
        pt = {"x": value.x, "y": value.y}
        if hasattr(value, "z"):
            pt["z"] = value.z
        return pt
    if isinstance(value, list):
        return [neo4j_to_python(v) for v in value]
    return value


def section(title: str, char: str = "─", width: int = 60) -> str:
    return f"\n{char * 4} {title} {char * max(0, width - len(title) - 6)}"


# ── Fetch Functions ───────────────────────────────────────────────────────────

def fetch_nodes(session) -> list[dict]:
    """Lấy tất cả nodes, trả về list đã được nhóm theo label."""
    result = session.run(
        "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props "
        "ORDER BY labels(n)[0], id(n)"
    )
    nodes = []
    for record in result:
        props = {k: neo4j_to_python(v) for k, v in record["props"].items()}
        nodes.append({
            "id":         record["id"],
            "labels":     record["labels"],
            "properties": props,
        })
    return nodes


def fetch_relationships(session) -> list[dict]:
    """Lấy tất cả relationships."""
    result = session.run(
        """
        MATCH (s)-[r]->(t)
        RETURN id(r)        AS id,
               id(s)        AS source_id,
               labels(s)[0] AS source_label,
               id(t)        AS target_id,
               labels(t)[0] AS target_label,
               type(r)      AS type,
               properties(r) AS props
        ORDER BY type(r), id(r)
        """
    )
    rels = []
    for record in result:
        props = {k: neo4j_to_python(v) for k, v in record["props"].items()}
        rels.append({
            "id":           record["id"],
            "type":         record["type"],
            "source": {
                "id":    record["source_id"],
                "label": record["source_label"],
            },
            "target": {
                "id":    record["target_id"],
                "label": record["target_label"],
            },
            "properties": props,
        })
    return rels


def fetch_schema(session) -> dict:
    """Lấy schema đầy đủ: labels, rel types, indexes, constraints."""
    schema = {}

    labels_result = session.run("CALL db.labels()")
    schema["node_labels"] = sorted([r["label"] for r in labels_result])

    rel_result = session.run("CALL db.relationshipTypes()")
    schema["relationship_types"] = sorted([r["relationshipType"] for r in rel_result])

    try:
        schema["indexes"] = [
            {k: (str(v) if not isinstance(v, (str, int, float, bool, list, type(None))) else v)
             for k, v in dict(r).items()}
            for r in session.run("SHOW INDEXES")
        ]
    except Exception:
        schema["indexes"] = []

    try:
        schema["constraints"] = [
            {k: (str(v) if not isinstance(v, (str, int, float, bool, list, type(None))) else v)
             for k, v in dict(r).items()}
            for r in session.run("SHOW CONSTRAINTS")
        ]
    except Exception:
        schema["constraints"] = []

    return schema


# ── Aggregation ───────────────────────────────────────────────────────────────

def group_nodes_by_label(nodes: list[dict]) -> dict:
    """Nhóm nodes theo label chính (label đầu tiên)."""
    grouped = defaultdict(list)
    for node in nodes:
        primary_label = node["labels"][0] if node["labels"] else "Unknown"
        grouped[primary_label].append(node)
    return dict(sorted(grouped.items()))


def group_rels_by_type(rels: list[dict]) -> dict:
    """Nhóm relationships theo type."""
    grouped = defaultdict(list)
    for rel in rels:
        grouped[rel["type"]].append(rel)
    return dict(sorted(grouped.items()))


def build_statistics(nodes: list[dict], rels: list[dict]) -> dict:
    """Tính thống kê tổng hợp về graph."""
    node_label_counts = defaultdict(int)
    for node in nodes:
        for label in node["labels"]:
            node_label_counts[label] += 1

    rel_type_counts = defaultdict(int)
    for rel in rels:
        rel_type_counts[rel["type"]] += 1

    # Degree (in/out) per node
    out_degree = defaultdict(int)
    in_degree  = defaultdict(int)
    for rel in rels:
        out_degree[rel["source"]["id"]] += 1
        in_degree[rel["target"]["id"]]  += 1

    degrees = [out_degree[n["id"]] + in_degree[n["id"]] for n in nodes]
    avg_degree = round(sum(degrees) / len(degrees), 2) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    return {
        "total_nodes":         len(nodes),
        "total_relationships": len(rels),
        "avg_degree":          avg_degree,
        "max_degree":          max_degree,
        "nodes_by_label":      dict(sorted(node_label_counts.items(),
                                           key=lambda x: -x[1])),
        "relationships_by_type": dict(sorted(rel_type_counts.items(),
                                             key=lambda x: -x[1])),
    }


# ── Main Export ───────────────────────────────────────────────────────────────

def export_graph(uri: str, user: str, password: str, output_file: str):
    print(section("Kết nối Neo4j", "═"))
    print(f"  URI  : {uri}")
    print(f"  User : {user}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            print(section("Thu thập dữ liệu"))
            print("  [1/3] Schema …")
            schema = fetch_schema(session)

            print("  [2/3] Nodes …")
            nodes = fetch_nodes(session)

            print("  [3/3] Relationships …")
            rels = fetch_relationships(session)
    finally:
        driver.close()

    # ── Build structured output ──────────────────────────────────────────────
    stats    = build_statistics(nodes, rels)
    by_label = group_nodes_by_label(nodes)
    by_type  = group_rels_by_type(rels)

    graph_data = {
        # ── 1. Metadata ──────────────────────────────────────────────────────
        "metadata": {
            "source_uri":      uri,
            "exported_at":     datetime.utcnow().isoformat() + "Z",
            "format_version":  "2.0",
        },

        # ── 2. Statistics ────────────────────────────────────────────────────
        "statistics": stats,

        # ── 3. Schema ────────────────────────────────────────────────────────
        "schema": schema,

        # ── 4. Nodes (grouped by label) ──────────────────────────────────────
        "nodes_by_label": by_label,

        # ── 5. Relationships (grouped by type) ───────────────────────────────
        "relationships_by_type": by_type,

        # ── 6. Flat lists (dùng cho graph rendering tools) ───────────────────
        "flat": {
            "nodes":         nodes,
            "relationships": rels,
        },
    }

    # ── Write JSON ───────────────────────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2, default=str)

    # ── Terminal summary ─────────────────────────────────────────────────────
    print(section("Kết quả xuất", "═"))
    print(f"  ✅ File          : {output_file}")
    print(f"  📦 Nodes         : {stats['total_nodes']:,}")
    print(f"  🔗 Relationships : {stats['total_relationships']:,}")
    print(f"  📐 Avg degree    : {stats['avg_degree']}")
    print(f"  🏷  Node labels  : {', '.join(schema['node_labels'])}")
    print(f"  🔀 Rel types     : {', '.join(schema['relationship_types'])}")

    print(section("Nodes by label"))
    for label, count in stats["nodes_by_label"].items():
        bar = "█" * min(count, 30)
        print(f"  {label:<25} {bar} {count:,}")

    print(section("Relationships by type"))
    for rtype, count in stats["relationships_by_type"].items():
        bar = "█" * min(count, 30)
        print(f"  {rtype:<30} {bar} {count:,}")

    # ── JSON Preview ─────────────────────────────────────────────────────────
    print(section("JSON Preview (3 nodes · 3 rels)"))
    preview = {
        "metadata":   graph_data["metadata"],
        "statistics": graph_data["statistics"],
        "schema": {
            "node_labels":       schema["node_labels"],
            "relationship_types": schema["relationship_types"],
        },
        "nodes_preview": {
            label: items[:2]
            for label, items in list(by_label.items())[:3]
        },
        "relationships_preview": {
            rtype: items[:2]
            for rtype, items in list(by_type.items())[:3]
        },
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))

    return graph_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Export Neo4j Knowledge Graph to structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python export_kg_neo4j.py
  python export_kg_neo4j.py --uri bolt://localhost:7687 --user neo4j --password secret
  python export_kg_neo4j.py --output my_knowledge_graph.json
        """,
    )
    p.add_argument("--uri",      default=DB_URL,      help="Neo4j connection URI")
    p.add_argument("--user",     default=DB_USER,     help="Neo4j username")
    p.add_argument("--password", default=DB_PASSWORD, help="Neo4j password")
    p.add_argument("--output",   default=OUTPUT_FILE, help="Output JSON filename")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_graph(
        uri=args.uri,
        user=args.user,
        password=args.password,
        output_file=args.output,
    )
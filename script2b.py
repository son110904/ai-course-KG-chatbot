"""

Xóa MAJOR nodes trùng (code = name) bằng Cypher thuần — không cần APOC.
Chiến lược:
  1. Tìm cặp: MAJOR có code hợp lệ (bắt đầu bằng số) ↔ MAJOR cùng name nhưng code = name
  2. Chuyển tất cả relationships từ node code=name sang node có code hợp lệ (tạo lại bằng Cypher)
  3. Xóa node code=name
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def step0_diagnose(session):
    """Xem hiện trạng trước khi fix."""
    print("\n[Diagnose] MAJOR nodes hiện tại:")
    result = session.run("""
        MATCH (m:MAJOR)
        RETURN m.name AS name, m.code AS code, elementId(m) AS node_id,
               CASE 
                 WHEN m.code IS NULL THEN "NO CODE"
                 WHEN m.code =~ '^\\d+' THEN "VALID_CODE"
                 WHEN m.code = m.name THEN "CODE_IS_NAME (duplicate)"
                 ELSE "OTHER"
               END AS status
        ORDER BY m.name, m.code
    """).data()

    for r in result:
        code_str = f"code={r['code']}" if r['code'] else "NO CODE"
        print(f"  id={r['node_id']}  {r['name']}  ({code_str})  [{r['status']}]")

    # Tìm các tên bị trùng
    dupes = session.run("""
        MATCH (m:MAJOR)
        WITH m.name AS name, collect(m) AS nodes, count(m) AS cnt
        WHERE cnt > 1
        RETURN name, cnt
        ORDER BY name
    """).data()

    print(f"\n  → {len(dupes)} tên MAJOR bị trùng lặp:")
    for d in dupes:
        print(f"    '{d['name']}' xuất hiện {d['cnt']} lần")

    return len(dupes)


def step1_reconnect_relationships(session):
    """
    Với mỗi cặp (has_code, no_code) cùng name:
    Tạo lại TẤT CẢ relationships của node no_code → trỏ vào has_code.
    
    Xử lý từng loại relationship riêng vì Cypher không cho phép
    dynamic relationship type trong MERGE.
    """
    print("\n[Step 1] Chuyển relationships từ node code=name → node có code hợp lệ...")

    rel_types = [
        # (rel_type, direction)
        # direction = "out"  → (no_code)-[r]->(x)  cần tạo (has_code)-[r]->(x)
        # direction = "in" → (x)-[r]->(no_code)  cần tạo (x)-[r]->(has_code)
        ("LEADS_TO",       "out"),   # MAJOR -[LEADS_TO]-> CAREER
        ("OFFERS",         "out"),   # MAJOR -[OFFERS]-> SUBJECT
        ("MENTIONED_IN",   "out"),   # MAJOR -[MENTIONED_IN]-> DOCUMENT
    ]

    total = 0
    for rel_type, direction in rel_types:
        if direction == "out":
            # (no_code)-[rel]->(x) → tạo (has_code)-[rel]->(x)
            cypher = f"""
                MATCH (has_code:MAJOR)  
                WHERE has_code.code IS NOT NULL AND has_code.code =~ '^\\d+'
                MATCH (no_code:MAJOR)
                    WHERE no_code.code IS NOT NULL AND no_code.code = no_code.name
                    AND toLower(no_code.name) = toLower(has_code.name)
                MATCH (no_code)-[:{rel_type}]->(x)
                MERGE (has_code)-[:{rel_type}]->(x)
                RETURN count(*) AS cnt
            """
        else:
            # (x)-[rel]->(no_code) → tạo (x)-[rel]->(has_code)
            cypher = f"""
                MATCH (has_code:MAJOR)  
                WHERE has_code.code IS NOT NULL AND has_code.code =~ '^\\d+'
                MATCH (no_code:MAJOR)
                    WHERE no_code.code IS NOT NULL AND no_code.code = no_code.name
                    AND toLower(no_code.name) = toLower(has_code.name)
                MATCH (x)-[:{rel_type}]->(no_code)
                MERGE (x)-[:{rel_type}]->(has_code)
                RETURN count(*) AS cnt
            """
        cnt = session.run(cypher).single()["cnt"]
        if cnt > 0:
            print(f"  ✓ [{rel_type}] {cnt} relationships tái tạo")
        total += cnt

    print(f"  → Tổng: {total} relationships đã chuyển")
    return total


def step2_delete_no_code_nodes(session):
    """Xóa MAJOR nodes có code = name (đã là duplicate của node có code hợp lệ)."""
    print("\n[Step 2] Xóa MAJOR nodes có code = name (duplicate)...")

    # Chỉ xóa node no_code nếu tồn tại node has_code cùng tên
    result = session.run("""
        MATCH (has_code:MAJOR) 
        WHERE has_code.code IS NOT NULL AND has_code.code =~ '^\\d+'
        MATCH (no_code:MAJOR)
            WHERE no_code.code IS NOT NULL AND no_code.code = no_code.name
            AND toLower(no_code.name) = toLower(has_code.name)
        RETURN no_code.name AS name, elementId(no_code) AS node_id
    """).data()

    if not result:
        print("  Không có node nào cần xóa.")
        return 0

    print(f"  Sẽ xóa {len(result)} nodes:")
    for r in result:
        print(f"    id={r['node_id']}  '{r['name']}'")

    # Xóa (DETACH xử lý luôn các relationships còn sót)
    deleted = session.run("""
        MATCH (has_code:MAJOR) 
        WHERE has_code.code IS NOT NULL AND has_code.code =~ '^\\d+'
        MATCH (no_code:MAJOR)
            WHERE no_code.code IS NOT NULL AND no_code.code = no_code.name
            AND toLower(no_code.name) = toLower(has_code.name)
        DETACH DELETE no_code
        RETURN count(*) AS cnt
    """).single()["cnt"]

    print(f"  ✓ Đã xóa {deleted} nodes")
    return deleted


def step3_verify(session):
    """Kiểm tra kết quả sau khi fix."""
    print("\n[Verify] MAJOR nodes sau khi fix:")

    result = session.run("""
        MATCH (m:MAJOR)
        OPTIONAL MATCH (m)-[:LEADS_TO]->(c:CAREER)
        OPTIONAL MATCH (m)-[:OFFERS]->(s:SUBJECT)
        RETURN m.name AS name, m.code AS code,
               count(DISTINCT c) AS career_count,
               count(DISTINCT s) AS subject_count
        ORDER BY m.name
    """).data()

    for r in result:
        code_str = r['code'] if r['code'] else "⚠ NO CODE"
        print(f"  [{code_str}] {r['name']}")
        print(f"         → {r['career_count']} careers,  {r['subject_count']} subjects")

    # Kiểm tra còn node trùng không
    dupes = session.run("""
        MATCH (m:MAJOR)
        WITH m.name AS name, count(m) AS cnt
        WHERE cnt > 1
        RETURN name, cnt
    """).data()

    if dupes:
        print(f"\n  ⚠ Vẫn còn {len(dupes)} tên trùng:")
        for d in dupes:
            print(f"    '{d['name']}' × {d['cnt']}")
    else:
        print("\n  ✅ Không còn MAJOR node trùng lặp!")

    return len(dupes) == 0


def main():
    print("=" * 55)
    print("Fix Duplicate MAJOR Nodes (no APOC needed)")
    print("=" * 55)

    driver = get_driver()
    with driver.session() as session:

        # Xem hiện trạng
        dupe_count = step0_diagnose(session)

        if dupe_count == 0:
            print("\n✅ Không có node trùng. Không cần làm gì.")
            driver.close()
            return

        input(f"\n→ Tìm thấy {dupe_count} tên trùng. Nhấn Enter để tiến hành fix...")

        # Fix
        step1_reconnect_relationships(session)
        step2_delete_no_code_nodes(session)
        ok = step3_verify(session)

        if ok:
            print("\n✅ Done! Graph đã sạch.")
        else:
            print("\n⚠ Vẫn còn vấn đề. Xem output phía trên để debug.")

    driver.close()


if __name__ == "__main__":
    main()
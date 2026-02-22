"""
Script 2b: Post-processing — Ghép MAJOR nodes và tạo LEADS_TO relationships
Chạy sau script2 để đảm bảo:
1. MAJOR từ career_description (chỉ có name) được ghép với MAJOR từ curriculum (có code)
2. LEADS_TO relationships được tạo đầy đủ
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def merge_duplicate_majors(session):
    """
    Tìm các cặp MAJOR node có cùng name nhưng 1 có code, 1 không có code.
    Chuyển tất cả relationship từ node không có code sang node có code, rồi xóa node trùng.
    """
    print("\n[Step 1] Tìm MAJOR nodes trùng tên...")

    # Tìm các cặp duplicate
    dupes = session.run("""
        MATCH (m_with_code:MAJOR)
        WHERE m_with_code.code IS NOT NULL
        MATCH (m_no_code:MAJOR {name: m_with_code.name})
        WHERE m_no_code.code IS NULL AND id(m_with_code) <> id(m_no_code)
        RETURN m_with_code.name AS name,
               m_with_code.code AS code,
               id(m_with_code)  AS id_with_code,
               id(m_no_code)    AS id_no_code
    """).data()

    if not dupes:
        print("  Không tìm thấy MAJOR nodes trùng lặp.")
        return 0

    print(f"  Tìm thấy {len(dupes)} cặp trùng lặp:")
    for d in dupes:
        print(f"    '{d['name']}' (code={d['code']}) ↔ node không có code")

    merged = 0
    for d in dupes:
        id_keep   = d["id_with_code"]   # giữ node có code
        id_remove = d["id_no_code"]     # xóa node không có code

        # Chuyển tất cả relationship từ node không code sang node có code
        # Incoming relationships (X)-[r]->(m_no_code) → (X)-[r]->(m_with_code)
        session.run("""
            MATCH (keep:MAJOR)   WHERE id(keep)   = $id_keep
            MATCH (remove:MAJOR) WHERE id(remove) = $id_remove
            MATCH (x)-[r]->(remove)
            WHERE id(x) <> id(keep)
            CALL apoc.refactor.to(r, keep)
            YIELD input RETURN input
        """, id_keep=id_keep, id_remove=id_remove)

        # Outgoing relationships (m_no_code)-[r]->(X) → (m_with_code)-[r]->(X)
        session.run("""
            MATCH (keep:MAJOR)   WHERE id(keep)   = $id_keep
            MATCH (remove:MAJOR) WHERE id(remove) = $id_remove
            MATCH (remove)-[r]->(x)
            WHERE id(x) <> id(keep)
            CALL apoc.refactor.from(r, keep)
            YIELD input RETURN input
        """, id_keep=id_keep, id_remove=id_remove)

        # Xóa node trùng
        session.run("""
            MATCH (remove:MAJOR) WHERE id(remove) = $id_remove
            DETACH DELETE remove
        """, id_remove=id_remove)

        merged += 1
        print(f"  ✓ Merged '{d['name']}'")

    return merged


def create_leads_to_from_career_docs(session):
    """
    Fallback: Nếu không có APOC, tạo LEADS_TO trực tiếp bằng cách
    match MAJOR (có code) với CAREER qua tên ngành trong DOCUMENT.
    Đọc từ node DOCUMENT loại career_description và career nodes.
    """
    print("\n[Step 2] Tạo LEADS_TO relationships từ career_description documents...")

    # Lấy tất cả CAREER và MAJOR đang tồn tại
    result = session.run("""
        MATCH (c:CAREER)-[:MENTIONED_IN]->(doc:DOCUMENT {doctype: 'career_description'})
        MATCH (m:MAJOR)-[:MENTIONED_IN]->(doc)
        WHERE m.code IS NOT NULL
        MERGE (m)-[:LEADS_TO]->(c)
        RETURN m.name AS major, m.code AS code, c.name AS career
    """).data()

    if result:
        print(f"  ✓ Tạo {len(result)} LEADS_TO relationships:")
        for r in result[:10]:
            print(f"    {r['major']} ({r['code']}) → {r['career']}")
        if len(result) > 10:
            print(f"    ... và {len(result)-10} relationships khác")
    else:
        print("  Không tạo được qua DOCUMENT. Thử cách khác...")

        # Cách 2: Match theo tên ngành giống nhau (case-insensitive)
        result2 = session.run("""
            MATCH (m_code:MAJOR) WHERE m_code.code IS NOT NULL
            MATCH (m_nocode:MAJOR) WHERE m_nocode.code IS NULL
              AND toLower(m_nocode.name) = toLower(m_code.name)
            MATCH (m_nocode)-[:LEADS_TO]->(c:CAREER)
            MERGE (m_code)-[:LEADS_TO]->(c)
            RETURN m_code.name AS major, m_code.code AS code, c.name AS career
        """).data()

        if result2:
            print(f"  ✓ Tạo {len(result2)} LEADS_TO relationships qua name matching:")
            for r in result2[:10]:
                print(f"    {r['major']} ({r['code']}) → {r['career']}")
        else:
            print("  ⚠ Không tạo được relationship nào. Kiểm tra lại dữ liệu.")

    return len(result) if result else 0


def verify_links(session):
    """Kiểm tra kết quả sau khi ghép."""
    print("\n[Verify] Kiểm tra MAJOR → CAREER links:")
    result = session.run("""
        MATCH (m:MAJOR)-[:LEADS_TO]->(c:CAREER)
        RETURN m.name AS major, m.code AS code, collect(c.name) AS careers
        ORDER BY m.name
    """).data()

    if result:
        for r in result:
            print(f"  ✓ {r['major']} ({r['code']}) → {r['careers']}")
    else:
        print("  Vẫn chưa có MAJOR → CAREER links nào!")

    return len(result)


def main():
    print("Starting post-link pipeline...")
    driver = get_driver()

    with driver.session() as session:
        # Thử dùng APOC để merge duplicate nodes
        try:
            merged = merge_duplicate_majors(session)
            print(f"\n  Merged {merged} duplicate MAJOR nodes")
        except Exception as e:
            print(f"\n  APOC không khả dụng ({e}), dùng fallback...")
            # Fallback không cần APOC
            create_leads_to_from_career_docs(session)

        verify_links(session)

    driver.close()
    print("\n Post-link complete.")


if __name__ == "__main__":
    main()
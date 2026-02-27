"""
Script 2b: 
  PHẦN 1 — Xóa MAJOR nodes trùng (code = name) bằng Cypher thuần — không cần APOC.
  Chiến lược:
    1. Tìm cặp: MAJOR có code hợp lệ (bắt đầu bằng số) ↔ MAJOR cùng name nhưng code = name
    2. Chuyển tất cả relationships từ node code=name sang node có code hợp lệ (tạo lại bằng Cypher)
    3. Xóa node code=name

  PHẦN 2 — Gắn thuộc tính major_codes vào CAREER nodes.
  Chiến lược:
    - Mỗi CAREER có quan hệ LEADS_TO từ các MAJOR (hướng: MAJOR -[LEADS_TO]-> CAREER
      HOẶC CAREER -[SUITABLE_FOR]-> MAJOR, tuỳ schema thực tế — script tự detect).
    - Thu thập tất cả mã ngành hợp lệ (code =~ '^\\d+') của các MAJOR liên quan.
    - SET career.major_codes = [list mã ngành đó].
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


# ═══════════════════════════════════════════════════════════════
#  PHẦN 1: FIX DUPLICATE MAJOR NODES
# ═══════════════════════════════════════════════════════════════

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
        # direction = "in"   → (x)-[r]->(no_code)  cần tạo (x)-[r]->(has_code)
        ("LEADS_TO",     "out"),   # MAJOR -[LEADS_TO]-> CAREER
        ("OFFERS",       "out"),   # MAJOR -[OFFERS]-> SUBJECT
        ("MENTIONED_IN", "out"),   # MAJOR -[MENTIONED_IN]-> DOCUMENT
    ]

    total = 0
    for rel_type, direction in rel_types:
        if direction == "out":
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


# ═══════════════════════════════════════════════════════════════
#  PHẦN 2: GẮN major_codes VÀO CAREER NODES
# ═══════════════════════════════════════════════════════════════

def detect_career_major_rel_direction(session):
    """
    Tự detect hướng quan hệ giữa CAREER và MAJOR trong graph.
    Trả về: "MAJOR_TO_CAREER" | "CAREER_TO_MAJOR" | None
    """
    # Thử MAJOR -[LEADS_TO]-> CAREER
    r1 = session.run("""
        MATCH (m:MAJOR)-[:LEADS_TO]->(c:CAREER)
        RETURN count(*) AS cnt
    """).single()["cnt"]

    # Thử CAREER -[SUITABLE_FOR|RELATED_TO|LEADS_TO]-> MAJOR (một số schema ngược)
    r2 = session.run("""
        MATCH (c:CAREER)-[r]->(m:MAJOR)
        RETURN type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC LIMIT 1
    """).data()

    if r1 > 0:
        print(f"  Phát hiện: MAJOR -[LEADS_TO]-> CAREER ({r1} rels)")
        return "MAJOR_TO_CAREER", "LEADS_TO"
    elif r2:
        rel_type = r2[0]["rel_type"]
        cnt = r2[0]["cnt"]
        print(f"  Phát hiện: CAREER -[{rel_type}]-> MAJOR ({cnt} rels)")
        return "CAREER_TO_MAJOR", rel_type
    else:
        print("  ⚠ Không tìm thấy quan hệ nào giữa CAREER và MAJOR!")
        return None, None


def step4_diagnose_career_major_codes(session):
    """Xem hiện trạng CAREER nodes trước khi gắn major_codes."""
    print("\n[Step 4 - Diagnose] CAREER nodes và MAJOR liên quan:")

    direction, rel_type = detect_career_major_rel_direction(session)
    if direction is None:
        return direction, rel_type

    if direction == "MAJOR_TO_CAREER":
        query = f"""
            MATCH (c:CAREER)
            OPTIONAL MATCH (m:MAJOR)-[:{rel_type}]->(c)
            WITH c, collect({{name: m.name, code: m.code}}) AS majors
            RETURN c.name AS career,
                   c.major_codes AS existing_codes,
                   majors
            ORDER BY c.name
        """
    else:
        query = f"""
            MATCH (c:CAREER)
            OPTIONAL MATCH (c)-[:{rel_type}]->(m:MAJOR)
            WITH c, collect({{name: m.name, code: m.code}}) AS majors
            RETURN c.name AS career,
                   c.major_codes AS existing_codes,
                   majors
            ORDER BY c.name
        """

    result = session.run(query).data()
    total_careers = len(result)
    has_valid_major = 0

    for r in result:
        valid_codes = [
            m["code"] for m in r["majors"]
            if m["code"] and m["code"] != m["name"]
            # code hợp lệ: có giá trị và không phải là tên ngành
        ]
        if valid_codes:
            has_valid_major += 1
        existing = r["existing_codes"] or "—"
        print(f"  CAREER: {r['career']}")
        print(f"    existing major_codes : {existing}")
        print(f"    MAJOR rels found     : {[m['name'] for m in r['majors'] if m['name']]}")
        print(f"    valid codes to assign: {valid_codes}")

    print(f"\n  → Tổng {total_careers} CAREER nodes, "
          f"{has_valid_major} có MAJOR với code hợp lệ")
    return direction, rel_type


def step5_set_major_codes(session, direction: str, rel_type: str):
    """
    Gắn thuộc tính major_codes vào mỗi CAREER node.
    major_codes = list mã ngành hợp lệ (code =~ '^\\d+') của các MAJOR liên quan.

    Với MAJOR không có code hợp lệ: thử fuzzy-match tên sang MAJOR có code hợp lệ.
    """
    print(f"\n[Step 5] Gắn major_codes vào CAREER nodes (rel direction: {direction})...")

    # ── 5a. Gắn từ MAJOR có code hợp lệ trực tiếp ──────────────────────────
    if direction == "MAJOR_TO_CAREER":
        cypher_direct = f"""
            MATCH (c:CAREER)
            OPTIONAL MATCH (m:MAJOR)-[:{rel_type}]->(c)
            WHERE m.code IS NOT NULL AND m.code =~ '^\\d+'
            WITH c, collect(DISTINCT m.code) AS codes
            SET c.major_codes = codes
            RETURN count(c) AS updated, 
                   sum(size(codes)) AS total_codes
        """
    else:
        cypher_direct = f"""
            MATCH (c:CAREER)
            OPTIONAL MATCH (c)-[:{rel_type}]->(m:MAJOR)
            WHERE m.code IS NOT NULL AND m.code =~ '^\\d+'
            WITH c, collect(DISTINCT m.code) AS codes
            SET c.major_codes = codes
            RETURN count(c) AS updated,
                   sum(size(codes)) AS total_codes
        """

    r = session.run(cypher_direct).single()
    print(f"  ✓ Đã SET major_codes cho {r['updated']} CAREER nodes "
          f"(tổng {r['total_codes']} code assignments)")

    # ── 5b. Với CAREER chưa có code nào: thử map qua tên MAJOR ──────────────
    # Tìm CAREER có major_codes rỗng nhưng có MAJOR rels (với code=name / no code)
    if direction == "MAJOR_TO_CAREER":
        cypher_fallback_find = f"""
            MATCH (c:CAREER)
            WHERE size(c.major_codes) = 0
            MATCH (m_bad:MAJOR)-[:{rel_type}]->(c)
            WHERE m_bad.code IS NULL OR NOT (m_bad.code =~ '^\\d+')
            // Tìm MAJOR có code hợp lệ cùng tên (case-insensitive)
            OPTIONAL MATCH (m_good:MAJOR)
            WHERE m_good.code IS NOT NULL AND m_good.code =~ '^\\d+'
              AND toLower(m_good.name) = toLower(m_bad.name)
            RETURN c.name AS career,
                   m_bad.name AS bad_name, m_bad.code AS bad_code,
                   m_good.code AS resolved_code
        """
    else:
        cypher_fallback_find = f"""
            MATCH (c:CAREER)
            WHERE size(c.major_codes) = 0
            MATCH (c)-[:{rel_type}]->(m_bad:MAJOR)
            WHERE m_bad.code IS NULL OR NOT (m_bad.code =~ '^\\d+')
            OPTIONAL MATCH (m_good:MAJOR)
            WHERE m_good.code IS NOT NULL AND m_good.code =~ '^\\d+'
              AND toLower(m_good.name) = toLower(m_bad.name)
            RETURN c.name AS career,
                   m_bad.name AS bad_name, m_bad.code AS bad_code,
                   m_good.code AS resolved_code
        """

    fallback_rows = session.run(cypher_fallback_find).data()

    # Group by career
    career_extra: dict[str, list[str]] = {}
    for row in fallback_rows:
        if row["resolved_code"]:
            career_extra.setdefault(row["career"], [])
            if row["resolved_code"] not in career_extra[row["career"]]:
                career_extra[row["career"]].append(row["resolved_code"])
        else:
            print(f"  ⚠ Không map được MAJOR '{row['bad_name']}' "
                  f"(code='{row['bad_code']}') cho CAREER '{row['career']}'")

    for career_name, extra_codes in career_extra.items():
        session.run("""
            MATCH (c:CAREER {name: $name})
            SET c.major_codes = $codes
        """, name=career_name, codes=extra_codes)
        print(f"  ✓ [fallback] '{career_name}' → {extra_codes}")

    return True


def step6_verify_career_codes(session):
    """Kiểm tra kết quả sau khi gắn major_codes."""
    print("\n[Verify] CAREER nodes sau khi gắn major_codes:")

    result = session.run("""
        MATCH (c:CAREER)
        RETURN c.name AS career, c.major_codes AS codes
        ORDER BY c.name
    """).data()

    no_codes = 0
    for r in result:
        codes = r["codes"] or []
        status = f"{codes}" if codes else "⚠ EMPTY"
        if not codes:
            no_codes += 1
        print(f"  {r['career']}")
        print(f"    major_codes: {status}")

    if no_codes:
        print(f"\n  ⚠ {no_codes} CAREER nodes vẫn chưa có major_codes "
              f"(có thể không có MAJOR nào liên kết với chúng)")
    else:
        print(f"\n  ✅ Tất cả CAREER nodes đều có major_codes!")

    return no_codes == 0


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Script 2b: Fix MAJOR Duplicates + Gắn major_codes cho CAREER")
    print("=" * 60)

    driver = get_driver()
    with driver.session() as session:

        # ── PHẦN 1: Fix MAJOR duplicates ──────────────────────────────────
        print("\n" + "─" * 60)
        print("PHẦN 1: Fix MAJOR duplicate nodes")
        print("─" * 60)

        dupe_count = step0_diagnose(session)

        if dupe_count > 0:
            input(f"\n→ Tìm thấy {dupe_count} tên trùng. Nhấn Enter để tiến hành fix...")
            step1_reconnect_relationships(session)
            step2_delete_no_code_nodes(session)
            step3_verify(session)
        else:
            print("\n✅ Không có MAJOR node trùng. Bỏ qua Phần 1.")

        # ── PHẦN 2: Gắn major_codes vào CAREER ────────────────────────────
        print("\n" + "─" * 60)
        print("PHẦN 2: Gắn major_codes vào CAREER nodes")
        print("─" * 60)

        direction, rel_type = step4_diagnose_career_major_codes(session)

        if direction is None:
            print("⚠ Không thể xác định quan hệ CAREER ↔ MAJOR. Bỏ qua Phần 2.")
        else:
            input(f"\n→ Nhấn Enter để gắn major_codes cho CAREER nodes...")
            step5_set_major_codes(session, direction, rel_type)
            step6_verify_career_codes(session)

    driver.close()
    print("\n✅ Hoàn tất.")


if __name__ == "__main__":
    main()
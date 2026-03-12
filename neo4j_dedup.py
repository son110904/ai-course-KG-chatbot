"""
Neo4j Graph Deduplication Script
=================================
Phân tích và xóa dữ liệu trùng lặp trong graph, bao gồm:

  1. SKILL nodes  — trùng skill_key  (285 node dư / 184 key)
  2. TEACHER nodes — trùng teacher_key (599 node dư / 400 key)
  3. Relationships — trùng (source, target, type) (35 rel dư)

Chiến lược giữ node "tốt nhất":
  - SKILL  : giữ node có name ngắn nhất (tên chuẩn, ít "râu" nhất)
  - TEACHER: giữ node có name ngắn nhất (không bị prefix title)
  Tất cả relationship của node bị xóa sẽ được chuyển sang node được giữ.

Yêu cầu:
    pip install neo4j

Cách dùng:
    python neo4j_dedup.py                         # dry-run (chỉ báo cáo)
    python neo4j_dedup.py --execute               # thực sự xóa
    python neo4j_dedup.py --uri bolt://... --execute
"""

import argparse
from collections import defaultdict
from neo4j import GraphDatabase

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DB_URL= "neo4j+s://bdfc7297.databases.neo4j.io"
DB_USER= "bdfc7297"
DB_PASSWORD= "0WWAdQtwxMMeoPqTT62bTLBb0DOVqlZs3bjNlASrPDs"
# ─────────────────────────────────────────────────────────────────────────────


def best_node(nodes: list[dict], key_field: str) -> dict:
    """Chọn node tốt nhất: ưu tiên name ngắn nhất (tên chuẩn, ít prefix nhất)."""
    return min(nodes, key=lambda n: len(n["properties"].get("name", "")))


# ══════════════════════════════════════════════════════════════════════════════
#  PHÂN TÍCH
# ══════════════════════════════════════════════════════════════════════════════

def analyze(session) -> dict:
    report = {}

    # ── 1. SKILL duplicates ───────────────────────────────────────────────────
    result = session.run(
        "MATCH (s:SKILL) RETURN id(s) AS id, s.skill_key AS key, s.name AS name"
    )
    skill_groups = defaultdict(list)
    for r in result:
        if r["key"]:
            skill_groups[r["key"]].append({"id": r["id"], "name": r["name"]})

    skill_dups = {k: v for k, v in skill_groups.items() if len(v) > 1}
    report["skill_dup_keys"]   = len(skill_dups)
    report["skill_extra_nodes"]= sum(len(v) - 1 for v in skill_dups.values())
    report["skill_groups"]     = skill_dups

    # ── 2. TEACHER duplicates ─────────────────────────────────────────────────
    result = session.run(
        "MATCH (t:TEACHER) RETURN id(t) AS id, t.teacher_key AS key, t.name AS name"
    )
    teacher_groups = defaultdict(list)
    for r in result:
        if r["key"]:
            teacher_groups[r["key"]].append({"id": r["id"], "name": r["name"]})

    teacher_dups = {k: v for k, v in teacher_groups.items() if len(v) > 1}
    report["teacher_dup_keys"]    = len(teacher_dups)
    report["teacher_extra_nodes"] = sum(len(v) - 1 for v in teacher_dups.values())
    report["teacher_groups"]      = teacher_dups

    # ── 3. Relationship duplicates ────────────────────────────────────────────
    result = session.run(
        """
        MATCH (a)-[r]->(b)
        WITH id(a) AS src, id(b) AS tgt, type(r) AS typ, collect(id(r)) AS ids
        WHERE size(ids) > 1
        RETURN src, tgt, typ, ids
        """
    )
    rel_dups = []
    total_extra_rels = 0
    for r in result:
        ids = r["ids"]
        rel_dups.append({"src": r["src"], "tgt": r["tgt"], "type": r["typ"],
                         "keep": ids[0], "delete": ids[1:]})
        total_extra_rels += len(ids) - 1

    report["rel_dup_pairs"]   = len(rel_dups)
    report["rel_extra_rels"]  = total_extra_rels
    report["rel_dups"]        = rel_dups

    return report


def print_report(report: dict):
    print("\n" + "═" * 60)
    print("  BÁO CÁO TRÙNG LẶP NEO4J GRAPH")
    print("═" * 60)

    print(f"\n📌 SKILL nodes")
    print(f"   Số skill_key bị trùng  : {report['skill_dup_keys']}")
    print(f"   Node dư cần xóa        : {report['skill_extra_nodes']}")
    print("   Ví dụ (3 key đầu):")
    for key, nodes in list(report["skill_groups"].items())[:3]:
        print(f"     skill_key='{key}'")
        for n in nodes:
            print(f"       id={n['id']}  name='{n['name']}'")

    print(f"\n📌 TEACHER nodes")
    print(f"   Số teacher_key bị trùng: {report['teacher_dup_keys']}")
    print(f"   Node dư cần xóa        : {report['teacher_extra_nodes']}")
    print("   Ví dụ (3 key đầu):")
    for key, nodes in list(report["teacher_groups"].items())[:3]:
        print(f"     teacher_key='{key}'")
        for n in nodes:
            print(f"       id={n['id']}  name='{n['name']}'")

    print(f"\n📌 Relationships trùng (cùng source + target + type)")
    print(f"   Số cặp bị trùng        : {report['rel_dup_pairs']}")
    print(f"   Relationship dư cần xóa: {report['rel_extra_rels']}")

    total_nodes  = report["skill_extra_nodes"] + report["teacher_extra_nodes"]
    total_rels   = report["rel_extra_rels"]
    print(f"\n📊 TỔNG KẾT")
    print(f"   Node sẽ bị xóa     : {total_nodes}")
    print(f"   Relation sẽ bị xóa : {total_rels}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  THỰC HIỆN XÓA
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_delete_node_dups(session, groups: dict, key_field: str, label: str):
    """
    Với mỗi nhóm node trùng:
      1. Chọn node 'winner' (name ngắn nhất).
      2. Chuyển tất cả relationship của node dư sang winner.
      3. Xóa node dư.
    """
    deleted = 0
    merged_rels = 0

    for key, nodes in groups.items():
        winner = min(nodes, key=lambda n: len(n.get("name") or ""))
        losers = [n for n in nodes if n["id"] != winner["id"]]

        for loser in losers:
            # Chuyển OUTGOING relationships
            result = session.run(
                """
                MATCH (loser)-[r]->(other)
                WHERE id(loser) = $loser_id AND id(other) <> $winner_id
                MATCH (winner) WHERE id(winner) = $winner_id
                WITH winner, other, type(r) AS rtype, properties(r) AS rprops, r
                CALL apoc.merge.relationship(winner, rtype, {}, rprops, other) YIELD rel
                DELETE r
                RETURN count(r) AS cnt
                """,
                loser_id=loser["id"], winner_id=winner["id"]
            )
            # Nếu không có APOC → dùng cách thủ công
            # (xem bên dưới fallback)

            # Chuyển INCOMING relationships
            session.run(
                """
                MATCH (other)-[r]->(loser)
                WHERE id(loser) = $loser_id AND id(other) <> $winner_id
                MATCH (winner) WHERE id(winner) = $winner_id
                WITH winner, other, type(r) AS rtype, properties(r) AS rprops, r
                CALL apoc.merge.relationship(other, rtype, {}, rprops, winner) YIELD rel
                DELETE r
                """,
                loser_id=loser["id"], winner_id=winner["id"]
            )

            # Xóa node dư (và toàn bộ rel còn lại với winner)
            session.run(
                "MATCH (n) WHERE id(n) = $id DETACH DELETE n",
                id=loser["id"]
            )
            deleted += 1

    return deleted


def merge_and_delete_node_dups_no_apoc(session, groups: dict):
    """
    Fallback không cần APOC:
    Tạo lại relationship mới trỏ tới winner, rồi xóa node dư.
    Relationship types được lấy động qua Cypher.
    """
    deleted = 0

    for key, nodes in groups.items():
        winner = min(nodes, key=lambda n: len(n.get("name") or ""))
        losers = [n for n in nodes if n["id"] != winner["id"]]

        for loser in losers:
            # Lấy danh sách outgoing rels
            out_rels = session.run(
                """
                MATCH (loser)-[r]->(other)
                WHERE id(loser) = $lid AND id(other) <> $wid
                RETURN id(other) AS other_id, type(r) AS rtype, properties(r) AS rprops
                """,
                lid=loser["id"], wid=winner["id"]
            ).data()

            for rel in out_rels:
                session.run(
                    f"""
                    MATCH (w) WHERE id(w) = $wid
                    MATCH (o) WHERE id(o) = $oid
                    MERGE (w)-[r:`{rel['rtype']}`]->(o)
                    SET r += $props
                    """,
                    wid=winner["id"], oid=rel["other_id"], props=rel["rprops"]
                )

            # Lấy danh sách incoming rels
            in_rels = session.run(
                """
                MATCH (other)-[r]->(loser)
                WHERE id(loser) = $lid AND id(other) <> $wid
                RETURN id(other) AS other_id, type(r) AS rtype, properties(r) AS rprops
                """,
                lid=loser["id"], wid=winner["id"]
            ).data()

            for rel in in_rels:
                session.run(
                    f"""
                    MATCH (w) WHERE id(w) = $wid
                    MATCH (o) WHERE id(o) = $oid
                    MERGE (o)-[r:`{rel['rtype']}`]->(w)
                    SET r += $props
                    """,
                    wid=winner["id"], oid=rel["other_id"], props=rel["rprops"]
                )

            # Xóa node dư
            session.run(
                "MATCH (n) WHERE id(n) = $id DETACH DELETE n",
                id=loser["id"]
            )
            deleted += 1

    return deleted


def delete_dup_relationships(session, rel_dups: list) -> int:
    deleted = 0
    for dup in rel_dups:
        for rel_id in dup["delete"]:
            session.run(
                "MATCH ()-[r]-() WHERE id(r) = $id DELETE r",
                id=rel_id
            )
            deleted += 1
    return deleted


def check_apoc(session) -> bool:
    try:
        session.run("RETURN apoc.version() AS v")
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(uri, user, password, execute: bool):
    print(f"[+] Kết nối: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            print("[+] Đang phân tích graph …")
            report = analyze(session)
            print_report(report)

            if not execute:
                print("\n⚠️  Chế độ DRY-RUN — chưa xóa gì cả.")
                print("   Thêm flag --execute để thực sự xóa trùng.\n")
                return

            has_apoc = check_apoc(session)
            print(f"\n[+] APOC available: {has_apoc}")
            print("[+] Bắt đầu dedup …\n")

            # ── Xóa relationship trùng trước ────────────────────────────────
            print("  ► Xóa relationship trùng …")
            n_rels = delete_dup_relationships(session, report["rel_dups"])
            print(f"    ✅ Đã xóa {n_rels} relationship trùng")

            # ── Xóa SKILL nodes trùng ────────────────────────────────────────
            print("  ► Merge & xóa SKILL nodes trùng …")
            if has_apoc:
                n_skills = merge_and_delete_node_dups(
                    session, report["skill_groups"], "skill_key", "SKILL"
                )
            else:
                n_skills = merge_and_delete_node_dups_no_apoc(
                    session, report["skill_groups"]
                )
            print(f"    ✅ Đã xóa {n_skills} SKILL node trùng")

            # ── Xóa TEACHER nodes trùng ──────────────────────────────────────
            print("  ► Merge & xóa TEACHER nodes trùng …")
            if has_apoc:
                n_teachers = merge_and_delete_node_dups(
                    session, report["teacher_groups"], "teacher_key", "TEACHER"
                )
            else:
                n_teachers = merge_and_delete_node_dups_no_apoc(
                    session, report["teacher_groups"]
                )
            print(f"    ✅ Đã xóa {n_teachers} TEACHER node trùng")

            # ── Xóa rel trùng phát sinh sau merge ────────────────────────────
            print("  ► Xóa relationship trùng phát sinh sau merge …")
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WITH id(a) AS src, id(b) AS tgt, type(r) AS typ, collect(id(r)) AS ids
                WHERE size(ids) > 1
                RETURN ids
                """
            )
            extra_ids = []
            for rec in result:
                extra_ids.extend(rec["ids"][1:])

            if extra_ids:
                session.run(
                    "MATCH ()-[r]-() WHERE id(r) IN $ids DELETE r",
                    ids=extra_ids
                )
            print(f"    ✅ Đã xóa {len(extra_ids)} relationship trùng bổ sung")

            print(f"\n🎉 HOÀN THÀNH!")
            print(f"   Node đã xóa     : {n_skills + n_teachers}")
            print(f"   Relation đã xóa : {n_rels + len(extra_ids)}")

    finally:
        driver.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Dedup Neo4j graph")
    parser.add_argument("--uri",      default=DB_URL)
    parser.add_argument("--user",     default=DB_USER)
    parser.add_argument("--password", default=DB_PASSWORD)
    parser.add_argument(
        "--execute", action="store_true",
        help="Thực sự xóa dữ liệu trùng (mặc định chỉ dry-run)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.uri, args.user, args.password, args.execute)
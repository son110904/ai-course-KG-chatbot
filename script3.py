"""
Script 3: Knowledge Graph Q&A Chatbot
v9 — GraphRAG 3-Tier Community Detection (synchronized với script1 v2, script2 v4)
"""

import os
import re
import json
import uuid
import datetime
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")

MAX_HOPS = int(os.getenv("MAX_HOPS", "3"))
LOG_DIR  = Path("./qa_logs")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: ĐỊNH NGHĨA 3 TẦNG CỘNG ĐỒNG (GRAPHRAG COMMUNITY SCHEMA)
# ══════════════════════════════════════════════════════════════════════════════

RELATIONSHIP_WEIGHTS: dict[str, int] = {
    "PROVIDES":             3,
    "REQUIRES":             3,
    "TEACH":                2,
    "LEADS_TO":             2,
    "MAJOR_OFFERS_SUBJECT": 1,
}

COMMUNITY_LEVELS: dict[str, dict] = {

    "L1_GLOBAL": {
        "id":          "L1_GLOBAL",
        "level":       1,
        "name":        "Hệ sinh thái Đào tạo & Nghề nghiệp",
        "node_labels": {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER"},
        "rel_weights": RELATIONSHIP_WEIGHTS,
        "purpose": (
            "Trả lời câu hỏi chiến lược: xu hướng đào tạo, liên kết toàn diện "
            "giữa chương trình học và thị trường lao động."
        ),
    },

    "L2_ACADEMIC": {
        "id":          "L2_ACADEMIC",
        "level":       2,
        "name":        "Cụm Học thuật (Academic Cluster)",
        # community_L2: MAJOR=2, SUBJECT=2, TEACHER=0 — không đồng nhất, dùng label filter
        "node_labels": {"MAJOR", "SUBJECT", "TEACHER"},
        "rel_weights": {
            "TEACH":                2,
            "MAJOR_OFFERS_SUBJECT": 1,
        },
        "purpose": (
            "Trả lời về chương trình ngành, môn học, giảng viên phụ trách. "
            "Kết nối Teacher ↔ Subject ↔ Major."
        ),
    },

    "L2_CAREER_ALIGNMENT": {
        "id":          "L2_CAREER_ALIGNMENT",
        "level":       2,
        "name":        "Cụm Năng lực & Việc làm (Career Alignment Cluster)",
        # community_L2: SKILL=0, CAREER=1, SUBJECT=2 — không đồng nhất, dùng label filter
        "node_labels": {"SKILL", "CAREER", "SUBJECT"},
        "rel_weights": {
            "PROVIDES": 3,
            "REQUIRES": 3,
        },
        "purpose": (
            "Kết nối đầu ra môn học (Subject→Skill) với yêu cầu thực tế (Career→Skill). "
            "Trả lời về kỹ năng cần thiết, môn học liên quan đến nghề nghiệp."
        ),
    },

    "L3_MAJOR_CENTRIC": {
        "id":          "L3_MAJOR_CENTRIC",
        "level":       3,
        "name":        "Cộng đồng theo Ngành (Major-centric)",
        # community_L3: SUBJECT=0, TEACHER=1, SKILL=2 — không đồng nhất, dùng label filter
        "node_labels": {"SUBJECT", "TEACHER", "SKILL"},
        "rel_weights": {
            "MAJOR_OFFERS_SUBJECT": 1,
            "TEACH":                2,
            "PROVIDES":             3,
        },
        "purpose": (
            "Chi tiết lộ trình một ngành cụ thể: môn học, giảng viên, kỹ năng đầu ra. "
            "Kích hoạt khi câu hỏi nhắc tới Major Code cụ thể."
        ),
    },

    "L3_SKILL_CENTRIC": {
        "id":          "L3_SKILL_CENTRIC",
        "level":       3,
        "name":        "Cộng đồng theo Kỹ năng (Skill-centric)",
        "node_labels": {"SUBJECT", "CAREER"},
        "rel_weights": {
            "PROVIDES": 3,
            "REQUIRES": 3,
        },
        "purpose": (
            "Giá trị của một kỹ năng cụ thể: môn nào dạy + nghề nào yêu cầu. "
            "Kích hoạt khi câu hỏi nhắc tới Skill cụ thể."
        ),
    },
}

# ── Ánh xạ intent → community ID ─────────────────────────────────────────────
INTENT_TO_COMMUNITY: dict[tuple, str] = {
    # Academic cluster
    ("MAJOR",   "SUBJECT"):  "L2_ACADEMIC",
    ("MAJOR",   "TEACHER"):  "L2_ACADEMIC",
    ("SUBJECT", "TEACHER"):  "L2_ACADEMIC",
    ("TEACHER", "SUBJECT"):  "L2_ACADEMIC",
    ("TEACHER", "MAJOR"):    "L2_ACADEMIC",
    # Self-queries học thuật
    ("SUBJECT", "SUBJECT"):  "L2_ACADEMIC",
    ("TEACHER", "TEACHER"):  "L2_ACADEMIC",
    ("MAJOR",   "MAJOR"):    "L1_GLOBAL",

    # Career cluster
    ("MAJOR",   "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("MAJOR",   "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("CAREER",  "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("CAREER",  "SUBJECT"):  "L2_CAREER_ALIGNMENT",
    ("CAREER",  "MAJOR"):    "L2_CAREER_ALIGNMENT",
    ("SKILL",   "MAJOR"):    "L2_CAREER_ALIGNMENT",
    ("SKILL",   "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("SKILL",   "SUBJECT"):  "L2_CAREER_ALIGNMENT",
    ("SUBJECT", "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("SUBJECT", "CAREER"):   "L2_CAREER_ALIGNMENT",
    # Self-queries nghề nghiệp
    ("CAREER",  "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("SKILL",   "SKILL"):    "L2_CAREER_ALIGNMENT",
}


def route_to_community(intent: dict) -> tuple[str, dict]:
    mentioned = intent.get("mentioned_labels") or []
    asked     = intent.get("asked_label", "UNKNOWN")
    keywords  = intent.get("keywords", [])

    # L3_MAJOR_CENTRIC: keyword là mã ngành 7 chữ số
    MAJOR_CODE_PATTERN = re.compile(r"\b\d{7}\b")
    for kw in keywords:
        if MAJOR_CODE_PATTERN.search(str(kw)):
            return "L3_MAJOR_CENTRIC", COMMUNITY_LEVELS["L3_MAJOR_CENTRIC"]

    # L3_SKILL_CENTRIC: hỏi về skill cụ thể → career hoặc subject
    if asked in ("CAREER", "SUBJECT") and "SKILL" in mentioned:
        long_kws = [k for k in keywords if len(k.split()) >= 2]
        if long_kws:
            return "L3_SKILL_CENTRIC", COMMUNITY_LEVELS["L3_SKILL_CENTRIC"]

    # Lookup intent map
    first_mentioned = mentioned[0] if mentioned else None
    cid = INTENT_TO_COMMUNITY.get((first_mentioned, asked))
    if not cid:
        for m in mentioned:
            cid = INTENT_TO_COMMUNITY.get((m, asked))
            if cid:
                break
    if not cid:
        cid = "L1_GLOBAL"

    return cid, COMMUNITY_LEVELS[cid]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: LOUVAIN COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def run_louvain_and_write(driver, community_def: dict) -> dict:
    level      = community_def["level"]
    cid        = community_def["id"]
    prop_key   = f"community_L{level}"
    graph_name = f"neo_edu_{cid.lower()}"

    stats = {"community_id": cid, "level": level, "nodes_written": 0, "error": None}

    if level == 1:
        with driver.session() as session:
            r = session.run(
                "MATCH (n) WHERE (n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER) "
                f"SET n.{prop_key} = 0 RETURN count(n) AS cnt"
            ).single()
            stats["nodes_written"] = r["cnt"] if r else 0
        return stats

    with driver.session() as session:
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        node_labels = list(community_def["node_labels"])
        rel_proj    = {
            rtype: {"type": rtype, "orientation": "UNDIRECTED",
                    "properties": {"weight": {"defaultValue": w}}}
            for rtype, w in community_def["rel_weights"].items()
        }

        try:
            session.run(
                "CALL gds.graph.project($gname, $nlabels, $rproj)",
                gname=graph_name, nlabels=node_labels, rproj=rel_proj,
            )
        except Exception as e:
            stats["error"] = f"GDS project error: {e}"
            _fallback_community_assignment(driver, community_def, prop_key)
            return stats

        try:
            session.run(
                f"CALL gds.louvain.write('{graph_name}', "
                f"{{relationshipWeightProperty: 'weight', writeProperty: '{prop_key}'}})"
            )
            r = session.run(
                f"MATCH (n) WHERE n.{prop_key} IS NOT NULL RETURN count(n) AS cnt"
            ).single()
            stats["nodes_written"] = r["cnt"] if r else 0
        except Exception as e:
            stats["error"] = f"GDS Louvain error: {e}"
            _fallback_community_assignment(driver, community_def, prop_key)
        finally:
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            except Exception:
                pass

    return stats


def _fallback_community_assignment(driver, community_def: dict, prop_key: str):
    """
    Fallback assignment khớp với dữ liệu thực tế trong DB:
      L2: MAJOR=2, SUBJECT=2, CAREER=1, SKILL=0, TEACHER=0
      L3: SUBJECT=0, TEACHER=1, CAREER=1, SKILL=2
    """
    cid = community_def["id"]
    label_to_community = {
        "L2_ACADEMIC":          {"TEACHER": 0, "SUBJECT": 2, "MAJOR": 2},
        "L2_CAREER_ALIGNMENT":  {"SKILL": 0, "CAREER": 1, "SUBJECT": 2},
        "L3_MAJOR_CENTRIC":     {"SUBJECT": 0, "TEACHER": 1, "SKILL": 2},
        "L3_SKILL_CENTRIC":     {"SUBJECT": 0, "CAREER": 1},
    }.get(cid, {})

    with driver.session() as session:
        for label, comm_val in label_to_community.items():
            session.run(f"MATCH (n:{label}) SET n.{prop_key} = {comm_val}")


def initialize_communities(driver, force_rebuild: bool = False):
    print("\n[Community Init] Bắt đầu khởi tạo 3 tầng cộng đồng...")

    if not force_rebuild:
        with driver.session() as session:
            r = session.run(
                "MATCH (n) WHERE n.community_L2 IS NOT NULL RETURN count(n) AS cnt LIMIT 1"
            ).single()
            if r and r["cnt"] > 0:
                print("[Community Init] Community L2/L3 đã tồn tại, bỏ qua rebuild.")
                return

    BUILD_ORDER = ["L1_GLOBAL", "L2_ACADEMIC", "L2_CAREER_ALIGNMENT",
                   "L3_MAJOR_CENTRIC", "L3_SKILL_CENTRIC"]

    for cid in BUILD_ORDER:
        cdef  = COMMUNITY_LEVELS[cid]
        level = cdef["level"]
        print(f"  [L{level}] Building: {cdef['name']}...")
        stats = run_louvain_and_write(driver, cdef)
        if stats.get("error"):
            print(f"    ⚠ Fallback (no GDS): {stats['error'][:80]}")
        else:
            print(f"    ✓ {stats['nodes_written']} nodes tagged (community_L{level})")

    print("[Community Init] Hoàn tất.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: AGGREGATION QUERY ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_AGG_ALL_MAJOR_TOKENS = (
    r"tất cả(?: các)? ngành|mọi ngành|"
    r"các ngành đều|"
    r"ngành nào cũng|"
    r"chung cho(?: tất cả| mọi| các)?(?: các)? ngành|"
    r"môn chung|môn bắt buộc chung|môn(?: học)? bắt buộc"
)

AGGREGATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"môn(?: học)?(?: nào)?(?: là)? chung(?: giữa| của)?(.*?)(?:\s+và\s+)(.*?)(?:\?|$)",
        re.IGNORECASE | re.UNICODE,
    ), "subject_intersection_two"),
    (re.compile(
        r"(?:môn(?: học)?(?: gì| nào)?.*?(?:" + _AGG_ALL_MAJOR_TOKENS + r")"
        r"|(?:" + _AGG_ALL_MAJOR_TOKENS + r").*?(?:môn|học phần))",
        re.IGNORECASE | re.UNICODE,
    ), "subject_intersection_all"),
    (re.compile(
        r"ngành(?: nào)?.{0,20}(?:nhiều môn|nhiều học phần).{0,15}nhất",
        re.IGNORECASE | re.UNICODE,
    ), "major_most_subjects"),
    (re.compile(
        r"(?:nghề|career|vị trí).{0,20}(?:nhiều kỹ năng|nhiều skill).{0,15}nhất",
        re.IGNORECASE | re.UNICODE,
    ), "career_most_skills"),
    (re.compile(
        r"môn(?: học)?(?: nào)?.{0,30}(?:nhiều ngành|phổ biến nhất|nhiều nhất)",
        re.IGNORECASE | re.UNICODE,
    ), "subject_most_majors"),
    (re.compile(
        r"(?:kỹ năng|skill)(?: nào)?.{0,30}(?:nhiều môn|phổ biến nhất)",
        re.IGNORECASE | re.UNICODE,
    ), "skill_most_subjects"),
    (re.compile(
        r"(?:có|tổng)(?: tất cả)? bao nhiêu (ngành|môn|nghề|kỹ năng|giảng viên)",
        re.IGNORECASE | re.UNICODE,
    ), "count_entities"),
]


def detect_aggregation_type(question: str) -> str | None:
    for pattern, agg_type in AGGREGATION_PATTERNS:
        if pattern.search(question):
            return agg_type
    return None


def run_aggregation_query(driver, question: str, agg_type: str) -> list[dict]:
    results = []
    with driver.session() as session:

        if agg_type == "subject_intersection_all":
            rows = session.run("""
                MATCH (m:MAJOR)
                WITH count(m) AS total_majors
                MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                WITH s, count(DISTINCT m) AS major_count, total_majors
                WHERE major_count = total_majors
                RETURN s.name AS name, s.code AS code, major_count
                ORDER BY s.name ASC
            """).data()
            if not rows:
                rows = session.run("""
                    MATCH (m:MAJOR)
                    WITH count(m) AS total_majors
                    MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                    WITH s, count(DISTINCT m) AS major_count, total_majors
                    WHERE major_count >= toInteger(total_majors * 0.8)
                    RETURN s.name AS name, s.code AS code,
                           major_count, total_majors
                    ORDER BY major_count DESC LIMIT 30
                """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_count": r.get("major_count"), "hops": 1,
                    "_agg_meta": f"Xuất hiện trong {r.get('major_count')} ngành",
                })

        elif agg_type == "subject_intersection_two":
            rows = session.run("""
                MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                WITH s, collect(DISTINCT toLower(m.name)) AS major_names,
                     count(DISTINCT m) AS major_count
                WHERE major_count >= 2
                RETURN s.name AS name, s.code AS code,
                       major_names, major_count
                ORDER BY major_count DESC LIMIT 50
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_names": r.get("major_names"),
                    "major_count": r.get("major_count"), "hops": 1,
                })

        elif agg_type == "major_most_subjects":
            rows = session.run("""
                MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(s:SUBJECT)
                WITH m, count(DISTINCT s) AS subject_count
                RETURN m.name AS name, m.code AS code, subject_count
                ORDER BY subject_count DESC LIMIT 10
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "MAJOR", "code": r["code"],
                    "subject_count": r.get("subject_count"), "hops": 1,
                    "_agg_meta": f"{r.get('subject_count')} môn học",
                })

        elif agg_type == "career_most_skills":
            rows = session.run("""
                MATCH (c:CAREER)-[:REQUIRES]->(sk:SKILL)
                WITH c, count(DISTINCT sk) AS skill_count
                RETURN c.name AS name, skill_count
                ORDER BY skill_count DESC LIMIT 10
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "CAREER",
                    "skill_count": r.get("skill_count"), "hops": 1,
                    "_agg_meta": f"{r.get('skill_count')} kỹ năng",
                })

        elif agg_type == "subject_most_majors":
            rows = session.run("""
                MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(s:SUBJECT)
                WITH s, count(DISTINCT m) AS major_count
                RETURN s.name AS name, s.code AS code, major_count
                ORDER BY major_count DESC LIMIT 15
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_count": r.get("major_count"), "hops": 1,
                    "_agg_meta": f"Được dạy trong {r.get('major_count')} ngành",
                })

        elif agg_type == "skill_most_subjects":
            rows = session.run("""
                MATCH (s:SUBJECT)-[:PROVIDES]->(sk:SKILL)
                WITH sk, count(DISTINCT s) AS subject_count
                RETURN sk.name AS name, subject_count
                ORDER BY subject_count DESC LIMIT 15
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SKILL",
                    "subject_count": r.get("subject_count"), "hops": 1,
                    "_agg_meta": f"Được cung cấp bởi {r.get('subject_count')} môn",
                })

        elif agg_type == "count_entities":
            q_lower = question.lower()
            if "ngành" in q_lower:        label, vn = "MAJOR",   "ngành"
            elif "nghề" in q_lower:       label, vn = "CAREER",  "nghề"
            elif "kỹ năng" in q_lower or "skill" in q_lower:
                                          label, vn = "SKILL",   "kỹ năng"
            elif "giảng viên" in q_lower: label, vn = "TEACHER", "giảng viên"
            else:                         label, vn = "SUBJECT", "môn học"
            cnt = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()["cnt"]
            results.append({
                "name": f"Tổng số {vn}: {cnt}", "label": label,
                "count": cnt, "hops": 0,
                "_agg_meta": f"count={cnt}",
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: SCHEMA + CONSTRAINTS + SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_DESC = """
Nodes (dữ liệu thực tế trong DB):
  MAJOR   (37 ngành):   code, name, name_vi, name_en
                        + philosophy_and_objectives, admission_requirements,
                          learning_outcomes, po_plo_matrix,
                          training_process_and_graduation_conditions,
                          curriculum_structure_and_content,
                          teaching_and_assessment_methods,
                          reference_programs, lecturer_and_teaching_assistant_standards,
                          facilities_and_learning_resources

  SUBJECT (802 môn):    code, name, name_vi, name_en
                        + course_description, courses_goals, assessment,
                          learning_resources, course_requirements_and_expectations,
                          syllabus_adjustment_time, week_1..week_N (kế hoạch giảng dạy)

  CAREER  (27 nghề):    career_key, name, name_vi, name_en, field_name
                        + description (JSON: short_description, role_in_organization),
                          job_tasks, education_certification, market

  SKILL   (5217 kỹ năng): skill_key, name, skill_type (hard|soft)

  TEACHER (695 GV):     teacher_key, name, email, title

Relationships (đồng bộ script1 v2, script2 v4):
  (MAJOR)  -[:MAJOR_OFFERS_SUBJECT {semester, required_type}]-> (SUBJECT)  (1421)
  (SUBJECT)-[:PROVIDES {mastery_level}]->                       (SKILL)    (8069)
  (TEACHER)-[:TEACH]->                                          (SUBJECT)  (3981)
  (CAREER) -[:REQUIRES {required_level}]->                      (SKILL)    (223)
  (SUBJECT)-[:PREREQUISITE_FOR]->                               (SUBJECT)  (24)
  (MAJOR)  -[:LEADS_TO]->                                       (CAREER)   (6)
"""

RELATIONSHIP_CONSTRAINTS = {
    ("MAJOR", "CAREER"):   (
        "MAJOR -[:LEADS_TO]-> CAREER. "
        "Liệt kê Career mà Major dẫn đến. KHÔNG đề cập SUBJECT trừ khi được hỏi."
    ),
    ("CAREER", "SKILL"):   (
        "CAREER -[:REQUIRES]-> SKILL và SUBJECT -[:PROVIDES]-> SKILL. "
        "Trả lời kỹ năng cần + môn cung cấp kỹ năng đó."
    ),
    ("MAJOR", "SKILL"):    (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT -[:PROVIDES]-> SKILL. "
        "Kỹ năng đạt được từ các môn trong chương trình. Kèm tên môn (mã môn)."
    ),
    ("SKILL", "MAJOR"):    (
        "SKILL <-[:PROVIDES]- SUBJECT <-[:MAJOR_OFFERS_SUBJECT]- MAJOR. "
        "Ngành học có môn cung cấp kỹ năng đó. Kèm mã ngành, tên môn trung gian."
    ),
    ("CAREER", "SUBJECT"): (
        "CAREER -[:REQUIRES]-> SKILL <-[:PROVIDES]- SUBJECT. "
        "Môn học cung cấp kỹ năng nghề yêu cầu. Kèm mã môn + kỹ năng tương ứng."
    ),
    ("MAJOR", "SUBJECT"):  (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT. "
        "Môn học thuộc chương trình ngành, kèm mã môn, học kỳ (semester), "
        "loại (required_type: required=bắt buộc, elective=tự chọn)."
    ),
    ("SKILL", "CAREER"):   (
        "SKILL <-[:REQUIRES]- CAREER. Nghề nghiệp yêu cầu kỹ năng đó."
    ),
    ("CAREER", "MAJOR"):   (
        "MAJOR -[:LEADS_TO]-> CAREER. Ngành học dẫn đến nghề đó, kèm mã ngành."
    ),
    ("SUBJECT", "SKILL"):  (
        "SUBJECT -[:PROVIDES]-> SKILL. Kỹ năng đạt được sau khi học môn đó."
    ),
    ("SKILL", "SUBJECT"):  (
        "SKILL <-[:PROVIDES]- SUBJECT. Môn học (kèm mã môn) cung cấp kỹ năng đó."
    ),
    ("SUBJECT", "TEACHER"): (
        "TEACHER -[:TEACH]-> SUBJECT. Giảng viên phụ trách môn đó."
    ),
    ("TEACHER", "SUBJECT"): (
        "TEACHER -[:TEACH]-> SUBJECT. Môn học thầy/cô đó phụ trách, kèm mã môn."
    ),
    ("MAJOR", "TEACHER"):  (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT <-[:TEACH]- TEACHER. "
        "Giảng viên dạy trong chương trình ngành đó."
    ),
    ("TEACHER", "MAJOR"):  (
        "TEACHER -[:TEACH]-> SUBJECT <-[:MAJOR_OFFERS_SUBJECT]- MAJOR. "
        "Ngành học thầy/cô đó tham gia giảng dạy."
    ),
    ("MAJOR", "MAJOR"):    (
        "So sánh: MAJOR -[:LEADS_TO]-> CAREER và MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT. "
        "So sánh cơ hội nghề nghiệp và môn học đặc trưng của từng ngành."
    ),
    # Self-queries
    ("SUBJECT", "SUBJECT"): (
        "Trả lời: mã môn (code), mô tả môn học (course_description), "
        "mục tiêu (courses_goals), đánh giá (assessment), "
        "môn tiên quyết nếu có (PREREQUISITE_FOR)."
    ),
    ("CAREER", "CAREER"):  (
        "Trả lời: mô tả nghề (description.short_description), "
        "công việc chính (job_tasks), lĩnh vực (field_name), "
        "thị trường (market), kỹ năng yêu cầu (REQUIRES→SKILL)."
    ),
    ("TEACHER", "TEACHER"): (
        "Trả lời: học hàm/học vị (title), email, "
        "môn đang dạy (TEACH→SUBJECT)."
    ),
    ("MAJOR", "MAJOR_DETAIL"): (
        "Trả lời chi tiết ngành: mục tiêu đào tạo (philosophy_and_objectives), "
        "chuẩn đầu ra (learning_outcomes), cơ hội nghề nghiệp (LEADS_TO→CAREER)."
    ),
}

ANSWER_SYSTEM_BASE = """Bạn là trợ lý tư vấn học thuật cho Đại học Kinh tế Quốc dân (NEU).

{schema}

==================================================
LUẬT TUYỆT ĐỐI:
==================================================
A. CHỈ dùng đúng tên/code/thông tin có trong [DỮ LIỆU GRAPH].
B. TUYỆT ĐỐI KHÔNG thêm kỹ năng, môn học, nghề nghiệp từ kiến thức bên ngoài.
C. TUYỆT ĐỐI KHÔNG liệt kê mục chung chung nếu không có trong [DỮ LIỆU GRAPH].
D. Mọi tên SKILL/SUBJECT/CAREER/MAJOR phải lấy nguyên văn từ [DỮ LIỆU GRAPH].
E. Mọi mã môn (code) phải lấy nguyên văn từ field "code".
F. Nếu [DỮ LIỆU GRAPH] trống → trả lời:
   "Dữ liệu hiện tại chưa đủ để tư vấn về [chủ đề]. Bạn có thể liên hệ phòng đào tạo."

ĐỊNH DẠNG:
- Tiếng Việt tự nhiên, thân thiện.
- Môn học: "Tên môn (mã môn)" — VD: "Toán rời rạc (TOCB1107)".
- Ngành: "Tên ngành (mã ngành)" — VD: "Công nghệ thông tin (7480201)".
- Môn bắt buộc/tự chọn: lấy từ field required_type (required=bắt buộc, elective=tự chọn).
- Khi người dùng phủ định (không giỏi X) → bỏ X khỏi gợi ý.
- KHÔNG hỏi ngược lại người dùng.

SỬ DỤNG THUỘC TÍNH MỞ RỘNG KHI CÓ:
- SUBJECT: dùng course_description, courses_goals khi hỏi nội dung môn học.
- CAREER:  dùng description, job_tasks, market khi hỏi về nghề nghiệp.
- MAJOR:   dùng philosophy_and_objectives, learning_outcomes khi hỏi về ngành.
- Nếu field là JSON string → parse và trình bày ngắn gọn phần liên quan.

RÀNG BUỘC THEO LOẠI CÂU HỎI:
{constraint}

CỘNG ĐỒNG ĐÃ ĐƯỢC ĐỊNH TUYẾN:
{community_context}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: ABBREVIATION EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

ABBREVIATION_MAP: dict[str, list[str]] = {
    "da":   ["data analyst", "phân tích dữ liệu"],
    "de":   ["data engineer", "kỹ sư dữ liệu"],
    "ds":   ["data scientist", "khoa học dữ liệu"],
    "data analyst":     ["phân tích dữ liệu", "chuyên viên phân tích dữ liệu"],
    "data engineer":    ["kỹ sư dữ liệu"],
    "data scientist":   ["khoa học dữ liệu", "nhà khoa học dữ liệu"],
    "data engineering": ["kỹ sư dữ liệu"],
    "data analysis":    ["phân tích dữ liệu"],
    "ba":   ["business analyst", "phân tích kinh doanh"],
    "pm":   ["project manager", "quản lý dự án"],
    "po":   ["product owner"],
    "qa":   ["kiểm thử", "quality assurance"],
    "dev":  ["lập trình viên", "developer"],
    "fe":   ["front end", "lập trình viên frontend"],
    "be":   ["back end", "lập trình viên backend"],
    "ml":   ["machine learning", "học máy"],
    "ai":   ["trí tuệ nhân tạo", "artificial intelligence"],
    "cntt": ["công nghệ thông tin"],
    "ktpm": ["kỹ thuật phần mềm"],
    "httt": ["hệ thống thông tin"],
    "qtkd": ["quản trị kinh doanh"],
    "tcnh": ["tài chính ngân hàng"],
    "kt":   ["kế toán", "kinh tế"],
    "mkt":  ["marketing"],
    "hr":   ["quản trị nhân lực", "nhân sự"],
}


def expand_abbreviations(question: str) -> tuple[str, list[str]]:
    q_lower  = question.lower()
    expanded = question
    extras   = []
    found    = {}

    for abbrev, expansions in ABBREVIATION_MAP.items():
        if len(abbrev) <= 3:
            pat = r"(?<![\w\u00C0-\u024F])" + re.escape(abbrev.upper()) + r"(?![\w\u00C0-\u024F])"
            if not re.search(pat, question, re.UNICODE):
                continue
        pattern = r"(?<![\w\u00C0-\u024F])" + re.escape(abbrev) + r"(?![\w\u00C0-\u024F])"
        if re.search(pattern, q_lower, re.IGNORECASE | re.UNICODE):
            found[abbrev] = expansions
            extras.extend(expansions)

    if found:
        hints    = "; ".join(f"{k.upper()} = {' / '.join(v)}" for k, v in found.items())
        expanded = question + f"  [GHI CHÚ: {hints}]"

    return expanded, extras


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 6: INTENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_query_intent(ai_client: OpenAI, question: str) -> dict:
    system_msg = (
        "Bạn phân tích câu hỏi tư vấn học thuật và trả về JSON.\n"
        "Schema Node labels: MAJOR, SUBJECT, SKILL, CAREER, TEACHER\n\n"
        "Chuẩn hóa keyword:\n"
        "  data analyst/DA → phân tích dữ liệu, data analyst\n"
        "  business analyst/BA → phân tích kinh doanh\n"
        "  CNTT/IT → công nghệ thông tin\n"
        "  KTPM → kỹ thuật phần mềm | HTTT → hệ thống thông tin\n"
        "  developer/DEV → lập trình viên | tester/QA → kiểm thử\n\n"
        "Quy tắc xác định asked_label:\n"
        "  - Hỏi thông tin môn học (mô tả, mã môn, nội dung, kế hoạch giảng dạy) → asked=SUBJECT\n"
        "  - Hỏi thông tin nghề nghiệp (mô tả nghề, công việc, thị trường lao động) → asked=CAREER\n"
        "  - Hỏi thông tin giảng viên (email, học hàm, dạy môn gì) → asked=TEACHER\n"
        "  - Hỏi thông tin ngành học (chương trình, chuẩn đầu ra, mục tiêu) → asked=MAJOR\n"
        "  - Hỏi kỹ năng → asked=SKILL\n\n"
        "Trả về JSON:\n"
        "{\n"
        '  "keywords": ["tên thực thể để tìm trong KG"],\n'
        '  "mentioned_labels": ["MAJOR|SUBJECT|SKILL|CAREER|TEACHER"],\n'
        '  "asked_label": "MAJOR|SUBJECT|SKILL|CAREER|TEACHER|UNKNOWN",\n'
        '  "negated_keywords": ["thực thể bị phủ định"],\n'
        '  "is_comparison": false\n'
        "}\n"
    )
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Phân tích: {question}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return {
        "keywords":         parsed.get("keywords", []),
        "mentioned_labels": parsed.get("mentioned_labels", []),
        "asked_label":      parsed.get("asked_label", "UNKNOWN"),
        "negated_keywords": parsed.get("negated_keywords", []),
        "is_comparison":    parsed.get("is_comparison", False),
    }


def get_relationship_constraint(intent: dict) -> str:
    mentioned = intent.get("mentioned_labels", [])
    asked     = intent.get("asked_label", "UNKNOWN")
    is_comp   = intent.get("is_comparison", False)

    if is_comp and "MAJOR" in mentioned:
        return RELATIONSHIP_CONSTRAINTS.get(("MAJOR", "MAJOR"), "")

    for m in ([mentioned[0]] if mentioned else []) + mentioned:
        key = (m, asked)
        if key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[key]

    # Self-query fallback
    if asked != "UNKNOWN":
        self_key = (asked, asked)
        if self_key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[self_key]

    return "Trả lời theo đúng câu hỏi, chỉ dùng dữ liệu trong Knowledge Graph."


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7: COMMUNITY-AWARE TRAVERSAL
# ══════════════════════════════════════════════════════════════════════════════

# Extended props được fetch từ DB và đưa vào context cho LLM
EXTENDED_PROPS: dict[str, list[str]] = {
    "SUBJECT": [
        "course_description", "courses_goals", "assessment",
        "learning_resources", "course_requirements_and_expectations",
    ],
    "CAREER": [
        "description", "job_tasks", "field_name", "market",
    ],
    "MAJOR": [
        "philosophy_and_objectives", "admission_requirements",
        "learning_outcomes", "curriculum_structure_and_content",
    ],
    "TEACHER": ["email", "title"],
    "SKILL":   ["skill_type"],
}

# Targeted Queries — trả về các columns chuẩn: name, label, code, rel_types, node_names, hops
# + extended cols: course_description, semester, required_type
TARGETED_QUERIES: dict[tuple[str, str], str] = {

    # ── Academic ──────────────────────────────────────────────────────────────
    ("MAJOR", "SUBJECT"): """
        MATCH (start:MAJOR)-[r:MAJOR_OFFERS_SUBJECT]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['MAJOR_OFFERS_SUBJECT'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               r.semester AS semester,
               r.required_type AS required_type,
               n.course_description AS course_description
        ORDER BY r.required_type DESC, r.semester ASC, n.name ASC
        LIMIT 100
    """,
    ("MAJOR", "TEACHER"): """
        MATCH (n:TEACHER)-[:TEACH]->(sub:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(start:MAJOR)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['MAJOR_OFFERS_SUBJECT','TEACH'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("TEACHER", "SUBJECT"): """
        MATCH (start:TEACHER)-[:TEACH]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['TEACH'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("TEACHER", "MAJOR"): """
        MATCH (start:TEACHER)-[:TEACH]->(sub:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(n:MAJOR)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['TEACH','MAJOR_OFFERS_SUBJECT'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "TEACHER"): """
        MATCH (n:TEACHER)-[:TEACH]->(start:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['TEACH'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    # Self: thông tin chi tiết môn học + môn tiên quyết
    ("SUBJECT", "SUBJECT"): """
        MATCH (start:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN start.name AS name, labels(start)[0] AS label, start.code AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type,
               start.course_description AS course_description
        ORDER BY start.name LIMIT 10
        UNION
        MATCH (start:SUBJECT)-[:PREREQUISITE_FOR]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['PREREQUISITE_FOR'] AS rel_types,
               [start.name, n.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 30
    """,
    # Self: thông tin giảng viên
    ("TEACHER", "TEACHER"): """
        MATCH (start:TEACHER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
    """,

    # ── Career cluster ────────────────────────────────────────────────────────
    ("MAJOR", "CAREER"): """
        MATCH (start:MAJOR)-[:LEADS_TO]->(n:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['LEADS_TO'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("CAREER", "SKILL"): """
        MATCH (start:CAREER)-[:REQUIRES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['REQUIRES'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("CAREER", "SUBJECT"): """
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        OPTIONAL MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(n)
        WHERE m.code IN start.major_codes
        WITH start, sk, n,
             count(DISTINCT m) AS major_match,
             size([(s2:SUBJECT)-[:PROVIDES]->(sk) | s2]) AS skill_breadth
        ORDER BY major_match DESC, skill_breadth ASC, n.name ASC
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['REQUIRES','PROVIDES'] AS rel_types,
               [start.name, sk.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        LIMIT 30
    """,
    ("CAREER", "MAJOR"): """
        MATCH (n:MAJOR)-[:LEADS_TO]->(start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['LEADS_TO'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("MAJOR", "SKILL"): """
        MATCH (start:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(sub:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['MAJOR_OFFERS_SUBJECT','PROVIDES'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "MAJOR"): """
        MATCH (n:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(sub:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['MAJOR_OFFERS_SUBJECT','PROVIDES'] AS rel_types,
               [n.name, sub.name, start.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "CAREER"): """
        MATCH (n:CAREER)-[:REQUIRES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['REQUIRES'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "SUBJECT"): """
        MATCH (n:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['PROVIDES'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "SKILL"): """
        MATCH (start:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['PROVIDES'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "CAREER"): """
        MATCH (start:SUBJECT)-[:PROVIDES]->(sk:SKILL)<-[:REQUIRES]-(n:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['PROVIDES','REQUIRES'] AS rel_types,
               [start.name, sk.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    # Self: thông tin chi tiết nghề nghiệp
    ("CAREER", "CAREER"): """
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
    """,
    # Skill self-lookup
    ("SKILL", "SKILL"): """
        MATCH (start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
    """,
}


def _add_node_and_paths(rec, all_nodes: list, all_paths: list):
    """Thêm node và path vào context, kèm extended props."""
    node = {
        "name":  rec["name"],
        "label": rec["label"],
        "code":  rec.get("code"),
        "hops":  rec["hops"],
    }
    # Extended props từ targeted query
    for field in ("course_description", "semester", "required_type"):
        val = rec.get(field)
        if val is not None:
            node[field] = val

    all_nodes.append(node)

    node_names = rec.get("node_names") or []
    rel_types  = rec.get("rel_types") or []
    for i, rel in enumerate(rel_types):
        all_paths.append({
            "from":     node_names[i]   if i < len(node_names) else "",
            "to":       node_names[i+1] if i+1 < len(node_names) else "",
            "relation": rel,
            "hop":      i + 1,
        })


def fetch_node_details(driver, nodes: list[dict]) -> list[dict]:
    """
    Enrich nodes với extended properties từ DB.
    Chỉ fetch khi node chưa có extended props và là SUBJECT/CAREER/MAJOR.
    """
    to_fetch: dict[str, list[str]] = {"SUBJECT": [], "CAREER": [], "MAJOR": []}
    node_map: dict[str, dict] = {}

    for n in nodes:
        label = n.get("label", "")
        name  = n.get("name", "")
        if not name:
            continue
        node_map[name] = n
        if label in to_fetch:
            has_extended = any(n.get(p) for p in EXTENDED_PROPS.get(label, []))
            if not has_extended:
                to_fetch[label].append(name)

    with driver.session() as session:
        if to_fetch["SUBJECT"]:
            rows = session.run("""
                MATCH (n:SUBJECT) WHERE n.name IN $names
                RETURN n.name AS name,
                       n.course_description AS course_description,
                       n.courses_goals AS courses_goals,
                       n.assessment AS assessment,
                       n.learning_resources AS learning_resources,
                       n.course_requirements_and_expectations AS course_requirements_and_expectations
            """, names=to_fetch["SUBJECT"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

        if to_fetch["CAREER"]:
            rows = session.run("""
                MATCH (n:CAREER) WHERE n.name IN $names
                RETURN n.name AS name,
                       n.description AS description,
                       n.job_tasks AS job_tasks,
                       n.field_name AS field_name,
                       n.market AS market
            """, names=to_fetch["CAREER"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

        if to_fetch["MAJOR"]:
            rows = session.run("""
                MATCH (n:MAJOR) WHERE n.name IN $names
                RETURN n.name AS name,
                       n.philosophy_and_objectives AS philosophy_and_objectives,
                       n.admission_requirements AS admission_requirements,
                       n.learning_outcomes AS learning_outcomes,
                       n.curriculum_structure_and_content AS curriculum_structure_and_content
            """, names=to_fetch["MAJOR"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

    return nodes


def multihop_traversal_community_aware(
    driver,
    keywords:      list[str],
    max_hops:      int = MAX_HOPS,
    intent:        dict | None = None,
    community_def: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Traversal 3-phase community-aware:
    Phase 1 — TARGETED Cypher theo intent.
    Phase 2 — BFS label-scoped (KHÔNG dùng community number filter vì không đồng nhất).
    Phase 3 — CROSS-CLUSTER BRIDGE (L2/L3).
    """
    all_nodes:  list[dict] = []
    all_paths:  list[dict] = []
    seen_names: set[str]   = set()

    mentioned_labels = (intent or {}).get("mentioned_labels", [])
    asked_label      = (intent or {}).get("asked_label", "UNKNOWN")
    first_mentioned  = mentioned_labels[0] if mentioned_labels else None

    if community_def:
        allowed_labels = community_def["node_labels"]
        level          = community_def["level"]
        comm_id        = community_def["id"]
    else:
        allowed_labels = {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER"}
        level          = 1
        comm_id        = "L1_GLOBAL"

    print(f"  [community] Routing to: {comm_id} (Level {level})")
    print(f"  [community] Scope labels: {allowed_labels}")

    # ── Phase 1: Targeted query ───────────────────────────────────────────────
    targeted_key    = (first_mentioned, asked_label) if first_mentioned else None
    targeted_cypher = TARGETED_QUERIES.get(targeted_key) if targeted_key else None

    # Fallback: self-lookup
    if not targeted_cypher and asked_label != "UNKNOWN":
        self_key = (asked_label, asked_label)
        if self_key in TARGETED_QUERIES:
            targeted_key    = self_key
            targeted_cypher = TARGETED_QUERIES[self_key]

    if targeted_cypher:
        with driver.session() as session:
            for kw in keywords:
                try:
                    for rec in session.run(targeted_cypher, kw=kw):
                        _add_node_and_paths(rec, all_nodes, all_paths)
                except Exception as e:
                    print(f"  [targeted] WARNING: {e}")
        if all_nodes:
            print(f"  [targeted] ({targeted_key}) → {len(all_nodes)} nodes")

    # ── Phase 2: BFS label-scoped ─────────────────────────────────────────────
    # Dùng allowed_labels filter, KHÔNG filter theo community number
    # (vì MAJOR=2, SUBJECT=2, TEACHER=0 tại L2 — không đồng nhất)
    label_clauses = " OR ".join(f"n:{lbl}" for lbl in allowed_labels)

    with driver.session() as session:
        for kw in keywords:
            seed_rows = session.run("""
                MATCH (seed)
                WHERE (seed:MAJOR OR seed:SUBJECT OR seed:SKILL
                       OR seed:CAREER OR seed:TEACHER)
                  AND (toLower(seed.name) CONTAINS toLower($kw)
                       OR (seed.code IS NOT NULL AND seed.code = $kw)
                       OR (seed.career_key IS NOT NULL
                           AND toLower(seed.career_key) CONTAINS toLower($kw))
                       OR (seed.teacher_key IS NOT NULL
                           AND toLower(seed.teacher_key) CONTAINS toLower($kw))
                       OR (seed.skill_key IS NOT NULL
                           AND toLower(seed.skill_key) CONTAINS toLower($kw)))
                WITH seed, size([(seed)-[]-() | 1]) AS degree
                RETURN seed
                ORDER BY degree DESC
                LIMIT 3
            """, kw=kw).data()
            seeds = [r["seed"] for r in seed_rows]

            for seed in seeds:
                seed_name = seed.get("name", "")
                if seed_name in seen_names:
                    continue
                seen_names.add(seed_name)

                # Thêm seed node vào context (kèm extended props)
                seed_labels = list(seed.labels) if hasattr(seed, "labels") else []
                seed_label  = seed_labels[0] if seed_labels else "UNKNOWN"
                seed_node   = {
                    "name":  seed_name,
                    "label": seed_label,
                    "code":  seed.get("code"),
                    "hops":  0,
                }
                for prop in EXTENDED_PROPS.get(seed_label, []):
                    val = seed.get(prop)
                    if val is not None:
                        seed_node[prop] = val
                all_nodes.append(seed_node)

                # BFS label-scoped traversal
                traversal_query = f"""
                    MATCH path = (start)-[*1..{max_hops}]-(n)
                    WHERE start.name = $seed_name
                      AND ({label_clauses})
                    WITH n, path,
                         [r IN relationships(path) | type(r)] AS rel_types,
                         [x IN nodes(path) | x.name]          AS node_names
                    RETURN DISTINCT
                        n.name                 AS name,
                        labels(n)[0]           AS label,
                        n.code                 AS code,
                        n.course_description   AS course_description,
                        null                   AS semester,
                        null                   AS required_type,
                        rel_types,
                        node_names,
                        length(path)           AS hops
                    ORDER BY hops ASC
                    LIMIT 60
                """
                try:
                    for rec in session.run(traversal_query, seed_name=seed_name):
                        _add_node_and_paths(rec, all_nodes, all_paths)
                except Exception as e:
                    print(f"  [BFS] WARNING seed={seed_name}: {e}")

    # ── Phase 3: Cross-cluster bridge (L2/L3) ────────────────────────────────
    if level >= 2 and asked_label not in (None, "UNKNOWN"):
        bridge_pairs = [
            ("L2_ACADEMIC", "CAREER",
             "MATCH (m:MAJOR)-[:LEADS_TO]->(n:CAREER) WHERE m.name IN $names "
             "RETURN n.name AS name, 'CAREER' AS label, null AS code, "
             "['LEADS_TO'] AS rel_types, [m.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description"),

            ("L2_CAREER_ALIGNMENT", "SUBJECT",
             "MATCH (c:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT) "
             "WHERE c.name IN $names "
             "OPTIONAL MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(n) WHERE m.code IN c.major_codes "
             "WITH c, sk, n, count(DISTINCT m) AS major_match, "
             "size([(s2:SUBJECT)-[:PROVIDES]->(sk) | s2]) AS skill_breadth "
             "ORDER BY major_match DESC, skill_breadth ASC "
             "RETURN n.name AS name, 'SUBJECT' AS label, n.code AS code, "
             "['REQUIRES','PROVIDES'] AS rel_types, [c.name, sk.name, n.name] AS node_names, 2 AS hops, "
             "null AS semester, null AS required_type, n.course_description AS course_description "
             "LIMIT 20"),
        ]
        seed_names = list({n["name"] for n in all_nodes if n.get("name")})[:20]

        if seed_names:
            with driver.session() as session:
                for bridge_cid, bridge_label, bridge_q in bridge_pairs:
                    if comm_id != bridge_cid:
                        continue
                    if bridge_label != asked_label and asked_label != "UNKNOWN":
                        continue
                    try:
                        for rec in session.run(bridge_q, names=seed_names):
                            _add_node_and_paths(rec, all_nodes, all_paths)
                        print(f"  [bridge] {bridge_cid}→{bridge_label}: added")
                    except Exception as e:
                        print(f"  [bridge] WARNING: {e}")

    return all_nodes, all_paths


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8: GENERATE ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    ai_client:    OpenAI,
    question:     str,
    ranked_nodes: list[dict],
    traversal_paths: list[dict],
    intent:       dict,
    community_def: dict | None = None,
    override_constraint: str | None = None,
) -> str:
    context = json.dumps({
        "ranked_results":  ranked_nodes,
        "traversal_paths": traversal_paths[:60],
    }, ensure_ascii=False, indent=2)

    constraint = (
        override_constraint if override_constraint is not None
        else get_relationship_constraint(intent)
    )

    negated = intent.get("negated_keywords", [])
    if negated:
        constraint += (
            f"\n\nLƯU Ý PHỦ ĐỊNH: Người dùng KHÔNG giỏi/thích: {negated}. "
            "Loại bỏ khỏi gợi ý."
        )

    if community_def:
        community_context = (
            f"Tầng {community_def['level']} — {community_def['name']}\n"
            f"Mục tiêu: {community_def['purpose']}"
        )
    else:
        community_context = "L1 Global — Toàn bộ hệ sinh thái đào tạo"

    system_prompt = ANSWER_SYSTEM_BASE.format(
        schema=SCHEMA_DESC,
        constraint=constraint,
        community_context=community_context,
    )

    no_data_hint = ""
    if not ranked_nodes:
        no_data_hint = (
            "\n[CẢNH BÁO: Không tìm thấy dữ liệu trong Knowledge Graph. "
            "Thông báo lịch sự, không bịa thông tin.]"
        )

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"Câu hỏi: {question}\n\n"
                f"[DỮ LIỆU GRAPH]:\n{context}"
                f"{no_data_hint}\n\n"
                "Trả lời CHỈ dùng tên/code từ [DỮ LIỆU GRAPH]:"
            )},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 9: PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def ask(driver, ai_client: OpenAI, question: str, query_id: str | None = None) -> dict:
    if query_id is None:
        query_id = "q" + uuid.uuid4().hex[:6]

    print(f"\n{'='*60}")
    print(f"Q [{query_id}]: {question}")

    # ── Bước 0: Aggregation Router ────────────────────────────────────────────
    agg_type = detect_aggregation_type(question)
    if agg_type:
        print(f"  [aggregation] {agg_type}")
        agg_nodes = run_aggregation_query(driver, question, agg_type)
        print(f"  [aggregation] {len(agg_nodes)} nodes")

        agg_intent = {
            "keywords": [], "mentioned_labels": [], "negated_keywords": [],
            "is_comparison": False, "agg_type": agg_type,
            "asked_label": (
                "SUBJECT" if "subject" in agg_type else
                "MAJOR"   if "major"   in agg_type else
                "CAREER"  if "career"  in agg_type else
                "SKILL"   if "skill"   in agg_type else "UNKNOWN"
            ),
        }
        agg_constraint = (
            "Câu hỏi thống kê/tập hợp. Dữ liệu đã tổng hợp từ graph. "
            "Trình bày rõ ràng, kèm mã môn/ngành, số liệu (_agg_meta). "
            "Nếu intersection: giải thích đây là môn tất cả ngành đều học. "
            "Nếu ranking: liệt kê từ cao xuống thấp."
        )
        answer = generate_answer(
            ai_client, question, agg_nodes, [],
            intent=agg_intent,
            community_def=COMMUNITY_LEVELS["L1_GLOBAL"],
            override_constraint=agg_constraint,
        )
        print(f"\nA: {answer}")
        return _build_record(query_id, question, answer, [], agg_intent,
                             agg_nodes, [], "aggregation")

    # ── Bước 0b: Expand viết tắt ──────────────────────────────────────────────
    expanded_question, abbrev_keywords = expand_abbreviations(question)
    if abbrev_keywords:
        print(f"  [abbrev] {abbrev_keywords}")

    # ── Bước 1: Extract intent ────────────────────────────────────────────────
    intent = extract_query_intent(ai_client, expanded_question)
    intent["keywords"] = list(dict.fromkeys(intent["keywords"] + abbrev_keywords))
    keywords = intent["keywords"]
    print(f"  Keywords: {keywords}")
    print(f"  Intent: mentioned={intent['mentioned_labels']} "
          f"asked={intent['asked_label']} negated={intent['negated_keywords']}")

    # ── Bước 2: Community Routing ─────────────────────────────────────────────
    community_id, community_def = route_to_community(intent)
    intent["community_id"] = community_id

    # ── Bước 3: Community-aware Traversal ────────────────────────────────────
    raw_nodes, traversal_paths = multihop_traversal_community_aware(
        driver, keywords, max_hops=MAX_HOPS,
        intent=intent, community_def=community_def,
    )
    print(f"  Traversal: {len(raw_nodes)} nodes | {len(traversal_paths)} paths")

    # ── Bước 4: Dedup + Negation filter ──────────────────────────────────────
    negated_lower = [kw.lower() for kw in intent.get("negated_keywords", [])]
    seen: dict[tuple, dict] = {}
    for n in raw_nodes:
        key = (n.get("label", ""), n.get("name", ""))
        if key not in seen or (n.get("hops") or 99) < (seen[key].get("hops") or 99):
            seen[key] = n
    context_nodes = [
        n for n in seen.values()
        if not any(neg in (n.get("name") or "").lower() for neg in negated_lower)
    ]
    print(f"  Context nodes (dedup+negation): {len(context_nodes)}")

    # ── Bước 4b: Enrich extended props khi cần ───────────────────────────────
    asked = intent.get("asked_label", "UNKNOWN")
    if asked in ("SUBJECT", "CAREER", "MAJOR") and len(context_nodes) <= 20:
        context_nodes = fetch_node_details(driver, context_nodes)
        print(f"  [enrich] Extended props fetched for: {asked}")

    # ── Bước 5: LLM answer ───────────────────────────────────────────────────
    answer = generate_answer(
        ai_client, question, context_nodes, traversal_paths,
        intent=intent, community_def=community_def,
    )
    print(f"\nA: {answer}")

    return _build_record(
        query_id, question, answer, keywords, intent,
        context_nodes, traversal_paths,
        f"Targeted+BFS label-scoped [{community_id}]",
    )


def _build_record(
    query_id, question, answer, keywords, intent,
    context_nodes, traversal_paths, algorithm_desc,
) -> dict:
    return {
        "query_id":         query_id,
        "query":            question,
        "generated_answer": answer,
        "keywords":         keywords,
        "intent":           intent,
        "community_id":     intent.get("community_id", ""),
        "retrieved_nodes": [
            {
                "node_id":  f"node{i+1:03d}",
                "content":  json.dumps(n, ensure_ascii=False),
                "entities": [n.get("name", "")],
            }
            for i, n in enumerate(context_nodes)
        ],
        "traversal_path": traversal_paths[:20],
        "timestamp":      datetime.datetime.now().isoformat(),
        "algorithm": {
            "community_detection": "Louvain weighted (GDS) + rule-based fallback",
            "traversal":           algorithm_desc,
            "weights":             RELATIONSHIP_WEIGHTS,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 10: MAIN + INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def interactive_loop(driver, ai_client: OpenAI):
    print("\n🎓 Knowledge Graph Chatbot v9 — GraphRAG 3-Tier (label-scoped BFS)")
    print("DB: 6778 nodes | 13724 rels")
    print("    MAJOR=37 | SUBJECT=802 | CAREER=27 | SKILL=5217 | TEACHER=695")
    print("Rels: TEACH | PROVIDES | REQUIRES | LEADS_TO | MAJOR_OFFERS_SUBJECT | PREREQUISITE_FOR")
    print(f"max_hops={MAX_HOPS} | BFS dùng label filter (không dùng community number filter)")
    print("Gõ câu hỏi. Nhập 'exit' để thoát.\n")

    counter = 1
    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "thoat", "thoát"):
            print("Tạm biệt!")
            break
        ask(driver, ai_client, question, query_id=f"q{counter:03d}")
        counter += 1


def main():
    print("Starting KG Chatbot v9 (GraphRAG 3-Tier, label-scoped BFS)...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()
    try:
        initialize_communities(driver, force_rebuild=False)
        interactive_loop(driver, ai_client)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
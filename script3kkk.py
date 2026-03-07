"""
Script 3: Knowledge Graph Q&A Chatbot
v9 — GraphRAG 3-Tier Community Detection (synchronized với script1 v2, script2 v4)

Dữ liệu thực tế trong Neo4j (6778 nodes, 13724 rels):
  MAJOR    (37):   code, name, name_vi, name_en, philosophy_and_objectives,
                   admission_requirements, learning_outcomes, po_plo_matrix,
                   training_process_and_graduation_conditions,
                   curriculum_structure_and_content, teaching_and_assessment_methods,
                   reference_programs, lecturer_and_teaching_assistant_standards,
                   facilities_and_learning_resources
                   community_L2=2

  SUBJECT  (802):  code, name, name_vi, name_en,
                   course_description, courses_goals, assessment,
                   learning_resources, course_requirements_and_expectations,
                   syllabus_adjustment_time, week_1..week_N
                   community_L2=2, community_L3=0

  CAREER   (27):   career_key, name, name_vi, name_en, field_name,
                   description, job_tasks, education_certification, market, major_codes
                   community_L2=1, community_L3=1

  SKILL    (5217): skill_key, name, skill_type
                   community_L2=0, community_L3=2

  TEACHER  (695):  teacher_key, name, email, title
                   community_L2=0, community_L3=1

Relationships:
  MAJOR    -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT   (1421)
  SUBJECT  -[:PROVIDES]->             SKILL     (8069)
  TEACHER  -[:TEACH]->               SUBJECT   (3981)
  CAREER   -[:REQUIRES]->            SKILL     (223)
  SUBJECT  -[:PREREQUISITE_FOR]->    SUBJECT   (24)
  MAJOR    -[:LEADS_TO]->            CAREER    (6)

QUAN TRỌNG — Community filter:
  Community numbers KHÔNG đồng nhất trong 1 cluster:
    L2_ACADEMIC:         MAJOR(L2=2), SUBJECT(L2=2), TEACHER(L2=0) — khác nhau
    L2_CAREER_ALIGNMENT: SKILL(L2=0), CAREER(L2=1), SUBJECT(L2=2) — khác nhau
    L3_MAJOR_CENTRIC:    SUBJECT(L3=0), TEACHER(L3=1), SKILL(L3=2) — khác nhau
  → BFS dùng allowed_labels filter (label-based), KHÔNG dùng community number filter.
  → community_Lx props chỉ dùng cho initialize_communities / Louvain.
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

    # L3_SKILL_CENTRIC: hỏi về 1 SKILL CỤ THỂ (keyword là tên kỹ năng, dài ≥ 2 từ)
    # Điều kiện: SKILL là label duy nhất được đề cập (không có CAREER trong mentioned)
    # VD đúng: "Kỹ năng Python có giá trị thế nào?" → mentioned=[SKILL], asked=CAREER
    # VD sai:  "Nghề Tester cần kỹ năng gì?" → mentioned=[CAREER, SKILL], asked=CAREER → L2
    skill_only = "SKILL" in mentioned and "CAREER" not in mentioned
    if asked in ("CAREER", "SUBJECT") and skill_only:
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
    # ── Skill constraints: luôn kèm môn học ─────────────────────────────────
    ("SUBJECT", "SKILL"):  (
        "SUBJECT -[:PROVIDES]-> SKILL. "
        "Liệt kê kỹ năng đạt được sau khi học môn đó, kèm tên môn (mã môn)."
    ),
    ("SKILL", "SUBJECT"):  (
        "SKILL <-[:PROVIDES]- SUBJECT. "
        "Liệt kê môn học (tên + mã môn) cung cấp kỹ năng đó."
    ),
    ("MAJOR", "SKILL"):    (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT -[:PROVIDES]-> SKILL. "
        "Với mỗi kỹ năng, kèm tên môn (mã môn) trung gian cung cấp kỹ năng đó."
    ),
    ("SKILL", "MAJOR"):    (
        "SKILL <-[:PROVIDES]- SUBJECT <-[:MAJOR_OFFERS_SUBJECT]- MAJOR. "
        "Kèm tên môn (mã môn) trung gian giữa skill và ngành."
    ),
    # ── Self-queries: truy thuộc tính thực thể ────────────────────────────────
    ("SUBJECT", "SUBJECT"): (
        "Trả lời theo đúng nội dung câu hỏi, ưu tiên field tương ứng:\n"
        "- Hỏi mã môn → dùng field code.\n"
        "- Hỏi mô tả / giới thiệu → dùng course_description.\n"
        "- Hỏi mục tiêu / chuẩn đầu ra → dùng courses_goals.\n"
        "- Hỏi tài liệu / sách tham khảo → dùng learning_resources.\n"
        "- Hỏi đánh giá / hình thức thi → dùng assessment.\n"
        "- Hỏi yêu cầu / điều kiện → dùng course_requirements_and_expectations.\n"
        "- Hỏi môn tiên quyết → dùng PREREQUISITE_FOR.\n"
        "Nếu câu hỏi hỏi chung (giới thiệu, thông tin, cho biết về...) → "
        "trả lời: code, course_description, courses_goals, môn tiên quyết nếu có."
    ),
    ("CAREER", "CAREER"):  (
        "Trả lời đầy đủ 4 phần:\n"
        "1. Mô tả nghề: dùng description.short_description hoặc role_in_organization.\n"
        "2. Công việc chính: liệt kê từ job_tasks.\n"
        "3. Kỹ năng và môn học tương ứng:\n"
        "   - Dùng field skill_to_subjects trong [DỮ LIỆU GRAPH] để biết skill nào được dạy ở môn nào.\n"
        "   - Format mỗi dòng: • [Tên skill] → [Tên môn (mã môn)].\n"
        "   - Nếu skill không có trong skill_to_subjects → chỉ liệt kê tên skill, không bịa môn.\n"
        "4. ĐỀ XUẤT NGÀNH HỌC: liệt kê recommended_majors (tên + mã ngành). "
        "   Nếu không có → dùng education_certification.recommended_majors làm gợi ý.\n"
        "Thị trường lao động: tóm tắt market nếu câu hỏi hỏi về cơ hội việc làm."
    ),
    ("TEACHER", "TEACHER"): (
        "Trả lời theo đúng câu hỏi, ưu tiên field tương ứng:\n"
        "- Hỏi thông tin chung → title, email, danh sách môn đang dạy (TEACH→SUBJECT).\n"
        "- Hỏi email → dùng field email.\n"
        "- Hỏi học hàm/học vị → dùng field title.\n"
        "- Hỏi dạy môn gì → liệt kê SUBJECT (tên + mã môn)."
    ),
    ("MAJOR", "MAJOR"):    (
        "Trả lời theo đúng câu hỏi, ưu tiên field tương ứng:\n"
        "- Hỏi thông tin chung / giới thiệu → code, philosophy_and_objectives, learning_outcomes.\n"
        "- Hỏi cơ hội nghề nghiệp → LEADS_TO→CAREER (tên nghề).\n"
        "- Hỏi môn học → MAJOR_OFFERS_SUBJECT→SUBJECT (tên + mã môn).\n"
        "- So sánh 2 ngành → so sánh CAREER và SUBJECT đặc trưng.\n"
        "- Hỏi yêu cầu đầu vào → dùng admission_requirements.\n"
        "- Hỏi chuẩn đầu ra → dùng learning_outcomes."
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
- skill_to_subjects: map sẵn skill → danh sách môn dạy skill đó. Khi liệt kê kỹ năng, LUÔN kiểm tra field này và kèm môn học tương ứng.
- Nếu field là JSON string → parse và trình bày ngắn gọn phần liên quan.

ĐỀ XUẤT NGÀNH HỌC (BẮT BUỘC khi trả lời về CAREER):
- Luôn kiểm tra field "recommended_majors" trong dữ liệu — đây là các MAJOR node được map qua major_codes.
- Nếu có → liệt kê "Tên ngành (mã ngành)" ở cuối câu trả lời.
- Nếu không có recommended_majors nhưng có education_certification → dùng tên trong recommended_majors của nó làm gợi ý (không có mã).
- KHÔNG bịa ngành không có trong [DỮ LIỆU GRAPH].

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


# Stopwords xuất hiện đầu keyword do LLM thêm vào
_KW_STOPWORDS = re.compile(
    r"^(môn học|học phần|môn |kỹ năng |kỹ năng$|nghề |ngành học|ngành |"
    r"giảng viên |thầy |cô |sinh viên ngành |tài liệu môn |"
    r"mã môn |mã của môn |thông tin môn |giới thiệu môn )",
    re.IGNORECASE | re.UNICODE,
)

# Pattern loại bỏ suffix thừa
_KW_SUFFIX = re.compile(
    r"\s*(là gì|như thế nào|có gì|không|ạ|\?|\.)$",
    re.IGNORECASE | re.UNICODE,
)

def _normalize_keyword(kw: str) -> str:
    """
    Loại bỏ prefix/suffix stopword thừa khỏi keyword để match DB.
    VD: "môn lập trình web" → "lập trình web"
        "mã của môn lập trình web" → "lập trình web"
        "kỹ năng python là gì" → "python"
    """
    kw = kw.strip()
    # Bỏ suffix thừa
    kw = _KW_SUFFIX.sub("", kw).strip()
    # Lặp bỏ prefix thừa (tối đa 4 lần cho nested: "mã của môn học lập trình web")
    for _ in range(4):
        new = _KW_STOPWORDS.sub("", kw).strip()
        if new == kw:
            break
        kw = new
    # Bỏ "của " đầu còn sót (VD: "của môn lập trình" sau khi bỏ "mã")
    kw = re.sub(r"^của\s+", "", kw, flags=re.IGNORECASE | re.UNICODE).strip()
    return kw


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
        "Quy tắc xác định asked_label (chỉ 1 giá trị, KHÔNG dùng | hay /):\n"
        "  - Hỏi về nghề (kỹ năng cần, cơ hội việc làm, phù hợp ngành nào) → asked=CAREER\n"
        "  - Hỏi thông tin môn học (mô tả, mã môn, nội dung, kế hoạch giảng dạy) → asked=SUBJECT\n"
        "  - Hỏi thông tin giảng viên (email, học hàm, dạy môn gì) → asked=TEACHER\n"
        "  - Hỏi thông tin ngành học (chương trình, chuẩn đầu ra, mục tiêu) → asked=MAJOR\n"
        "  - Hỏi kỹ năng đơn thuần (kỹ năng X là gì, môn nào dạy) → asked=SKILL\n"
        "  - Khi hỏi kết hợp nghề + kỹ năng + ngành → ưu tiên asked=CAREER\n\n"
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
    asked_raw = parsed.get("asked_label", "UNKNOWN")
    # Normalize: LLM đôi khi trả "SKILL|CAREER", "MAJOR|CAREER" → tách lấy label đầu
    valid_labels = {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER"}
    if "|" in str(asked_raw):
        parts = [p.strip() for p in asked_raw.split("|")]
        asked_raw = next((p for p in parts if p in valid_labels), "UNKNOWN")
    elif asked_raw not in valid_labels:
        asked_raw = "UNKNOWN"

    return {
        "keywords":         parsed.get("keywords", []),
        "mentioned_labels": [l for l in parsed.get("mentioned_labels", []) if l in valid_labels],
        "asked_label":      asked_raw,
        "negated_keywords": parsed.get("negated_keywords", []),
        "is_comparison":    parsed.get("is_comparison", False),
    }


# Keyword → field hint mapping cho single-entity queries
_KEYWORD_FIELD_HINTS: list[tuple[list[str], str]] = [
    # SUBJECT fields
    (["tài liệu", "sách", "giáo trình", "tham khảo", "học liệu"],
     "→ dùng field learning_resources"),
    (["mã môn", "mã của", "mã số"],
     "→ dùng field code"),
    (["mô tả", "giới thiệu", "nội dung", "về môn"],
     "→ dùng field course_description"),
    (["mục tiêu", "chuẩn đầu ra", "sau khi học"],
     "→ dùng field courses_goals"),
    (["đánh giá", "thi", "kiểm tra", "điểm", "hình thức thi"],
     "→ dùng field assessment"),
    (["yêu cầu", "điều kiện", "tiên quyết", "cần gì để học"],
     "→ dùng field course_requirements_and_expectations"),
    (["kế hoạch", "lịch học", "tuần", "week"],
     "→ dùng field week_1..week_N (kế hoạch giảng dạy từng tuần)"),
    # CAREER fields
    (["mô tả", "là gì", "làm gì", "vai trò"],
     "→ dùng field description.short_description"),
    (["công việc", "nhiệm vụ", "task"],
     "→ dùng field job_tasks"),
    (["thị trường", "cơ hội", "nhu cầu", "lương", "triển vọng"],
     "→ dùng field market"),
    (["chứng chỉ", "bằng cấp", "học vấn"],
     "→ dùng field education_certification"),
    # MAJOR fields
    (["triết lý", "mục tiêu đào tạo", "tầm nhìn"],
     "→ dùng field philosophy_and_objectives"),
    (["chuẩn đầu ra", "kết quả học tập", "năng lực"],
     "→ dùng field learning_outcomes"),
    (["yêu cầu đầu vào", "điều kiện tuyển sinh", "xét tuyển"],
     "→ dùng field admission_requirements"),
    # TEACHER fields
    (["email", "liên hệ", "địa chỉ"],
     "→ dùng field email"),
    (["học hàm", "học vị", "tiến sĩ", "thạc sĩ", "giáo sư", "phó giáo sư"],
     "→ dùng field title"),
]


# SUBJECT-only hints (chỉ apply khi asked=SUBJECT)
_SUBJECT_ONLY_HINTS: set[str] = {
    "→ dùng field course_requirements_and_expectations",
    "→ dùng field week_1..week_N (kế hoạch giảng dạy từng tuần)",
    "→ dùng field courses_goals",
    "→ dùng field course_description",
    "→ dùng field assessment",
    "→ dùng field learning_resources",
    "→ dùng field code",
}
# CAREER-only hints
_CAREER_ONLY_HINTS: set[str] = {
    "→ dùng field job_tasks",
    "→ dùng field market",
    "→ dùng field education_certification",
    "→ dùng field description.short_description",
}


def _detect_field_hint(question: str, asked: str) -> str:
    """Phát hiện keyword trong câu hỏi → gợi ý field cần dùng, lọc theo asked label."""
    q_lower = question.lower()
    hints = []
    for keywords, hint in _KEYWORD_FIELD_HINTS:
        if not any(kw in q_lower for kw in keywords):
            continue
        # Lọc hint không phù hợp với asked label
        if hint in _SUBJECT_ONLY_HINTS and asked != "SUBJECT":
            continue
        if hint in _CAREER_ONLY_HINTS and asked != "CAREER":
            continue
        hints.append(hint)
    if hints:
        return "Câu hỏi này yêu cầu: " + "; ".join(hints[:3])
    return ""


def get_relationship_constraint(intent: dict, question: str = "") -> str:
    mentioned = intent.get("mentioned_labels", [])
    asked     = intent.get("asked_label", "UNKNOWN")
    is_comp   = intent.get("is_comparison", False)

    if is_comp and "MAJOR" in mentioned:
        return RELATIONSHIP_CONSTRAINTS.get(("MAJOR", "MAJOR"), "")

    base_constraint = ""
    for m in ([mentioned[0]] if mentioned else []) + mentioned:
        key = (m, asked)
        if key in RELATIONSHIP_CONSTRAINTS:
            base_constraint = RELATIONSHIP_CONSTRAINTS[key]
            break

    # Self-query fallback
    if not base_constraint and asked != "UNKNOWN":
        self_key = (asked, asked)
        base_constraint = RELATIONSHIP_CONSTRAINTS.get(self_key, "")

    if not base_constraint:
        base_constraint = "Trả lời theo đúng câu hỏi, chỉ dùng dữ liệu trong Knowledge Graph."

    # Inject dynamic field hint nếu là single-entity self-query
    is_self_query = (len(set(mentioned)) <= 1 or not mentioned) and asked != "UNKNOWN"
    if is_self_query and question:
        field_hint = _detect_field_hint(question, asked)
        if field_hint:
            base_constraint = field_hint + "\n" + base_constraint

    return base_constraint


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
    # Self: thông tin chi tiết môn học + môn tiên quyết (đầy đủ extended fields)
    ("SUBJECT", "SUBJECT"): """
        MATCH (start:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        WITH start LIMIT 10
        RETURN start.name AS name, labels(start)[0] AS label, start.code AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type,
               start.course_description AS course_description,
               start.courses_goals AS courses_goals,
               start.learning_resources AS learning_resources,
               start.assessment AS assessment,
               start.course_requirements_and_expectations AS course_requirements_and_expectations
        UNION ALL
        MATCH (start:SUBJECT)-[:PREREQUISITE_FOR]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['PREREQUISITE_FOR'] AS rel_types,
               [start.name, n.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type,
               null AS course_description, null AS courses_goals,
               null AS learning_resources, null AS assessment,
               null AS course_requirements_and_expectations
        ORDER BY name LIMIT 30
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
        UNION
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(sub:SUBJECT)
              <-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
        WHERE (toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw))
          AND m.code IN start.major_codes
        RETURN sub.name AS name, labels(sub)[0] AS label, sub.code AS code,
               ['REQUIRES','PROVIDES'] AS rel_types,
               [start.name, sk.name, sub.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, sub.course_description AS course_description
        ORDER BY sk.name, sub.name LIMIT 20
        UNION
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        MATCH (sub:SUBJECT)
        WHERE toLower(sub.name) CONTAINS toLower(sk.name)
           OR toLower(sk.name) CONTAINS toLower(sub.name)
        RETURN sub.name AS name, labels(sub)[0] AS label, sub.code AS code,
               ['REQUIRES','NAME_MATCH'] AS rel_types,
               [start.name, sk.name, sub.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, sub.course_description AS course_description
        ORDER BY sk.name LIMIT 15
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
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        // Ưu tiên 1: MAJOR via LEADS_TO
        OPTIONAL MATCH (m1:MAJOR)-[:LEADS_TO]->(start)
        // Ưu tiên 2: MAJOR via major_codes
        OPTIONAL MATCH (m2:MAJOR) WHERE m2.code IN start.major_codes
        WITH start, collect(DISTINCT m1) + collect(DISTINCT m2) AS all_majors
        UNWIND all_majors AS m
        WITH DISTINCT m, start
        WHERE m IS NOT NULL
        RETURN m.name AS name, labels(m)[0] AS label, m.code AS code,
               ['RECOMMENDED_MAJOR'] AS rel_types, [start.name, m.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY m.name LIMIT 10
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
        ORDER BY n.name LIMIT 30
        UNION
        MATCH (sub:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN sub.name AS name, labels(sub)[0] AS label, sub.code AS code,
               ['PROVIDES'] AS rel_types, [sub.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, sub.course_description AS course_description
        ORDER BY sub.name LIMIT 30
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
    # Self: thông tin chi tiết nghề nghiệp + kỹ năng + môn học + ngành học đề xuất
    ("CAREER", "CAREER"): """
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
        UNION
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN sk.name AS name, labels(sk)[0] AS label, null AS code,
               ['REQUIRES'] AS rel_types,
               [start.name, sk.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY sk.name LIMIT 50
        UNION
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(sub:SUBJECT)
              <-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
        WHERE (toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw))
          AND m.code IN start.major_codes
        RETURN sub.name AS name, labels(sub)[0] AS label, sub.code AS code,
               ['REQUIRES','PROVIDES'] AS rel_types,
               [start.name, sk.name, sub.name] AS node_names, 2 AS hops,
               null AS semester, null AS required_type, sub.course_description AS course_description
        ORDER BY sk.name, sub.name LIMIT 20
        UNION
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)
        WHERE (toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw))
          AND NOT (sk)<-[:PROVIDES]-(:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(:MAJOR {code: start.major_codes[0]})
        MATCH (sub:SUBJECT)
        WHERE toLower(sub.name) CONTAINS toLower(sk.name)
           OR toLower(sk.name) CONTAINS toLower(sub.name)
        RETURN sub.name AS name, labels(sub)[0] AS label, sub.code AS code,
               ['REQUIRES','NAME_MATCH'] AS rel_types,
               [start.name, sk.name, sub.name] AS node_names, 2 AS hops,
               null AS semester, null AS required_type, sub.course_description AS course_description
        ORDER BY sk.name, sub.name LIMIT 15
        UNION
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        MATCH (m:MAJOR) WHERE m.code IN start.major_codes
        RETURN m.name AS name, labels(m)[0] AS label, m.code AS code,
               ['RECOMMENDED_MAJOR'] AS rel_types,
               [start.name, m.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY m.name LIMIT 10
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
    # Capture tất cả extended fields có trong record (không hardcode)
    _EXTRA_FIELDS = (
        "course_description", "courses_goals", "learning_resources",
        "assessment", "course_requirements_and_expectations",
        "description", "job_tasks", "field_name", "market",
        "philosophy_and_objectives", "learning_outcomes",
        "email", "title", "skill_type",
        "semester", "required_type",
    )
    for field in _EXTRA_FIELDS:
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
                OPTIONAL MATCH (m:MAJOR) WHERE m.code IN n.major_codes
                WITH n, collect({name: m.name, code: m.code}) AS recommended_majors
                RETURN n.name AS name,
                       n.description AS description,
                       n.job_tasks AS job_tasks,
                       n.field_name AS field_name,
                       n.market AS market,
                       n.education_certification AS education_certification,
                       n.major_codes AS major_codes,
                       recommended_majors
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

def _build_skill_subject_map(traversal_paths: list[dict]) -> dict[str, list[str]]:
    """
    Từ traversal_paths, build mapping: skill_name → [subject_name (code)]
    Dùng khi LLM cần trả lời "kỹ năng X được dạy ở môn nào".
    """
    # Bước 1: Collect skill→subject pairs từ paths
    # Path chain: A -REQUIRES-> B -PROVIDES/NAME_MATCH-> C
    # → requires_targets[A] = {B}, provides_sources[B] = {C}
    requires_map: dict[str, set[str]] = {}   # career → skills
    provides_map: dict[str, set[str]] = {}   # skill → subjects

    for p in traversal_paths:
        rel = p.get("relation", "")
        frm = p.get("from", "")
        to  = p.get("to", "")
        if rel == "REQUIRES":
            requires_map.setdefault(frm, set()).add(to)
        elif rel in ("PROVIDES", "NAME_MATCH"):
            provides_map.setdefault(frm, set()).add(to)

    # Bước 2: Chỉ giữ skill có subject
    result: dict[str, list[str]] = {}
    for career_skills in requires_map.values():
        for skill in career_skills:
            subjects = provides_map.get(skill)
            if subjects:
                result[skill] = sorted(subjects)
    return result


def generate_answer(
    ai_client:    OpenAI,
    question:     str,
    ranked_nodes: list[dict],
    traversal_paths: list[dict],
    intent:       dict,
    community_def: dict | None = None,
    override_constraint: str | None = None,
) -> str:
    # Pre-compute skill→subject mapping để LLM không cần tự trace paths
    skill_subject_map = _build_skill_subject_map(traversal_paths)

    # Build subject code lookup
    subject_code: dict[str, str] = {}
    for n in ranked_nodes:
        if n.get("label") == "SUBJECT" and n.get("code"):
            subject_code[n["name"]] = n["code"]

    # Enrich map với code
    skill_subject_display: dict[str, list[str]] = {}
    for skill, subjects in skill_subject_map.items():
        display = []
        for s in subjects:
            code = subject_code.get(s, "")
            display.append(f"{s} ({code})" if code else s)
        skill_subject_display[skill] = display

    context = json.dumps({
        "ranked_results":      ranked_nodes,
        "traversal_paths":     traversal_paths[:80],
        "skill_to_subjects":   skill_subject_display,   # pre-computed map
    }, ensure_ascii=False, indent=2)

    constraint = (
        override_constraint if override_constraint is not None
        else get_relationship_constraint(intent, question=question)
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
    # Normalize keywords: bỏ stopwords đầu/cuối để tránh miss match
    intent["keywords"] = [_normalize_keyword(kw) for kw in intent["keywords"]]
    intent["keywords"] = list(dict.fromkeys(kw for kw in intent["keywords"] if kw))
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
    deduplicated = [
        n for n in seen.values()
        if not any(neg in (n.get("name") or "").lower() for neg in negated_lower)
    ]

    # Post-filter MAJOR: nếu có MAJOR từ targeted query (RECOMMENDED_MAJOR),
    # loại bỏ MAJOR từ BFS (hops >= 2) để tránh ngành không liên quan
    has_recommended_major = any(
        n.get("label") == "MAJOR" and n.get("hops", 99) <= 1
        for n in deduplicated
    )
    if has_recommended_major:
        context_nodes = [
            n for n in deduplicated
            if not (n.get("label") == "MAJOR" and (n.get("hops") or 99) >= 2)
        ]
    else:
        context_nodes = deduplicated
    print(f"  Context nodes (dedup+negation): {len(context_nodes)}")

    # ── Bước 4b: Enrich extended props khi cần ───────────────────────────────
    asked = intent.get("asked_label", "UNKNOWN")
    # Enrich: luôn fetch nếu là self-query (ít SUBJECT/CAREER/MAJOR nodes),
    # hoặc khi tổng context nodes <= 50
    subject_nodes = [n for n in context_nodes if n.get("label") == asked]
    should_enrich = (
        asked in ("SUBJECT", "CAREER", "MAJOR")
        and (len(context_nodes) <= 50 or len(subject_nodes) <= 5)
    )
    if should_enrich:
        context_nodes = fetch_node_details(driver, context_nodes)
        print(f"  [enrich] Extended props fetched for: {asked} ({len(subject_nodes)} target nodes)")

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
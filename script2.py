"""
Script 2 (OPTIMIZED v3): Load extracted KG JSON → generate Cypher TRỰC TIẾP (không dùng LLM)
→ push to Neo4j Aura

Hỗ trợ 3 schema thực tế:
  - CUR  (curriculum):         type MAJOR/SUBJECT/CAREER, rel: major_offers_subject, major_leads_to_career
  - SYL  (syllabus):           type SUBJECT/TEACHER/SKILL, rel: teach, provides, prerequisite_for
  - CAR  (career_description): type CAREER/SKILL,          rel: requires

Relationship names trong Neo4j (đồng bộ với script 1 & 3):
  MAJOR    -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT
  MAJOR    -[:LEADS_TO]->             CAREER
  TEACHER  -[:TEACH]->               SUBJECT
  SUBJECT  -[:PROVIDES]->            SKILL
  SUBJECT  -[:PREREQUISITE_FOR]->    SUBJECT
  CAREER   -[:REQUIRES]->            SKILL

Tất cả idempotent (MERGE everywhere), không dùng LLM.
"""

import os
import json
import logging
from pathlib import Path
from neo4j import GraphDatabase, exceptions as neo4j_exc
from dotenv import load_dotenv

load_dotenv()

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./cache/ingestion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")

LOCAL_OUT_DIR = Path("./cache/output")
FOLDERS = ["curriculum", "career_description", "syllabus"]
# ──────────────────────────────────────────────────────────────────────────────


def _s(value) -> str:
    """Safe strip: trả về chuỗi rỗng nếu value là None."""
    if value is None:
        return ""
    return str(value).strip()


def _esc(value) -> str:
    """Escape single quotes cho Cypher string literals. An toàn với None."""
    return _s(value).replace("\\", "\\\\").replace("'", "\\'")


# ─── SCHEMA DETECTOR ──────────────────────────────────────────────────────────

def detect_schema(kg_data: dict) -> str:
    """
    Tự động nhận dạng loại file từ nội dung JSON.
    Returns: 'curriculum' | 'syllabus' | 'career' | 'unknown'
    """
    nodes     = kg_data.get("nodes", [])
    rels      = kg_data.get("relationships", [])
    node_types = {n.get("type", "") for n in nodes}
    rel_types  = {r.get("rel_type", "") for r in rels}

    # career_description: có CAREER + career_requires_skill, KHÔNG có MAJOR
    if ("CAREER" in node_types or "career_requires_skill" in rel_types) \
            and "MAJOR" not in node_types:
        return "career"
    # syllabus: có TEACHER hoặc teacher_instructs_subject
    if "TEACHER" in node_types or "teacher_instructs_subject" in rel_types:
        return "syllabus"
    # curriculum: có MAJOR hoặc major_offers_subject
    if "MAJOR" in node_types or "major_offers_subject" in rel_types:
        return "curriculum"

    return "unknown"


# ─── CYPHER BUILDERS PER SCHEMA ───────────────────────────────────────────────

# ══════════════════════════════
#  CURRICULUM
# ══════════════════════════════

def cur_node_cypher(node: dict) -> str | None:
    """CUR: MAJOR | SUBJECT | CAREER"""
    t = node.get("type", "")

    if t == "MAJOR":
        code    = _esc(node.get("major_code"))
        name_vi = _esc(node.get("major_name_vi"))
        name_en = _esc(node.get("major_name_en"))
        if not code:
            return None
        return (
            f"MERGE (n:MAJOR {{code: '{code}'}})"
            f" SET n.name = '{name_vi}', n.name_vi = '{name_vi}', n.name_en = '{name_en}'"
        )

    if t == "SUBJECT":
        code    = _esc(node.get("subject_code"))
        name_vi = _esc(node.get("subject_name_vi"))
        name_en = _esc(node.get("subject_name_en"))
        if not code:
            return None
        return (
            f"MERGE (n:SUBJECT {{code: '{code}'}})"
            f" SET n.name = '{name_vi}', n.name_vi = '{name_vi}', n.name_en = '{name_en}'"
        )

    if t == "CAREER":
        key     = _esc(node.get("career_key"))
        name_vi = _esc(node.get("career_name_vi"))
        name_en = _esc(node.get("career_name_en"))
        name    = name_vi or name_en
        if not name:
            return None
        stmt = f"MERGE (n:CAREER {{name: '{name}'}})"
        sets = []
        if key:     sets.append(f"n.career_key = '{key}'")
        if name_vi: sets.append(f"n.name_vi = '{name_vi}'")
        if name_en: sets.append(f"n.name_en = '{name_en}'")
        if sets:
            stmt += " SET " + ", ".join(sets)
        return stmt

    return None


def cur_rel_cypher(rel: dict) -> str | None:
    """CUR: major_offers_subject | major_leads_to_career"""
    rtype = rel.get("rel_type", "")

    # ── MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT ──────────────────────────────
    if rtype == "major_offers_subject":
        major_code   = _esc(rel.get("from_major_code"))
        subject_code = _esc(rel.get("to_subject_code"))
        semester     = rel.get("semester", "")
        req_type     = _esc(rel.get("required_type"))
        if not major_code or not subject_code:
            return None
        if semester != "" and req_type:
            return (
                f"MATCH (a:MAJOR {{code: '{major_code}'}}), (b:SUBJECT {{code: '{subject_code}'}})"
                f" MERGE (a)-[:MAJOR_OFFERS_SUBJECT {{semester: {int(semester)}, required_type: '{req_type}'}}]->(b)"
            )
        else:
            return (
                f"MATCH (a:MAJOR {{code: '{major_code}'}}), (b:SUBJECT {{code: '{subject_code}'}})"
                f" MERGE (a)-[:MAJOR_OFFERS_SUBJECT]->(b)"
            )

    # ── MAJOR -[:LEADS_TO]-> CAREER ───────────────────────────────────────────
    if rtype == "major_leads_to_career":
        major_code  = _esc(rel.get("from_major_code"))
        career_key  = _esc(rel.get("to_career_key"))
        if not major_code or not career_key:
            return None
        # CAREER node được MERGE bằng name, match bằng career_key property
        return (
            f"MATCH (a:MAJOR {{code: '{major_code}'}}), (b:CAREER {{career_key: '{career_key}'}})"
            f" MERGE (a)-[:LEADS_TO]->(b)"
        )

    return None


# ══════════════════════════════
#  SYLLABUS
# ══════════════════════════════

def syl_node_cypher(node: dict) -> str | None:
    """SYL: SUBJECT | TEACHER | SKILL"""
    t = node.get("type", "")

    if t == "SUBJECT":
        code    = _esc(node.get("subject_code"))
        name_vi = _esc(node.get("subject_name_vi"))
        name_en = _esc(node.get("subject_name_en"))
        if not code:
            return None
        return (
            f"MERGE (n:SUBJECT {{code: '{code}'}})"
            f" SET n.name = '{name_vi}', n.name_vi = '{name_vi}', n.name_en = '{name_en}'"
        )

    if t == "TEACHER":
        name  = _esc(node.get("name"))
        email = _esc(node.get("email"))
        title = _esc(node.get("title"))
        key   = _esc(node.get("teacher_key"))
        if not name:
            return None
        stmt = f"MERGE (n:TEACHER {{name: '{name}'}})"
        sets = []
        if email: sets.append(f"n.email = '{email}'")
        if title: sets.append(f"n.title = '{title}'")
        if key:   sets.append(f"n.teacher_key = '{key}'")
        if sets:
            stmt += " SET " + ", ".join(sets)
        return stmt

    if t == "SKILL":
        key        = _esc(node.get("skill_key"))
        name       = _esc(node.get("skill_name"))
        skill_type = _esc(node.get("skill_type"))
        if not name:
            return None
        stmt = f"MERGE (n:SKILL {{name: '{name}'}})"
        sets = []
        if key:        sets.append(f"n.skill_key = '{key}'")
        if skill_type: sets.append(f"n.skill_type = '{skill_type}'")
        if sets:
            stmt += " SET " + ", ".join(sets)
        return stmt

    return None


def syl_rel_cypher(rel: dict) -> str | None:
    """
    SYL:
      teacher_instructs_subject → TEACHER -[:TEACH]-> SUBJECT
      subject_provides_skill    → SUBJECT -[:PROVIDES]-> SKILL
      subject_is_prerequisite_of_subject → SUBJECT -[:PREREQUISITE_FOR]-> SUBJECT
    """
    rtype = rel.get("rel_type", "")

    # ── TEACHER -[:TEACH]-> SUBJECT ───────────────────────────────────────────
    if rtype == "teacher_instructs_subject":
        teacher_key  = _esc(rel.get("from_teacher_key"))
        subject_code = _esc(rel.get("to_subject_code"))
        if not teacher_key or not subject_code:
            return None
        return (
            f"MATCH (a:TEACHER {{teacher_key: '{teacher_key}'}}), (b:SUBJECT {{code: '{subject_code}'}})"
            f" MERGE (a)-[:TEACH]->(b)"
        )

    # ── SUBJECT -[:PROVIDES]-> SKILL ──────────────────────────────────────────
    if rtype == "subject_provides_skill":
        subject_code = _esc(rel.get("from_subject_code"))
        skill_key    = _esc(rel.get("to_skill_key"))
        mastery      = _esc(rel.get("mastery_level"))
        if not subject_code or not skill_key:
            return None
        if mastery:
            return (
                f"MATCH (a:SUBJECT {{code: '{subject_code}'}}), (b:SKILL {{skill_key: '{skill_key}'}})"
                f" MERGE (a)-[:PROVIDES {{mastery_level: '{mastery}'}}]->(b)"
            )
        return (
            f"MATCH (a:SUBJECT {{code: '{subject_code}'}}), (b:SKILL {{skill_key: '{skill_key}'}})"
            f" MERGE (a)-[:PROVIDES]->(b)"
        )

    # ── SUBJECT -[:PREREQUISITE_FOR]-> SUBJECT ────────────────────────────────
    if rtype == "subject_is_prerequisite_of_subject":
        from_code = _esc(rel.get("from_subject_code"))
        to_code   = _esc(rel.get("to_subject_code"))
        if not from_code or not to_code:
            return None
        return (
            f"MATCH (a:SUBJECT {{code: '{from_code}'}}), (b:SUBJECT {{code: '{to_code}'}})"
            f" MERGE (a)-[:PREREQUISITE_FOR]->(b)"
        )

    return None


# ══════════════════════════════
#  CAREER DESCRIPTION
# ══════════════════════════════

def car_node_cypher(node: dict) -> str | None:
    """CAR: CAREER | SKILL"""
    t = node.get("type", "")

    if t == "CAREER":
        key      = _esc(node.get("career_key"))
        name_vi  = _esc(node.get("career_name_vi"))
        name_en  = _esc(node.get("career_name_en"))
        field    = _esc(node.get("field_name"))
        majors   = node.get("major_codes", [])
        name     = name_vi or name_en
        if not name:
            return None
        stmt = f"MERGE (n:CAREER {{name: '{name}'}})"
        sets = []
        if key:     sets.append(f"n.career_key = '{key}'")
        if name_vi: sets.append(f"n.name_vi = '{name_vi}'")
        if name_en: sets.append(f"n.name_en = '{name_en}'")
        if field:   sets.append(f"n.field_name = '{field}'")
        if majors:
            codes_lit = "[" + ", ".join(f"'{_esc(str(c))}'" for c in majors) + "]"
            sets.append(f"n.major_codes = {codes_lit}")
        if sets:
            stmt += " SET " + ", ".join(sets)
        return stmt

    if t == "SKILL":
        key        = _esc(node.get("skill_key"))
        name       = _esc(node.get("skill_name"))
        skill_type = _esc(node.get("skill_type"))
        if not name:
            return None
        stmt = f"MERGE (n:SKILL {{name: '{name}'}})"
        sets = []
        if key:        sets.append(f"n.skill_key = '{key}'")
        if skill_type: sets.append(f"n.skill_type = '{skill_type}'")
        if sets:
            stmt += " SET " + ", ".join(sets)
        return stmt

    return None


def car_rel_cypher(rel: dict) -> str | None:
    """
    CAR: career_requires_skill → CAREER -[:REQUIRES]-> SKILL
    Match CAREER bằng career_key (consistent với MERGE node).
    """
    if rel.get("rel_type") != "career_requires_skill":
        return None

    career_key = _esc(rel.get("from_career_key"))
    skill_key  = _esc(rel.get("to_skill_key"))
    req_level  = _esc(rel.get("required_level"))

    if not career_key or not skill_key:
        return None

    if req_level:
        return (
            f"MATCH (a:CAREER {{career_key: '{career_key}'}}), (b:SKILL {{skill_key: '{skill_key}'}})"
            f" MERGE (a)-[:REQUIRES {{required_level: '{req_level}'}}]->(b)"
        )
    return (
        f"MATCH (a:CAREER {{career_key: '{career_key}'}}), (b:SKILL {{skill_key: '{skill_key}'}})"
        f" MERGE (a)-[:REQUIRES]->(b)"
    )


# ─── DISPATCH ─────────────────────────────────────────────────────────────────

NODE_BUILDERS = {
    "curriculum": cur_node_cypher,
    "syllabus":   syl_node_cypher,
    "career":     car_node_cypher,
}

REL_BUILDERS = {
    "curriculum": cur_rel_cypher,
    "syllabus":   syl_rel_cypher,
    "career":     car_rel_cypher,
}


def kg_to_cypher_statements(kg_data: dict, schema: str) -> list[str]:
    """Chuyển KG JSON → list Cypher statements theo schema đã detect."""
    node_fn = NODE_BUILDERS.get(schema)
    rel_fn  = REL_BUILDERS.get(schema)

    if not node_fn:
        log.warning(f"  Schema '{schema}' không có builder, bỏ qua.")
        return []

    statements = []

    for node in kg_data.get("nodes", []):
        stmt = node_fn(node)
        if stmt:
            statements.append(stmt)
        else:
            log.debug(f"  [skip node] {node.get('type')} – thiếu key bắt buộc")

    for rel in kg_data.get("relationships", []):
        stmt = rel_fn(rel)
        if stmt:
            statements.append(stmt)
        else:
            log.debug(f"  [skip rel] {rel.get('rel_type')} – thiếu key bắt buộc")

    return statements


# ─── NEO4J ────────────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_pool_size=50,
        connection_timeout=30,
    )


def create_indexes(session):
    stmts = [
        # Unique constraints
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MAJOR)   REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:SUBJECT) REQUIRE n.code IS UNIQUE",
        # Node indexes
        "CREATE INDEX IF NOT EXISTS FOR (n:SKILL)   ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:SKILL)   ON (n.skill_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:CAREER)  ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:CAREER)  ON (n.career_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TEACHER) ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TEACHER) ON (n.teacher_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:MAJOR)   ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:SUBJECT) ON (n.name)",
    ]
    for stmt in stmts:
        try:
            session.run(stmt)
        except Exception as e:
            log.warning(f"[Index] {e}")
    log.info("Indexes/constraints ready.")


def run_statements_in_tx(session, statements: list[str], label: str) -> tuple[int, int]:
    """Chạy tất cả statements trong 1 write transaction."""
    if not statements:
        return 0, 0

    ok = fail = 0

    def tx_func(tx):
        nonlocal ok, fail
        for stmt in statements:
            try:
                tx.run(stmt)
                ok += 1
            except neo4j_exc.CypherSyntaxError as e:
                fail += 1
                log.error(f"  [CypherError] {e.message}\n  → {stmt[:300]}")
            except neo4j_exc.ConstraintError:
                ok += 1  # MERGE race condition, bỏ qua
            except Exception as e:
                fail += 1
                log.error(f"  [StmtError] {e}\n  → {stmt[:300]}")

    try:
        session.execute_write(tx_func)
    except Exception as e:
        log.error(f"  [TX FAILED] {label}: {e}")
        return 0, len(statements)

    return ok, fail


# ─── PROCESS FILES ────────────────────────────────────────────────────────────

def process_files(driver):
    with driver.session() as session:
        log.info("Creating indexes / constraints...")
        create_indexes(session)

        total_ok = total_fail = total_files = 0

        for folder in FOLDERS:
            folder_path = LOCAL_OUT_DIR / folder
            if not folder_path.exists():
                log.warning(f"Folder không tồn tại: {folder_path}")
                continue

            files = sorted(folder_path.glob("*.json"))
            if not files:
                log.warning(f"Không có file JSON trong {folder_path}")
                continue

            log.info(f"\n{'='*60}")
            log.info(f"Folder: {folder} ({len(files)} files)")

            for jf in files:
                total_files += 1
                log.info(f"  → {jf.name}")

                try:
                    with open(jf, encoding="utf-8") as f:
                        kg_data = json.load(f)
                except json.JSONDecodeError as e:
                    log.error(f"    JSON parse error: {e}")
                    total_fail += 1
                    continue

                schema = detect_schema(kg_data)
                log.info(f"    Detected schema: {schema}")

                if schema == "unknown":
                    log.warning(f"    Không nhận dạng được schema, bỏ qua.")
                    continue

                statements = kg_to_cypher_statements(kg_data, schema)
                log.info(f"    Generated {len(statements)} statements")

                if not statements:
                    log.warning(f"    Không có statements nào, bỏ qua.")
                    continue

                ok, fail = run_statements_in_tx(session, statements, jf.name)
                total_ok   += ok
                total_fail += fail
                log.info(f"    ✓ {ok}  ✗ {fail}")

        log.info(f"\n{'='*60}")
        log.info(f"TỔNG KẾT: {total_files} files | ✓ {total_ok} statements | ✗ {total_fail} lỗi")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting Neo4j ingestion pipeline (v3 – synchronized)...")

    if not NEO4J_URI:
        raise ValueError("DB_URL không tìm thấy trong .env")

    driver = get_driver()
    try:
        process_files(driver)
    finally:
        driver.close()

    log.info("\n✅ Ingestion complete.")


if __name__ == "__main__":
    main()
"""
Script 2 (OPTIMIZED): Load extracted KG JSON → generate Cypher TRỰC TIẾP (không dùng LLM)
→ push to Neo4j Aura

IMPROVEMENTS vs original:
- Bỏ hoàn toàn LLM để sinh Cypher → nhanh ~10x, không hallucinate, không tốn token
- Sinh Cypher xác định từ JSON schema đã biết
- Batch MERGE trong single transaction per file
- Connection pool tối ưu
- Structured logging với thống kê chi tiết
- Idempotent hoàn toàn (MERGE everywhere)
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
FOLDERS       = ["curriculum", "career_description", "syllabus"]
# ──────────────────────────────────────────────────────────────────────────────


# ─── CYPHER GENERATION (NO LLM) ───────────────────────────────────────────────

# Merge key theo label: dùng trường nào để MERGE (tránh duplicate)
MERGE_KEY = {
    "MAJOR":    "code",   # MAJOR merge theo code
    "SUBJECT":  "code",   # SUBJECT merge theo code
    "DOCUMENT": "docid",  # DOCUMENT merge theo docid
    "SKILL":    "name",   # SKILL merge theo name (không có code)
    "CAREER":   "name",   # CAREER merge theo name
    "TEACHER":  "name",   # TEACHER merge theo name
}

# Labels dùng MERGE theo name (không có code unique)
NAME_ONLY_LABELS = {"SKILL", "CAREER", "TEACHER"}

# Labels dùng MERGE theo code (hoặc docid)
CODE_LABELS = {"MAJOR", "SUBJECT"}

def _esc(value: str) -> str:
    """Escape single quotes cho Cypher string literals."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def build_node_cypher(node: dict) -> str | None:
    """
    Sinh MERGE statement cho 1 node.
    Returns None nếu node thiếu thông tin bắt buộc.
    """
    label  = node.get("label", "")
    props  = node.get("properties", {})
    name   = props.get("name", "").strip()
    code   = props.get("code", "").strip()
    docid  = props.get("docid", "").strip()
    doctype = props.get("doctype", "").strip()

    if not label or not name:
        return None

    if label == "DOCUMENT":
        if not docid:
            return None
        stmt = (
            f"MERGE (n:DOCUMENT {{docid: '{_esc(docid)}'}})"
            f" SET n.name = '{_esc(name)}', n.doctype = '{_esc(doctype)}'"
        )

    elif label in CODE_LABELS:
        if not code:
            return None  # bắt buộc có code
        stmt = (
            f"MERGE (n:{label} {{code: '{_esc(code)}'}})"
            f" SET n.name = '{_esc(name)}'"
        )

    elif label in NAME_ONLY_LABELS:
        stmt = (
            f"MERGE (n:{label} {{name: '{_esc(name)}'}})"
        )

    else:
        # Unknown label - skip
        return None

    return stmt


def build_rel_cypher(rel: dict, node_map: dict) -> str | None:
    """
    Sinh MERGE statement cho 1 relationship.
    node_map: {local_id -> node dict}
    """
    src_id  = rel.get("source")
    tgt_id  = rel.get("target")
    rtype   = rel.get("type", "")

    src_node = node_map.get(src_id)
    tgt_node = node_map.get(tgt_id)

    if not src_node or not tgt_node or not rtype:
        return None

    src_match = _build_match_clause("a", src_node)
    tgt_match = _build_match_clause("b", tgt_node)

    if not src_match or not tgt_match:
        return None

    return f"MATCH {src_match}, {tgt_match} MERGE (a)-[:{rtype}]->(b)"


def _build_match_clause(var: str, node: dict) -> str | None:
    """Sinh MATCH clause cho 1 node dùng unique key."""
    label  = node.get("label", "")
    props  = node.get("properties", {})
    name   = props.get("name", "").strip()
    code   = props.get("code", "").strip()
    docid  = props.get("docid", "").strip()

    if not label:
        return None

    if label == "DOCUMENT":
        if not docid:
            return None
        return f"({var}:DOCUMENT {{docid: '{_esc(docid)}'}})"

    elif label in CODE_LABELS:
        if not code:
            return None
        return f"({var}:{label} {{code: '{_esc(code)}'}})"

    elif label in NAME_ONLY_LABELS:
        if not name:
            return None
        return f"({var}:{label} {{name: '{_esc(name)}'}})"

    # MAJOR trong career_description không có code → match bằng name
    if label == "MAJOR" and not code and name:
        return f"({var}:MAJOR {{name: '{_esc(name)}'}})"

    return None


def kg_to_cypher_statements(kg_data: dict) -> list[str]:
    """
    Chuyển KG JSON → list Cypher statements (không cần LLM).
    """
    nodes = kg_data.get("nodes", [])
    rels  = kg_data.get("relationships", [])

    # Build local id → node dict
    node_map = {n["id"]: n for n in nodes if "id" in n}

    statements = []

    # 1. Node MERGE statements
    for node in nodes:
        stmt = build_node_cypher(node)
        if stmt:
            statements.append(stmt)
        else:
            label = node.get("label", "?")
            nid   = node.get("id", "?")
            log.debug(f"  [skip node] id={nid} label={label} (thiếu key)")

    # 2. Relationship MERGE statements
    for rel in rels:
        stmt = build_rel_cypher(rel, node_map)
        if stmt:
            statements.append(stmt)
        else:
            log.debug(f"  [skip rel] {rel.get('source')} -[{rel.get('type')}]-> {rel.get('target')}")

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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MAJOR)    REQUIRE n.code  IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:SUBJECT)  REQUIRE n.code  IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DOCUMENT) REQUIRE n.docid IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (n:SKILL)   ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:CAREER)  ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TEACHER) ON (n.name)",
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
    """
    Chạy tất cả statements trong 1 write transaction.
    Returns (ok_count, fail_count).
    """
    if not statements:
        return 0, 0

    ok = 0
    fail = 0

    def tx_func(tx):
        nonlocal ok, fail
        for stmt in statements:
            try:
                tx.run(stmt)
                ok += 1
            except neo4j_exc.CypherSyntaxError as e:
                fail += 1
                log.error(f"  [CypherError] {e.message}\n  → {stmt[:200]}")
            except neo4j_exc.ConstraintError as e:
                # Có thể xảy ra với concurrent writes, bỏ qua
                log.debug(f"  [ConstraintSkip] {e}")
                ok += 1
            except Exception as e:
                fail += 1
                log.error(f"  [StmtError] {e}\n  → {stmt[:200]}")

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

                # Sinh Cypher trực tiếp (không LLM)
                statements = kg_to_cypher_statements(kg_data)
                log.info(f"    Generated {len(statements)} statements (no LLM)")

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
    log.info("Starting Neo4j ingestion pipeline (OPTIMIZED - No LLM)...")

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
"""
Script 2: Load extracted KG JSON → generate Cypher via LLM → push to Neo4j Aura
Reads extracted JSON from: ./cache/output/{curriculum,career_description,syllabus}/
Uses OpenAI API
"""

import os
import json
from pathlib import Path
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("DB_URL", )
NEO4J_USERNAME = os.getenv("DB_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "your-password")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

LOCAL_OUT_DIR = Path("./cache/output")
FOLDERS       = ["curriculum", "career_description", "syllabus"]
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher expert. Given a Knowledge Graph JSON with nodes and relationships, generate valid Cypher MERGE statements to load all data into Neo4j.

RULES:
1. Use MERGE (not CREATE) so re-runs are idempotent.
2. MERGE nodes on their unique key: code for MAJOR/SUBJECT, docid for DOCUMENT, name for SKILL/CAREER/TEACHER.
3. After MERGE, use SET to update remaining properties.
4. For relationships: MATCH both endpoints then MERGE the relationship.
5. Return ONLY a JSON array of Cypher strings. No markdown, no explanation.

Example:
[
  "MERGE (n:MAJOR {code: 'SE'}) SET n.name = 'KỸ THUẬT PHẦN MỀM'",
  "MERGE (n:SUBJECT {code: 'IT001'}) SET n.name = 'LẬP TRÌNH JAVA'",
  "MATCH (a:MAJOR {code: 'SE'}), (b:SUBJECT {code: 'IT001'}) MERGE (a)-[:OFFERS]->(b)"
]"""


def generate_cypher(ai_client: OpenAI, kg_data: dict) -> list[str]:
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CYPHER_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Convert this KG JSON to Cypher:\n{json.dumps(kg_data, ensure_ascii=False, indent=2)}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    # The model returns {"statements": [...]} or just [...]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    # handle wrapped object
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def create_indexes(session):
    stmts = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MAJOR)    REQUIRE n.code  IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:SUBJECT)  REQUIRE n.code  IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DOCUMENT) REQUIRE n.docid IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (n:SKILL)   ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:CAREER)  ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TEACHER) ON (n.name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:MAJOR)   ON (n.name)",
    ]
    for stmt in stmts:
        try:
            session.run(stmt)
        except Exception as e:
            print(f"  [index] WARNING: {e}")


def run_statements(session, statements: list[str], label: str):
    ok = fail = 0
    for stmt in statements:
        try:
            session.run(stmt)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"    [ERROR] {e}\n    → {stmt[:120]}")
    print(f"    [{label}] ✓ {ok}  ✗ {fail}")


def process_files(driver, ai_client: OpenAI):
    with driver.session() as session:
        print("Creating indexes / constraints...")
        create_indexes(session)

        for folder in FOLDERS:
            folder_path = LOCAL_OUT_DIR / folder
            if not folder_path.exists():
                print(f"  Folder not found: {folder_path}")
                continue

            files = sorted(folder_path.glob("*.json"))
            if not files:
                print(f"  No files in {folder_path}")
                continue

            print(f"\n{'='*60}\nFolder: {folder} ({len(files)} files)")

            for jf in files:
                print(f"\n  → {jf.name}")
                with open(jf, encoding="utf-8") as f:
                    kg_data = json.load(f)

                try:
                    statements = generate_cypher(ai_client, kg_data)
                    print(f"    Generated {len(statements)} statements")
                except Exception as e:
                    print(f"    ERROR generating Cypher: {e}"); continue

                run_statements(session, statements, jf.name)


def main():
    print("Starting Neo4j ingestion pipeline...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()
    try:
        process_files(driver, ai_client)
    finally:
        driver.close()
    print("\n✅ Ingestion complete.")


if __name__ == "__main__":
    main()
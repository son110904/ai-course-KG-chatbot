"""
Script 3: Knowledge Graph Q&A chatbot
- User asks questions in natural language
- LLM generates Cypher → runs against Neo4j
- LLM synthesizes answer from graph results
- Each QA round is logged to a JSON file
Uses OpenAI API
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

LOG_DIR = Path("./qa_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_DESC = """
Neo4j Knowledge Graph Schema:
Nodes: MAJOR {name, code}, SUBJECT {name, code}, SKILL {name}, CAREER {name}, TEACHER {name}, DOCUMENT {name, docid, doctype}
Relationships:
  (MAJOR)-[:OFFERS]->(SUBJECT)
  (TEACHER)-[:TEACH]->(SUBJECT)
  (SUBJECT)-[:PROVIDES]->(SKILL)
  (CAREER)-[:REQUIRES]->(SKILL)
  (SUBJECT)-[:PREREQUISITE_FOR]->(SUBJECT)
  (MAJOR)-[:LEADS_TO]->(CAREER)
  (MAJOR|TEACHER|SUBJECT|CAREER|SKILL)-[:MENTIONED_IN]->(DOCUMENT)

All name/text properties are stored in UPPERCASE Vietnamese.
"""

CYPHER_GEN_SYSTEM = f"""You are a Neo4j Cypher expert for a Vietnamese university Knowledge Graph.
{SCHEMA_DESC}
Given a user question, generate 1-3 Cypher READ queries (no writes) that best answer it.
Return ONLY a JSON object with key "queries" containing an array of Cypher strings.
Use toLower() / CONTAINS for flexible name matching.
Example: {{"queries": ["MATCH (t:TEACHER)-[:TEACH]->(s:SUBJECT) WHERE toLower(t.name) CONTAINS 'lâm' RETURN t.name, s.name, s.code"]}}"""

ANSWER_SYSTEM = f"""You are a helpful Vietnamese university assistant. Answer questions about academic programs, subjects, skills, teachers, and careers using Knowledge Graph data provided.
{SCHEMA_DESC}
Synthesize a clear, natural Vietnamese answer from the graph results. If results are empty, say so politely."""


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def generate_cypher(ai_client: OpenAI, question: str) -> list[str]:
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CYPHER_GEN_SYSTEM},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    # Accept {"queries": [...]} or bare list
    if isinstance(parsed, list):
        return parsed
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []


def run_queries(driver, queries: list[str]) -> tuple[list[dict], list[dict]]:
    all_records = []
    traversal   = []
    with driver.session() as session:
        for cypher in queries:
            try:
                records = [dict(r) for r in session.run(cypher)]
                all_records.extend(records)
                # Infer simple traversal hint from result shape
                for rec in records:
                    keys = list(rec.keys())
                    if len(keys) >= 2:
                        rel = cypher.split("[:")[1].split("]")[0] if "[:" in cypher else "RELATED"
                        traversal.append({
                            "from":     str(rec.get(keys[0], "")),
                            "to":       str(rec.get(keys[1], "")),
                            "relation": rel,
                            "hop":      1,
                        })
            except Exception as e:
                print(f"  [cypher ERROR] {e}")
    return all_records, traversal


def generate_answer(ai_client: OpenAI, question: str, graph_results: list[dict]) -> str:
    context = json.dumps(graph_results, ensure_ascii=False, indent=2)
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",   "content": f"Câu hỏi: {question}\n\nKết quả từ Knowledge Graph:\n{context}\n\nHãy trả lời bằng tiếng Việt."},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_entities(records: list[dict]) -> list[str]:
    entities = set()
    for rec in records:
        for v in rec.values():
            if isinstance(v, str) and len(v) > 1:
                entities.add(v)
    return list(entities)


def build_retrieved_nodes(records: list[dict]) -> list[dict]:
    nodes = []
    seen  = set()
    for i, rec in enumerate(records):
        key = json.dumps(rec, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        nodes.append({
            "node_id":  f"node{i+1:03d}",
            "content":  json.dumps(rec, ensure_ascii=False),
            "score":    1.0,
            "entities": extract_entities([rec]),
        })
    return nodes


def save_qa_log(qa_record: dict) -> Path:
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = LOG_DIR / f"{ts}_{qa_record['query_id']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(qa_record, f, ensure_ascii=False, indent=2)
    print(f"  [log] saved → {filename}")
    return filename


def ask(driver, ai_client: OpenAI, question: str, query_id: str | None = None) -> dict:
    if query_id is None:
        query_id = "q" + uuid.uuid4().hex[:6]

    print(f"\n{'='*60}")
    print(f"Q [{query_id}]: {question}")

    # 1. Generate Cypher
    try:
        cypher_stmts = generate_cypher(ai_client, question)
        print(f"  Cypher ({len(cypher_stmts)}):")
        for c in cypher_stmts:
            print(f"    {c}")
    except Exception as e:
        print(f"  ERROR generating Cypher: {e}")
        cypher_stmts = []

    # 2. Run queries
    graph_results, traversal = run_queries(driver, cypher_stmts)
    print(f"  Records returned: {len(graph_results)}")

    # 3. Generate answer
    answer = generate_answer(ai_client, question, graph_results)
    print(f"\nA: {answer}")

    # 4. Build & save log
    qa_record = {
        "query_id":           query_id,
        "query":              question,
        "generated_answer":   answer,
        "context_text":       "\n".join(json.dumps(r, ensure_ascii=False) for r in graph_results),
        "retrieved_nodes":    build_retrieved_nodes(graph_results),
        "traversal_path":     traversal[:20],
        "communities_covered": extract_entities(graph_results),
        "cypher_queries":     cypher_stmts,
        "timestamp":          datetime.datetime.now().isoformat(),
    }
    save_qa_log(qa_record)
    return qa_record


def interactive_loop(driver, ai_client: OpenAI):
    print("\n🎓 Knowledge Graph Chatbot (NEU)")
    print("Gõ câu hỏi của bạn. Nhập 'exit' để thoát.\n")
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
    print("Starting KG Chatbot...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()
    try:
        interactive_loop(driver, ai_client)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
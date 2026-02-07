from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx_reader import read_docx_from_directory
import os
import time

# ======================================================
# ENV
# ======================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")        # bolt://localhost:7687
NEO4J_USER = os.getenv("NEO4J_USER")      # neo4j
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ======================================================
# CONFIG
# ======================================================
MAX_WORKERS = 12
CHUNK_SIZE = 2000
OVERLAP_SIZE = 300
USE_MINI_MODEL = True


# ======================================================
# NEO4J HELPERS
# ======================================================
def save_entity(tx, name):
    tx.run(
        "MERGE (:Entity {name: $name})",
        name=name
    )


def save_relation(tx, src, rel, tgt):
    tx.run(
        """
        MERGE (a:Entity {name: $src})
        MERGE (b:Entity {name: $tgt})
        MERGE (a)-[:REL {type: $rel}]->(b)
        """,
        src=src, tgt=tgt, rel=rel
    )


# ======================================================
# CLEAN CYPHER
# ======================================================
def clean_cypher(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    return text.strip().rstrip(";")


# ======================================================
# BATCH LLM EXTRACTION
# ======================================================
def batch_extract(chunks):
    system_prompt = """
You are an information extraction system.

Extract entities and relationships from the text.

STRICT FORMAT ONLY:

ENTITY: <entity name>
RELATION: <entity_1> -> <relation> -> <entity_2>

Rules:
- Use Vietnamese if text is Vietnamese
- Max 5 words per entity
- Use '->' exactly
- No explanations
"""

    results = []

    def process(chunk):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini" if USE_MINI_MODEL else "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk[:1500]}
                ],
                max_tokens=350
            )
            return resp.choices[0].message.content
        except Exception as e:
            print("[WARN] Extract failed:", e)
            return ""

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(process, c) for c in chunks]
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    return results


# ======================================================
# INGEST PIPELINE
# ======================================================
def ingest_documents(documents):
    print("\n[1/3] Chunking documents...")
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), CHUNK_SIZE - OVERLAP_SIZE):
            chunks.append(doc[i:i + CHUNK_SIZE])
    print(f"  ✓ {len(chunks)} chunks")

    print("\n[2/3] Extracting entities & relations...")
    extracted = batch_extract(chunks)
    print(f"  ✓ {len(extracted)} extraction results")

    print("\n[3/3] Saving to Neo4j...")
    with driver.session() as session:
        for block in extracted:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for line in lines:
                if line.startswith("ENTITY:"):
                    name = line.replace("ENTITY:", "").strip()
                    if len(name) > 2:
                        session.execute_write(save_entity, name)

                elif line.startswith("RELATION:"):
                    try:
                        _, body = line.split(":", 1)
                        src, rel, tgt = [p.strip() for p in body.split("->")]
                        if len(src) > 2 and len(tgt) > 2:
                            session.execute_write(
                                save_relation, src, rel, tgt
                            )
                    except:
                        pass

    print("  ✓ Neo4j ingest complete")


# ======================================================
# QUERY PIPELINE (GRAPH RAG)
# ======================================================
def answer_question(question):
    cypher_prompt = f"""
Convert the question into a Cypher query.

Graph schema:
(:Entity)-[:REL {{type}}]->(:Entity)

RULES:
- Output ONLY Cypher
- NO markdown
- NO ``` blocks
- Start with MATCH or RETURN

Question:
{question}
"""

    raw_cypher = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": cypher_prompt}],
        max_tokens=200
    ).choices[0].message.content

    cypher = clean_cypher(raw_cypher)

    print("\n[Cypher generated]")
    print(cypher)

    with driver.session() as session:
        records = session.run(cypher)
        facts = []
        for r in records:
            facts.append(" - ".join(str(v) for v in r.values()))

    context = "\n".join(facts[:30])

    final_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Dựa trên dữ liệu đồ thị, hãy trả lời chính xác bằng tiếng Việt."
            },
            {
                "role": "user",
                "content": f"""
Câu hỏi: {question}

Dữ liệu đồ thị:
{context}
"""
            }
        ],
        max_tokens=800
    )

    return final_answer.choices[0].message.content


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    print("=" * 80)
    print("NEO4J GRAPH RAG (NO EMBEDDING) – FIXED")
    print("=" * 80)

    docs = read_docx_from_directory("example_docx")
    ingest_documents(docs)

    while True:
        q = input("\nCâu hỏi (enter để thoát): ").strip()
        if not q:
            break
        ans = answer_question(q)
        print("\nANSWER")
        print("-" * 80)
        print(ans)

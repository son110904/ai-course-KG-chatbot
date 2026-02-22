"""
Script 3: Knowledge Graph Q&A Chatbot"""

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
NEO4J_URI      = os.getenv("DB_URL",)
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")

MAX_HOPS    = int(os.getenv("MAX_HOPS"))   # giới hạn BFS
TOP_K       = int(os.getenv("TOP_K"))  # số node trả về sau ranking
LOG_DIR     = Path("./qa_logs")   # dùng khi chạy evaluation
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_DESC = """
Nodes: MAJOR{name,code,community_id,pagerank}, SUBJECT{name,code,community_id,pagerank},
       SKILL{name,community_id,pagerank}, CAREER{name,community_id,pagerank},
       TEACHER{name,community_id}, DOCUMENT{name,docid,doctype}
Relationships:
  (MAJOR)-[:OFFERS]->(SUBJECT)
  (TEACHER)-[:TEACH]->(SUBJECT)
  (SUBJECT)-[:PROVIDES]->(SKILL)
  (CAREER)-[:REQUIRES]->(SKILL)
  (SUBJECT)-[:PREREQUISITE_FOR]->(SUBJECT)
  (MAJOR)-[:LEADS_TO]->(CAREER)
  (*)-[:MENTIONED_IN]->(DOCUMENT)
All name values are UPPERCASE Vietnamese.
"""

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 0: SETUP — Chạy Community Detection + PageRank (offline, 1 lần)
# ══════════════════════════════════════════════════════════════════════════════

def setup_graph_algorithms(driver):
    """
    Community Detection (Louvain) + PageRank tính bằng NetworkX (Python),
    sau đó ghi community_id và pagerank ngược lên Neo4j.
    Không cần GDS plugin — hoạt động trên Aura Free.
    Chỉ cần chạy 1 lần sau khi load xong dữ liệu.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        print("  Cài networkx: pip install networkx")
        return

    print("\n[Setup] Pull graph từ Neo4j → tính Louvain + PageRank bằng NetworkX...")

    G = nx.Graph()
    node_labels = {}   # node_name → label

    with driver.session() as session:
        # ── Pull toàn bộ nodes ────────────────────────────────────────────────
        nodes = session.run("""
            MATCH (n)
            WHERE n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER
            RETURN n.name AS name, labels(n)[0] AS label
        """).data()
        for row in nodes:
            if row["name"]:
                G.add_node(row["name"])
                node_labels[row["name"]] = row["label"]

        # ── Pull toàn bộ relationships ────────────────────────────────────────
        rels = session.run("""
            MATCH (a)-[r]->(b)
            WHERE (a:MAJOR OR a:SUBJECT OR a:SKILL OR a:CAREER OR a:TEACHER)
              AND (b:MAJOR OR b:SUBJECT OR b:SKILL OR b:CAREER OR b:TEACHER)
              AND a.name IS NOT NULL AND b.name IS NOT NULL
            RETURN a.name AS src, b.name AS tgt
        """).data()
        for row in rels:
            G.add_edge(row["src"], row["tgt"])

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Community Detection: Louvain trong từng label group ─────────────────
    # Chiến lược: mỗi label (TEACHER, SKILL, CAREER, MAJOR, SUBJECT) là 1 "super-community"
    # Bên trong mỗi label, dùng Louvain để tìm sub-community
    # → Đảm bảo TEACHER không bị lẫn vào community của SUBJECT
    print("  Chạy Louvain community detection (per-label)...")

    LABEL_BASE_ID = {
        "TEACHER": 0,
        "SKILL":   1000,
        "CAREER":  2000,
        "MAJOR":   3000,
        "SUBJECT": 4000,
    }

    node_community = {}

    for label, base_id in LABEL_BASE_ID.items():
        # Lấy các node thuộc label này
        label_nodes = [n for n, lbl in node_labels.items() if lbl == label]
        if not label_nodes:
            continue

        # Tạo subgraph chỉ gồm các node cùng label
        subG = G.subgraph(label_nodes).copy()

        if subG.number_of_edges() > 0:
            # Có edges → dùng Louvain để tìm sub-community
            sub_communities = louvain_communities(subG, seed=42)
            for sub_cid, community in enumerate(sub_communities):
                for node in community:
                    node_community[node] = base_id + sub_cid
        else:
            # Không có edges giữa các node cùng label → mỗi node 1 community
            for i, node in enumerate(label_nodes):
                node_community[node] = base_id + i

    total_communities = len(set(node_community.values()))
    print(f"  Tìm thấy {total_communities} communities "
          f"(TEACHER:0xxx, SKILL:1xxx, CAREER:2xxx, MAJOR:3xxx, SUBJECT:4xxx)")

    # ── PageRank ──────────────────────────────────────────────────────────────
    print("  Chạy PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    # ── Ghi ngược lên Neo4j ───────────────────────────────────────────────────
    print("  Ghi community_id + pagerank lên Neo4j...")
    with driver.session() as session:
        BATCH_SIZE = 500
        items = list(node_community.items())
        for i in range(0, len(items), BATCH_SIZE):
            batch = [
                {"name": name, "cid": cid, "pr": round(pagerank.get(name, 0.0), 8)}
                for name, cid in items[i:i+BATCH_SIZE]
            ]
            session.run("""
                UNWIND $batch AS row
                MATCH (n) WHERE n.name = row.name
                SET n.community_id = row.cid,
                    n.pagerank      = row.pr
            """, batch=batch)

    total_written = len(node_community)
    print(f"  Đã ghi {total_written} nodes")
    print("[Setup] Xong. community_id và pagerank đã được ghi vào graph.\n")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: COMMUNITY DETECTION — Tìm community liên quan đến câu hỏi
# ══════════════════════════════════════════════════════════════════════════════

def find_relevant_communities(driver, keywords: list[str]) -> list[int]:
    """
    Tìm community_id của các node có tên chứa keyword.
    Trả về danh sách community_id liên quan.
    """
    if not keywords:
        return []

    with driver.session() as session:
        community_ids = set()
        for kw in keywords:
            result = session.run("""
                MATCH (n)
                WHERE (n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER)
                  AND toLower(n.name) CONTAINS toLower($kw)
                  AND n.community_id IS NOT NULL
                RETURN DISTINCT n.community_id AS cid
                LIMIT 5
            """, kw=kw)
            for rec in result:
                community_ids.add(rec["cid"])

    return list(community_ids)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: MULTI-HOP TRAVERSAL — BFS trong community, giới hạn max_hops
# ══════════════════════════════════════════════════════════════════════════════

def multihop_traversal(driver, keywords: list[str],
                       community_ids: list[int],
                       max_hops: int = MAX_HOPS) -> tuple[list[dict], list[dict]]:
    """
    BFS traversal từ các seed node (khớp keyword) mở rộng tối đa max_hops bước.
    Nếu có community_id → chỉ tìm trong community đó.
    Trả về (nodes, traversal_paths).
    """
    all_nodes  = []
    all_paths  = []
    seen_names = set()

    with driver.session() as session:
        for kw in keywords:
            # Tìm seed nodes
            seed_query = """
                MATCH (seed)
                WHERE (seed:MAJOR OR seed:SUBJECT OR seed:SKILL OR seed:CAREER OR seed:TEACHER)
                  AND toLower(seed.name) CONTAINS toLower($kw)
                RETURN seed
                LIMIT 3
            """
            seeds = [rec["seed"] for rec in session.run(seed_query, kw=kw)]

            for seed in seeds:
                seed_name = seed.get("name", "")
                if seed_name in seen_names:
                    continue
                seen_names.add(seed_name)

                # BFS multi-hop: traversal tới max_hops bước
                # Lọc theo community_id nếu có
                community_filter = ""
                params: dict = {"seed_name": seed_name, "max_hops": max_hops}

                if community_ids:
                    community_filter = "AND (n.community_id IN $cids OR n.community_id IS NULL)"
                    params["cids"] = community_ids

                traversal_query = f"""
                    MATCH path = (start)-[*1..{max_hops}]-(n)
                    WHERE start.name = $seed_name
                      AND (n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER)
                      {community_filter}
                    WITH n, path,
                         [r IN relationships(path) | type(r)] AS rel_types,
                         [x IN nodes(path) | x.name]          AS node_names
                    RETURN DISTINCT
                        n.name         AS name,
                        labels(n)[0]   AS label,
                        n.code         AS code,
                        n.pagerank     AS pagerank,
                        n.community_id AS community_id,
                        rel_types,
                        node_names,
                        length(path)   AS hops
                    ORDER BY hops ASC
                    LIMIT 50
                """
                results = session.run(traversal_query, **params)

                for rec in results:
                    node_info = {
                        "name":         rec["name"],
                        "label":        rec["label"],
                        "code":         rec["code"],
                        "pagerank":     rec["pagerank"],
                        "community_id": rec["community_id"],
                        "hops":         rec["hops"],
                    }
                    all_nodes.append(node_info)

                    # Build traversal path log
                    node_names = rec["node_names"]
                    rel_types  = rec["rel_types"]
                    for i, rel in enumerate(rel_types):
                        path_entry = {
                            "from":     node_names[i]   if i < len(node_names) else "",
                            "to":       node_names[i+1] if i+1 < len(node_names) else "",
                            "relation": rel,
                            "hop":      i + 1,
                        }
                        all_paths.append(path_entry)

    return all_nodes, all_paths


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: PAGERANK RANKING — Xếp hạng và lọc top-K node quan trọng nhất
# ══════════════════════════════════════════════════════════════════════════════

def rank_nodes(nodes: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Xếp hạng nodes theo PageRank (đã tính sẵn trên Neo4j).
    Ưu tiên: node có pagerank cao + hop ít (gần seed).
    """
    def score(n: dict) -> float:
        pr   = n.get("pagerank") or 0.0
        hops = n.get("hops")     or 1
        # Score = pagerank / hops  → node quan trọng + gần seed được ưu tiên
        return pr / hops

    ranked = sorted(nodes, key=score, reverse=True)

    # Dedup theo name
    seen  = set()
    dedup = []
    for n in ranked:
        key = (n.get("label",""), n.get("name",""))
        if key not in seen:
            seen.add(key)
            dedup.append(n)

    return dedup[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# LLM: Extract keywords + Generate answer
# ══════════════════════════════════════════════════════════════════════════════

def extract_keywords(ai_client: OpenAI, question: str) -> list[str]:
    """Extract các từ khoá thực thể từ câu hỏi để làm seed BFS."""
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": (
                "Extract entity keywords from the user question for a university Knowledge Graph search. "
                "Return JSON: {\"keywords\": [\"keyword1\", \"keyword2\", ...]}. "
                "Keywords should be names of: careers, subjects, skills, majors, or teachers. "
                "Keep original Vietnamese text."
            )},
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("keywords", [])


def generate_answer(ai_client: OpenAI, question: str,
                    ranked_nodes: list[dict], traversal_paths: list[dict]) -> str:
    context = json.dumps({
        "ranked_results": ranked_nodes,
        "traversal_paths": traversal_paths[:30],
    }, ensure_ascii=False, indent=2)

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": (
                "Bạn là trợ lý tư vấn học thuật. Tổng hợp câu trả lời rõ ràng bằng tiếng Việt "
                "từ kết quả Knowledge Graph đã xếp hạng theo PageRank. "
                "Ưu tiên các node có pagerank cao. Nếu không có dữ liệu, thông báo lịch sự.\n"
                f"{SCHEMA_DESC}\n\n"
                "QUY TẮC ĐỊNH DẠNG KẾT QUẢ:\n"
                "- Khi đề cập đến MAJOR (chương trình đào tạo / ngành học): "
                "luôn viết theo dạng 'Tên ngành (Mã ngành)'. "
                "Ví dụ: 'Công nghệ thông tin (7480201)', 'Kỹ thuật phần mềm (7480103)'.\n"
                "- Khi đề cập đến SUBJECT (môn học): "
                "luôn viết theo dạng 'Tên môn (Mã môn)'. "
                "Ví dụ: 'Cơ sở dữ liệu (IT001)'.\n"
                "- Nếu không có mã trong dữ liệu thì chỉ ghi tên, không bịa mã."
            )},
            {"role": "user", "content": (
                f"Câu hỏi: {question}\n\n"
                f"Kết quả Knowledge Graph (đã xếp hạng PageRank):\n{context}\n\n"
                "Hãy trả lời bằng tiếng Việt, nhớ kèm mã ngành/mã môn khi có:"
            )},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def ask(driver, ai_client: OpenAI, question: str, query_id: str | None = None) -> dict:
    if query_id is None:
        query_id = "q" + uuid.uuid4().hex[:6]

    print(f"\n{'='*60}")
    print(f"Q [{query_id}]: {question}")

    # ── Bước 1: Extract keywords ──────────────────────────────────────────────
    keywords = extract_keywords(ai_client, question)
    print(f"  Keywords: {keywords}")

    # ── Bước 2: Community Detection — thu hẹp không gian tìm kiếm ────────────
    community_ids = find_relevant_communities(driver, keywords)
    print(f"  Communities: {community_ids}")

    # ── Bước 3: Multi-hop BFS Traversal ──────────────────────────────────────
    raw_nodes, traversal_paths = multihop_traversal(
        driver, keywords, community_ids, max_hops=MAX_HOPS
    )
    print(f"  BFS nodes found: {len(raw_nodes)}  |  paths: {len(traversal_paths)}")

    # ── Bước 4: PageRank Ranking ──────────────────────────────────────────────
    ranked_nodes = rank_nodes(raw_nodes, top_k=TOP_K)
    print(f"  After PageRank ranking (top {TOP_K}): {len(ranked_nodes)} nodes")
    if ranked_nodes:
        top3 = [(n["name"], round(n.get("pagerank") or 0, 4)) for n in ranked_nodes[:3]]
        print(f"  Top 3: {top3}")

    # ── Bước 5: LLM tổng hợp câu trả lời ─────────────────────────────────────
    answer = generate_answer(ai_client, question, ranked_nodes, traversal_paths)
    print(f"\nA: {answer}")

    # ── Build record (trả về để eval pipeline dùng, không tự động lưu file) ──
    qa_record = {
        "query_id":            query_id,
        "query":               question,
        "generated_answer":    answer,
        "keywords":            keywords,
        "communities_covered": community_ids,
        "context_text":        json.dumps(ranked_nodes, ensure_ascii=False),
        "retrieved_nodes": [
            {
                "node_id":  f"node{i+1:03d}",
                "content":  json.dumps(n, ensure_ascii=False),
                "score":    round(n.get("pagerank") or 0, 6),
                "entities": [n.get("name","")],
            }
            for i, n in enumerate(ranked_nodes)
        ],
        "traversal_path":      traversal_paths[:20],
        "timestamp":           datetime.datetime.now().isoformat(),
        "algorithm": {
            "community_detection": "Louvain (Neo4j GDS)",
            "traversal":           f"BFS multi-hop (max_hops={MAX_HOPS})",
            "ranking":             "PageRank (damping=0.85)",
        },
    }

    return qa_record


# ── Neo4j ─────────────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ── Interactive loop ──────────────────────────────────────────────────────────

def interactive_loop(driver, ai_client: OpenAI):
    print("\n🎓 Knowledge Graph Chatbot (NEU)")
    print(f"Pipeline: Community Detection → BFS multi-hop (max={MAX_HOPS}) → PageRank → LLM")
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
    print("Starting KG Chatbot...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()

    try:
        # Hỏi người dùng có muốn chạy setup không
        print("\nBạn có muốn chạy Community Detection + PageRank không?")
        print("(Chỉ cần chạy 1 lần sau khi load dữ liệu lên Neo4j)")
        ans = input("Nhập 'yes' để chạy, Enter để bỏ qua: ").strip().lower()
        if ans == "yes":
            setup_graph_algorithms(driver)

        interactive_loop(driver, ai_client)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
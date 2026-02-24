"""
Script 3: Knowledge Graph Q&A Chatbot
Fixes v2:

  - Intent detection: phân loại thực thể đề cập / thực thể được hỏi
  - Relationship constraints per query type: ràng buộc đường truy xuất theo loại câu hỏi
  - Negation handling: nhận diện "ko / k / không / chẳng / kém / chưa giỏi" → lọc thực thể phủ định
  - Prompt AI trả lời sát trọng tâm, không thêm thông tin ngoài lề
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from collections import deque
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")

MAX_HOPS    = int(os.getenv("MAX_HOPS", "3"))
TOP_K       = int(os.getenv("TOP_K", "15"))
LOG_DIR     = Path("./qa_logs")
# ─────────────────────────────────────────────────────────────────────────────

# Từ đồng nghĩa phủ định — nhận diện câu hỏi có từ phủ định / "không giỏi"
NEGATION_SYNONYMS = {
    "ko", "k", "không", "chẳng", "chả", "kém", "chưa giỏi",
    "không giỏi", "ko giỏi", "k giỏi", "yếu", "dở",
    "không thích", "ko thích", "k thích", "chán",
    "không muốn", "ko muốn", "không có", "ko có",
    "không biết", "ko biết", "chưa biết",
}

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

# ── Ràng buộc quan hệ theo loại câu hỏi ──────────────────────────────────────
# Key: (thực thể đề cập, thực thể được hỏi)
RELATIONSHIP_CONSTRAINTS = {
    # Đề cập MAJOR → hỏi CAREER
    ("MAJOR", "CAREER"): (
        "Đường truy xuất: MAJOR -[:LEADS_TO]-> CAREER.\n"
        "Chỉ liệt kê các nghề nghiệp (CAREER) mà ngành (MAJOR) dẫn đến.\n"
        "KHÔNG đề cập SUBJECT (môn học) trừ khi được hỏi thêm."
    ),
    # Đề cập CAREER → hỏi SKILL
    ("CAREER", "SKILL"): (
        "Đường truy xuất: CAREER -[:REQUIRES]-> SKILL và SUBJECT -[:PROVIDES]-> SKILL.\n"
        "Trả lời: kỹ năng cần thiết cho nghề đó + môn học cung cấp kỹ năng tương ứng.\n"
        "Kèm mã môn học nếu có."
    ),
    # Đề cập MAJOR → hỏi SKILL
    ("MAJOR", "SKILL"): (
        "Đường truy xuất: MAJOR -[:OFFERS]-> SUBJECT -[:PROVIDES]-> SKILL.\n"
        "Trả lời: kỹ năng đạt được từ các môn học trong chương trình đào tạo.\n"
        "Kèm tên môn học (mã môn) cung cấp kỹ năng đó."
    ),
    # Đề cập SKILL → hỏi MAJOR
    ("SKILL", "MAJOR"): (
        "Đường truy xuất: SKILL <-[:PROVIDES]- SUBJECT <-[:OFFERS]- MAJOR.\n"
        "Trả lời: ngành học (MAJOR) có môn học cung cấp kỹ năng đó.\n"
        "Kèm mã ngành, tên môn trung gian."
    ),
    # Đề cập CAREER → hỏi SUBJECT (môn học)
    ("CAREER", "SUBJECT"): (
        "Đường truy xuất: CAREER -[:REQUIRES]-> SKILL <-[:PROVIDES]- SUBJECT.\n"
        "Trả lời: các môn học cung cấp kỹ năng mà nghề đó yêu cầu.\n"
        "Kèm mã môn học và kỹ năng tương ứng."
    ),
    # Đề cập MAJOR → hỏi SUBJECT (môn học)
    ("MAJOR", "SUBJECT"): (
        "Đường truy xuất: MAJOR -[:OFFERS]-> SUBJECT.\n"
        "Trả lời: các môn học thuộc chương trình ngành đó, kèm mã môn và kỹ năng cung cấp (SKILL)."
    ),
    # Đề cập SKILL → hỏi CAREER
    ("SKILL", "CAREER"): (
        "Đường truy xuất: SKILL <-[:REQUIRES]- CAREER.\n"
        "Trả lời: danh sách nghề nghiệp yêu cầu kỹ năng đó."
    ),
    # Đề cập CAREER → hỏi MAJOR
    ("CAREER", "MAJOR"): (
        "Đường truy xuất: MAJOR -[:LEADS_TO]-> CAREER.\n"
        "Trả lời: ngành học (MAJOR) dẫn đến nghề đó, kèm mã ngành."
    ),
    # Đề cập SUBJECT → hỏi SKILL
    ("SUBJECT", "SKILL"): (
        "Đường truy xuất: SUBJECT -[:PROVIDES]-> SKILL.\n"
        "Trả lời: kỹ năng đạt được sau khi học môn đó."
    ),
    # Đề cập SKILL → hỏi SUBJECT
    ("SKILL", "SUBJECT"): (
        "Đường truy xuất: SKILL <-[:PROVIDES]- SUBJECT.\n"
        "Trả lời: môn học (kèm mã môn) cung cấp kỹ năng đó, và ngành nào chứa môn đó."
    ),
    # Đề cập MAJOR → so sánh nhiều ngành
    ("MAJOR", "MAJOR"): (
        "Đây là câu so sánh giữa các ngành.\n"
        "Truy xuất: MAJOR -[:LEADS_TO]-> CAREER và MAJOR -[:OFFERS]-> SUBJECT.\n"
        "Trả lời: so sánh cơ hội nghề nghiệp và môn học đặc trưng của từng ngành.\n"
        "Kèm mã ngành, mã môn học nếu có. Trích dẫn nguồn tài liệu (DOCUMENT) nếu có."
    ),
    # Đề cập MAJOR/CAREER → hỏi CAREER/MAJOR (tổng quát)
    ("MAJOR", "MAJOR_CAREER"): (
        "Đường truy xuất: MAJOR -[:LEADS_TO]-> CAREER và MAJOR -[:OFFERS]-> SUBJECT -[:PROVIDES]-> SKILL.\n"
        "Trả lời nghề nghiệp + kỹ năng đặc trưng + môn học trong ngành đó."
    ),
}

# ── Prompt hệ thống chính cho generate_answer ─────────────────────────────────
ANSWER_SYSTEM_BASE = """Bạn là trợ lý tư vấn học thuật cho Đại học Kinh tế Quốc dân (NEU).
Tổng hợp câu trả lời rõ ràng, tự nhiên bằng tiếng Việt từ kết quả Knowledge Graph đã xếp hạng.

{schema}

QUY TẮC QUAN TRỌNG:
1. Trả lời ĐÚNG TRỌNG TÂM câu hỏi. Không thêm thông tin không được hỏi đến.
2. Không dùng câu "ngoài ra..." để mở rộng ngoài phạm vi câu hỏi.
3. Nếu dữ liệu không đủ để trả lời → nói rõ "Dữ liệu hiện tại chưa đủ để tư vấn về [chủ đề], bạn có thể liên hệ phòng đào tạo để biết thêm."
4. KHÔNG bịa thông tin không có trong Knowledge Graph.
5. Luôn kèm mã ngành (MAJOR.code) và mã môn học (SUBJECT.code) khi có trong dữ liệu.
6. Khi người dùng đề cập thực thể mà họ KHÔNG giỏi / không thích → loại bỏ thực thể đó khỏi câu trả lời.
7. Ngôn ngữ tự nhiên, thân thiện — KHÔNG máy móc, lý thuyết.

RÀNG BUỘC THEO LOẠI CÂU HỎI:
{constraint}
"""

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 0: SETUP — Chạy Community Detection + PageRank (offline, 1 lần)
# ══════════════════════════════════════════════════════════════════════════════

def setup_graph_algorithms(driver):
    """
    Global Community Detection (Louvain) + PageRank
    Chạy trên toàn bộ graph (không chia theo label).
    Phù hợp cho GraphRAG reasoning đa thực thể.
    """

    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        print("Cài networkx: pip install networkx")
        return

    print("\n[Setup] Pull graph từ Neo4j → tính Global Louvain + PageRank...")

    G = nx.Graph()

    # ─── 1. Pull nodes ───────────────────────────────────
    with driver.session() as session:

        nodes = session.run("""
            MATCH (n)
            WHERE n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER
            RETURN n.name AS name
        """).data()

        for row in nodes:
            if row["name"]:
                G.add_node(row["name"])

        # ─── 2. Pull relationships ────────────────────────
        rels = session.run("""
            MATCH (a)-[r]->(b)
            WHERE (a:MAJOR OR a:SUBJECT OR a:SKILL OR a:CAREER OR a:TEACHER)
              AND (b:MAJOR OR b:SUBJECT OR b:SKILL OR b:CAREER OR b:TEACHER)
              AND a.name IS NOT NULL AND b.name IS NOT NULL
            RETURN a.name AS src, b.name AS tgt
        """).data()

        for row in rels:
            G.add_edge(row["src"], row["tgt"])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ─── 3. Global Louvain ───────────────────────────────
    print("Chạy Global Louvain community detection...")
    communities = louvain_communities(G, seed=42)

    node_community = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_community[node] = cid

    print(f"Tìm thấy {len(communities)} communities")

    # ─── 4. PageRank ─────────────────────────────────────
    print("Chạy PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    # ─── 5. Ghi lại Neo4j ───────────────────────────────
    print("Ghi community_id + pagerank lên Neo4j...")

    with driver.session() as session:
        BATCH_SIZE = 500
        items = list(node_community.items())

        for i in range(0, len(items), BATCH_SIZE):
            batch = [
                {
                    "name": name,
                    "cid": cid,
                    "pr": round(pagerank.get(name, 0.0), 8)
                }
                for name, cid in items[i:i+BATCH_SIZE]
            ]

            session.run("""
                UNWIND $batch AS row
                MATCH (n) WHERE n.name = row.name
                SET n.community_id = row.cid,
                    n.pagerank      = row.pr
            """, batch=batch)

    print(f"Đã ghi {len(node_community)} nodes")
    print("[Setup] Xong.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MỚI: EXTRACT QUERY INTENT — Phân loại ý định câu hỏi
# ══════════════════════════════════════════════════════════════════════════════

def extract_query_intent(ai_client: OpenAI, question: str) -> dict:
    """
    Trích xuất:
    - keywords: từ khoá tìm kiếm thực thể trong KG
    - mentioned_labels: loại thực thể được đề cập trong câu hỏi
    - asked_label: loại thực thể người dùng muốn biết
    - negated_keywords: từ khoá người dùng phủ định (không giỏi, không thích, ...)
    - is_comparison: câu hỏi so sánh
    """

    system_msg = (
        "Bạn phân tích câu hỏi tư vấn học thuật và trả về JSON.\n"
        "Schema Knowledge Graph:\n"
        "  Node labels: MAJOR (ngành học), SUBJECT (môn học), SKILL (kỹ năng), "
        "CAREER (nghề nghiệp / vị trí việc làm), TEACHER (giảng viên)\n\n"
        "Từ đồng nghĩa phủ định: ko, k, không, chẳng, kém, yếu, dở, chưa giỏi, "
        "không giỏi, không thích, không muốn, không biết\n\n"
        "PHÂN BIỆT QUAN TRỌNG:\n"
        "  - Hỏi 'môn học / môn nào / học môn gì' → asked_label: 'SUBJECT'\n"
        "  - Hỏi 'ngành nào / học ngành gì / chuyên ngành' → asked_label: 'MAJOR'\n"
        "QUAN TRỌNG - Chuẩn hóa keyword về tiếng Việt theo graph:\n"
        "  data analyst → chuyên viên phân tích dữ liệu\n"
        "  software engineer / developer → lập trình viên, kỹ sư phần mềm\n"
        "  tester / QA → kiểm thử\n"
        "  IT / information technology → công nghệ thông tin\n"
        "  AI / machine learning → trí tuệ nhân tạo, học máy\n"
        "  Nếu không biết tên tiếng Việt → giữ nguyên tiếng Anh\n\n"
        "Trả về JSON với đúng các trường sau:\n"
        "{\n"
        '  "keywords": ["từ khoá thực thể để tìm trong KG"],\n'
        '  "mentioned_labels": ["MAJOR|SUBJECT|SKILL|CAREER|TEACHER"],\n'
        '  "asked_label": "MAJOR|SUBJECT|SKILL|CAREER|TEACHER|UNKNOWN",\n'
        '  "negated_keywords": ["thực thể / kỹ năng / môn bị phủ định"],\n'
        '  "is_comparison": true\n'
        "}\n\n"
        "Ví dụ:\n"
        '  Câu: "Giỏi giao tiếp thì học ngành nào?" → mentioned_labels: ["SKILL"], asked_label: "MAJOR"\n'
        '  Câu: "Ngành CNTT có những nghề gì?" → mentioned_labels: ["MAJOR"], asked_label: "CAREER"\n'
        '  Câu: "Ko giỏi toán thì theo nghề lập trình viên được không?" '
        '→ negated_keywords: ["toán"], mentioned_labels: ["CAREER"]\n'
        '  Câu: "CNTT hay KTPM phù hợp hơn?" → is_comparison: true, mentioned_labels: ["MAJOR"]\n'
        '  Câu: "Học môn gì để làm lập trình viên?" → mentioned_labels: ["CAREER"], asked_label: "SUBJECT"\n'
        '  Câu: "Môn nào giúp tôi trở thành data analyst?" → mentioned_labels: ["CAREER"], asked_label: "SUBJECT"\n'
        '  Câu: "Cần học những môn gì cho nghề kế toán?" → mentioned_labels: ["CAREER"], asked_label: "SUBJECT"\n'
    )

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Phan tich cau hoi sau va tra ve json: {question}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return {
        "keywords":        parsed.get("keywords", []),
        "mentioned_labels": parsed.get("mentioned_labels", []),
        "asked_label":     parsed.get("asked_label", "UNKNOWN"),
        "negated_keywords": parsed.get("negated_keywords", []),
        "is_comparison":   parsed.get("is_comparison", False),
    }


def detect_negation_in_question(question: str) -> bool:
    """Kiểm tra nhanh câu hỏi có chứa từ phủ định không."""
    q_lower = question.lower()
    for neg in NEGATION_SYNONYMS:
        if neg in q_lower:
            return True
    return False


def get_relationship_constraint(intent: dict) -> str:
    """Lấy ràng buộc quan hệ dựa trên intent."""
    mentioned = intent.get("mentioned_labels", [])
    asked     = intent.get("asked_label", "UNKNOWN")
    is_comp   = intent.get("is_comparison", False)

    if is_comp and "MAJOR" in mentioned:
        return RELATIONSHIP_CONSTRAINTS.get(("MAJOR", "MAJOR"), "")

    # Lấy label đề cập đầu tiên
    first_mentioned = mentioned[0] if mentioned else None

    if first_mentioned and asked and asked != "UNKNOWN":
        key = (first_mentioned, asked)
        if key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[key]

    # Thử tổ hợp khác
    for m in mentioned:
        key = (m, asked)
        if key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[key]

    return "Trả lời theo đúng câu hỏi, chỉ dùng dữ liệu có trong Knowledge Graph."


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_relevant_communities(driver, keywords: list[str]) -> list[int]:
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
# BƯỚC 2: MULTI-HOP TRAVERSAL
# ══════════════════════════════════════════════════════════════════════════════

def _add_node_and_paths(rec, all_nodes, all_paths):
    """Helper: parse 1 record từ traversal query vào all_nodes / all_paths."""
    node_info = {
        "name":         rec["name"],
        "label":        rec["label"],
        "code":         rec["code"],
        "pagerank":     rec["pagerank"],
        "community_id": rec["community_id"],
        "hops":         rec["hops"],
    }
    all_nodes.append(node_info)
    node_names = rec["node_names"]
    rel_types  = rec["rel_types"]
    for i, rel in enumerate(rel_types):
        all_paths.append({
            "from":     node_names[i]   if i < len(node_names) else "",
            "to":       node_names[i+1] if i+1 < len(node_names) else "",
            "relation": rel,
            "hop":      i + 1,
        })


# Bảng các targeted query theo intent (mentioned_label, asked_label)
# Dùng khi BFS thông thường bị chặn bởi community filter
TARGETED_QUERIES: dict[tuple[str, str], str] = {
    ("MAJOR", "CAREER"): """
        MATCH (start:MAJOR)-[:LEADS_TO]->(n:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['LEADS_TO'] AS rel_types, [start.name, n.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("CAREER", "SKILL"): """
        MATCH (start:CAREER)-[:REQUIRES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.name) CONTAINS 'phân tích'
           OR toLower(start.name) CONTAINS 'analyst'
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['REQUIRES'] AS rel_types, [start.name, n.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("MAJOR", "SKILL"): """
        MATCH (start:MAJOR)-[:OFFERS]->(sub:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['OFFERS','PROVIDES'] AS rel_types, [start.name, sub.name, n.name] AS node_names, 2 AS hops
        LIMIT 30
    """,
    ("SKILL", "MAJOR"): """
        MATCH (n:MAJOR)-[:OFFERS]->(sub:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['OFFERS','PROVIDES'] AS rel_types, [n.name, sub.name, start.name] AS node_names, 2 AS hops
        LIMIT 30
    """,
    ("SKILL", "CAREER"): """
        MATCH (n:CAREER)-[:REQUIRES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['REQUIRES'] AS rel_types, [n.name, start.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("CAREER", "SUBJECT"): """
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['REQUIRES','PROVIDES'] AS rel_types, [start.name, sk.name, n.name] AS node_names, 2 AS hops
        LIMIT 30
    """,
    ("MAJOR", "SUBJECT"): """
        MATCH (start:MAJOR)-[:OFFERS]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['OFFERS'] AS rel_types, [start.name, n.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("SKILL", "SUBJECT"): """
        MATCH (n:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['PROVIDES'] AS rel_types, [n.name, start.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("SUBJECT", "SKILL"): """
        MATCH (start:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['PROVIDES'] AS rel_types, [start.name, n.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
    ("CAREER", "MAJOR"): """
        MATCH (n:MAJOR)-[:LEADS_TO]->(start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               n.pagerank AS pagerank, n.community_id AS community_id,
               ['LEADS_TO'] AS rel_types, [n.name, start.name] AS node_names, 1 AS hops
        LIMIT 30
    """,
}


def multihop_traversal(driver, keywords: list[str],
                       community_ids: list[int],
                       max_hops: int = MAX_HOPS,
                       intent: dict | None = None) -> tuple[list[dict], list[dict]]:
    all_nodes  = []
    all_paths  = []
    seen_names = set()

    mentioned_labels = (intent or {}).get("mentioned_labels", [])
    asked_label      = (intent or {}).get("asked_label", "UNKNOWN")
    first_mentioned  = mentioned_labels[0] if mentioned_labels else None

    # ── Phase 1: Targeted query theo intent (không bị chặn bởi community) ────
    targeted_key = (first_mentioned, asked_label) if first_mentioned else None
    targeted_cypher = TARGETED_QUERIES.get(targeted_key) if targeted_key else None

    if targeted_cypher:
        with driver.session() as session:
            for kw in keywords:
                try:
                    results = session.run(targeted_cypher, kw=kw)
                    for rec in results:
                        _add_node_and_paths(rec, all_nodes, all_paths)
                    if all_nodes:
                        print(f"  [targeted] ({targeted_key}) → {len(all_nodes)} nodes via direct path")
                except Exception as e:
                    print(f"  [targeted] WARNING: {e}")

    # ── Phase 2: BFS thông thường (community-filtered) để lấy context ────────
    with driver.session() as session:
        for kw in keywords:
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

                # Community filter KHÔNG áp dụng cho asked_label
                # để tránh chặn các node đích quan trọng
                community_filter = ""
                params: dict = {"seed_name": seed_name, "max_hops": max_hops}

                if community_ids and asked_label not in ("UNKNOWN", None):
                    community_filter = (
                        f"AND (n.community_id IN $cids "
                        f"OR n.community_id IS NULL "
                        f"OR labels(n)[0] = '{asked_label}')"
                    )
                    params["cids"] = community_ids
                elif community_ids:
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
                    _add_node_and_paths(rec, all_nodes, all_paths)

    return all_nodes, all_paths


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: PAGERANK RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_nodes(nodes: list[dict], top_k: int = TOP_K,
               negated_keywords: list[str] | None = None,
               asked_label: str | None = None) -> list[dict]:
    """
    Xếp hạng nodes theo PageRank với chiến lược 2 bucket:
    - Bucket 1 (ưu tiên): nodes khớp asked_label → lấy tối đa top_k * 2 / 3
    - Bucket 2 (context): các nodes còn lại → lấy phần còn lại
    Lọc bỏ nodes khớp negated_keywords và dedup theo (label, name).
    """
    negated_keywords = [kw.lower() for kw in (negated_keywords or [])]

    def score(n: dict) -> float:
        pr   = n.get("pagerank") or 0.0
        hops = n.get("hops")     or 1
        return pr / hops

    # Dedup toàn bộ trước (ưu tiên bản có hops nhỏ nhất = gần seed nhất)
    seen_keys: dict = {}
    for n in nodes:
        key = (n.get("label", ""), n.get("name", ""))
        if key not in seen_keys or (n.get("hops") or 99) < (seen_keys[key].get("hops") or 99):
            seen_keys[key] = n

    deduped = list(seen_keys.values())

    # Lọc thực thể bị phủ định
    if negated_keywords:
        deduped = [
            n for n in deduped
            if not any(neg in (n.get("name") or "").lower() for neg in negated_keywords)
        ]

    # Tách 2 bucket
    if asked_label and asked_label != "UNKNOWN":
        target_nodes  = [n for n in deduped if n.get("label") == asked_label]
        context_nodes = [n for n in deduped if n.get("label") != asked_label]

        target_nodes.sort(key=score, reverse=True)
        context_nodes.sort(key=score, reverse=True)

        target_slots  = max(top_k // 2, min(len(target_nodes), top_k))
        context_slots = top_k - min(len(target_nodes), target_slots)

        result = target_nodes[:target_slots] + context_nodes[:context_slots]
        print(f"  [rank] target({asked_label})={len(target_nodes)} dung {min(len(target_nodes), target_slots)} | "
              f"context={len(context_nodes)} dung {min(len(context_nodes), context_slots)}")
    else:
        deduped.sort(key=score, reverse=True)
        result = deduped[:top_k]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# LLM: Extract intent + Generate answer
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(ai_client: OpenAI, question: str,
                    ranked_nodes: list[dict], traversal_paths: list[dict],
                    intent: dict) -> str:
    """
    Tổng hợp câu trả lời từ KG context + intent constraints.
    """
    context = json.dumps({
        "ranked_results": ranked_nodes,
        "traversal_paths": traversal_paths[:60],
    }, ensure_ascii=False, indent=2)

    # Lấy ràng buộc quan hệ theo loại câu hỏi
    constraint = get_relationship_constraint(intent)

    # Bổ sung ghi chú phủ định nếu có
    negated = intent.get("negated_keywords", [])
    if negated:
        constraint += (
            f"\n\nLƯU Ý PHỦ ĐỊNH: Người dùng đề cập họ KHÔNG giỏi / không thích: {negated}. "
            "Loại bỏ các môn/kỹ năng/ngành này khỏi gợi ý. "
            "Thay vào đó gợi ý những lựa chọn phù hợp hơn."
        )

    system_prompt = ANSWER_SYSTEM_BASE.format(
        schema=SCHEMA_DESC,
        constraint=constraint,
    )

    # Cảnh báo về dữ liệu trống
    no_data_hint = ""
    if not ranked_nodes:
        no_data_hint = (
            "\n[CẢNH BÁO: Không tìm thấy dữ liệu liên quan trong Knowledge Graph. "
            "Thông báo lịch sự rằng dữ liệu chưa đủ, không bịa thông tin.]"
        )

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"Câu hỏi: {question}\n\n"
                f"Kết quả Knowledge Graph (đã xếp hạng PageRank):\n{context}"
                f"{no_data_hint}\n\n"
                "Hướng dẫn trả lời:\n"
                "- Dùng TẤT CẢ thông tin có trong kết quả trên để trả lời.\n"
                "- Nếu có node SUBJECT với code (mã môn) → nhắc đến tên môn và mã môn.\n"
                "- Nếu có node CAREER → nhắc đến nghề nghiệp cụ thể.\n"
                "- Nếu có node SKILL → liệt kê kỹ năng.\n"
                "- KHÔNG nói 'dữ liệu chưa đủ' nếu đã có nodes trong kết quả.\n"
                "- Trả lời tự nhiên bằng tiếng Việt, kèm mã ngành/mã môn khi có:"
            )},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════


def fetch_seed_entities(driver, keywords: list[str], mentioned_labels: list[str]) -> list[dict]:
    """
    Fetch trực tiếp các seed entity (MAJOR/CAREER/...) khớp keyword.
    Đảm bảo code/name của thực thể gốc luôn có trong context dù bị ranking đẩy xuống.
    """
    if not keywords or not mentioned_labels:
        return []
    label_filter = " OR ".join([f"n:{lbl}" for lbl in mentioned_labels])
    results = []
    with driver.session() as session:
        for kw in keywords:
            rows = session.run(f"""
                MATCH (n)
                WHERE ({label_filter})
                  AND toLower(n.name) CONTAINS toLower($kw)
                RETURN n.name AS name, labels(n)[0] AS label,
                       n.code AS code, n.pagerank AS pagerank,
                       n.community_id AS community_id
                LIMIT 3
            """, kw=kw).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": r["label"],
                    "code": r["code"], "pagerank": r["pagerank"],
                    "community_id": r["community_id"], "hops": 0,
                })
    return results

def ask(driver, ai_client: OpenAI, question: str,
        query_id: str | None = None) -> dict:
    if query_id is None:
        query_id = "q" + uuid.uuid4().hex[:6]

    print(f"\n{'='*60}")
    print(f"Q [{query_id}]: {question}")

    # ── Bước 1: Extract intent (keywords + labels + negation) ─────────────────
    intent = extract_query_intent(ai_client, question, memory)
    keywords         = intent["keywords"]
    negated_keywords = intent["negated_keywords"]
    print(f"  Keywords: {keywords}")
    print(f"  Intent: mentioned={intent['mentioned_labels']} asked={intent['asked_label']} "
          f"negated={negated_keywords} comparison={intent['is_comparison']}")

    # ── Bước 1b: Fetch seed entities (đảm bảo code luôn có trong context) ─────
    seed_entities = fetch_seed_entities(driver, keywords, intent.get("mentioned_labels", []))
    print(f"  Seed entities: {[(e['name'], e['code']) for e in seed_entities]}")

    # ── Bước 2: Community Detection ───────────────────────────────────────────
    community_ids = find_relevant_communities(driver, keywords)
    print(f"  Communities: {community_ids}")

    # ── Bước 3: Multi-hop BFS Traversal ──────────────────────────────────────
    raw_nodes, traversal_paths = multihop_traversal(
        driver, keywords, community_ids, max_hops=MAX_HOPS, intent=intent
    )
    print(f"  BFS nodes found: {len(raw_nodes)}  |  paths: {len(traversal_paths)}")

    # ── Bước 4: PageRank Ranking (có lọc thực thể phủ định) ──────────────────
    ranked_nodes = rank_nodes(raw_nodes, top_k=TOP_K, negated_keywords=negated_keywords,
                              asked_label=intent.get("asked_label"))
    print(f"  After PageRank ranking (top {TOP_K}): {len(ranked_nodes)} nodes")
    if ranked_nodes:
        top3 = [(n["name"], round(n.get("pagerank") or 0, 4)) for n in ranked_nodes[:3]]
        print(f"  Top 3: {top3}")

    # ── Bước 4b: Inject seed entities vào đầu context (đảm bảo code luôn có) ─
    # Dedup: loại seed_entities đã có trong ranked_nodes
    ranked_names = {n.get("name") for n in ranked_nodes}
    extra_seeds  = [e for e in seed_entities if e.get("name") not in ranked_names]
    context_nodes = extra_seeds + ranked_nodes

    # ── Bước 5: LLM tổng hợp câu trả lời (có intent constraints) ────
    answer = generate_answer(
        ai_client, question, context_nodes, traversal_paths,
        intent=intent
    )
    print(f"\nA: {answer}")

    qa_record = {
        "query_id":            query_id,
        "query":               question,
        "generated_answer":    answer,
        "keywords":            keywords,
        "intent":              intent,
        "communities_covered": community_ids,
        "context_text":        json.dumps(ranked_nodes, ensure_ascii=False),
        "retrieved_nodes": [
            {
                "node_id":  f"node{i+1:03d}",
                "content":  json.dumps(n, ensure_ascii=False),
                "score":    round(n.get("pagerank") or 0, 6),
                "entities": [n.get("name", "")],
            }
            for i, n in enumerate(ranked_nodes)
        ],
        "traversal_path":      traversal_paths[:20],
        "timestamp":           datetime.datetime.now().isoformat(),
        "algorithm": {
            "community_detection": "Louvain (NetworkX)",
            "traversal":           f"BFS multi-hop (max_hops={MAX_HOPS})",
            "ranking":             "PageRank (damping=0.85) + negation filter",
        },
    }

    return qa_record


# ── Neo4j ─────────────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ── Interactive loop với Memory ───────────────────────────────────────────────

def interactive_loop(driver, ai_client: OpenAI):
    print("\n🎓 Knowledge Graph Chatbot (NEU)")
    print(f"Pipeline: Intent Detection → Community → BFS (max={MAX_HOPS}) → PageRank → LLM")
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

        qa_record = ask(
            driver, ai_client, question,
            query_id=f"q{counter:03d}"
        )

        counter += 1


def main():
    print("Starting KG Chatbot...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()

    try:
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
from openai import OpenAI
import networkx as nx
from cdlib import algorithms
import os
from dotenv import load_dotenv
from docx_reader import read_docx_from_directory
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =========================================================
# ENV
# =========================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# ULTRA FAST CONFIG
# =========================================================
MAX_WORKERS = 15
CHUNK_SIZE = 2000
OVERLAP_SIZE = 300
USE_MINI_MODEL = True

# =========================================================
# BATCH LLM CALL
# =========================================================
def batch_api_call(items, system_prompt, user_template, max_tokens=300):
    results = [None] * len(items)

    def process_item(item_data):
        index, item = item_data
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini" if USE_MINI_MODEL else "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(item=item[:1500])}
                ],
                max_tokens=max_tokens
            )
            return index, response.choices[0].message.content
        except Exception as e:
            print(f"[WARN] Chunk {index} failed: {e}")
            return index, ""

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_item, (i, item)): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            idx, res = future.result()
            results[idx] = res

    return [r for r in results if r]


# =========================================================
# MAIN PIPELINE
# =========================================================
def ultra_fast_pipeline(documents, query):
    start_time = time.time()

    print("\n" + "=" * 80)
    print("ULTRA-FAST GRAPH RAG PIPELINE (EDGE-AWARE)")
    print("=" * 80)

    # -----------------------------------------------------
    # STEP 1: CHUNKING
    # -----------------------------------------------------
    print(f"\n[1/5] Chunking {len(documents)} documents...")
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), CHUNK_SIZE - OVERLAP_SIZE):
            chunks.append(doc[i:i + CHUNK_SIZE])
    print(f"  ✓ {len(chunks)} chunks created")

    # -----------------------------------------------------
    # STEP 2: ENTITY + RELATION EXTRACTION
    # -----------------------------------------------------
    print(f"\n[2/5] Extracting entities & relations (parallel)...")

    system_prompt = """
You are an information extraction system.

Extract ENTITIES and RELATIONSHIPS from the text.

STRICT FORMAT (no explanation, no markdown):

ENTITY: <entity name>
RELATION: <entity_1> -> <relation> -> <entity_2>

Rules:
- Use '->' exactly for relations
- Entity names: max 5 words
- Use Vietnamese if the text is Vietnamese
- Do NOT invent relations not present in text

Example:
ENTITY: Hệ điều hành
ENTITY: Tiến trình
RELATION: Hệ điều hành -> quản lý -> Tiến trình
"""

    elements = batch_api_call(
        chunks,
        system_prompt=system_prompt,
        user_template="{item}",
        max_tokens=350
    )

    print(f"  ✓ {len(elements)} extraction results")

    # -----------------------------------------------------
    # STEP 3: BUILD GRAPH (EDGE-FIRST)
    # -----------------------------------------------------
    print(f"\n[3/5] Building graph...")
    G = nx.Graph()

    for elem in elements:
        lines = [l.strip() for l in elem.split("\n") if l.strip()]
        for line in lines:
            if line.startswith("RELATION:"):
                try:
                    _, rel = line.split(":", 1)
                    src, _, tgt = [p.strip() for p in rel.split("->")]
                    if len(src) > 2 and len(tgt) > 2:
                        G.add_edge(src, tgt)
                except:
                    pass

            elif line.startswith("ENTITY:"):
                node = line.replace("ENTITY:", "").strip()
                if len(node) > 2:
                    G.add_node(node)

    print(f"  ✓ Nodes: {G.number_of_nodes()}")
    print(f"  ✓ Edges: {G.number_of_edges()}")
    print(f"  ✓ Sample edges: {list(G.edges())[:5]}")

    # -----------------------------------------------------
    # STEP 4: COMMUNITY DETECTION (LEIDEN)
    # -----------------------------------------------------
    print(f"\n[4/5] Detecting communities...")

    communities = []

    for component in nx.connected_components(G):
        if len(component) > 2:
            subgraph = G.subgraph(component)
            try:
                comms = algorithms.leiden(subgraph)
                communities.extend([list(c) for c in comms.communities])
            except:
                communities.append(list(component))
        else:
            communities.append(list(component))

    large_communities = [c for c in communities if len(c) >= 3]
    if not large_communities:
        large_communities = communities

    print(f"  ✓ {len(large_communities)} communities")

    # -----------------------------------------------------
    # STEP 5: FINAL ANSWER GENERATION
    # -----------------------------------------------------
    print(f"\n[5/5] Generating answer...")

    comm_desc = []
    for i, comm in enumerate(large_communities[:12]):
        comm_desc.append(f"Nhóm {i+1}: {', '.join(comm[:10])}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Dựa trên các nhóm khái niệm và quan hệ, hãy trả lời câu hỏi một cách chi tiết bằng tiếng Việt."
            },
            {
                "role": "user",
                "content": f"""
Câu hỏi: {query}

Các nhóm kiến thức chính:
{chr(10).join(comm_desc)}
"""
            }
        ],
        max_tokens=1000
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✓ HOÀN THÀNH – {elapsed:.1f}s")
    print("=" * 80)

    return response.choices[0].message.content


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GRAPH RAG – EDGE-AWARE ULTRA FAST")
    print("=" * 80)

    documents = read_docx_from_directory("example_docx")

    if not documents:
        print("❌ Không tìm thấy file .docx")
        exit(1)

    print(f"✓ Loaded {len(documents)} documents")
    print(f"✓ Total characters: {sum(len(d) for d in documents):,}")

    query = input("\nCâu hỏi: ").strip()
    if not query:
        query = "Tổng hợp nội dung chính của các tài liệu"

    answer = ultra_fast_pipeline(documents, query)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)
    print("=" * 80)

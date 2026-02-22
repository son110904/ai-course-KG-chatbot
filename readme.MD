# GraphRAG Chatbot — Neo4j Aura

Hệ thống hỏi đáp học thuật dựa trên Knowledge Graph, sử dụng Neo4j Aura + OpenAI.

## Pipeline

```
MinIO (docx/pdf)
      ↓
script1.py   — LLM trích xuất entities + relations → cache/output/
      ↓
script2_.py      — Sinh Cypher MERGE + push lên Neo4j Aura
      ↓
script2b.py — Ghép MAJOR ↔ CAREER (post-processing)
      ↓
script3.py         — Chatbot hỏi đáp
script4.py               — Đánh giá độ chính xác
```



## Cài đặt
```bash
git clone https://github.com/your-username/ai-course-KG-chatbot.git
cd ai-course-KG-chatbot
```

```bash
pip install -r requirements.txt
```


Tạo file `.env`:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET=syllabus
MINIO_BASE_FOLDER=courses-processed
MAX_WORKERS=5
NEO4J_URI=neo4j+s://<aura-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
MAX_HOPS=3
TOP_K=20
```

## Chạy

### 1. Build Knowledge Graph (chạy 1 lần, Sơn đã chạy rồi nên mọi người chạy cái script3 thôi)

```bash
python script1.py    # Extract từ MinIO
python script2.py       # Push lên Neo4j
python script2b.py  # Fix MAJOR ↔ CAREER links
```

### 2. Chatbot

```bash
python script3.py
# Lần đầu chọn "yes" để chạy Community Detection + PageRank
```

### 3. Đánh giá

```bash
# Tạo ground truth (nếu chưa có)
python scripts/generate_groundtruth.py --llm openai \
    --domain "Vietnamese university curriculum knowledge graph"

# Chạy eval
python script4.py --dataset data/ground_truth/dataset.json --k 5
```

Kết quả lưu tại `reports/eval_{timestamp}.json` + `reports/charts/`.

## Thuật toán retrieval (script3)

```
Câu hỏi
  ↓ Extract keywords (strip học hàm: TS, PGS, Thầy...)
  ↓ Community Detection (Louvain per-label) → thu hẹp phạm vi tìm kiếm
  ↓ BFS Multi-hop Traversal (max_hops=3)
  ↓ PageRank Ranking (score = pagerank / hops) → top 20 nodes
  ↓ LLM tổng hợp câu trả lời (kèm mã ngành/mã môn)
```


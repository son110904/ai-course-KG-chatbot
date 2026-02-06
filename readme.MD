# GraphRAG chatbot (Neo4j Aura)

Pipeline hiện tại:
1. **Load dữ liệu** từ MinIO (`docx`, `pdf`).
2. **Chunking** tài liệu.
3. **LLM extraction** để lấy `entities` + `relations` có mô tả.
4. **Ingest vào Neo4j Aura** với schema:
   - `(:Entity {id, description, ...})`
   - `(:Chunk {id, text, source})`
   - `(Chunk)-[:MENTIONS]->(Entity)`
   - `(Entity)-[:RELATION_TYPE {description, weight}]->(Entity)`
5. **GraphRAG retrieval**:
   - fulltext search trên entity (`entity_id_ft`)
   - mở rộng neighborhood để lấy fact liên quan
   - trả context cho LLM trả lời.

## Biến môi trường cần có

```env
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET=
MINIO_PREFIX=
MINIO_SECURE=false

NEO4J_URI=neo4j+s://<aura-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0

GRAPH_RAG_MAX_HITS=8
GRAPH_RAG_MAX_HOPS=2
GRAPH_RAG_MAX_FACTS=30
```

## Chạy ingest + demo hỏi đáp

```bash
python main.py
```

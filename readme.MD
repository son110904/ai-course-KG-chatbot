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

## System prompt for LLM
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
## Chạy ingest + demo hỏi đáp

```bash
python app.py
```

# GraphRAG chatbot (Neo4j Aura)

pip install -r requirements.txt
down Neo4j destop về, cài plugin: APOC, Graph Data Science

```env
# ===== Neo4j =====
DB_URL=
DB_USER=
DB_PASSWORD=
DB_DATABASE=

# ===== OpenAI =====
OPENAI_API_KEY=
OPENAI_MODEL=
OPENAI_TEMPERATURE=
K_HOPS=
TOP_K_SEEDS=
MAX_SUBGRAPH_NODES=

# ===== Entity Linking =====
FUZZY_THRESHOLD=
```

## Chạy ingest + demo hỏi đáp

```bash
python build_graph_v3.py
```


```bash
python query_cli_v3.py
```

import os
from dotenv import load_dotenv

load_dotenv()

# ===== MinIO =====
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# ===== Neo4j Aura =====
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ===== LLM =====
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

# ===== GraphRAG Retrieval =====
GRAPH_RAG_MAX_HITS = int(os.getenv("GRAPH_RAG_MAX_HITS", 8))
GRAPH_RAG_MAX_HOPS = int(os.getenv("GRAPH_RAG_MAX_HOPS", 2))
GRAPH_RAG_MAX_FACTS = int(os.getenv("GRAPH_RAG_MAX_FACTS", 30))

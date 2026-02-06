import logging
from uuid import uuid4

from chatbot import chatbot
from chunking import chunk_documents
from config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_PREFIX,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)
from graph_extractor import extract_graph
from minio_loader import create_document_loader
from neo4j_store import ensure_schema, save_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest() -> None:
    loader = create_document_loader(
        "minio",
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket=MINIO_BUCKET,
        prefix=MINIO_PREFIX,
        secure=MINIO_SECURE,
    )
    docs = loader.load()
    chunks = chunk_documents(docs)

    ensure_schema()

    for idx, chunk in enumerate(chunks):
        chunk_payload = {
            "id": f"chunk-{idx}-{uuid4().hex[:8]}",
            "content": chunk["content"],
            "source": chunk.get("source", ""),
        }
        graph = extract_graph(chunk_payload["content"])
        save_graph(graph, chunk_payload)

    logger.info("Ingestion hoàn tất: %s chunks", len(chunks))


if __name__ == "__main__":
    ingest()
    print(chatbot("Môn học tiên quyết của Hệ điều hành là gì?"))

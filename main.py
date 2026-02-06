from minio_loader import LoadedDocument
from chunking import chunk_documents
from graph_extractor import extract_graph
from neo4j_store import save_graph
from chatbot import chatbot

def ingest():
    docs = load_documents()
    chunks = chunk_documents(docs)

    for c in chunks:
        graph = extract_graph(c["content"])
        save_graph(graph)

if __name__ == "__main__":
    ingest()
    print(chatbot("Môn học tiên quyết của Hệ điều hành là gì?"))

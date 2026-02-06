from dataclasses import asdict

from langchain.text_splitter import RecursiveCharacterTextSplitter


def _normalize_document(doc):
    if isinstance(doc, dict):
        return {
            "content": doc.get("content", ""),
            "source": doc.get("source", ""),
            "metadata": doc.get("metadata", {})
        }
    if hasattr(doc, "__dataclass_fields__"):
        return asdict(doc)
    return {
        "content": getattr(doc, "content", ""),
        "source": getattr(doc, "source", ""),
        "metadata": getattr(doc, "metadata", {})
    }

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        normalized = _normalize_document(doc)
        parts = splitter.split_text(normalized["content"])
        for p in parts:
            chunks.append({
                "content": p,
                "source": normalized["source"],
                "metadata": normalized.get("metadata", {})
            })

    return chunks

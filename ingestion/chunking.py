import re
from dataclasses import asdict
from typing import Dict, Iterable, List, Set


VI_STOPWORDS = {
    "và", "là", "của", "cho", "với", "các", "những", "một", "trong", "được", "theo", "từ",
    "này", "đó", "khi", "đến", "về", "để", "có", "không", "đang", "sẽ", "hoặc", "tại", "thì",
}


def _normalize_document(doc):
    if isinstance(doc, dict):
        return {
            "content": doc.get("content", ""),
            "source": doc.get("source", ""),
            "metadata": doc.get("metadata", {}),
        }
    if hasattr(doc, "__dataclass_fields__"):
        return asdict(doc)
    return {
        "content": getattr(doc, "content", ""),
        "source": getattr(doc, "source", ""),
        "metadata": getattr(doc, "metadata", {}),
    }


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def _keyword_set(text: str) -> Set[str]:
    tokens = re.findall(r"[\wÀ-ỹ]{3,}", text.lower())
    return {t for t in tokens if t not in VI_STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _emit_chunk(
    chunk_text: str,
    source: str,
    metadata: Dict,
    chunks: List[Dict],
):
    content = chunk_text.strip()
    if content:
        chunks.append({"content": content, "source": source, "metadata": metadata})


def chunk_documents(
    documents: Iterable,
    target_chunk_size: int = 1200,
    min_chunk_size: int = 400,
    semantic_similarity_threshold: float = 0.12,
):
    """
    Chunk theo ngữ nghĩa gần đúng:
    - Tách theo paragraph trước
    - Gộp paragraph liên tiếp khi độ tương đồng keyword đủ cao
    - Cắt chunk khi đổi chủ đề hoặc vượt target size
    """
    chunks = []

    for doc in documents:
        normalized = _normalize_document(doc)
        paragraphs = _split_paragraphs(normalized.get("content", ""))

        current_parts: List[str] = []
        current_keywords: Set[str] = set()

        for paragraph in paragraphs:
            paragraph_keywords = _keyword_set(paragraph)
            candidate_text = "\n\n".join(current_parts + [paragraph])

            if not current_parts:
                current_parts = [paragraph]
                current_keywords = paragraph_keywords
                continue

            similarity = _jaccard(current_keywords, paragraph_keywords)
            should_split_for_topic = (
                similarity < semantic_similarity_threshold
                and len("\n\n".join(current_parts)) >= min_chunk_size
            )
            should_split_for_size = len(candidate_text) > target_chunk_size

            if should_split_for_topic or should_split_for_size:
                _emit_chunk("\n\n".join(current_parts), normalized["source"], normalized.get("metadata", {}), chunks)
                current_parts = [paragraph]
                current_keywords = paragraph_keywords
            else:
                current_parts.append(paragraph)
                current_keywords = current_keywords | paragraph_keywords

        if current_parts:
            _emit_chunk("\n\n".join(current_parts), normalized["source"], normalized.get("metadata", {}), chunks)

    return chunks

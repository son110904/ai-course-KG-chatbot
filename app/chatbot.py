from langchain_openai import ChatOpenAI

from config.config import OPENAI_MODEL, OPENAI_TEMPERATURE
from retrieval.retriever import graph_retrieve

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


def chatbot(question: str):
    graph_context = graph_retrieve(question)

    prompt = f"""
Bạn là trợ lý học vụ. Hãy trả lời CHỈ dựa trên dữ liệu GraphRAG bên dưới.
Nếu dữ liệu chưa đủ, nói rõ chưa đủ thông tin và nêu cần thêm gì.

Ngữ cảnh GraphRAG:
{graph_context}

Câu hỏi:
{question}
"""

    return llm.invoke(prompt).content

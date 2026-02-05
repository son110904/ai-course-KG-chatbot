from langchain_openai import ChatOpenAI
from config import *
from retriever import graph_retrieve

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE
)

def chatbot(question: str):
    graph_context = graph_retrieve(question)

    prompt = f"""
    Dữ liệu đào tạo:
    {graph_context}

    Trả lời câu hỏi:
    {question}
    """

    return llm.invoke(prompt).content

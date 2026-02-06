import json

from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_TEMPERATURE
from neo4j_store import driver

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE
)


def _load_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None


def extract_query_terms(question: str):
    prompt = f"""
    Trích xuất các thực thể hoặc từ khóa chính từ câu hỏi sau.
    Trả về JSON array thuần, ví dụ: ["Hệ điều hành", "tiên quyết"].
    Câu hỏi: {question}
    """
    response = llm.invoke(prompt).content
    data = _load_json(response)
    if isinstance(data, list) and data:
        return [str(item).strip() for item in data if str(item).strip()]
    return [question]


def graph_retrieve(question: str):
    terms = extract_query_terms(question)
    query = """
    MATCH (n)-[r]->(m)
    WHERE any(term IN $terms WHERE toLower(n.id) CONTAINS toLower(term)
        OR toLower(m.id) CONTAINS toLower(term))
    RETURN n.id AS source,
           labels(n) AS source_labels,
           type(r) AS relation,
           m.id AS target,
           labels(m) AS target_labels
    LIMIT 20
    """

    with driver.session() as session:
        results = session.run(query, terms=terms).data()

    if not results:
        return "Không tìm thấy dữ liệu phù hợp trong đồ thị."

    lines = []
    for record in results:
        source = record.get("source")
        target = record.get("target")
        relation = record.get("relation")
        lines.append(f"{source} -[{relation}]-> {target}")
    return "\n".join(lines)

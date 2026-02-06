import json
from typing import List

from langchain_openai import ChatOpenAI

from config.config import (
    GRAPH_RAG_MAX_FACTS,
    GRAPH_RAG_MAX_HITS,
    GRAPH_RAG_MAX_HOPS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from storage.neo4j_store import driver
from config.config import NEO4J_DATABASE

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


def _load_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def extract_query_terms(question: str) -> List[str]:
    prompt = f"""
Trích xuất thực thể/từ khóa cho truy vấn GraphRAG từ câu hỏi sau.
Trả về JSON array thuần. Giữ cả cụm từ quan trọng.
Câu hỏi: {question}
"""
    response = llm.invoke(prompt).content
    data = _load_json(response)
    if isinstance(data, list):
        terms = [str(item).strip() for item in data if str(item).strip()]
        if terms:
            return terms
    return [question]


def graph_retrieve(question: str) -> str:
    terms = extract_query_terms(question)

    max_hops = max(1, GRAPH_RAG_MAX_HOPS)

    cypher = f"""
    CALL db.index.fulltext.queryNodes('entity_id_ft', $query) YIELD node, score
    WITH node, score
    ORDER BY score DESC
    LIMIT $max_hits
    MATCH path = (node)-[r*1..{max_hops}]-(nbr:Entity)
    WITH node, score, relationships(path) AS rels, nodes(path) AS nodes_in_path
    UNWIND rels AS rel
    WITH node, score, rel, nodes_in_path
    WITH node, score,
         startNode(rel) AS source,
         endNode(rel) AS target,
         type(rel) AS rel_type,
         coalesce(rel.description, '') AS rel_description,
         coalesce(rel.weight, 1) AS rel_weight,
         nodes_in_path
    RETURN node.id AS seed,
           score,
           source.id AS source,
           rel_type AS relation,
           target.id AS target,
           rel_description AS relation_description,
           rel_weight AS relation_weight,
           [n IN nodes_in_path | n.id][0..5] AS path_nodes
    ORDER BY score DESC, relation_weight DESC
    LIMIT $max_facts
    """

    query_text = " OR ".join(terms)

    with driver.session(database=NEO4J_DATABASE) as session:
        records = session.run(
            cypher,
            query=query_text,
            max_hits=GRAPH_RAG_MAX_HITS,
            max_facts=GRAPH_RAG_MAX_FACTS,
        ).data()

    if not records:
        return "Không tìm thấy dữ liệu phù hợp trong đồ thị tri thức."

    lines = []
    for idx, record in enumerate(records, 1):
        lines.append(
            f"[{idx}] seed={record['seed']} | {record['source']} -[{record['relation']}]-> {record['target']}"
            f" | weight={record['relation_weight']} | mô tả={record['relation_description']}"
        )
    return "\n".join(lines)

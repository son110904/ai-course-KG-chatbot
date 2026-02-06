import json

from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_TEMPERATURE

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE
)


def _load_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None


def _dedupe_entities(entities):
    seen = set()
    cleaned = []
    for entity in entities:
        entity_id = entity.get("id", "").strip()
        entity_type = entity.get("type", "Entity").strip() or "Entity"
        key = (entity_id, entity_type)
        if entity_id and key not in seen:
            cleaned.append({"id": entity_id, "type": entity_type})
            seen.add(key)
    return cleaned


def _dedupe_relations(relations):
    seen = set()
    cleaned = []
    for relation in relations:
        source = relation.get("source", "").strip()
        target = relation.get("target", "").strip()
        relation_type = relation.get("type", "RELATED_TO").strip() or "RELATED_TO"
        key = (source, relation_type, target)
        if source and target and key not in seen:
            cleaned.append({
                "source": source,
                "target": target,
                "type": relation_type
            })
            seen.add(key)
    return cleaned


def extract_graph(text: str):
    prompt = f"""
    Bạn là trợ lý trích xuất tri thức. Hãy trích xuất các thực thể và quan hệ từ đoạn văn sau.

    Yêu cầu:
    - Trả về JSON thuần (không markdown).
    - Định dạng:
    {{
      "entities": [{{"id": "...", "type": "..."}}],
      "relations": [{{"source": "...", "target": "...", "type": "..."}}]
    }}
    - "type" của quan hệ dùng UPPER_SNAKE_CASE (VD: HAS_SUBJECT).
    - Không suy diễn ngoài nội dung.

    Văn bản:
    {text}
    """

    response = llm.invoke(prompt).content
    data = _load_json(response)
    if not data:
        return {"entities": [], "relations": []}

    entities = _dedupe_entities(data.get("entities", []))
    relations = _dedupe_relations(data.get("relations", []))

    return {"entities": entities, "relations": relations}

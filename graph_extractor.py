import json
import re
from typing import Dict, List

from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_TEMPERATURE

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


def _load_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _normalize_relation_type(value: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9_]", "_", value.upper())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "RELATED_TO"


def _dedupe_entities(entities: List[Dict]) -> List[Dict]:
    seen = set()
    cleaned = []
    for entity in entities:
        entity_id = str(entity.get("id", "")).strip()
        entity_type = str(entity.get("type", "Entity")).strip() or "Entity"
        description = str(entity.get("description", "")).strip()

        key = (entity_id.casefold(), entity_type.casefold())
        if entity_id and key not in seen:
            cleaned.append(
                {
                    "id": entity_id,
                    "type": entity_type,
                    "description": description,
                }
            )
            seen.add(key)
    return cleaned


def _dedupe_relations(relations: List[Dict]) -> List[Dict]:
    seen = set()
    cleaned = []
    for relation in relations:
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        relation_type = _normalize_relation_type(str(relation.get("type", "RELATED_TO")).strip())
        description = str(relation.get("description", "")).strip()

        key = (source.casefold(), relation_type, target.casefold())
        if source and target and key not in seen:
            cleaned.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "description": description,
                }
            )
            seen.add(key)
    return cleaned


def extract_graph(text: str):
    prompt = f"""
Bạn là trợ lý trích xuất tri thức cho pipeline GraphRAG.

Hãy trích xuất thực thể + quan hệ từ đoạn văn bên dưới và trả về JSON THUẦN:
{{
  "entities": [
    {{"id": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}}
  ]
}}

Quy tắc:
- Không markdown, không giải thích ngoài JSON.
- Quan hệ phải ở UPPER_SNAKE_CASE.
- Chỉ dùng thông tin xuất hiện trong văn bản.
- "description" ngắn gọn, chứa ngữ cảnh giúp trả lời câu hỏi sau này.

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

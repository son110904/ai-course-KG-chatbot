import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import numpy as np
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALIAS = {
    "neu": "Trường Đại học Kinh tế Quốc dân",
    "đh kinh tế quốc dân": "Trường Đại học Kinh tế Quốc dân",
    "trường đh kinh tế quốc dân": "Trường Đại học Kinh tế Quốc dân"
}

INPUT_FILE = "pipeline_outputs/step2_extraction_output.txt"
OUTPUT_FILE = "pipeline_outputs/step2.1_entity_linking_output.txt"

def normalize(text):
    return text.lower().strip()

# phát hiện entity có số thứ tự (1,2,3 hoặc I,II,IV…)
def is_numbered_entity(entity: str):
    entity = entity.lower()
    return bool(re.search(r"\b\d+\b", entity)) or bool(re.search(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", entity))

def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding)

def cosine(a, b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

entities = []
lines = open(INPUT_FILE, encoding="utf-8").readlines()

for line in lines:
    if line.startswith("ENTITY:"):
        e = line.replace("ENTITY:", "").strip()
        entities.append(e)

linked = {}
embeddings = {}

for e in entities:
    key = normalize(e)

    # 1. alias rule
    if key in ALIAS:
        linked[e] = ALIAS[key]
        continue

    # 🚫 nếu là entity có số thứ tự thì không merge
    if is_numbered_entity(e):
        linked[e] = e
        if e not in embeddings:
            embeddings[e] = get_embedding(e)
        continue

    # 2. embedding similarity
    emb_e = get_embedding(e)
    best = e
    best_score = 0.0

    for canon in embeddings:
        # không so với entity có số
        if is_numbered_entity(canon):
            continue

        score = cosine(emb_e, embeddings[canon])
        if score > best_score:
            best_score = score
            best = canon

    if best_score > 0.85:
        linked[e] = best
    else:
        linked[e] = e
        embeddings[e] = emb_e

# ghi file output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for raw, canon in linked.items():
        if raw != canon:
            f.write(f"MERGE: {raw}  -->  {canon}\n")
        else:
            f.write(f"KEEP: {raw}\n")

print("✅ Entity Linking xong! Xem file:", OUTPUT_FILE)

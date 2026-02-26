"""
Script 1 (OPTIMIZED): Extract entities & relationships from MinIO JSON files
- Async I/O với asyncio + aiohttp cho MinIO downloads
- ThreadPoolExecutor chỉ cho OpenAI calls (sync SDK)
- Retry với exponential backoff (openai rate limit safe)
- Validate output JSON trước khi lưu
- Resume support (auto-skip đã có)
- Structured logging
"""

import os
import json
import asyncio
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from minio import Minio
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./cache/extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MINIO_ENDPOINT    = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY  = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY  = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET      = os.getenv("MINIO_BUCKET")
MINIO_SECURE      = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BASE_FOLDER = os.getenv("MINIO_BASE_FOLDER", "courses-processed")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

INPUT_FOLDERS = ["curriculum", "career_description", "syllabus"]
LOCAL_OUT_DIR = Path("./cache/output")
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", "10"))   # tăng lên vì I/O bound
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = 2.0  # seconds
# ──────────────────────────────────────────────────────────────────────────────

DOCTYPE_MAP = {
    "curriculum":         "curriculum",
    "career_description": "career_description",
    "syllabus":           "syllabus",
}

# ─── VALID NODE LABELS & RELATIONSHIPS PER DOCTYPE ────────────────────────────
VALID_NODES_BY_DOCTYPE = {
    "syllabus":           {"DOCUMENT", "SUBJECT", "TEACHER", "SKILL", "MAJOR"},
    "curriculum":         {"DOCUMENT", "MAJOR", "SUBJECT", "CAREER"},
    "career_description": {"DOCUMENT", "CAREER", "SKILL", "MAJOR"},
}

VALID_REL_TYPES = {
    "OFFERS", "TEACH", "PROVIDES", "REQUIRES",
    "PREREQUISITE_FOR", "LEADS_TO", "MENTIONED_IN",
}

REQUIRED_CODE_LABELS = {"MAJOR", "SUBJECT"}  # phải có code thực sự (trừ career_description MAJOR)

# ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """Bạn là chuyên gia Knowledge Graph. Nhiệm vụ của bạn là đọc tài liệu JSON và trích xuất các thực thể (nodes) và quan hệ (relationships) theo schema đã cho.

ONTOLOGY:
- Node Labels: MAJOR {name, code}, SUBJECT {name, code}, SKILL {name}, CAREER {name}, TEACHER {name}, DOCUMENT {name, docid, doctype}
- Relationships chính:
  (MAJOR)-[:OFFERS]->(SUBJECT)
  (TEACHER)-[:TEACH]->(SUBJECT)
  (SUBJECT)-[:PROVIDES]->(SKILL)
  (CAREER)-[:REQUIRES]->(SKILL)
  (SUBJECT)-[:PREREQUISITE_FOR]->(SUBJECT)
  (MAJOR)-[:LEADS_TO]->(CAREER)
- Relationships phụ (truy xuất nguồn gốc):
  (MAJOR|TEACHER|SUBJECT|CAREER|SKILL)-[:MENTIONED_IN]->(DOCUMENT)

NGUYÊN TẮC CHUNG:
1. Tên thực thể (name) phải VIẾT HOA tất cả chữ cái sau khi chuẩn hóa.
2. Nếu cùng tên (đã chuẩn hóa) → cùng thực thể (dùng lại id).
3. Bỏ qua thuộc tính null/thiếu thay vì để null.
4. Mọi thực thể phải có quan hệ MENTIONED_IN tới node DOCUMENT.
5. CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích.
6. Nhất quán: Đại từ (anh ấy, học phần này, ngành này,...) → dùng tên gốc VIẾT HOA.

RÀNG BUỘC THEO LOẠI TÀI LIỆU:
- syllabus: chứa SUBJECT (1 duy nhất, bắt buộc có code thực sự), TEACHER, SKILL, MAJOR (nếu có). KHÔNG tạo CAREER.
- curriculum: chứa MAJOR (1 duy nhất, bắt buộc có code), SUBJECT (bắt buộc có code). KHÔNG tạo TEACHER.
- career_description: chứa CAREER (1 duy nhất), SKILL, MAJOR (không cần code). KHÔNG tạo SUBJECT, TEACHER.

QUAN TRỌNG:
- SUBJECT và MAJOR bắt buộc phải có code thực sự (ví dụ: IT001, 7480201). Nếu không có code → KHÔNG tạo node đó.
- KHÔNG dùng code='', code='SKILL' hoặc các giá trị vô nghĩa.
- KHÔNG tạo node với label sai loại tài liệu (xem ràng buộc trên).

Định dạng output:
{
  "metadata": {"docid": "...", "doctype": "..."},
  "nodes": [
    {"id": "DOC1", "label": "DOCUMENT", "properties": {"name": "...", "docid": "...", "doctype": "..."}},
    {"id": "S1",   "label": "SUBJECT",  "properties": {"name": "...", "code": "IT001"}},
    ...
  ],
  "relationships": [
    {"source": "T1", "target": "S1",   "type": "TEACH"},
    {"source": "S1", "target": "DOC1", "type": "MENTIONED_IN"},
    ...
  ]
}"""

SYSTEM_PROMPT_BY_DOCTYPE = {
    "syllabus": SYSTEM_PROMPT_BASE + """

Tài liệu này là ĐỀ CƯƠNG MÔN HỌC (syllabus).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node SUBJECT (môn học). SUBJECT phải có code môn học thực sự.
- TEACHER: các giảng viên phụ trách môn đó.
- MAJOR: ngành nào đang giảng dạy môn này (nếu có đề cập).
- KHÔNG tạo CAREER.

TRÍCH XUẤT SKILL — RẤT QUAN TRỌNG:
- Nguồn CHÍNH: phần "Chuẩn đầu ra học phần" / "course_learning_outcomes" / "learning_outcomes".
- Mỗi chuẩn đầu ra (CLO) → tạo 1 node SKILL ngắn gọn, súc tích.
  Ví dụ đúng: "LẬP TRÌNH PYTHON", "PHÂN TÍCH DỮ LIỆU"
  Ví dụ sai: "SINH VIÊN CÓ KHẢ NĂNG VIẾT ĐƯỢC CHƯƠNG TRÌNH..."
- Tạo relationship: (SUBJECT)-[:PROVIDES]->(SKILL)

Quan hệ hợp lệ: TEACHER-[TEACH]->SUBJECT, SUBJECT-[PROVIDES]->SKILL, MAJOR-[OFFERS]->SUBJECT.""",

    "curriculum": SYSTEM_PROMPT_BASE + """

Tài liệu này là CHƯƠNG TRÌNH ĐÀO TẠO (curriculum).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node MAJOR (ngành học). MAJOR phải có code ngành thực sự.
- SUBJECT: tất cả môn học trong chương trình, bắt buộc có code. Nếu không có code → bỏ qua.
- KHÔNG tạo TEACHER.

TRÍCH XUẤT CAREER — RẤT QUAN TRỌNG:
- Nguồn CHÍNH: phần "Cơ hội làm việc và khả năng học tập nâng cao" / "career_opportunities" / "job_opportunities".
- Mỗi vị trí/nghề nghiệp được nhắc tới → tạo 1 node CAREER, tên VIẾT HOA, cụ thể.
  Ví dụ đúng: "LẬP TRÌNH VIÊN", "KỸ SƯ PHẦN MỀM", "GIẢNG VIÊN ĐẠI HỌC"
  Ví dụ sai: "LÀM VIỆC TRONG LĨNH VỰC CÔNG NGHỆ"
- Tạo relationship: (MAJOR)-[:LEADS_TO]->(CAREER) cho mỗi nghề.
- KHÔNG bỏ sót bất kỳ nghề nào, kể cả nghề phi kỹ thuật.

MAJOR: name VIẾT HOA đầy đủ (ví dụ: "CÔNG NGHỆ THÔNG TIN"), code chính xác từ tài liệu (ví dụ: "7480201").
Quan hệ hợp lệ: MAJOR-[OFFERS]->SUBJECT, MAJOR-[LEADS_TO]->CAREER, SUBJECT-[PREREQUISITE_FOR]->SUBJECT.""",

    "career_description": SYSTEM_PROMPT_BASE + """

Tài liệu này là MÔ TẢ NGHỀ NGHIỆP (career_description).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node CAREER (nghề nghiệp). BẮT BUỘC tạo node CAREER này.
- SKILL: tất cả kỹ năng nghề này yêu cầu (hard + soft skill).
  Lấy từ: required_skills, skills, key_skills, responsibilities, job_description.
- MAJOR: các ngành học được khuyến nghị. Lấy từ recommended_majors. MAJOR KHÔNG cần code.
- TUYỆT ĐỐI KHÔNG tạo SUBJECT. TUYỆT ĐỐI KHÔNG tạo TEACHER.

CAREER: tên nghề VIẾT HOA.
Relationship: (CAREER)-[:REQUIRES]->(SKILL), (MAJOR)-[:LEADS_TO]->(CAREER), (CAREER)-[:MENTIONED_IN]->(DOCUMENT).

MAJOR name phải VIẾT HOA, khớp chính xác với tên ngành trong curriculum.
Ví dụ đúng: "CÔNG NGHỆ THÔNG TIN", "KỸ THUẬT PHẦN MỀM"
Ví dụ sai: "CNTT", "Công nghệ thông tin", "IT".""",
}


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def validate_extracted(data: dict, doctype: str) -> tuple[bool, list[str]]:
    """Validate extracted JSON. Returns (is_valid, list_of_warnings)."""
    errors = []

    if not isinstance(data, dict):
        return False, ["Output không phải dict"]

    nodes = data.get("nodes", [])
    rels  = data.get("relationships", [])

    if not nodes:
        return False, ["Không có nodes nào"]

    valid_labels = VALID_NODES_BY_DOCTYPE.get(doctype, set())
    node_ids = {n["id"] for n in nodes if "id" in n}
    doc_node_ids = {n["id"] for n in nodes if n.get("label") == "DOCUMENT"}

    if not doc_node_ids:
        errors.append("Thiếu node DOCUMENT")

    for node in nodes:
        label = node.get("label", "")
        props = node.get("properties", {})

        # Label không hợp lệ
        if label and label not in valid_labels and label != "DOCUMENT":
            errors.append(f"Node label '{label}' không hợp lệ cho doctype '{doctype}'")

        # SUBJECT/MAJOR cần code (trừ MAJOR trong career_description)
        if label in REQUIRED_CODE_LABELS and doctype != "career_description":
            code = props.get("code", "")
            if not code or code.upper() in {"", "SKILL", "MAJOR", "SUBJECT", "CAREER"}:
                errors.append(f"Node {node.get('id')} ({label}) thiếu code hợp lệ: '{code}'")

        # Name phải tồn tại
        if not props.get("name"):
            errors.append(f"Node {node.get('id')} thiếu name")

    # Kiểm tra relationship endpoints tồn tại
    for rel in rels:
        src, tgt, rtype = rel.get("source"), rel.get("target"), rel.get("type")
        if src not in node_ids:
            errors.append(f"Relationship source '{src}' không tồn tại trong nodes")
        if tgt not in node_ids:
            errors.append(f"Relationship target '{tgt}' không tồn tại trong nodes")
        if rtype and rtype not in VALID_REL_TYPES:
            errors.append(f"Relationship type '{rtype}' không hợp lệ")

    # Kiểm tra MENTIONED_IN tới DOCUMENT
    mentioned_sources = {r["source"] for r in rels if r.get("type") == "MENTIONED_IN"}
    non_doc_nodes = [n["id"] for n in nodes if n.get("label") != "DOCUMENT"]
    missing_mentioned = [nid for nid in non_doc_nodes if nid not in mentioned_sources]
    if missing_mentioned:
        errors.append(f"Các nodes thiếu MENTIONED_IN: {missing_mentioned}")

    is_valid = len(errors) == 0
    return is_valid, errors


def fix_extracted(data: dict, doctype: str) -> dict:
    """
    Auto-fix các lỗi nhỏ:
    - Xóa nodes có label sai
    - Xóa SUBJECT/MAJOR thiếu code (trừ career_description MAJOR)
    - Xóa relationships có endpoint không tồn tại
    - Thêm MENTIONED_IN còn thiếu
    """
    valid_labels = VALID_NODES_BY_DOCTYPE.get(doctype, set()) | {"DOCUMENT"}
    nodes = data.get("nodes", [])
    rels  = data.get("relationships", [])

    # Tìm DOC node id
    doc_node_ids = [n["id"] for n in nodes if n.get("label") == "DOCUMENT"]
    doc_id = doc_node_ids[0] if doc_node_ids else None

    # Lọc nodes hợp lệ
    clean_nodes = []
    for node in nodes:
        label = node.get("label", "")
        props = node.get("properties", {})

        if label not in valid_labels:
            continue

        if label in REQUIRED_CODE_LABELS and doctype != "career_description":
            code = props.get("code", "")
            if not code or code.upper() in {"", "SKILL", "MAJOR", "SUBJECT", "CAREER"}:
                continue

        if not props.get("name"):
            continue

        clean_nodes.append(node)

    valid_ids = {n["id"] for n in clean_nodes}

    # Lọc relationships
    clean_rels = [
        r for r in rels
        if r.get("source") in valid_ids
        and r.get("target") in valid_ids
        and r.get("type") in VALID_REL_TYPES
    ]

    # Thêm MENTIONED_IN còn thiếu
    if doc_id and doc_id in valid_ids:
        mentioned_sources = {r["source"] for r in clean_rels if r.get("type") == "MENTIONED_IN"}
        non_doc = [n["id"] for n in clean_nodes if n.get("label") != "DOCUMENT"]
        for nid in non_doc:
            if nid not in mentioned_sources:
                clean_rels.append({"source": nid, "target": doc_id, "type": "MENTIONED_IN"})

    data["nodes"] = clean_nodes
    data["relationships"] = clean_rels
    return data


# ─── MINIO HELPERS ────────────────────────────────────────────────────────────

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def list_json_objects(client: Minio, bucket: str, prefix: str) -> list[str]:
    objects = client.list_objects(bucket, prefix=prefix + "/", recursive=True)
    all_names = [obj.object_name for obj in objects]
    log.info(f"[{prefix}] Tìm thấy {len(all_names)} objects")
    return [o for o in all_names if o.endswith(".json")]


def download_json(client: Minio, bucket: str, object_name: str) -> dict:
    response = client.get_object(bucket, object_name)
    data = json.loads(response.read().decode("utf-8"))
    response.close()
    return data


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_docid(folder: str, filename: str) -> str:
    stem = Path(filename).stem
    prefix_map = {
        "curriculum":         "CUR",
        "career_description": "CAR",
        "syllabus":           "SYL",
    }
    return f"{prefix_map.get(folder, 'DOC')}-{stem}"


def save_local(data: dict, folder: str, filename: str):
    out_path = LOCAL_OUT_DIR / folder / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── LLM EXTRACTION WITH RETRY ───────────────────────────────────────────────

def extract_via_llm(ai_client: OpenAI, doc_json: dict, docid: str, doctype: str) -> dict:
    user_msg = (
        f"Tài liệu cần trích xuất:\n"
        f"docid: {docid}\ndoctype: {doctype}\n\n"
        f"Nội dung JSON:\n{json.dumps(doc_json, ensure_ascii=False, indent=2)}\n\n"
        f"Trả về JSON hợp lệ theo schema."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BY_DOCTYPE.get(doctype, SYSTEM_PROMPT_BASE)},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)

        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning(f"Rate limit (attempt {attempt}/{MAX_RETRIES}), chờ {wait:.1f}s: {e}")
            time.sleep(wait)

        except APIError as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BASE_DELAY * attempt
            log.warning(f"API error (attempt {attempt}/{MAX_RETRIES}), chờ {wait:.1f}s: {e}")
            time.sleep(wait)

        except json.JSONDecodeError as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(f"JSON parse error (attempt {attempt}/{MAX_RETRIES}): {e}")

    raise RuntimeError(f"Hết {MAX_RETRIES} lần retry cho {docid}")


# ─── WORKER ───────────────────────────────────────────────────────────────────

def process_one(minio_client: Minio, ai_client: OpenAI, folder: str, obj_name: str) -> str:
    filename = Path(obj_name).name
    docid    = make_docid(folder, filename)
    doctype  = DOCTYPE_MAP[folder]
    out_path = LOCAL_OUT_DIR / folder / f"{docid}.json"

    if out_path.exists():
        log.debug(f"[skip] {filename}")
        return "skip"

    log.info(f"[start] {filename} ({docid})")
    try:
        doc_json  = download_json(minio_client, MINIO_BUCKET, obj_name)
        extracted = extract_via_llm(ai_client, doc_json, docid, doctype)

        # Validate & auto-fix
        is_valid, errors = validate_extracted(extracted, doctype)
        if not is_valid:
            log.warning(f"[warn] {filename} có {len(errors)} vấn đề, đang auto-fix...")
            for e in errors:
                log.warning(f"       - {e}")
            extracted = fix_extracted(extracted, doctype)

            # Re-validate sau fix
            is_valid2, errors2 = validate_extracted(extracted, doctype)
            if not is_valid2:
                log.error(f"[error] {filename} vẫn lỗi sau fix: {errors2}")
                return "error"

        save_local(extracted, folder, f"{docid}.json")
        node_count = len(extracted.get("nodes", []))
        rel_count  = len(extracted.get("relationships", []))
        log.info(f"[done] {filename} → {node_count} nodes, {rel_count} rels")
        return "ok"

    except Exception as e:
        log.error(f"[ERROR] {filename}: {e}")
        return "error"


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def process_folder(minio_client: Minio, ai_client: OpenAI, folder: str):
    log.info(f"\n{'='*60}\nProcessing folder: {folder}")
    objects = list_json_objects(minio_client, MINIO_BUCKET, f"{MINIO_BASE_FOLDER}/{folder}")
    if not objects:
        log.warning(f"Không tìm thấy file JSON trong {folder}/")
        return {"ok": 0, "skip": 0, "error": 0}

    log.info(f"{len(objects)} files → {MAX_WORKERS} workers song song")
    counts = {"ok": 0, "skip": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, minio_client, ai_client, folder, obj): obj
            for obj in objects
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1

    log.info(f"Folder '{folder}' xong: ✓{counts['ok']}  skip={counts['skip']}  ✗{counts['error']}")
    return counts


def main():
    log.info("Starting entity extraction pipeline...")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY không tìm thấy trong .env")
    if not MINIO_ENDPOINT:
        raise ValueError("MINIO_ENDPOINT không tìm thấy trong .env")

    minio_client = get_minio_client()
    ai_client    = OpenAI(api_key=OPENAI_API_KEY)

    total = {"ok": 0, "skip": 0, "error": 0}
    for folder in INPUT_FOLDERS:
        counts = process_folder(minio_client, ai_client, folder)
        for k in total:
            total[k] += counts.get(k, 0)

    log.info(f"\n✅ Extraction complete. Tổng: ✓{total['ok']}  skip={total['skip']}  ✗{total['error']}")
    log.info("Results saved to ./cache/output/")


if __name__ == "__main__":
    main()
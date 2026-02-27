"""
Script 1 (OPTIMIZED): Extract entities & relationships from MinIO JSON files
- ThreadPoolExecutor cho OpenAI calls (sync SDK)
- Retry với exponential backoff (openai rate limit safe)
- Validate output JSON trước khi lưu
- Resume support (auto-skip đã có)
- Structured logging

Schema v2:
  Nodes: MAJOR, SUBJECT, SKILL, CAREER, TEACHER  (bỏ DOCUMENT)
  Fields:
    MAJOR:   {major_code, major_name_vi, major_name_en}
    SUBJECT: {subject_code, subject_name_vi, subject_name_en}
    SKILL:   {skill_key, skill_name, skill_type}
    CAREER:  {career_key, career_name_vi, career_name_en, field_name}
    TEACHER: {teacher_key, name, email, title}
  Relationships (lowercase snake_case):
    major_offers_subject      {from_major_code, to_subject_code, semester, required_type}
    major_leads_to_career     {from_major_code, to_career_key}
    subject_provides_skill    {from_subject_code, to_skill_key, mastery_level}
    career_requires_skill     {from_career_key, to_skill_key, required_level}
    teacher_instructs_subject {from_teacher_key, to_subject_code}

PHASE 2: Sau khi extract xong TẤT CẢ files, tự động mapping mã ngành
  cho CAREER nodes từ curriculum JSONs đã extract.
"""

import os
import re
import json
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

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")

INPUT_FOLDERS    = ["curriculum", "career_description", "syllabus"]
LOCAL_OUT_DIR    = Path("./cache/output")
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", "10"))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = 2.0  # seconds
# ──────────────────────────────────────────────────────────────────────────────

DOCTYPE_MAP = {
    "curriculum":         "curriculum",
    "career_description": "career_description",
    "syllabus":           "syllabus",
}

# ─── VALID NODE TYPES & REL TYPES PER DOCTYPE ─────────────────────────────────
VALID_NODES_BY_DOCTYPE = {
    "syllabus":           {"SUBJECT", "TEACHER", "SKILL"},
    "curriculum":         {"MAJOR", "SUBJECT", "CAREER", "TEACHER"},
    "career_description": {"CAREER", "SKILL"},
}

VALID_REL_TYPES = {
    "major_offers_subject",
    "major_leads_to_career",
    "subject_provides_skill",
    "career_requires_skill",
    "teacher_instructs_subject",
}

# ─── PROMPTS ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """Bạn là chuyên gia trích xuất dữ liệu Knowledge Graph.
Nhiệm vụ: Trích xuất thông tin sang Nodes và Relationships theo Schema nghiêm ngặt.

1. DANH SÁCH NODES (CHỈ 5 LOẠI NÀY):
- MAJOR:   {major_code, major_name_vi, major_name_en}
- SUBJECT: {subject_code, subject_name_vi, subject_name_en}
- SKILL:   {skill_key, skill_name, skill_type}
- CAREER:  {career_key, career_name_vi, career_name_en, field_name}
- TEACHER: {teacher_key, name, email, title}

2. QUY TẮC TẠO KEY (SLUGIFY):
- CODE (MAJOR, SUBJECT): Dùng mã có sẵn (VD: CNTT1168). Nếu thiếu, tạo: [TEN_MON_KHONG_DAU].
- KEY (SKILL, CAREER, TEACHER): Viết thường, không dấu, thay khoảng trắng bằng "_".
  * TEACHER: Bỏ qua học hàm/học vị (TS, ThS, GS, PGS) khi tạo key.
    Ví dụ: "TS. Nguyễn Văn A" → teacher_key = "nguyen_van_a"
  * SKILL:   Tên kỹ năng ngắn gọn. Ví dụ: "lap_trinh_python", "phan_tich_du_lieu"
  * CAREER:  Tên nghề ngắn gọn.  Ví dụ: "lap_trinh_vien", "ky_su_phan_mem"

3. HỆ THỐNG QUAN HỆ (RELATIONSHIPS):
- major_offers_subject:      {from_major_code, to_subject_code, semester, required_type}
- major_leads_to_career:     {from_major_code, to_career_key}
- subject_provides_skill:    {from_subject_code, to_skill_key, mastery_level}
- career_requires_skill:     {from_career_key, to_skill_key, required_level}
- teacher_instructs_subject: {from_teacher_key, to_subject_code}

4. DEDUP (CHỐNG TRÙNG):
Luôn MERGE dựa trên Code hoặc Key. Không tạo Node mới nếu Key/Code đã tồn tại.

5. OUTPUT FORMAT — CHỈ trả về JSON hợp lệ, không markdown, không giải thích:
{
  "nodes": [
    {"type": "MAJOR",   "major_code": "...", "major_name_vi": "...", "major_name_en": "..."},
    {"type": "SUBJECT", "subject_code": "...", "subject_name_vi": "...", "subject_name_en": "..."},
    {"type": "SKILL",   "skill_key": "...", "skill_name": "...", "skill_type": "hard|soft"},
    {"type": "CAREER",  "career_key": "...", "career_name_vi": "...", "career_name_en": "...", "field_name": "..."},
    {"type": "TEACHER", "teacher_key": "...", "name": "...", "email": "...", "title": "..."}
  ],
  "relationships": [
    {"rel_type": "major_offers_subject",      "from_major_code": "...",   "to_subject_code": "...", "semester": 1, "required_type": "required|elective"},
    {"rel_type": "major_leads_to_career",     "from_major_code": "...",   "to_career_key": "..."},
    {"rel_type": "subject_provides_skill",    "from_subject_code": "...", "to_skill_key": "...", "mastery_level": "basic|intermediate|advanced"},
    {"rel_type": "career_requires_skill",     "from_career_key": "...",   "to_skill_key": "...", "required_level": "basic|intermediate|advanced"},
    {"rel_type": "teacher_instructs_subject", "from_teacher_key": "...",  "to_subject_code": "..."}
  ]
}"""

PROMPT_SYLLABUS = SYSTEM_PROMPT_BASE + """

Tài liệu: SYLLABUS (Đề cương chi tiết môn học).

CÁC BƯỚC TRÍCH XUẤT:
1. SUBJECT: Trích xuất từ "course_code", "course_name_vi" (và "course_name_en" nếu có).
   - Đây là node trung tâm của tài liệu này.

2. TEACHER: Tìm trong "management.instructors" (hoặc field tương đương).
   - Tạo node TEACHER {teacher_key, name, email, title}.
   - Bỏ qua học hàm/học vị khi tạo teacher_key. Ví dụ: "TS. Nguyễn Văn A" → "nguyen_van_a".
   - Tạo relationship: teacher_instructs_subject {from_teacher_key, to_subject_code}.
   - Ghi evidence_ref = "instructors".

3. SKILL: Tìm trong "course_learning_outcomes" / "learning_outcomes" / CLO.
   - Mỗi CLO → tạo 1 node SKILL {skill_key, skill_name, skill_type}.
   - skill_name phải NGẮN GỌN, súc tích (không phải cả câu CLO).
     Ví dụ đúng: "Lập trình Python", "Phân tích dữ liệu"
     Ví dụ sai:  "Sinh viên có khả năng viết được chương trình..."
   - skill_type: "hard" cho kỹ năng kỹ thuật, "soft" cho kỹ năng mềm.
   - Tạo relationship: subject_provides_skill {from_subject_code, to_skill_key, mastery_level}.
   - Lưu mastery_level (basic/intermediate/advanced) nếu có, ghi vào field "note" của quan hệ.
   - Ghi evidence_ref = "course_learning_outcomes".

CHỈ tạo SUBJECT, TEACHER, SKILL. KHÔNG tạo MAJOR, CAREER."""

PROMPT_CURRICULUM = SYSTEM_PROMPT_BASE + """

Tài liệu: CURRICULUM (Chương trình đào tạo).

CÁC BƯỚC TRÍCH XUẤT:
1. MAJOR: Trích xuất từ "major.code" và "major.name_vi" (và "name_en" nếu có).
   - Đây là node trung tâm của tài liệu này.

2. SUBJECT LIST: Duyệt "teaching_plan_and_course_list.courses" (hoặc field tương đương).
   - Mỗi môn học → tạo node SUBJECT {subject_code, subject_name_vi}.
   - Nếu môn không có code → BỎ QUA.
   - Tạo relationship: major_offers_subject {from_major_code, to_subject_code, semester, required_type}.
   - Lưu "semester_no" và "required_type" vào field "note" của quan hệ.
   - "semester" = semester_no (số nguyên). "required_type" = "required" hoặc "elective".

3. CAREER: Tìm trong "career_opportunities" / "job_opportunities" (hoặc field tương đương).
   - Mỗi vị trí/nghề nghiệp → tạo node CAREER {career_key, career_name_vi}.
   - Tên nghề cụ thể, không bỏ sót.
   - Tạo relationship: major_leads_to_career {from_major_code, to_career_key}.

4. TEACHER (Lãnh đạo ngành): Trích xuất tên người ký / Viện trưởng ở cuối file (nếu có).
   - Tạo node TEACHER {teacher_key, name, title}.
   - Bỏ qua học hàm/học vị khi tạo teacher_key.
   - Có thể dùng quan hệ tự định nghĩa để ghi nhận (không bắt buộc rel chuẩn).

CHỈ tạo MAJOR, SUBJECT, CAREER, TEACHER. KHÔNG tạo SKILL."""

PROMPT_CAREER = SYSTEM_PROMPT_BASE + """

Tài liệu: CAREER_DESCRIPTION (Mô tả nghề nghiệp).

CÁC BƯỚC TRÍCH XUẤT:
1. CAREER: Trích xuất từ "name_vi" (và "name_en", "field_name" nếu có).
   - Đây là node trung tâm duy nhất. BẮT BUỘC tạo node CAREER này.

2. SKILL: Duyệt "hard_skills" và "soft_skills" (hoặc "required_skills", "skills").
   - Mỗi kỹ năng → tạo node SKILL {skill_key, skill_name, skill_type}.
   - skill_type = "hard" hoặc "soft" tương ứng với nguồn.
   - Tạo relationship: career_requires_skill {from_career_key, to_skill_key, required_level}.
   - BẮT BUỘC lưu "required_level" (basic/intermediate/advanced) dựa trên nội dung.
   - Ghi evidence_ref = "hard_skills" hoặc "soft_skills".

CHỈ tạo CAREER, SKILL. KHÔNG tạo MAJOR, SUBJECT, TEACHER."""

PROMPTS_BY_DOCTYPE = {
    "syllabus":           PROMPT_SYLLABUS,
    "curriculum":         PROMPT_CURRICULUM,
    "career_description": PROMPT_CAREER,
}


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def _get_node_key(node: dict) -> str | None:
    """Lấy key/code định danh của node."""
    t = node.get("type", "")
    if t == "MAJOR":   return node.get("major_code")
    if t == "SUBJECT": return node.get("subject_code")
    if t == "SKILL":   return node.get("skill_key")
    if t == "CAREER":  return node.get("career_key")
    if t == "TEACHER": return node.get("teacher_key")
    return None


def _get_node_name(node: dict) -> str | None:
    """Lấy tên chính của node."""
    t = node.get("type", "")
    if t == "MAJOR":   return node.get("major_name_vi")
    if t == "SUBJECT": return node.get("subject_name_vi")
    if t == "SKILL":   return node.get("skill_name")
    if t == "CAREER":  return node.get("career_name_vi")
    if t == "TEACHER": return node.get("name")
    return None


def validate_extracted(data: dict, doctype: str) -> tuple[bool, list[str]]:
    """Validate extracted JSON. Returns (is_valid, list_of_errors)."""
    errors = []

    if not isinstance(data, dict):
        return False, ["Output không phải dict"]

    nodes = data.get("nodes", [])
    rels  = data.get("relationships", [])

    if not nodes:
        return False, ["Không có nodes nào"]

    valid_labels = VALID_NODES_BY_DOCTYPE.get(doctype, set())

    # Build key sets cho rel validation
    all_major_codes   = {n["major_code"]   for n in nodes if n.get("type") == "MAJOR"   and n.get("major_code")}
    all_subject_codes = {n["subject_code"] for n in nodes if n.get("type") == "SUBJECT" and n.get("subject_code")}
    all_skill_keys    = {n["skill_key"]    for n in nodes if n.get("type") == "SKILL"   and n.get("skill_key")}
    all_career_keys   = {n["career_key"]   for n in nodes if n.get("type") == "CAREER"  and n.get("career_key")}
    all_teacher_keys  = {n["teacher_key"]  for n in nodes if n.get("type") == "TEACHER" and n.get("teacher_key")}

    for node in nodes:
        ntype = node.get("type", "")
        if ntype not in valid_labels:
            errors.append(f"Node type '{ntype}' không hợp lệ cho doctype '{doctype}'")
            continue
        if not _get_node_key(node):
            errors.append(f"Node {ntype} thiếu key/code: {node}")
        if not _get_node_name(node):
            errors.append(f"Node {ntype} (key={_get_node_key(node)}) thiếu name")

    for rel in rels:
        rtype = rel.get("rel_type", "")
        if rtype not in VALID_REL_TYPES:
            errors.append(f"rel_type '{rtype}' không hợp lệ")
            continue

        if rtype == "major_offers_subject":
            if rel.get("from_major_code") not in all_major_codes:
                errors.append(f"major_offers_subject: from_major_code '{rel.get('from_major_code')}' không tồn tại")
            if rel.get("to_subject_code") not in all_subject_codes:
                errors.append(f"major_offers_subject: to_subject_code '{rel.get('to_subject_code')}' không tồn tại")
        elif rtype == "major_leads_to_career":
            if rel.get("from_major_code") not in all_major_codes:
                errors.append(f"major_leads_to_career: from_major_code '{rel.get('from_major_code')}' không tồn tại")
            if rel.get("to_career_key") not in all_career_keys:
                errors.append(f"major_leads_to_career: to_career_key '{rel.get('to_career_key')}' không tồn tại")
        elif rtype == "subject_provides_skill":
            if rel.get("from_subject_code") not in all_subject_codes:
                errors.append(f"subject_provides_skill: from_subject_code '{rel.get('from_subject_code')}' không tồn tại")
            if rel.get("to_skill_key") not in all_skill_keys:
                errors.append(f"subject_provides_skill: to_skill_key '{rel.get('to_skill_key')}' không tồn tại")
        elif rtype == "career_requires_skill":
            if rel.get("from_career_key") not in all_career_keys:
                errors.append(f"career_requires_skill: from_career_key '{rel.get('from_career_key')}' không tồn tại")
            if rel.get("to_skill_key") not in all_skill_keys:
                errors.append(f"career_requires_skill: to_skill_key '{rel.get('to_skill_key')}' không tồn tại")
        elif rtype == "teacher_instructs_subject":
            if rel.get("from_teacher_key") not in all_teacher_keys:
                errors.append(f"teacher_instructs_subject: from_teacher_key '{rel.get('from_teacher_key')}' không tồn tại")
            if rel.get("to_subject_code") not in all_subject_codes:
                errors.append(f"teacher_instructs_subject: to_subject_code '{rel.get('to_subject_code')}' không tồn tại")

    return len(errors) == 0, errors


def fix_extracted(data: dict, doctype: str) -> dict:
    """
    Auto-fix các lỗi nhỏ:
    - Xóa nodes có type sai hoặc thiếu key/name
    - Xóa relationships có endpoint không tồn tại hoặc rel_type sai
    """
    valid_labels = VALID_NODES_BY_DOCTYPE.get(doctype, set())
    nodes = data.get("nodes", [])
    rels  = data.get("relationships", [])

    clean_nodes = [
        n for n in nodes
        if n.get("type") in valid_labels
        and _get_node_key(n)
        and _get_node_name(n)
    ]

    all_major_codes   = {n["major_code"]   for n in clean_nodes if n.get("type") == "MAJOR"   and n.get("major_code")}
    all_subject_codes = {n["subject_code"] for n in clean_nodes if n.get("type") == "SUBJECT" and n.get("subject_code")}
    all_skill_keys    = {n["skill_key"]    for n in clean_nodes if n.get("type") == "SKILL"   and n.get("skill_key")}
    all_career_keys   = {n["career_key"]   for n in clean_nodes if n.get("type") == "CAREER"  and n.get("career_key")}
    all_teacher_keys  = {n["teacher_key"]  for n in clean_nodes if n.get("type") == "TEACHER" and n.get("teacher_key")}

    clean_rels = []
    for rel in rels:
        rtype = rel.get("rel_type", "")
        if rtype not in VALID_REL_TYPES:
            continue
        ok = True
        if rtype == "major_offers_subject":
            ok = rel.get("from_major_code") in all_major_codes and rel.get("to_subject_code") in all_subject_codes
        elif rtype == "major_leads_to_career":
            ok = rel.get("from_major_code") in all_major_codes and rel.get("to_career_key") in all_career_keys
        elif rtype == "subject_provides_skill":
            ok = rel.get("from_subject_code") in all_subject_codes and rel.get("to_skill_key") in all_skill_keys
        elif rtype == "career_requires_skill":
            ok = rel.get("from_career_key") in all_career_keys and rel.get("to_skill_key") in all_skill_keys
        elif rtype == "teacher_instructs_subject":
            ok = rel.get("from_teacher_key") in all_teacher_keys and rel.get("to_subject_code") in all_subject_codes
        if ok:
            clean_rels.append(rel)

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


def _normalize_name(name: str) -> str:
    """Chuẩn hóa tên ngành để so sánh: uppercase + strip whitespace."""
    return re.sub(r"\s+", " ", name.strip().upper())


# ─── LLM EXTRACTION WITH RETRY ───────────────────────────────────────────────

def extract_via_llm(ai_client: OpenAI, doc_json: dict, docid: str, doctype: str) -> dict:
    user_msg = (
        f"Tài liệu cần trích xuất:\n"
        f"docid: {docid}\ndoctype: {doctype}\n\n"
        f"Nội dung JSON:\n{json.dumps(doc_json, ensure_ascii=False, indent=2)}\n\n"
        f"Trả về JSON hợp lệ theo schema."
    )

    system_prompt = PROMPTS_BY_DOCTYPE.get(doctype, SYSTEM_PROMPT_BASE)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
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


# ─── PHASE 2: MAJOR CODE MAPPING FOR CAREER NODES ────────────────────────────

def build_major_code_index() -> dict[str, str]:
    """
    Đọc toàn bộ curriculum JSONs đã extract.
    Trả về dict: normalized_major_name → major_code

    Ví dụ:
      {"CÔNG NGHỆ THÔNG TIN": "7480201", "KỸ THUẬT PHẦN MỀM": "7480103"}
    """
    index: dict[str, str] = {}
    cur_dir = LOCAL_OUT_DIR / "curriculum"

    if not cur_dir.exists():
        log.warning("[Phase 2] Thư mục curriculum không tồn tại, bỏ qua mapping.")
        return index

    files = list(cur_dir.glob("*.json"))
    log.info(f"[Phase 2] Đọc {len(files)} curriculum files để build major index...")

    for jf in files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[Phase 2] Lỗi đọc {jf.name}: {e}")
            continue

        for node in data.get("nodes", []):
            if node.get("type") != "MAJOR":
                continue
            name = node.get("major_name_vi", "").strip()
            code = node.get("major_code", "").strip()
            if not name or not code:
                continue
            norm = _normalize_name(name)
            if norm in index and index[norm] != code:
                log.warning(
                    f"[Phase 2] Tên ngành '{norm}' có nhiều code: "
                    f"'{index[norm]}' vs '{code}' — giữ code đầu tiên."
                )
            else:
                index[norm] = code

    log.info(f"[Phase 2] Major index: {len(index)} ngành")
    for name, code in sorted(index.items()):
        log.info(f"  {code}  {name}")
    return index


def _partial_match(norm_name: str, major_index: dict[str, str]) -> str | None:
    """
    Partial match: norm_name là substring của key hoặc ngược lại.
    Chỉ nhận khi unambiguous (duy nhất 1 kết quả).
    """
    candidates = []
    for key, code in major_index.items():
        if norm_name in key or key in norm_name:
            candidates.append(code)
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def map_major_codes_for_career(data: dict, major_index: dict[str, str]) -> tuple[dict, list[str], list[str]]:
    """
    Với 1 career_description JSON (schema v2 không có MAJOR nodes):
    - Nếu có field "major_names" trong CAREER node → map sang codes
    - Gắn major_codes vào CAREER node

    Trả về: (updated_data, mapped_codes, unmatched_names)
    """
    nodes = data.get("nodes", [])

    career_node = next((n for n in nodes if n.get("type") == "CAREER"), None)
    if not career_node:
        return data, [], []

    # Schema v2: career_description chỉ extract CAREER + SKILL
    # major_names có thể được LLM ghi vào field phụ nếu có trong tài liệu
    major_names = career_node.get("major_names", [])
    if not major_names:
        career_node["major_codes"] = []
        return data, [], []

    mapped_codes: list[str] = []
    unmatched:    list[str] = []

    for name in major_names:
        norm = _normalize_name(name)
        code = major_index.get(norm)
        if code:
            if code not in mapped_codes:
                mapped_codes.append(code)
        else:
            partial = _partial_match(norm, major_index)
            if partial and partial not in mapped_codes:
                mapped_codes.append(partial)
            else:
                unmatched.append(name)

    career_node["major_codes"] = mapped_codes
    return data, mapped_codes, unmatched


def run_phase2_mapping():
    """Phase 2: Map major_codes cho CAREER nodes trong career_description JSONs."""
    log.info(f"\n{'='*60}")
    log.info("PHASE 2: Mapping major_codes cho CAREER nodes")
    log.info(f"{'='*60}")

    major_index = build_major_code_index()
    if not major_index:
        log.error("[Phase 2] Major index rỗng — không thể mapping. Kiểm tra lại curriculum files.")
        return

    career_dir = LOCAL_OUT_DIR / "career_description"
    if not career_dir.exists():
        log.warning("[Phase 2] Thư mục career_description không tồn tại.")
        return

    files = list(career_dir.glob("*.json"))
    log.info(f"[Phase 2] Xử lý {len(files)} career_description files...")

    total_mapped    = 0
    total_unmatched = 0
    files_updated   = 0

    for jf in files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"[Phase 2] Lỗi đọc {jf.name}: {e}")
            continue

        career_node = next((n for n in data.get("nodes", []) if n.get("type") == "CAREER"), None)
        career_name = career_node.get("career_name_vi", "?") if career_node else "?"

        updated_data, mapped_codes, unmatched = map_major_codes_for_career(data, major_index)
        total_mapped    += len(mapped_codes)
        total_unmatched += len(unmatched)

        if mapped_codes:
            log.info(f"  ✓ {jf.name} | CAREER: {career_name} | major_codes: {mapped_codes}")
        else:
            log.warning(f"  ⚠ {jf.name} | CAREER: {career_name} | Không map được major_codes")

        if unmatched:
            log.warning(
                f"    Không tìm thấy code cho: {unmatched}\n"
                f"    → Kiểm tra xem tên ngành có khớp với curriculum không."
            )

        with open(jf, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        files_updated += 1

    log.info(
        f"\n[Phase 2] Hoàn tất: {files_updated} files cập nhật, "
        f"{total_mapped} codes mapped, {total_unmatched} tên ngành không match."
    )
    if total_unmatched > 0:
        log.warning(
            "[Phase 2] Có tên ngành không match. Nguyên nhân thường gặp: "
            "LLM viết tên ngành khác với tên trong curriculum."
        )


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def process_folder(minio_client: Minio, ai_client: OpenAI, folder: str) -> dict:
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

    # Phase 1: curriculum trước để Phase 2 có dữ liệu
    ordered_folders = ["curriculum", "syllabus", "career_description"]
    total = {"ok": 0, "skip": 0, "error": 0}
    for folder in ordered_folders:
        if folder not in INPUT_FOLDERS:
            continue
        counts = process_folder(minio_client, ai_client, folder)
        for k in total:
            total[k] += counts.get(k, 0)

    log.info(
        f"\n✅ Phase 1 complete. "
        f"Tổng: ✓{total['ok']}  skip={total['skip']}  ✗{total['error']}"
    )

    # Phase 2: Map major_codes cho CAREER nodes
    run_phase2_mapping()

    log.info("\n✅ Pipeline hoàn tất. Results saved to ./cache/output/")


if __name__ == "__main__":
    main()
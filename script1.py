"""
Script 1 (OPTIMIZED): Extract entities & relationships from MinIO JSON files
- ThreadPoolExecutor cho OpenAI calls (sync SDK)
- Retry với exponential backoff (openai rate limit safe)
- Validate output JSON trước khi lưu
- Resume support (auto-skip đã có)
- Structured logging

Schema v2:
  Nodes: MAJOR, SUBJECT, SKILL, CAREER, TEACHER 
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
    "curriculum":         {"MAJOR", "SUBJECT", "CAREER"},
    "career_description": {"CAREER", "SKILL", "MAJOR"},
}

VALID_REL_TYPES = {
    "major_offers_subject",
    "major_leads_to_career",
    "subject_provides_skill",
    "career_requires_skill",
    "teacher_instructs_subject",
    "subject_is_prerequisite_of_subject"
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
- CODE (MAJOR, SUBJECT): Dùng mã có sẵn (VD: CNTT1168). Nếu không tìm thấy CODE, không tạo node.
- KEY (SKILL, CAREER, TEACHER): Viết thường, không dấu, thay khoảng trắng bằng "_".
  * TEACHER: Bỏ qua học hàm/học vị (TS., ThS., GS., PGS.) khi tạo key.
    Ví dụ: "TS. Nguyễn Văn A" → teacher_key = "nguyen_van_a"
  * SKILL:   Tên kỹ năng ngắn gọn. Ví dụ: "lap_trinh_python", "phan_tich_du_lieu"
  * CAREER:  Tên nghề ngắn gọn.  Ví dụ: "lap_trinh_vien", "ky_su_phan_mem"

3. HỆ THỐNG QUAN HỆ (RELATIONSHIPS):
- major_offers_subject:      {from_major_code, to_subject_code, semester, required_type}
- major_leads_to_career:     {from_major_code, to_career_key}
- subject_provides_skill:    {from_subject_code, to_skill_key, mastery_level}
- career_requires_skill:     {from_career_key, to_skill_key, required_level}
- teacher_instructs_subject: {from_teacher_key, to_subject_code}
- subject_is_prerequisite_of_subject: {from_subject_code, to_subject_code}

4. DEDUP (CHỐNG TRÙNG):
Luôn MERGE dựa trên Code hoặc Key. Không tạo Node mới nếu Key/Code đã tồn tại.
KẾT NỐI QUAN HỆ VÀ GỘP NHẰM CHỐNG TRÙNG:
Các SKILL trích xuất ra từ đề cương sẽ được gộp với SKILL trích xuất ra từ mô tả nghề nghiệp thông qua tên skill hoặc skill key.
Các SUBJECT trích xuất ra từ chương trình đào tạo sẽ được gộp với SUBJECT đại diện cho đề cương thông qua subject_code
Các CAREER trích xuất ra từ chương trình đào tạo sẽ được gộp với CAREER đại diện cho mô tả nghề nghiệp thông qua tên career hoặc career key
Các MAJOR trích xuất ra từ mô tả nghề nghiệp sẽ được gộp với MAJOR đại diện cho chương trình đào tạo thông qua việc nhận diện tên major và nối major_code tương ứng.

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
    {"rel_type": "teacher_instructs_subject", "from_teacher_key": "...",  "to_subject_code": "..."},
    {"rel_type": "subject_is_prerequisite_of_subject", "from_subject_code": "...", "to_subject_code": "..."}
  ]
}"""

PROMPT_SYLLABUS = SYSTEM_PROMPT_BASE + """

Tài liệu: SYLLABUS (Đề cương chi tiết môn học).

CÁC BƯỚC TRÍCH XUẤT:
1. SUBJECT: Trích xuất từ "course_code", "course_name_vi" (và "course_name_en" nếu có).
   - Đây là node trung tâm của tài liệu này.
   Cần tìm các môn là tiên quyết (prerequisite) để tạo relationship subject_is_prerequisite_of_subject. Nếu không có môn tiên quyết -> không tạo relationship này.
   - QUAN TRỌNG: "from_subject_code" và "to_subject_code" trong relationship này BẮT BUỘC phải là MÃ MÔN HỌC (VD: KTTC1121), KHÔNG được dùng tên môn (VD: "Kế toán tài chính 2"). Nếu môn tiên quyết không có mã → KHÔNG tạo relationship này.

2. TEACHER: Tìm trong "management.instructors" (hoặc field tương đương).
   - Tạo node TEACHER {teacher_key, name, email, title}.
   - Bỏ qua học hàm/học vị khi tạo teacher_key. Ví dụ: "TS. Nguyễn Văn A" → "nguyen_van_a".
   - Nếu có email, ghi vào field "email". 
   - Tên giáo viên nên ghi đầy đủ, có dấu, không viết tắt. Ví dụ: "Nguyễn Văn A" chứ không phải "Nguyen Van A".
   - Học hàm, học vị (TS., ThS., GS., PGS.) ghi vào field "title" nếu có, nhưng KHÔNG ghi vào "name" hoặc "teacher_key".
   - Tạo relationship: teacher_instructs_subject {from_teacher_key, to_subject_code}.
   - Ghi evidence_ref = "instructors".
   - QUAN TRỌNG: Mỗi teacher_key trong relationship teacher_instructs_subject BẮT BUỘC phải có node TEACHER tương ứng trong "nodes". Tạo node TEACHER trước, rồi mới tạo relationship.

3. SKILL: Tìm trong "course_learning_outcomes" / "learning_outcomes" / CLO.
   - Mỗi CLO → tạo 1 node SKILL {skill_key, skill_name, skill_type}.
   - skill_name phải NGẮN GỌN, súc tích (không phải cả câu CLO).
     Ví dụ đúng: "Lập trình Python", "Phân tích dữ liệu"
     Ví dụ sai:  "Sinh viên có khả năng viết được chương trình..."
   - skill_type: "hard" cho kỹ năng kỹ thuật, dùng các công cụ; "soft" cho kỹ năng mềm, chẳng hạn như kỹ nằng làm việc nhóm, giao tiếp, quản lý thời gian...
   - Tạo relationship: subject_provides_skill {from_subject_code, to_skill_key, mastery_level}.
   - Lưu mastery_level (basic/intermediate/advanced) nếu có, ghi vào field "note" của quan hệ.
   - Ghi evidence_ref = "course_learning_outcomes".

CHỈ tạo SUBJECT, TEACHER, SKILL. KHÔNG tạo MAJOR, CAREER.

Đối với SUBJECT, hãy chú ý trích xuất bổ sung thêm nội dung văn bản những mục sau trong đề cương và đưa vào thuộc tính của node SUBJECT này:
Mô tả học phần (course_description)
Tài liệu học tập (learning_resources)
Mục tiêu học phần (courses_goals)
Đánh giá học phần (assessment)
Quy định của học phần (course_requirements_and_expectations)
Thời điểm điều chỉnh đề cương (syllabus_adjustment_time)
ĐẶC BIỆT QUAN TRỌNG cho Đề cương: Trích xuất phần Chuẩn đầu ra học phần (course_learning_outcomes) thành các node riêng có loai là SKILL, gắn với SUBJECT từ đề cương đang trích xuất hiện tại. Phần Chuẩn đầu ra học phần này có các CLO (clo_code), mỗi CLO sẽ tương ứng với 1 node SKILL, mỗi node SKILL này sẽ được nối với SUBJECT từ đề cương mà nó được trích xuất ra. 
Thêm vào đó, trích xuất phần kế hoạch dạy học (lesson_plan) và đưa vào làm thuộc tính của SUBJECT. Mỗi một tuần học sẽ tương ứng 1 thẻ thuộc tính, lấy từ mục Tuần (week_no) (ví dụ SUBJECT có các thuộc tính gồm week_1, week_2,...,), và nội dung của các, thuộc tính đó sẽ là văn bản gồm nội dung học (contents), tài liệu đọc (reading_materials), hoạt động dạy và học (teaching_learning_activities), đánh giá (assessment_activities) và CLO của tuần đó (clos)."""

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
   - Nếu một subject là bắt buộc trong một major thì subject đó sẽ được ưu tiên xuất hiện khi hỏi về ngành đó. Nếu một subject là tự chọn thì subject đó sẽ được xuất hiện sau khi đã xuất hết các subject bắt buộc khi hỏi về ngành đó.
   - "semester" = semester_no (số nguyên). "required_type" = "required"(bắt buộc) hoặc "elective"(tự chọn).

3. CAREER: Tìm trong "career_opportunities" / "job_opportunities" (hoặc field tương đương).
   - Mỗi vị trí/nghề nghiệp → tạo node CAREER {career_key, career_name_vi}.
   - Tên nghề cụ thể, không bỏ sót.
   - Tạo relationship: major_leads_to_career {from_major_code, to_career_key}.

CHỈ tạo MAJOR, SUBJECT, CAREER. KHÔNG tạo SKILL, TEACHER.
ĐỐI VỚI MAJOR, hãy chú ý trích xuất bổ sung thêm nội dung văn bản những mục sau trong chương trình đào tạo và đưa vào thuộc tính của node MAJOR này:
Triết lý, mục tiêu đào tạo và định hướng nơi làm việc sau tốt nghiệp (philosophy_and_objectives), trong đó có các mục tiêu cụ thể (specific_objectives) gồm nhiều PO code.
Chuẩn đầu vào (admission_requirements)
Chuẩn đầu ra (learning_outcomes), gồm các PLO (plo_groups)
Ma trận đáp ứng mục tiêu đào tạo và chuẩn đầu ra (po_plo_matrix)
Quy trình đào tạo, điều kiện tốt nghiệp (training_process_and_graduation_conditions)
Cấu trúc và nội dung của chương trình đào tạo (curriculum_structure_and_content)
Phương pháp giảng dạy và đánh giá kết quả học tập (teaching_and_assessment_methods)
Các chương trình đào tạo tham khảo (reference_programs)
Tiêu chuẩn đội ngũ giảng viên, trợ giảng (lecturer_and_teaching_assistant_standards)
Cơ sở vật chất, công nghệ và học liệu (facilities_and_learning_resources)
ĐẶC BIỆT QUAN TRỌNG cho Chương trình đào tạo: Trích xuất phần Nội dung và kế hoạch giảng dạy (teaching_plan_and_course_list) thành các node có loại là SUBJECT, gắn với MAJOR từ chương trình đào tạo đang trích xuất hiện tại. Mỗi một môn học (course) tương ứng với 1 node SUBJECT - gồm tên môn, mã môn (code), số tín chỉ, và được nối với MAJOR từ chương trình đào tạo mà nó được trích xuất ra.
Thêm vào đó, trích xuất phần Cơ hội làm việc và khả năng học tập nâng cao (career_and_further_study_opportunities) thành các node riêng có loại là CAREER, gắn với MAJOR từ chương trình đào tạo đang trích xuất hiện tại. Các node CAREER này được nối với node MAJOR từ chương trình đào tạo mà nó được trích xuất ra.
"""

PROMPT_CAREER = SYSTEM_PROMPT_BASE + """

Tài liệu: CAREER_DESCRIPTION (Mô tả nghề nghiệp).

CÁC BƯỚC TRÍCH XUẤT:
1. CAREER: Trích xuất từ "name_vi" (và "name_en", "field_name" nếu có).
   - Đây là node trung tâm duy nhất. BẮT BUỘC tạo node CAREER này.

2. SKILL: Duyệt "hard_skills" và "soft_skills" (hoặc "required_skills", "skills").
   - Mỗi kỹ năng → tạo node SKILL {skill_key, skill_name, skill_type}. 
   - skill_type = "hard" hoặc "soft" tương ứng với nguồn.
   - Tạo relationship: career_requires_skill {from_career_key, to_skill_key, required_level}.
   - BẮT BUỘC lưu "required_level" (basic/intermediate/advanced) dựa trên nội dung. Các level được liệt kê gồm cơ bản, trung cấp và thành thạo. Nếu không tìm thấy level nào phù hợp, hãy để trống trường này.
   - Ghi evidence_ref = "hard_skills" hoặc "soft_skills".

CHỈ tạo CAREER, SKILL. KHÔNG tạo MAJOR, SUBJECT, TEACHER.
Đối với CAREER, hãy chú ý trích xuất bổ sung thêm nội dung văn bản những mục sau trong mô tả nghề nghiệp và đưa vào thuộc tính của node:
Nhóm nghề / lĩnh vực (field_name)
Mô tả nghề nghiệp (description), gồm Mô tả ngắn (short_description) và Vai trò trong tổ chức/doanh nghiệp (role_in_organization)
Công việc chính (job_tasks)
Yêu cầu học vấn và chứng chỉ (education_certification)
Cơ hội việc làm và thị trường (market)
ĐẶC BIỆT QUAN TRỌNG cho Mô tả nghề nghiệp: Trích xuất phần Kỹ năng yêu cầu (skills) thành các node có loại là SKILL, gắn với CAREER. Các SKILL này gồm kỹ năng cứng (hard_skills) và kỹ năng mềm (soft_skills)
Thêm vào đó, trích xuất phần Ngành học phù hợp (recommended_majors) bên trong phần Yêu cầu học vấn và chứng chỉ thành các node riêng có loại là MAJOR - gồm tên ngành, gắn với CAREER từ mô tả nghề nghiệp đang trích xuất hiện tại. Yêu cầu chỉ lấy ra tên ngành ngắn gọn.
"""

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
        elif rtype == "subject_is_prerequisite_of_subject":
            if rel.get("from_subject_code") not in all_subject_codes:
                errors.append(f"subject_is_prerequisite_of_subject: from_subject_code '{rel.get('from_subject_code')}' không tồn tại")
            if rel.get("to_subject_code") not in all_subject_codes:
                errors.append(f"subject_is_prerequisite_of_subject: to_subject_code '{rel.get('to_subject_code')}' không tồn tại")

    return len(errors) == 0, errors


def _slugify(text: str) -> str:
    """Chuyển tên tiếng Việt → snake_case không dấu (dùng cho teacher_key)."""
    import unicodedata
    text = text.strip()
    # Bỏ học hàm/học vị ở đầu
    for prefix in ("GS.TS.", "PGS.TS.", "GS.", "PGS.", "TS.", "ThS.", "CN.", "Ths."):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Normalize unicode → ASCII
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def fix_extracted(data: dict, doctype: str) -> dict:
    """
    Auto-fix các lỗi phổ biến:
    1. Xóa nodes có type sai hoặc thiếu key/name
    2. teacher_instructs_subject: nếu thiếu node TEACHER → tự tạo stub node từ teacher_key
    3. subject_is_prerequisite_of_subject: nếu to_subject_code là tên môn (không phải mã) → bỏ rel đó
    4. Xóa relationships có endpoint không tồn tại sau khi đã recover
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

    # ── Fix 1: Teacher stub recovery ─────────────────────────────────────────
    # Nếu rel teacher_instructs_subject có from_teacher_key nhưng không có node TEACHER
    # → tự tạo node TEACHER stub với tên suy ra từ key
    existing_teacher_keys = {n["teacher_key"] for n in clean_nodes if n.get("type") == "TEACHER" and n.get("teacher_key")}
    stubs_added: set[str] = set()
    for rel in rels:
        if rel.get("rel_type") != "teacher_instructs_subject":
            continue
        tkey = rel.get("from_teacher_key", "")
        if not tkey or tkey in existing_teacher_keys or tkey in stubs_added:
            continue
        # Suy ra tên từ key: "nguyen_van_a" → "Nguyen Van A" (giữ dạng ASCII, không có dấu)
        stub_name = " ".join(w.capitalize() for w in tkey.split("_"))
        clean_nodes.append({
            "type":        "TEACHER",
            "teacher_key": tkey,
            "name":        stub_name,
            "email":       "",
            "title":       "",
        })
        stubs_added.add(tkey)
        log.warning(f"  [fix] Tạo stub TEACHER '{tkey}' (name='{stub_name}') cho relationship bị thiếu node")

    # Rebuild key sets sau khi đã recover
    all_major_codes   = {n["major_code"]   for n in clean_nodes if n.get("type") == "MAJOR"   and n.get("major_code")}
    all_subject_codes = {n["subject_code"] for n in clean_nodes if n.get("type") == "SUBJECT" and n.get("subject_code")}
    all_skill_keys    = {n["skill_key"]    for n in clean_nodes if n.get("type") == "SKILL"   and n.get("skill_key")}
    all_career_keys   = {n["career_key"]   for n in clean_nodes if n.get("type") == "CAREER"  and n.get("career_key")}
    all_teacher_keys  = {n["teacher_key"]  for n in clean_nodes if n.get("type") == "TEACHER" and n.get("teacher_key")}

    # ── Fix 2: prerequisite rel dùng tên môn thay vì mã môn ──────────────────
    # Nhận diện: to_subject_code không khớp bất kỳ subject_code nào và chứa khoảng trắng
    # → đây là tên môn → bỏ rel này (không thể ánh xạ an toàn)
    clean_rels = []
    skipped_prereq = 0
    for rel in rels:
        rtype = rel.get("rel_type", "")
        if rtype not in VALID_REL_TYPES:
            continue

        if rtype == "subject_is_prerequisite_of_subject":
            from_code = rel.get("from_subject_code", "")
            to_code   = rel.get("to_subject_code", "")
            # Nếu to_subject_code trông như tên môn (có khoảng trắng, không phải mã)
            if " " in str(to_code) and to_code not in all_subject_codes:
                skipped_prereq += 1
                log.warning(f"  [fix] Bỏ prerequisite rel: to_subject_code \'{to_code}\' là tên môn, không phải mã môn")
                continue
            if from_code not in all_subject_codes or to_code not in all_subject_codes:
                continue
            clean_rels.append(rel)
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

    if skipped_prereq:
        log.warning(f"  [fix] Đã bỏ {skipped_prereq} prerequisite rel dùng tên môn thay vì mã môn")

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


# ─── CAREER → MAJOR MAPPING TABLE ────────────────────────────────────────────
# Key: career_name_vi (chính xác như trong CAREER node)
# Value: danh sách major_code tương ứng
CAREER_MAJOR_MAP = {
    "Automation tester":                ["7480201", "7340405", "7480101"],
    "Business analyst":                 ["7480201", "7340405", "7480101"],
    "Chuyên viên dữ liệu":              ["7480201", "7310108", "7340405", "7480101"],
    "Customer Success":                 ["7340115"],
    "Data Analyst":                     ["7480201", "7340405", "7480101", "7310108"],
    "Data Engineer":                    ["7480201", "7480101", "7310108"],
    "Kế toán quản trị":                 ["7340201"],
    "Key Account Manager":              ["7340115"],
    "Kỹ sư cầu nối":                    ["7480201", "7480103"],
    "Kỹ sư phần mềm":                   ["7480101", "7480201", "7480103", "7480202"],
    "Lập trình game":                   ["7480101", "7480201", "7480103"],
    "Lập trình nhúng":                  ["7480101"],
    "Marketing Offline":                ["7340115"],
    "Media Planner":                    ["7340115"],
    "Nhân viên Bồi thường bảo hiểm":    ["7340204"],
    "Nhân viên kinh doanh tiếng Trung": ["7340120", "7340121"],
    "Nhân viên kinh doanh":             ["7340121", "7310101"],
    "Nhân viên triển khai phần mềm":    ["7480201", "7480101", "7480103"],
    "Quản lý kinh doanh":               [],   # QUẢN TRỊ KINH DOANH chưa có trong index
    "Sales Representative":             ["7340115"],
    "System Administrator":             ["7480201", "7480103", "7480101"],
    "Tester":                           ["7480201", "7480103", "7480101"],
}


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
    - Dùng CAREER_MAJOR_MAP (bảng mapping thủ công) để gắn major_codes vào CAREER node
    - Fallback sang major_index nếu career_name không có trong bảng thủ công

    Trả về: (updated_data, mapped_codes, unmatched_names)
    """
    nodes = data.get("nodes", [])

    career_node = next((n for n in nodes if n.get("type") == "CAREER"), None)
    if not career_node:
        return data, [], []

    career_name = career_node.get("career_name_vi", "")

    # Ưu tiên bảng mapping thủ công
    if career_name in CAREER_MAJOR_MAP:
        career_node["major_codes"] = CAREER_MAJOR_MAP[career_name]
        return data, CAREER_MAJOR_MAP[career_name], []

    # Fallback: dùng major_names field + major_index nếu có
    major_names = career_node.get("major_names", [])
    if not major_names:
        career_node["major_codes"] = []
        return data, [], [career_name] if career_name else []

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
            log.warning(f"  ⚠ {jf.name} | CAREER: {career_name} | Không map được major_codes — thêm vào CAREER_MAJOR_MAP nếu cần")

        if unmatched:
            log.warning(
                f"    Không tìm thấy code cho: {unmatched}\n"
                f"    → Thêm career_name_vi này vào CAREER_MAJOR_MAP để fix."
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
            "[Phase 2] Có career chưa được mapping. "
            "Hãy thêm career_name_vi tương ứng vào bảng CAREER_MAJOR_MAP trong script."
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

    # Phase 1: syllabus trước, rồi curriculum, cuối cùng career_description (để Phase 2 có dữ liệu)
    ordered_folders = ["syllabus", "curriculum", "career_description"]
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
"""
extract_hybrid.py — Hybrid Entity Extraction Pipeline (Token-Optimized)

Chiến lược theo từng doctype:
  CAREER      → 100% rule-based (parse paragraphs có cấu trúc text rõ ràng)
  SYLLABUS    → ~85% rule-based (teachers, CLOs, lesson_plan từ bảng structured)
               + LLM mini-call chỉ để rút gọn CLO text → skill_name ngắn gọn
  CURRICULUM  → LLM nhưng CHỈ gửi phần cần thiết (career_opps + PLO text),
               còn course list parse hoàn toàn bằng rule-based

Tiết kiệm ước tính so với gửi full JSON vào LLM:
  CAREER:      ~100% (không gọi LLM)
  SYLLABUS:    ~85%  (chỉ gửi list CLO text, ~200 tokens thay vì ~8000)
  CURRICULUM:  ~60%  (chỉ gửi phần text tự do, bỏ toàn bộ course list)

Schema v2 (giữ nguyên):
  Nodes: MAJOR, SUBJECT, SKILL, CAREER, TEACHER
  Relationships: major_offers_subject, major_leads_to_career,
                 subject_provides_skill, career_requires_skill,
                 teacher_instructs_subject, subject_is_prerequisite_of_subject
"""

import os
import re
import json
import logging
import time
import unicodedata
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
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # dùng mini cho skill naming

INPUT_FOLDERS    = ["curriculum", "career_description", "syllabus"]
LOCAL_OUT_DIR    = Path("./cache/output")
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", "10"))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = 2.0

# Đặt FORCE_REPROCESS=true để xử lý lại tất cả file (bỏ qua cache).
# Hoặc để mặc định: tự động reprocess nếu output cũ có 0 nodes (bị lỗi lần trước).
FORCE_REPROCESS  = os.getenv("FORCE_REPROCESS", "false").lower() == "true"


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
    """Chuyển tên tiếng Việt → snake_case không dấu."""
    text = text.strip()
    # Bỏ học hàm/học vị ở đầu
    for prefix in ("GS.TS.", "PGS.TS.", "GS.", "PGS.", "TS.", "ThS.", "T.S.", "Ths.", "CN."):
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].strip()
            break
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def extract_title(name: str) -> str:
    """Lấy học hàm/học vị từ tên."""
    for prefix in ("GS.TS.", "PGS.TS.", "GS.", "PGS.", "TS.", "ThS.", "T.S.", "Ths."):
        if name.strip().upper().startswith(prefix.upper()):
            return prefix.rstrip(".")
    return ""


def clean_name(name: str) -> str:
    """Bỏ học hàm/học vị khỏi tên."""
    name = name.strip()
    for prefix in ("GS.TS.", "PGS.TS.", "GS.", "PGS.", "TS.", "ThS.", "T.S.", "Ths.", "CN."):
        if name.upper().startswith(prefix.upper()):
            return name[len(prefix):].strip()
    return name


def get_paragraphs(doc: dict) -> list[dict]:
    """Trả về list các item type=paragraph từ content.stream."""
    return [
        item for item in doc.get("content", {}).get("stream", [])
        if item.get("type") == "paragraph"
    ]


def get_tables(doc: dict) -> list[dict]:
    """Trả về list các item là table từ content.stream."""
    return [
        item for item in doc.get("content", {}).get("stream", [])
        if "table_index" in item
    ]


def find_table_by_type(doc: dict, table_type: str) -> dict | None:
    for item in doc.get("content", {}).get("stream", []):
        if item.get("table_type") == table_type:
            return item
    return None


def find_tables_by_type(doc: dict, table_type: str) -> list[dict]:
    return [
        item for item in doc.get("content", {}).get("stream", [])
        if item.get("table_type") == table_type
    ]


def save_local(data: dict, folder: str, filename: str):
    out_path = LOCAL_OUT_DIR / folder / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_docid(folder: str, filename: str) -> str:
    stem = Path(filename).stem
    prefix_map = {"curriculum": "CUR", "career_description": "CAR", "syllabus": "SYL"}
    return f"{prefix_map.get(folder, 'DOC')}-{stem}"


# ═══════════════════════════════════════════════════════════════════════════════
# CAREER DESCRIPTION — 100% RULE-BASED
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping section header → field name
_CAREER_SECTION_PATTERNS = {
    r"tên nghề.*?:\s*(.+)":             "name_vi_raw",
    r"nhóm nghề.*?lĩnh vực.*?:\s*(.+)": "field_name",
    r"mô tả ngắn.*?:\s*(.+)":           "short_description",
    r"vai trò.*?tổ chức.*?:\s*(.+)":    "role_in_organization",
}

_HARD_SKILL_SECTION = re.compile(r"kỹ năng chuyên môn|hard skill", re.IGNORECASE)
_SOFT_SKILL_SECTION = re.compile(r"kỹ năng mềm|soft skill", re.IGNORECASE)
_JOB_TASK_SECTION   = re.compile(r"công việc chính|job task", re.IGNORECASE)
_MARKET_SECTION     = re.compile(r"cơ hội việc làm.*thị trường|market", re.IGNORECASE)
_EDU_SECTION        = re.compile(r"yêu cầu học vấn|chứng chỉ", re.IGNORECASE)
_MAJOR_SECTION      = re.compile(r"ngành học phù hợp", re.IGNORECASE)

# Required level keywords
def _infer_required_level(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["thành thạo", "nâng cao", "advanced", "chuyên sâu"]):
        return "advanced"
    if any(k in text_lower for k in ["trung cấp", "intermediate", "trung bình", "khá"]):
        return "intermediate"
    if any(k in text_lower for k in ["cơ bản", "basic", "nhập môn", "nền tảng"]):
        return "basic"
    return ""


def _parse_skill_line(text: str, skill_type: str) -> dict | None:
    """
    Từ 1 dòng text (bullet item), tạo node SKILL.
    Tách skill_name (phần trước ':') và level từ nội dung.
    """
    text = text.strip()
    if not text or len(text) < 4:
        return None

    # Bỏ bullet "-", "*", "+"
    text = re.sub(r"^[-*+•]\s*", "", text)
    if not text:
        return None

    # Tách tên kỹ năng: phần trước ":" hoặc toàn bộ nếu không có ":"
    parts = text.split(":", 1)
    raw_name = parts[0].strip()

    # Rút gọn: bỏ các cụm "Kỹ năng", "Khả năng" ở đầu
    raw_name = re.sub(r"^(kỹ năng|khả năng|kỹ nằn)\s+", "", raw_name, flags=re.IGNORECASE).strip()
    if not raw_name:
        return None

    skill_key = slugify(raw_name)
    if not skill_key:
        return None

    level = _infer_required_level(text)

    return {
        "node": {
            "type": "SKILL",
            "skill_key": skill_key,
            "skill_name": raw_name,
            "skill_type": skill_type,
        },
        "level": level,
    }


def extract_career(doc: dict) -> dict:
    """
    Parse career_description hoàn toàn bằng rule-based.
    Đọc từng paragraph, nhận diện section, extract thông tin.
    """
    paragraphs = get_paragraphs(doc)
    texts = [p["text"].strip() for p in paragraphs if p.get("text", "").strip()]

    # ── Bước 1: Thu thập metadata career ──────────────────────────────────────
    career_info = {
        "name_vi": "",
        "name_en": "",
        "field_name": "",
        "short_description": "",
        "role_in_organization": "",
        "job_tasks": [],
        "market": "",
        "education_certification": "",
    }

    for text in texts:
        # Tên nghề
        m = re.search(r"tên nghề\s*(?:\([^)]*\))?\s*[:\s]+(.+)", text, re.IGNORECASE)
        if m and not career_info["name_vi"]:
            raw = m.group(1).strip()
            # Tách vi / en nếu có dạng "Tên vi / Tên en"
            if "/" in raw:
                parts = raw.split("/", 1)
                # Bỏ phần dịch trong ngoặc: "Sales Representative (Đại diện kinh doanh)" → "Sales Representative"
                career_info["name_vi"] = re.sub(r"\s*\([^)]+\)\s*$", "", parts[0]).strip()
                career_info["name_en"] = re.sub(r"\s*\([^)]+\)\s*$", "", parts[1]).strip()
            else:
                cleaned = re.sub(r"\s*\([^)]+\)\s*$", "", raw).strip()
                career_info["name_vi"] = cleaned
                career_info["name_en"] = cleaned

        # Nhóm nghề
        m = re.search(r"nhóm nghề.*?lĩnh vực.*?[:\s]+(.+)", text, re.IGNORECASE)
        if m and not career_info["field_name"]:
            career_info["field_name"] = m.group(1).strip()

        # Mô tả ngắn
        m = re.search(r"mô tả ngắn\s*[:\s]+(.+)", text, re.IGNORECASE)
        if m and not career_info["short_description"]:
            career_info["short_description"] = m.group(1).strip()

        # Vai trò
        m = re.search(r"vai trò.*?tổ chức.*?[:\s]+(.+)", text, re.IGNORECASE)
        if m and not career_info["role_in_organization"]:
            career_info["role_in_organization"] = m.group(1).strip()

    # Fallback name: nếu vẫn chưa có, lấy từ source_file
    if not career_info["name_vi"]:
        src = doc.get("source_file", "")
        career_info["name_vi"] = Path(src).stem.replace("_", " ")
        career_info["name_en"] = career_info["name_vi"]

    career_key = slugify(career_info["name_vi"])

    # ── Bước 2: Parse skills theo section ─────────────────────────────────────
    skill_nodes = []
    skill_rels  = []
    seen_keys   = set()

    current_section = None  # "hard" | "soft" | "job_tasks" | "market" | "edu" | "major"
    job_tasks = []
    market_lines = []
    edu_lines = []
    major_names = []

    for text in texts:
        # Nhận diện section header — và parse phần inline nếu có nội dung sau ":"
        if _HARD_SKILL_SECTION.search(text):
            current_section = "hard"
            inline = re.sub(r".*?(?:hard skill[s]?|kỹ năng chuyên môn)\s*[:\(][^:)]*\)\s*[:\s]*\*?\s*", "", text, flags=re.IGNORECASE).strip()
            if inline:
                result = _parse_skill_line(inline, "hard")
                if result and result["node"]["skill_key"] not in seen_keys:
                    seen_keys.add(result["node"]["skill_key"])
                    skill_nodes.append(result["node"])
                    rel = {"rel_type": "career_requires_skill", "from_career_key": career_key, "to_skill_key": result["node"]["skill_key"]}
                    if result["level"]:
                        rel["required_level"] = result["level"]
                    skill_rels.append(rel)
            continue
        if _SOFT_SKILL_SECTION.search(text):
            current_section = "soft"
            inline = re.sub(r".*?(?:soft skill[s]?|kỹ năng mềm)\s*[:\(][^:)]*\)\s*[:\s]*\*?\s*", "", text, flags=re.IGNORECASE).strip()
            if inline:
                result = _parse_skill_line(inline, "soft")
                if result and result["node"]["skill_key"] not in seen_keys:
                    seen_keys.add(result["node"]["skill_key"])
                    skill_nodes.append(result["node"])
                    rel = {"rel_type": "career_requires_skill", "from_career_key": career_key, "to_skill_key": result["node"]["skill_key"]}
                    if result["level"]:
                        rel["required_level"] = result["level"]
                    skill_rels.append(rel)
            continue
        if _JOB_TASK_SECTION.search(text):
            current_section = "job_tasks"
            continue
        if _MARKET_SECTION.search(text):
            current_section = "market"
            continue
        if _EDU_SECTION.search(text):
            current_section = "edu"
            continue
        if _MAJOR_SECTION.search(text):
            current_section = "major"
            continue

        # Bỏ qua các tiêu đề section lớn
        if re.match(r"^\d+\.", text) and len(text) < 60:
            # Đây là header mục lớn, reset section context nếu không phải skill section
            if not any(kw in text.lower() for kw in ["kỹ năng", "skill"]):
                current_section = None
            continue

        # Parse theo section hiện tại
        if current_section in ("hard", "soft"):
            skill_type = "hard" if current_section == "hard" else "soft"
            result = _parse_skill_line(text, skill_type)
            if result and result["node"]["skill_key"] not in seen_keys:
                seen_keys.add(result["node"]["skill_key"])
                skill_nodes.append(result["node"])
                rel = {
                    "rel_type": "career_requires_skill",
                    "from_career_key": career_key,
                    "to_skill_key": result["node"]["skill_key"],
                }
                if result["level"]:
                    rel["required_level"] = result["level"]
                skill_rels.append(rel)
        elif current_section == "job_tasks":
            if text and not re.match(r"^\d+\.", text):
                job_tasks.append(text)

        elif current_section == "market":
            market_lines.append(text)

        elif current_section == "edu":
            edu_lines.append(text)

        elif current_section == "major":
            # Lọc tên ngành: loại bỏ dòng quá dài (là mô tả, không phải tên ngành)
            if text and len(text) < 100 and not re.match(r"các ngành đòi hỏi", text, re.IGNORECASE):
                clean = re.sub(r"^[-*+•\d.]+\s*", "", text).strip()
                if clean:
                    major_names.append(clean)

    # ── Bước 3: Build career node ──────────────────────────────────────────────
    career_node = {
        "type": "CAREER",
        "career_key": career_key,
        "career_name_vi": career_info["name_vi"],
        "career_name_en": career_info["name_en"],
        "field_name": career_info["field_name"],
        "description": {
            "short_description": career_info["short_description"],
            "role_in_organization": career_info["role_in_organization"],
        },
        "job_tasks": job_tasks,
        "market": " ".join(market_lines).strip(),
        "education_certification": " ".join(edu_lines).strip(),
        "major_names": major_names,   # dùng cho Phase 2 mapping
        "major_codes": [],            # sẽ được fill ở Phase 2
    }

    nodes = [career_node] + skill_nodes
    return {"nodes": nodes, "relationships": skill_rels}


# ═══════════════════════════════════════════════════════════════════════════════
# SYLLABUS — RULE-BASED + LLM mini-call cho skill_name
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_syllabus_info_table(doc: dict) -> dict:
    """
    Parse thông tin chung của syllabus từ paragraphs (không dùng table_index cứng).
    Hỗ trợ nhiều format khác nhau:
      - "Mã học phần: CNTT1197"          (paragraph riêng)
      - "Mã số học phần: NLKT1126"       (variant label)
      - "Mã HP: TIKT1134"                (viết tắt)
      - Thông tin nằm trong key_value table
    Fallback cuối: lấy code từ source_file nếu không parse được.
    """
    info = {
        "subject_code": "",
        "subject_name_vi": "",
        "subject_name_en": "",
        "credits": "",
        "prerequisites_raw": [],
    }

    # ── Bước 1: Thử parse từ key_value table (một số syllabus dùng format này) ──
    stream = doc.get("content", {}).get("stream", [])
    for item in stream:
        if item.get("table_type") != "key_value":
            continue
        data = item.get("data", {})
        for k, v in data.items():
            k_lower = k.lower()
            v_str = str(v).strip() if v else ""
            if not v_str:
                continue
            if re.search(r"mã.*?(học phần|hp|môn|số)", k_lower):
                m = re.search(r"([A-Z]{2,6}\d{4,})", v_str, re.IGNORECASE)
                if m:
                    info["subject_code"] = m.group(1).strip()
            elif re.search(r"tên.*?việt|tiếng việt", k_lower):
                info["subject_name_vi"] = v_str
            elif re.search(r"tên.*?anh|tiếng anh|english", k_lower):
                info["subject_name_en"] = v_str
            elif re.search(r"tín chỉ|credits", k_lower):
                m = re.search(r"\d+", v_str)
                if m and not info["credits"]:
                    info["credits"] = m.group(0)
            elif re.search(r"tiên quyết|prerequisite", k_lower):
                for p in re.split(r"[\n,;]+", v_str):
                    p = p.strip().lstrip("+-")
                    if p:
                        info["prerequisites_raw"].append(p)
        if info["subject_code"]:
            break

    # ── Bước 2: Parse từ paragraphs ───────────────────────────────────────────
    paragraphs = get_paragraphs(doc)
    texts = [p.get("text", "").strip() for p in paragraphs]

    # Pattern bắt mọi biến thể label mã học phần
    CODE_PATTERN = re.compile(
        r"(?:mã|ma)\s*(?:số\s*)?(?:học phần|hp|môn học|lớp|lh)?\s*[:\s]+([A-Z]{2,8}\d{3,})",
        re.IGNORECASE
    )

    in_prereq = False
    for text in texts:
        if not text:
            continue

        # Tên học phần tiếng Việt
        m = re.search(r"tên học phần.*?(?:tiếng việt|vi).*?[:\s]+(.+)", text, re.IGNORECASE)
        if m:
            info["subject_name_vi"] = m.group(1).strip()
            in_prereq = False
            continue

        # Tên học phần tiếng Anh
        m = re.search(r"tên học phần.*?(?:tiếng anh|en|english).*?[:\s]+(.+)", text, re.IGNORECASE)
        if m:
            info["subject_name_en"] = m.group(1).strip()
            in_prereq = False
            continue

        # Mã học phần — dùng pattern rộng hơn
        if not info["subject_code"]:
            m = CODE_PATTERN.search(text)
            if m:
                info["subject_code"] = m.group(1).strip()
                in_prereq = False
                continue

        # Số tín chỉ
        m = re.search(r"số tín chỉ\s*[:\s]+(\d+)", text, re.IGNORECASE)
        if m and not info["credits"]:
            info["credits"] = m.group(1)
            in_prereq = False
            continue

        # Header tiên quyết
        if re.search(r"các học phần tiên quyết|tiên quyết", text, re.IGNORECASE):
            rest = re.sub(r"^.*?tiên quyết\s*[:\s]*", "", text, flags=re.IGNORECASE).strip()
            if rest:
                for p in re.split(r"[\n,;]+", rest):
                    p = p.strip().lstrip("+-")
                    if p:
                        info["prerequisites_raw"].append(p)
            in_prereq = True
            continue

        # Dòng tiếp theo sau header tiên quyết
        if in_prereq:
            if re.match(r"^\d+\.", text) or re.search(
                r"(mã|số tín chỉ|giảng viên|khoa|viện|mô tả|số giờ|trình độ)",
                text, re.IGNORECASE
            ):
                in_prereq = False
            else:
                for p in re.split(r"[\n,;]+", text):
                    p = p.strip().lstrip("+-")
                    if p:
                        info["prerequisites_raw"].append(p)

    # ── Bước 3: Fallback — lấy code từ source_file nếu vẫn chưa có ──────────
    if not info["subject_code"]:
        src = doc.get("source_file", "")
        m = re.search(r"_([A-Z]{2,8}\d{3,})(?:\.\w+)?$", src, re.IGNORECASE)
        if m:
            info["subject_code"] = m.group(1).strip()
            log.debug(f"subject_code fallback từ filename: {info['subject_code']}")

    # ── Bước 4: Fallback tên môn từ source_file nếu chưa có ─────────────────
    if not info["subject_name_vi"]:
        src = doc.get("source_file", "")
        stem = Path(src).stem  # "An ninh không gian mạng_CNTT1197"
        name_part = re.sub(r"_[A-Z]{2,8}\d{3,}$", "", stem, flags=re.IGNORECASE).strip()
        if name_part:
            info["subject_name_vi"] = name_part

    return info


def _parse_teachers(doc: dict) -> list[dict]:
    """
    Parse bảng giảng viên bằng cách nhận diện header, không dùng table_index cứng.
    Thực tế: bảng GV ở table_index=1 và nhận diện qua cột 'Họ và tên'.
    """
    teachers = []
    stream = doc.get("content", {}).get("stream", [])
    for item in stream:
        if "table_index" not in item:
            continue
        headers = item.get("headers", [])
        # Nhận diện bảng giảng viên theo nội dung header
        if not any("họ" in str(h).lower() and "tên" in str(h).lower() for h in headers):
            continue
        for row in item.get("rows", []):
            # Tìm cột họ tên
            name_raw = ""
            email = ""
            for k, v in row.items():
                k_lower = k.lower()
                v_str = str(v).strip() if v else ""
                if ("họ" in k_lower and "tên" in k_lower) or "name" in k_lower:
                    name_raw = v_str
                elif "email" in k_lower:
                    email = v_str

            if not name_raw:
                continue

            title = extract_title(name_raw)
            name_clean = clean_name(name_raw)
            teacher_key = slugify(name_clean)
            if not teacher_key:
                continue

            teachers.append({
                "type": "TEACHER",
                "teacher_key": teacher_key,
                "name": name_clean,
                "email": email,
                "title": title,
            })
        break
    return teachers


def _parse_clos(doc: dict) -> list[dict]:
    """
    Parse bảng CLO bằng cách nhận diện header chứa 'CLO', không dùng table_index cứng.
    Thực tế: bảng CLO ở table_index=3, headers = ["Mục tiêu", "CLOij", "Mô tả CLO", "Mức độ đạt được"].
    """
    clos = []
    stream = doc.get("content", {}).get("stream", [])
    for item in stream:
        if "table_index" not in item:
            continue
        headers = item.get("headers", [])
        # Nhận diện bảng CLO theo header chứa "CLO" và "mô tả"
        headers_lower = [str(h).lower() for h in headers]
        if not (any("clo" in h for h in headers_lower) and any("mô tả" in h or "description" in h for h in headers_lower)):
            continue
        for row in item.get("rows", []):
            clo_code = ""
            description = ""
            mastery_level = ""

            for k, v in row.items():
                k_lower = k.lower()
                v_str = str(v).strip() if v else ""
                if not v_str or v_str == k:
                    continue
                # Cột mã CLO: "CLOij", "CLO", hoặc cột ngắn chứa "clo"
                if ("clo" in k_lower and len(k) < 15) or k_lower in ("cloij", "clo"):
                    if re.match(r"CLO\s*[\d.]+", v_str, re.IGNORECASE):
                        clo_code = re.sub(r"\s+", " ", v_str).strip()
                # Cột mô tả
                elif "mô tả" in k_lower or "description" in k_lower:
                    if len(v_str) > 10:
                        description = v_str
                # Cột mức độ / level
                elif "mức" in k_lower or "level" in k_lower or "đạt" in k_lower:
                    mastery_level = v_str

            # Fallback: dùng values theo vị trí nếu chưa lấy được clo_code
            if not clo_code:
                vals = list(row.values())
                if len(vals) >= 2:
                    clo_code = str(vals[1]).strip() if vals[1] else ""
                    description = str(vals[2]).strip() if len(vals) > 2 else ""
                    mastery_level = str(vals[3]).strip() if len(vals) > 3 else ""

            if clo_code and re.match(r"CLO\s*[\d.]+", clo_code, re.IGNORECASE):
                clos.append({
                    "clo_code": clo_code,
                    "description": description,
                    "mastery_level": mastery_level,
                })
        break
    return clos


def _parse_lesson_plan(doc: dict) -> dict:
    """
    Parse bảng kế hoạch dạy học (thường table_index=6).
    Trả về dict: {week_1: {...}, week_2: {...}, ...}
    """
    lesson_plan = {}
    stream = doc.get("content", {}).get("stream", [])

    # Tìm table lesson plan (có cột "Tuần")
    for item in stream:
        if "table_index" not in item:
            continue
        headers = item.get("headers", [])
        if not any("tuần" in str(h).lower() or "week" in str(h).lower() for h in headers):
            continue

        for row in item.get("rows", []):
            week_val = ""
            content = ""
            reading = ""
            activities = ""
            assessment = ""
            clos = ""

            for k, v in row.items():
                k_lower = k.lower()
                v_str = str(v).strip() if v else ""
                if "tuần" == k_lower or k_lower.startswith("tuần"):
                    week_val = v_str
                elif "nội dung" in k_lower or "content" in k_lower:
                    content = v_str
                elif "tài liệu" in k_lower or "reading" in k_lower:
                    reading = v_str
                elif "hoạt động" in k_lower or "activit" in k_lower:
                    activities = v_str
                elif "đánh giá" in k_lower or "assessment" in k_lower:
                    assessment = v_str
                elif "clo" in k_lower:
                    clos = v_str

            if not week_val:
                continue

            # Chuẩn hóa week key
            week_num = re.sub(r"\D", "", week_val)
            if week_num:
                key = f"week_{week_num}"
                lesson_plan[key] = {
                    "contents": content,
                    "reading_materials": reading,
                    "teaching_learning_activities": activities,
                    "assessment_activities": assessment,
                    "clos": clos,
                }
        break

    return lesson_plan


def _parse_other_syllabus_fields(doc: dict) -> dict:
    """Parse các trường text tự do từ paragraphs: mô tả, mục tiêu, đánh giá, quy định."""
    fields = {
        "course_description": "",
        "learning_resources": [],
        "courses_goals": "",
        "assessment": "",
        "course_requirements_and_expectations": "",
        "syllabus_adjustment_time": "",
    }

    paragraphs = get_paragraphs(doc)
    current_section = None
    buffer = []

    section_map = {
        r"mô tả học phần|course description":           "course_description",
        r"tài liệu học tập|learning resources":          "learning_resources",
        r"mục tiêu học phần|course goals":               "courses_goals",
        r"đánh giá học phần|course assessment":          "assessment",
        r"quy định.*học phần|course requirements":       "course_requirements_and_expectations",
        r"thời điểm điều chỉnh|syllabus adjustment":     "syllabus_adjustment_time",
    }

    def flush(section, buf):
        if not section or not buf:
            return
        content = " ".join(buf).strip()
        if section == "learning_resources":
            # Chia thành list theo dòng
            fields[section] = [b for b in buf if b]
        else:
            if fields[section]:
                fields[section] += " " + content
            else:
                fields[section] = content

    for p in paragraphs:
        text = p.get("text", "").strip()
        if not text:
            continue

        matched = False
        for pattern, field in section_map.items():
            if re.search(pattern, text, re.IGNORECASE) and len(text) < 100:
                flush(current_section, buffer)
                current_section = field
                buffer = []
                matched = True
                break

        if not matched and current_section:
            buffer.append(text)

    flush(current_section, buffer)
    return fields


def _clo_to_skill_name_heuristic(description: str) -> str:
    """
    Heuristic rule-based: rút gọn CLO description thành skill_name ngắn gọn.
    Bắt các pattern phổ biến, fallback sang 4 từ đầu.
    """
    desc = description.strip()

    # Bỏ prefix "Sinh viên có thể/có khả năng/được...", "Người học..."
    desc = re.sub(
        r"^(sinh viên|người học|học viên|sv)\s+(có thể|có khả năng|sẽ|được|có thể)\s+",
        "", desc, flags=re.IGNORECASE
    ).strip()

    # Pattern: "Hệ thống hóa được kiến thức về X" → "X"
    m = re.search(r"(?:kiến thức|kỹ năng|năng lực)\s+(?:về|về việc|trong)\s+(.+?)(?:\.|;|,|$)", desc, re.IGNORECASE)
    if m:
        skill = m.group(1).strip().rstrip(".,;")
        if len(skill) < 60:
            return skill[:1].upper() + skill[1:]

    # Pattern: "Vận dụng/Sử dụng/Xây dựng/Phân tích X" → "X"
    m = re.search(
        r"^(?:vận dụng|sử dụng|xây dựng|phân tích|thực hiện|giải thích|đánh giá|trình bày|áp dụng)\s+(?:được\s+)?(.+?)(?:\.|;|,|$)",
        desc, re.IGNORECASE
    )
    if m:
        skill = m.group(1).strip().rstrip(".,;")
        if 3 < len(skill) < 60:
            return skill[:1].upper() + skill[1:]

    # Fallback: lấy tối đa 5 từ đầu
    words = desc.split()[:5]
    return " ".join(words).rstrip(".,;")


def _infer_mastery(mastery_raw: str) -> str:
    """Chuẩn hóa mastery level từ số hoặc text."""
    text = str(mastery_raw).strip().lower()
    if text in ("4", "5", "advanced", "thành thạo", "nâng cao"):
        return "advanced"
    if text in ("3", "intermediate", "trung cấp", "trung bình"):
        return "intermediate"
    if text in ("1", "2", "basic", "cơ bản", "nhập môn"):
        return "basic"
    return "intermediate"   # default


def extract_syllabus(doc: dict, ai_client: OpenAI | None) -> dict:
    """
    Parse syllabus: rule-based cho tất cả cấu trúc,
    dùng LLM mini-call CHỈ để chuẩn hóa skill_name từ CLO text.
    """
    # ── 1. Parse bảng thông tin chung ─────────────────────────────────────────
    info = _parse_syllabus_info_table(doc)
    subject_code = info["subject_code"]
    subject_name_vi = info["subject_name_vi"]
    subject_name_en = info["subject_name_en"]

    if not subject_code:
        log.warning(f"Syllabus: Không tìm thấy subject_code — bỏ qua {doc.get('source_file', '')}")
        return {"nodes": [], "relationships": []}

    # ── 2. Parse teachers ──────────────────────────────────────────────────────
    teachers = _parse_teachers(doc)

    # ── 3. Parse CLOs ──────────────────────────────────────────────────────────
    clos = _parse_clos(doc)

    # ── 4. Parse lesson_plan ───────────────────────────────────────────────────
    lesson_plan = _parse_lesson_plan(doc)

    # ── 5. Parse các trường text khác ─────────────────────────────────────────
    other_fields = _parse_other_syllabus_fields(doc)

    # ── 6. Tạo SKILL nodes từ CLOs ────────────────────────────────────────────
    # Nếu có ai_client: gọi LLM 1 lần để rút gọn tất cả CLO descriptions
    # Nếu không: dùng heuristic rule-based
    skill_nodes = []
    skill_rels  = []

    if clos:
        # Thử LLM mini-call
        skill_names = _batch_clo_to_skill_names(clos, subject_code, ai_client)

        for clo, skill_name in zip(clos, skill_names):
            skill_key = slugify(skill_name)
            if not skill_key:
                continue

            # Phân loại hard/soft từ CLO description
            desc_lower = clo["description"].lower()
            skill_type = "soft" if any(
                k in desc_lower for k in ["nhóm", "giao tiếp", "tự chủ", "trách nhiệm",
                                          "hợp tác", "đạo đức", "tự học", "độc lập", "làm việc"]
            ) else "hard"

            skill_nodes.append({
                "type": "SKILL",
                "skill_key": skill_key,
                "skill_name": skill_name,
                "skill_type": skill_type,
                "clo_code": clo["clo_code"],
            })

            mastery = _infer_mastery(clo.get("mastery_level", ""))
            skill_rels.append({
                "rel_type": "subject_provides_skill",
                "from_subject_code": subject_code,
                "to_skill_key": skill_key,
                "mastery_level": mastery,
            })

    # ── 7. Tạo prerequisite relationships ─────────────────────────────────────
    prereq_rels = []
    for prereq_raw in info["prerequisites_raw"]:
        # Chỉ tạo rel nếu có vẻ là MÃ MÔN (không có khoảng trắng, và có chữ+số)
        code_candidate = prereq_raw.strip()
        if re.match(r"^[A-Z]{2,6}\d{4,}", code_candidate, re.IGNORECASE) and " " not in code_candidate:
            prereq_rels.append({
                "rel_type": "subject_is_prerequisite_of_subject",
                "from_subject_code": code_candidate,
                "to_subject_code": subject_code,
            })
        # Nếu là tên môn → bỏ qua (không thể map an toàn)

    # ── 8. Tạo teacher_instructs_subject relationships ─────────────────────────
    teacher_rels = []
    for t in teachers:
        teacher_rels.append({
            "rel_type": "teacher_instructs_subject",
            "from_teacher_key": t["teacher_key"],
            "to_subject_code": subject_code,
        })

    # ── 9. Build subject node ──────────────────────────────────────────────────
    subject_node = {
        "type": "SUBJECT",
        "subject_code": subject_code,
        "subject_name_vi": subject_name_vi,
        "subject_name_en": subject_name_en,
        "credits": info["credits"],
        "course_description": other_fields["course_description"],
        "learning_resources": other_fields["learning_resources"],
        "courses_goals": other_fields["courses_goals"],
        "assessment": other_fields["assessment"],
        "course_requirements_and_expectations": other_fields["course_requirements_and_expectations"],
        "syllabus_adjustment_time": other_fields["syllabus_adjustment_time"],
        "course_learning_outcomes": [
            {"clo_code": c["clo_code"], "description": c["description"]}
            for c in clos
        ],
        **{k: v for k, v in lesson_plan.items()},  # week_1, week_2, ...
    }

    nodes = [subject_node] + teachers + skill_nodes
    relationships = teacher_rels + skill_rels + prereq_rels

    return {"nodes": nodes, "relationships": relationships}


def _batch_clo_to_skill_names(clos: list[dict], subject_code: str, ai_client: OpenAI | None) -> list[str]:
    """
    Gọi LLM 1 lần duy nhất để rút gọn TẤT CẢ CLO descriptions thành skill names.
    Input LLM: chỉ list CLO descriptions (rất ít token).
    Fallback sang heuristic nếu không có ai_client hoặc lỗi.
    """
    # Heuristic fallback
    heuristic_names = [_clo_to_skill_name_heuristic(c["description"]) for c in clos]

    if ai_client is None:
        return heuristic_names

    # Build compact prompt — chỉ gửi CLO codes + descriptions
    clo_lines = "\n".join(
        f'{c["clo_code"]}: {c["description"]}'
        for c in clos
    )

    prompt = f"""Môn học: {subject_code}
Dưới đây là danh sách Chuẩn đầu ra (CLO). Với mỗi CLO, hãy trích xuất TÊN KỸ NĂNG ngắn gọn (2-5 từ tiếng Việt).

Quy tắc:
- Ngắn gọn, súc tích (VD: "Phân tích dữ liệu", "Lập trình Python", "Làm việc nhóm")
- KHÔNG viết cả câu CLO
- Trả về JSON object với key "skills" là array, thứ tự tương ứng với CLO đầu vào

CLOs:
{clo_lines}

Ví dụ format trả về: {{"skills": ["Phân tích dữ liệu", "Sử dụng phần mềm R", ...]}}"""

    try:
        response = ai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        # Có thể trả về {"skills": [...]} hoặc {"names": [...]} hoặc trực tiếp array
        if isinstance(parsed, list):
            names = parsed
        elif isinstance(parsed, dict):
            # Ưu tiên key "skills", fallback sang bất kỳ list nào
            names = parsed.get("skills") or next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
        else:
            return heuristic_names

        # Đảm bảo đúng số lượng
        if len(names) == len(clos):
            return [str(n).strip() for n in names]
        else:
            log.warning(f"LLM trả về {len(names)} names nhưng có {len(clos)} CLOs — dùng heuristic")
            return heuristic_names

    except Exception as e:
        log.warning(f"LLM skill naming failed ({e}), dùng heuristic")
        return heuristic_names


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM — RULE-BASED cho course list + LLM cho text tự do
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_major_info(doc: dict) -> dict:
    """Parse thông tin MAJOR từ bảng key_value đầu tiên."""
    major = {
        "major_code": "",
        "major_name_vi": "",
        "major_name_en": "",
    }

    stream = doc.get("content", {}).get("stream", [])
    for item in stream:
        if item.get("table_type") != "key_value":
            continue
        data = item.get("data", {})
        for k, v in data.items():
            k_lower = k.lower()
            v_str = str(v).strip() if v else ""
            if "mã ngành" in k_lower or "code" in k_lower:
                major["major_code"] = v_str
            elif "ngành đào tạo" in k_lower or "major" in k_lower:
                # Tách vi / en nếu có "/"
                if "/" in v_str:
                    parts = v_str.split("/", 1)
                    major["major_name_vi"] = parts[0].strip()
                    major["major_name_en"] = re.sub(r"[\n\r]+", " ", parts[1]).strip().strip("()")
                else:
                    major["major_name_vi"] = v_str
            elif "chương trình" in k_lower or "programme" in k_lower:
                if not major["major_name_vi"] and "/" in v_str:
                    parts = v_str.split("/", 1)
                    major["major_name_vi"] = parts[0].strip()
                    major["major_name_en"] = re.sub(r"[\n\r]+", " ", parts[1]).strip().strip("()")
        if major["major_code"]:
            break

    return major


def _parse_course_list(doc: dict, major_code: str) -> tuple[list[dict], list[dict]]:
    """
    Parse danh sách môn học từ table type="specialized_curriculum".
    Trả về (subject_nodes, major_offers_subject_rels).
    """
    subject_nodes = []
    rels = []
    seen_codes = set()

    stream = doc.get("content", {}).get("stream", [])
    for item in stream:
        if item.get("table_type") != "specialized_curriculum":
            continue

        semester_map = _build_semester_map(item)

        for row in item.get("rows", []):
            if row.get("row_type") != "course":
                continue

            code = str(row.get("code", "")).strip()
            if not code or code in ("GDTC", "GDQP", "NEU", "NNKC"):
                continue
            if code in seen_codes:
                continue
            seen_codes.add(code)

            # Tên môn: ưu tiên name_vi, fallback parse từ name
            name_vi = row.get("name_vi", "") or _split_course_name(row.get("name", ""))[0]
            name_en = row.get("name_en", "") or _split_course_name(row.get("name", ""))[1]
            credits = row.get("credits")

            subject_nodes.append({
                "type": "SUBJECT",
                "subject_code": code,
                "subject_name_vi": name_vi.strip(),
                "subject_name_en": name_en.strip(),
                "credits": str(credits) if credits else "",
            })

            # Xác định semester và required_type
            semester_no = semester_map.get(code) or _infer_semester(row)
            group_path = row.get("group_path", [])
            required_type = _infer_required_type(group_path)

            rel = {
                "rel_type": "major_offers_subject",
                "from_major_code": major_code,
                "to_subject_code": code,
                "semester": semester_no,
                "required_type": required_type,
            }
            rels.append(rel)

    return subject_nodes, rels


def _split_course_name(name: str) -> tuple[str, str]:
    """Tách tên môn vi/en từ field 'name' dạng 'Tên VN\nEnglish Name'."""
    if not name:
        return "", ""
    parts = name.strip().split("\n", 1)
    vi = parts[0].strip()
    en = parts[1].strip() if len(parts) > 1 else ""

    # Xóa prefix số "1 ", "2 "... ở đầu tên
    vi = re.sub(r"^\d+\s+", "", vi).strip()
    en = re.sub(r"^\d+\s+", "", en).strip()
    return vi, en


def _build_semester_map(curriculum_table: dict) -> dict[str, int]:
    """Build map: course_code → semester_no từ semester field hoặc hk_distribution."""
    semester_map = {}
    for row in curriculum_table.get("rows", []):
        if row.get("row_type") != "course":
            continue
        code = str(row.get("code", "")).strip()
        if not code:
            continue

        # Dạng 1: có field "semester" = "I", "II", ..., hoặc "HK1", "HK2",...
        sem = row.get("semester", "")
        if sem:
            no = _semester_str_to_int(str(sem))
            if no:
                semester_map[code] = no
                continue

        # Dạng 2: semester_distribution = {"HK1": "3", ...}
        dist = row.get("semester_distribution", {})
        if dist:
            first_hk = next(iter(dist), "")
            no = _semester_str_to_int(first_hk)
            if no:
                semester_map[code] = no

    return semester_map


def _semester_str_to_int(s: str) -> int | None:
    """Chuyển "HK3", "III", "3" → 3."""
    s = s.strip().upper()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
    if s in roman:
        return roman[s]
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))
    return None


def _infer_semester(row: dict) -> int | None:
    """Fallback: infer semester từ semester field trực tiếp."""
    sem = row.get("semester", "")
    if sem:
        return _semester_str_to_int(str(sem).split(",")[0].split("-")[0].strip())
    return None


def _infer_required_type(group_path: list) -> str:
    """Xác định bắt buộc/tự chọn từ group_path."""
    path_str = " ".join(group_path).lower()
    if "tự chọn" in path_str or "elective" in path_str:
        return "elective"
    return "required"


# ── LLM call cho curriculum: chỉ gửi phần text tự do ──────────────────────────

CURRICULUM_TEXT_PROMPT = """Trích xuất thông tin từ phần văn bản của chương trình đào tạo.
Chỉ trả về JSON hợp lệ, không markdown.

Cần trích xuất:
1. careers: list các vị trí nghề nghiệp từ phần "Cơ hội làm việc" (career_name_vi, career_name_en nếu có, field_name nếu có)
2. major_metadata: object chứa philosophy_and_objectives, learning_outcomes_summary, training_process_and_graduation_conditions

Format:
{
  "careers": [
    {"career_name_vi": "...", "career_name_en": "...", "field_name": "..."}
  ],
  "major_metadata": {
    "philosophy_and_objectives": "...",
    "learning_outcomes_summary": "...",
    "training_process_and_graduation_conditions": "..."
  }
}

Văn bản:
"""


def _extract_career_text_section(doc: dict) -> str:
    """
    Thu thập chỉ phần text liên quan đến cơ hội việc làm + mục tiêu đào tạo.
    Giới hạn ~2000 chars để tiết kiệm token.
    """
    paragraphs = get_paragraphs(doc)
    relevant = []
    capture = False
    chars = 0

    sections_of_interest = re.compile(
        r"cơ hội làm việc|khả năng học tập nâng cao|triết lý|mục tiêu đào tạo|"
        r"quy trình đào tạo|điều kiện tốt nghiệp|career|job opportunit",
        re.IGNORECASE
    )
    stop_sections = re.compile(
        r"cấu trúc.*chương trình|nội dung.*kế hoạch|phương pháp giảng dạy|"
        r"tiêu chuẩn đội ngũ|cơ sở vật chất|hướng dẫn thực hiện",
        re.IGNORECASE
    )

    for p in paragraphs:
        text = p.get("text", "").strip()
        if not text:
            continue
        if stop_sections.search(text) and len(text) < 80:
            capture = False
        if sections_of_interest.search(text):
            capture = True
        if capture:
            relevant.append(text)
            chars += len(text)
            if chars > 3000:
                break

    return "\n".join(relevant)


def extract_curriculum(doc: dict, ai_client: OpenAI) -> dict:
    """
    Parse curriculum:
    - Rule-based: major info + toàn bộ course list
    - LLM: chỉ gửi ~3000 chars text section để extract careers + metadata
    """
    # ── 1. Parse MAJOR info ────────────────────────────────────────────────────
    major_info = _parse_major_info(doc)
    major_code = major_info["major_code"]

    if not major_code:
        log.warning("Curriculum: Không tìm thấy major_code, thử fallback LLM full")
        return _extract_curriculum_llm_fallback(doc, ai_client)

    # ── 2. Parse course list (100% rule-based) ─────────────────────────────────
    subject_nodes, subject_rels = _parse_course_list(doc, major_code)
    log.info(f"  → Parsed {len(subject_nodes)} subjects rule-based")

    # ── 3. LLM call cho phần text tự do (careers + metadata) ──────────────────
    text_section = _extract_career_text_section(doc)
    career_nodes = []
    career_rels  = []
    major_metadata = {}

    if text_section and ai_client:
        llm_result = _call_curriculum_text_llm(text_section, ai_client)
        if llm_result:
            # Build CAREER nodes
            for c in llm_result.get("careers", []):
                name_vi = c.get("career_name_vi", "").strip()
                if not name_vi:
                    continue
                career_key = slugify(name_vi)
                career_nodes.append({
                    "type": "CAREER",
                    "career_key": career_key,
                    "career_name_vi": name_vi,
                    "career_name_en": c.get("career_name_en", ""),
                    "field_name": c.get("field_name", ""),
                })
                career_rels.append({
                    "rel_type": "major_leads_to_career",
                    "from_major_code": major_code,
                    "to_career_key": career_key,
                })
            major_metadata = llm_result.get("major_metadata", {})

    # ── 4. Build MAJOR node ────────────────────────────────────────────────────
    major_node = {
        "type": "MAJOR",
        "major_code": major_code,
        "major_name_vi": major_info["major_name_vi"],
        "major_name_en": major_info["major_name_en"],
        **major_metadata,
    }

    nodes = [major_node] + subject_nodes + career_nodes
    relationships = subject_rels + career_rels

    return {"nodes": nodes, "relationships": relationships}


def _call_curriculum_text_llm(text: str, ai_client: OpenAI) -> dict | None:
    """Gọi LLM với chỉ phần text tự do của curriculum."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": CURRICULUM_TEXT_PROMPT + text
                }],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=1500,
            )
            return json.loads(response.choices[0].message.content)
        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning(f"Rate limit (attempt {attempt}), wait {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"Curriculum LLM error (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_BASE_DELAY * attempt)
    return None


def _extract_curriculum_llm_fallback(doc: dict, ai_client: OpenAI) -> dict:
    """
    Fallback khi không parse được major_code bằng rule-based.
    Gửi phần compact của doc vào LLM nhưng vẫn giới hạn token.
    """
    log.warning("Curriculum: dùng LLM fallback")
    # Build compact representation: chỉ key_value tables + text sections quan trọng
    compact = {
        "source_file": doc.get("source_file", ""),
        "key_value_tables": [],
        "career_text": _extract_career_text_section(doc),
        "courses_sample": [],  # chỉ lấy 10 courses đầu để LLM biết format
    }

    for item in doc.get("content", {}).get("stream", []):
        if item.get("table_type") == "key_value":
            compact["key_value_tables"].append(item.get("data", {}))
        elif item.get("table_type") == "specialized_curriculum":
            courses = [r for r in item.get("rows", []) if r.get("row_type") == "course"]
            compact["courses_sample"] = courses[:10]
            break

    # Reuse full LLM extraction chỉ cho fallback này
    from_llm = _full_llm_extract(compact, ai_client, "curriculum")
    return from_llm


def _full_llm_extract(doc_json: dict, ai_client: OpenAI, doctype: str) -> dict:
    """Full LLM extraction (dùng như emergency fallback)."""
    # Import từ schema cũ
    PROMPT = """Trích xuất entities và relationships từ JSON chương trình đào tạo.
Chỉ tạo MAJOR, SUBJECT (có code), CAREER.
Trả về JSON: {"nodes": [...], "relationships": [...]}
Không markdown, không giải thích."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps(doc_json, ensure_ascii=False)[:8000]},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=4000,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == MAX_RETRIES:
                log.error(f"Full LLM fallback failed: {e}")
                return {"nodes": [], "relationships": []}
            time.sleep(RETRY_BASE_DELAY * attempt)
    return {"nodes": [], "relationships": []}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: CAREER → MAJOR MAPPING (giữ nguyên logic cũ)
# ═══════════════════════════════════════════════════════════════════════════════

# Map dùng cả tên VI lẫn EN để tăng khả năng match.
# career_description folder dùng tên VI; curriculum (LLM) có thể dùng tên khác.
CAREER_MAJOR_MAP = {
    # ── Tiếng Anh (từ career_description) ────────────────────────────────────
    "Automation tester":                ["7480201", "7340405", "7480101"],
    "Automation Tester":                ["7480201", "7340405", "7480101"],
    "Business analyst":                 ["7480201", "7340405", "7480101"],
    "Business Analyst":                 ["7480201", "7340405", "7480101"],
    "Customer Success":                 ["7340115"],
    "Data Analyst":                     ["7480201", "7340405", "7480101", "7310108","7460108"],
    "Data Engineer":                    ["7480201", "7480101", "7310108"],
    "Key Account Manager":              ["7340115"],
    "Marketing Offline":                ["7340115"],
    "Media Planner":                    ["7340115"],
    "Sales Representative":             ["7340115"],
    "System Administrator":             ["7480201", "7480103", "7480101"],
    "Tester":                           ["7480201", "7480103", "7480101"],
    # ── Tiếng Việt ────────────────────────────────────────────────────────────
    "Kiểm thử tự động":                 ["7480201", "7340405", "7480101"],
    "Chuyên viên dữ liệu":              ["7480201", "7310108", "7340405", "7480101"],
    "Lập trình viên":                   ["7480101", "7480201", "7480103", "7480202"],
    "Kế toán quản trị":                 ["7340201"],
    "Kỹ sư cầu nối":                    ["7480201", "7480103"],
    "Kỹ sư phần mềm":                   ["7480101", "7480201", "7480103", "7480202"],
    "Lập trình game":                   ["7480101", "7480201", "7480103"],
    "Lập trình nhúng":                  ["7480101"],
    "Nhân viên Bồi thường bảo hiểm":    ["7340204"],
    "Nhân viên kinh doanh tiếng Trung": ["7340120", "7340121"],
    "Nhân viên kinh doanh":             ["7340121", "7310101"],
    "Nhân viên triển khai phần mềm":    ["7480201", "7480101", "7480103"],
    "Quản lý kinh doanh":               ["7340101"],
    "Chuyên viên phân tích dữ liệu": ["7310107","7310108","7460108"],
    "System Admin": ["7480104"],
    "Sales Representative": ["7340115","7340101"]
}


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().upper())


def build_major_code_index() -> dict[str, str]:
    index: dict[str, str] = {}
    cur_dir = LOCAL_OUT_DIR / "curriculum"
    if not cur_dir.exists():
        return index
    for jf in cur_dir.glob("*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for node in data.get("nodes", []):
            if node.get("type") != "MAJOR":
                continue
            name = node.get("major_name_vi", "").strip()
            code = node.get("major_code", "").strip()
            if name and code:
                norm = _normalize_name(name)
                index.setdefault(norm, code)
    return index


def _lookup_career_major_codes(career_name_vi: str, career_name_en: str,
                               major_names: list, major_index: dict) -> list:
    """
    Tra cứu major_codes cho một CAREER node theo thứ tự ưu tiên:
    1. CAREER_MAJOR_MAP theo tên VI
    2. CAREER_MAJOR_MAP theo tên EN
    3. major_names field → major_index (tên ngành → major_code)
    """
    # 1. Lookup theo tên tiếng Việt
    if career_name_vi in CAREER_MAJOR_MAP:
        return CAREER_MAJOR_MAP[career_name_vi]
    # 2. Lookup theo tên tiếng Anh
    if career_name_en and career_name_en in CAREER_MAJOR_MAP:
        return CAREER_MAJOR_MAP[career_name_en]
    # 3. Lookup từ major_names
    codes = []
    for mn in major_names:
        code = major_index.get(_normalize_name(mn))
        if code and code not in codes:
            codes.append(code)
    return codes


def run_phase2_mapping():
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: Mapping major_codes cho CAREER nodes")
    log.info("=" * 60)

    major_index = build_major_code_index()

    # ── Xử lý career_description folder ──────────────────────────────────────
    career_dir = LOCAL_OUT_DIR / "career_description"
    if career_dir.exists():
        for jf in career_dir.glob("*.json"):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.error(f"Lỗi đọc {jf.name}: {e}")
                continue

            changed = False
            for node in data.get("nodes", []):
                if node.get("type") != "CAREER":
                    continue
                name_vi  = node.get("career_name_vi", "")
                name_en  = node.get("career_name_en", "")
                codes = _lookup_career_major_codes(
                    name_vi, name_en, node.get("major_names", []), major_index
                )
                node["major_codes"] = codes
                if not codes:
                    log.warning(f"  ⚠ {name_vi or name_en or '(unknown)'}: không map được major_codes")
                else:
                    changed = True

            if changed:
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Xử lý curriculum folder — CAREER nodes kèm MAJOR trong cùng file ─────
    # Với careers trích xuất từ curriculum, major_code lấy trực tiếp từ MAJOR
    # node cùng file (chúng đã có quan hệ major_leads_to_career).
    cur_dir = LOCAL_OUT_DIR / "curriculum"
    if cur_dir.exists():
        for jf in cur_dir.glob("*.json"):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.error(f"Lỗi đọc curriculum {jf.name}: {e}")
                continue

            # Lấy major_code từ MAJOR node trong file
            file_major_codes = [
                n["major_code"] for n in data.get("nodes", [])
                if n.get("type") == "MAJOR" and n.get("major_code")
            ]

            changed = False
            for node in data.get("nodes", []):
                if node.get("type") != "CAREER":
                    continue
                name_vi = node.get("career_name_vi", "")
                name_en = node.get("career_name_en", "")

                # Ưu tiên 1: bảng thủ công
                codes = _lookup_career_major_codes(
                    name_vi, name_en, node.get("major_names", []), major_index
                )
                # Ưu tiên 2: major_code từ MAJOR node cùng file
                if not codes and file_major_codes:
                    codes = file_major_codes

                node["major_codes"] = codes
                if not codes:
                    log.warning(f"  ⚠ curriculum career {name_vi or name_en or '(unknown)'}: không map được major_codes")
                else:
                    changed = True

            if changed:
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    log.info("[Phase 2] Hoàn tất")


# ═══════════════════════════════════════════════════════════════════════════════
# MINIO + MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

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
    return [o for o in all_names if o.endswith(".json")]


def download_json(client: Minio, bucket: str, object_name: str) -> dict:
    response = client.get_object(bucket, object_name)
    data = json.loads(response.read().decode("utf-8"))
    response.close()
    return data


def process_one(minio_client: Minio, ai_client: OpenAI, folder: str, obj_name: str) -> str:
    filename = Path(obj_name).name
    docid    = make_docid(folder, filename)
    out_path = LOCAL_OUT_DIR / folder / f"{docid}.json"

    if out_path.exists() and not FORCE_REPROCESS:
        # Tự động reprocess nếu output cũ có 0 nodes (lần chạy trước bị lỗi)
        try:
            with open(out_path, encoding="utf-8") as _f:
                _cached = json.load(_f)
            if len(_cached.get("nodes", [])) == 0:
                log.info(f"[reprocess] {filename} — cached output có 0 nodes")
            else:
                log.debug(f"[skip] {filename}")
                return "skip"
        except Exception:
            pass  # file lỗi → reprocess

    log.info(f"[start] {filename} ({docid})")
    try:
        doc_json = download_json(minio_client, MINIO_BUCKET, obj_name)
        doctype  = folder  # "syllabus" | "curriculum" | "career_description"

        if doctype == "career_description":
            extracted = extract_career(doc_json)
        elif doctype == "syllabus":
            extracted = extract_syllabus(doc_json, ai_client)
        elif doctype == "curriculum":
            extracted = extract_curriculum(doc_json, ai_client)
        else:
            extracted = {"nodes": [], "relationships": []}

        node_count = len(extracted.get("nodes", []))
        rel_count  = len(extracted.get("relationships", []))

        save_local(extracted, folder, f"{docid}.json")
        log.info(f"[done] {filename} → {node_count} nodes, {rel_count} rels")
        return "ok"

    except Exception as e:
        log.error(f"[ERROR] {filename}: {e}", exc_info=True)
        return "error"


def process_folder(minio_client: Minio, ai_client: OpenAI, folder: str) -> dict:
    log.info(f"\n{'=' * 60}\nProcessing folder: {folder}")
    prefix  = f"{MINIO_BASE_FOLDER}/{folder}"
    objects = list_json_objects(minio_client, MINIO_BUCKET, prefix)
    if not objects:
        log.warning(f"Không tìm thấy file trong {folder}/")
        return {"ok": 0, "skip": 0, "error": 0}

    counts = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, minio_client, ai_client, folder, obj): obj
            for obj in objects
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1

    log.info(f"Folder '{folder}': ✓{counts['ok']} skip={counts['skip']} ✗{counts['error']}")
    return counts


def main():
    log.info("Starting hybrid extraction pipeline...")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY không tìm thấy")
    if not MINIO_ENDPOINT:
        raise ValueError("MINIO_ENDPOINT không tìm thấy")

    minio_client = get_minio_client()
    ai_client    = OpenAI(api_key=OPENAI_API_KEY)

    total = {"ok": 0, "skip": 0, "error": 0}
    for folder in ["syllabus", "curriculum", "career_description"]:
        if folder not in INPUT_FOLDERS:
            continue
        counts = process_folder(minio_client, ai_client, folder)
        for k in total:
            total[k] += counts.get(k, 0)

    log.info(f"\n✅ Phase 1 done: ✓{total['ok']} skip={total['skip']} ✗{total['error']}")

    run_phase2_mapping()

    log.info("\n✅ Pipeline hoàn tất. Results saved to ./cache/output/")


if __name__ == "__main__":
    main()
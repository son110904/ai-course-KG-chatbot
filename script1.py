"""
Script 1: Extract entities & relationships from MinIO JSON files
Reads from: syllabus/courses-processed/{curriculum,career_description,syllabus}/
Saves extracted KG JSON locally to ./cache/output/{folder}/
- Parallel processing via ThreadPoolExecutor
- Auto-skip files already extracted (resume support)
Uses OpenAI API
"""

import os
import json
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from minio import Minio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
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
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", "5"))
# ─────────────────────────────────────────────────────────────────────────────

DOCTYPE_MAP = {
    "curriculum":         "curriculum",
    "career_description": "career_description",
    "syllabus":           "syllabus",
}

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

RÀNG BUỘC THEO LOẠI TÀI LIỆU:
- syllabus (đề cương): chứa SUBJECT (1 môn duy nhất, bắt buộc có code), TEACHER, SKILL. KHÔNG tạo CAREER, MAJOR.
- curriculum (chương trình đào tạo): chứa MAJOR (1 ngành duy nhất, bắt buộc có code), SUBJECT (bắt buộc có code), CAREER. KHÔNG tạo TEACHER.
- career_description (mô tả nghề): chứa CAREER (1 nghề duy nhất), SKILL, MAJOR (nếu có nhắc đến ngành khuyến nghị). KHÔNG tạo SUBJECT, TEACHER.

QUAN TRỌNG:
- SUBJECT bắt buộc phải có code thực sự (ví dụ: CNTT1234, IT001). Nếu không có code thì KHÔNG tạo node SUBJECT.
- KHÔNG dùng code='' hoặc code='SKILL' hoặc các giá trị vô nghĩa.

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

    # ── SYLLABUS ─────────────────────────────────────────────────────────────
    "syllabus": SYSTEM_PROMPT_BASE + """

Tài liệu này là ĐỀ CƯƠNG MÔN HỌC (syllabus).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node SUBJECT (môn học). SUBJECT phải có code môn học thực sự.
- TEACHER: các giảng viên phụ trách môn đó.
- MAJOR: ngành nào đang giảng dạy môn này (nếu có đề cập).
- KHÔNG tạo CAREER.

TRÍCH XUẤT SKILL — RẤT QUAN TRỌNG:
- Nguồn CHÍNH để lấy SKILL là phần "Chuẩn đầu ra học phần" / "course_learning_outcomes" / "learning_outcomes".
  Đây là nơi liệt kê rõ ràng các kiến thức, kỹ năng sinh viên đạt được sau môn học.
- Mỗi chuẩn đầu ra (CLO) hoặc kết quả học tập → tạo 1 node SKILL.
- Tên SKILL phải ngắn gọn, súc tích, thực chất (không sao chép nguyên câu dài).
  Ví dụ đúng: "LẬP TRÌNH PYTHON", "PHÂN TÍCH DỮ LIỆU", "THIẾT KẾ GIAO DIỆN WEB"
  Ví dụ sai: "SINH VIÊN CÓ KHẢ NĂNG VIẾT ĐƯỢC CHƯƠNG TRÌNH..."
- Tạo relationship: (SUBJECT)-[:PROVIDES]->(SKILL)

Quan hệ hợp lệ: TEACHER-[TEACH]->SUBJECT, SUBJECT-[PROVIDES]->SKILL, MAJOR-[OFFERS]->SUBJECT.""",

    # ── CURRICULUM ───────────────────────────────────────────────────────────
    "curriculum": SYSTEM_PROMPT_BASE + """

Tài liệu này là CHƯƠNG TRÌNH ĐÀO TẠO (curriculum).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node MAJOR (ngành học). MAJOR phải có code ngành thực sự.
- SUBJECT: tất cả các môn học trong chương trình, bắt buộc có code môn. Nếu không có code → bỏ qua.
- KHÔNG tạo TEACHER.

TRÍCH XUẤT CAREER — RẤT QUAN TRỌNG:
- Nguồn CHÍNH để lấy CAREER là phần "Cơ hội làm việc và khả năng học tập nâng cao" / "career_opportunities"
  / "vị trí công việc" / "cơ hội việc làm" / "job_opportunities".
- Đây là các nghề nghiệp, vị trí công việc mà sinh viên tốt nghiệp ngành này có thể đảm nhận.
- Mỗi vị trí/nghề nghiệp được nhắc tới → tạo 1 node CAREER riêng.
- Tên CAREER phải VIẾT HOA, cụ thể, súc tích.
  Ví dụ đúng: "LẬP TRÌNH VIÊN", "CHUYÊN VIÊN PHÂN TÍCH DỮ LIỆU", "KỸ SƯ PHẦN MỀM", "GIẢNG VIÊN ĐẠI HỌC"
  Ví dụ sai: "LÀM VIỆC TRONG LĨNH VỰC CÔNG NGHỆ"
- Tạo relationship: (MAJOR)-[:LEADS_TO]->(CAREER) cho mỗi nghề.
- KHÔNG bỏ sót bất kỳ nghề nào được nhắc đến, kể cả nghề phi kỹ thuật như giảng dạy, nghiên cứu.

QUAN TRỌNG về MAJOR:
- name: tên ngành đầy đủ, VIẾT HOA, ví dụ: "CÔNG NGHỆ THÔNG TIN".
- code: mã ngành chính xác từ tài liệu, ví dụ: "7480201". Không được bỏ qua code.

Quan hệ hợp lệ: MAJOR-[OFFERS]->SUBJECT, MAJOR-[LEADS_TO]->CAREER, SUBJECT-[PREREQUISITE_FOR]->SUBJECT.""",

    # ── CAREER DESCRIPTION ───────────────────────────────────────────────────
    "career_description": SYSTEM_PROMPT_BASE + """

Tài liệu này là MÔ TẢ NGHỀ NGHIỆP (career_description).

QUY TẮC TRÍCH XUẤT:
- Bản thân tài liệu đại diện cho DUY NHẤT 1 node CAREER (nghề nghiệp). Bắt buộc phải tạo node CAREER này.
- SKILL: tất cả kỹ năng mà nghề này yêu cầu (cả hard skill lẫn soft skill).
  Lấy từ các trường: required_skills, skills, key_skills, responsibilities, job_description.
- MAJOR: các ngành học được khuyến nghị cho nghề này. Lấy từ trường recommended_majors. MAJOR KHÔNG cần code.
- TUYỆT ĐỐI KHÔNG tạo SUBJECT.
- TUYỆT ĐỐI KHÔNG tạo TEACHER.

QUAN TRỌNG — BẮT BUỘC tạo node CAREER:
- Tên CAREER = tên nghề trong tài liệu, VIẾT HOA.
- Tạo relationship: (CAREER)-[:REQUIRES]->(SKILL) cho từng kỹ năng.
- Tạo relationship: (MAJOR)-[:LEADS_TO]->(CAREER) cho từng ngành được khuyến nghị.
- Tạo relationship: (CAREER)-[:MENTIONED_IN]->(DOCUMENT).

QUAN TRỌNG về MAJOR:
- Chỉ dùng name để định danh MAJOR, KHÔNG có code.
- Tên MAJOR phải VIẾT HOA và phải KHỚP CHÍNH XÁC với tên ngành trong các tài liệu curriculum.
  Ví dụ đúng: "CÔNG NGHỆ THÔNG TIN", "HỆ THỐNG THÔNG TIN QUẢN LÝ", "KỸ THUẬT PHẦN MỀM"
  Ví dụ sai: "CNTT", "Công nghệ thông tin", "IT".""",
}


# ── MinIO helpers ─────────────────────────────────────────────────────────────

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
    print(f"    [debug] bucket='{bucket}' prefix='{prefix}/' → {len(all_names)} objects found")
    if all_names:
        print(f"    [debug] sample: {all_names[:3]}")
    return [o for o in all_names if o.endswith(".json")]


def download_json(client: Minio, bucket: str, object_name: str) -> dict:
    response = client.get_object(bucket, object_name)
    data = json.loads(response.read().decode("utf-8"))
    response.close()
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── LLM extraction ────────────────────────────────────────────────────────────

def extract_via_llm(ai_client: OpenAI, doc_json: dict, docid: str, doctype: str) -> dict:
    user_msg = (
        f"Tài liệu cần trích xuất:\n"
        f"docid: {docid}\ndoctype: {doctype}\n\n"
        f"Nội dung JSON:\n{json.dumps(doc_json, ensure_ascii=False, indent=2)}\n\n"
        f"Trả về JSON hợp lệ theo schema."
    )
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


# ── Worker (chạy trong thread) ────────────────────────────────────────────────

def process_one(minio_client: Minio, ai_client: OpenAI, folder: str, obj_name: str) -> str:
    filename = Path(obj_name).name
    docid    = make_docid(folder, filename)
    out_path = LOCAL_OUT_DIR / folder / f"{docid}.json"

    if out_path.exists():
        print(f"  [skip]  {filename} (đã có)")
        return "skip"

    print(f"  [start] {filename}")
    try:
        doc_json  = download_json(minio_client, MINIO_BUCKET, obj_name)
        extracted = extract_via_llm(ai_client, doc_json, docid, DOCTYPE_MAP[folder])
        save_local(extracted, folder, f"{docid}.json")
        print(f"  [done]  {filename}")
        return "ok"
    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return "error"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_folder(minio_client: Minio, ai_client: OpenAI, folder: str):
    print(f"\n{'='*60}\nProcessing folder: {folder}")
    objects = list_json_objects(minio_client, MINIO_BUCKET, f"{MINIO_BASE_FOLDER}/{folder}")
    if not objects:
        print(f"  No JSON files found in {folder}/")
        return
    print(f"  {len(objects)} files → {MAX_WORKERS} workers song song\n")
    counts = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, minio_client, ai_client, folder, obj): obj
            for obj in objects
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
    print(f"\n  Folder '{folder}' xong: ✓{counts['ok']}  skip={counts['skip']}  ✗{counts['error']}")


def main():
    print("Starting entity extraction pipeline...")
    minio_client = get_minio_client()
    ai_client    = OpenAI(api_key=OPENAI_API_KEY)
    for folder in INPUT_FOLDERS:
        process_folder(minio_client, ai_client, folder)
    print("\n✅ Extraction complete. Results saved to ./cache/output/")


if __name__ == "__main__":
    main()
"""
csv_to_dataset.py
-----------------
Convert CSV exported từ Google Sheet → data/ground_truth/dataset.json

CSV format (tên cột có prefix x/v):
    x id | x query | x query_type | x ground_truth_answer | x gold_entities | x relevant_node_ids
    v gold_path | v gold_hop_count | v expected_communities

    Các list field ngăn cách bằng dấu chấm phẩy: "Sam Altman; OpenAI; GPT-4"

Usage:
    python scripts/csv_to_dataset.py ground_truth.csv
    python scripts/csv_to_dataset.py ground_truth.csv --output data/ground_truth/dataset.json
    python scripts/csv_to_dataset.py ground_truth.csv --validate
"""

import csv
import json
import sys
import argparse
from pathlib import Path


REQUIRED_COLS = ["id", "query", "query_type", "ground_truth_answer", "gold_entities", "relevant_node_ids"]
OPTIONAL_COLS = ["gold_path", "gold_hop_count", "expected_communities", "source_document", "annotator", "notes"]
VALID_QUERY_TYPES = {"single_hop", "multi_hop", "global"}


def clean_col_name(name: str) -> str:
    """
    Chuẩn hóa tên cột:
      - Strip khoảng trắng 2 đầu
      - Bỏ prefix 'x ' hoặc 'v ' (Google Sheet convention)
      - Bỏ emoji và ký tự đặc biệt đầu tên
    Ví dụ: 'x id' → 'id', 'v gold_path' → 'gold_path'
    """
    name = name.strip()
    # Bỏ emoji
    name = name.lstrip("\U0001f7e1\U0001f7e2 ").strip()
    # Bỏ prefix x/v theo convention của Google Sheet
    import re
    name = re.sub(r'^[xv]\s+', '', name).strip()
    return name


def parse_list_field(value: str) -> list:
    """'A; B; C' -> ['A', 'B', 'C']"""
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def is_skip_row(row: dict) -> bool:
    """
    Bỏ qua các dòng không phải data thật:
      - Dòng trống hoàn toàn
      - Dòng comment (bắt đầu bằng #)
      - Dòng ví dụ mẫu (id bắt đầu bằng 'VD:')
    """
    if not any(v.strip() for v in row.values()):
        return True
    row_id = row.get("id", "").strip()
    if row_id.startswith("#") or row_id == "id":
        return True
    if row_id.lower().startswith("vd:") or row_id.lower().startswith("vd"):
        return True
    return False


def convert_row(row: dict, row_num: int):
    """Convert 1 CSV row -> 1 dataset sample. Trả về (sample, errors)."""
    errors = []

    if is_skip_row(row):
        return None, []

    sample = {}

    for col in REQUIRED_COLS:
        val = row.get(col, "").strip()
        if not val and col not in ["id"]:  # id có thể auto-generate
            errors.append(f"Row {row_num}: Thiếu trường bắt buộc '{col}'")
        sample[col] = val

    qtype = sample.get("query_type", "").lower()
    if qtype not in VALID_QUERY_TYPES:
        errors.append(f"Row {row_num}: query_type '{qtype}' không hợp lệ. Phải là: {VALID_QUERY_TYPES}")
    sample["query_type"] = qtype

    sample["gold_entities"]      = parse_list_field(row.get("gold_entities", ""))
    sample["relevant_node_ids"]  = parse_list_field(row.get("relevant_node_ids", ""))
    sample["gold_path"]          = parse_list_field(row.get("gold_path", ""))

    hop_raw = row.get("gold_hop_count", "").strip()
    if hop_raw:
        try:
            sample["gold_hop_count"] = int(hop_raw)
        except ValueError:
            errors.append(f"Row {row_num}: gold_hop_count '{hop_raw}' phải là số nguyên")
            sample["gold_hop_count"] = 0
    else:
        sample["gold_hop_count"] = len(sample["gold_path"]) - 1 if sample["gold_path"] else 0

    sample["expected_communities"] = parse_list_field(row.get("expected_communities", ""))

    if qtype == "global" and not sample["expected_communities"]:
        errors.append(f"Row {row_num} ('{sample['id']}'): Global query nên có expected_communities")
    if qtype in {"single_hop", "multi_hop"} and not sample["gold_path"]:
        errors.append(f"Row {row_num} ('{sample['id']}'): {qtype} query nên có gold_path")

    return sample, errors


def convert_csv_to_dataset(
    csv_path: str,
    output_path: str = "data/ground_truth/dataset.json",
    validate_only: bool = False,
    skip_rows: int = 3,  # Số dòng bỏ qua ở đầu file (title + ghi chú + header mẫu)
) -> list:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File không tồn tại: {csv_path}")
        sys.exit(1)

    samples = []
    all_errors = []
    ids_seen = set()
    auto_id_counter = 1

    with open(csv_path, encoding="utf-8-sig") as f:
        # Đọc toàn bộ lines, bỏ qua các dòng đầu không phải header thật
        all_lines = f.readlines()

    # Tìm dòng header thật (dòng chứa 'id' hoặc 'x id')
    header_line_idx = None
    for i, line in enumerate(all_lines):
        stripped = line.strip().lower()
        if stripped.startswith("x id,") or stripped.startswith("id,"):
            header_line_idx = i
            break

    if header_line_idx is None:
        print("Không tìm thấy dòng header! Kiểm tra lại file CSV.")
        sys.exit(1)

    print(f"   Header found at line {header_line_idx + 1}")

    # Đọc lại từ dòng header
    import io
    csv_content = "".join(all_lines[header_line_idx:])
    reader = csv.DictReader(io.StringIO(csv_content))

    # Clean tên cột
    clean_fieldnames = [clean_col_name(name) for name in (reader.fieldnames or [])]
    reader.fieldnames = clean_fieldnames

    missing_required = set(REQUIRED_COLS) - set(clean_fieldnames)
    if missing_required:
        print(f"CSV thiếu các cột bắt buộc: {missing_required}")
        print(f"Các cột hiện có (sau khi clean): {set(clean_fieldnames)}")
        sys.exit(1)

    for row_num, row in enumerate(reader, start=header_line_idx + 2):
        # Clean key của từng row
        clean_row = {clean_col_name(k): v for k, v in row.items() if k}

        # Bỏ dòng ví dụ mẫu ngay sau header
        if row_num == header_line_idx + 2:
            row_id = clean_row.get("id", "").strip()
            if row_id.lower().startswith("vd") or "q001" in row_id.lower() or "q002" in row_id.lower():
                print(f"   Skipping example row at line {row_num}: '{row_id}'")
                continue

        sample, errors = convert_row(clean_row, row_num)
        if errors:
            all_errors.extend(errors)
        if sample is None:
            continue

        # Auto-generate ID nếu thiếu
        if not sample.get("id"):
            sample["id"] = f"q{auto_id_counter:03d}"
            auto_id_counter += 1

        sid = sample["id"]
        if sid in ids_seen:
            all_errors.append(f"Row {row_num}: ID '{sid}' bị trùng!")
        ids_seen.add(sid)
        samples.append(sample)

    print(f"\n{'='*55}")
    print(f"CSV -> dataset.json Converter")
    print(f"{'='*55}")
    print(f"Input:   {csv_path}")
    print(f"Samples: {len(samples)} rows hợp lệ")

    type_counts = {}
    for s in samples:
        type_counts[s["query_type"]] = type_counts.get(s["query_type"], 0) + 1
    for qtype, count in sorted(type_counts.items()):
        print(f"   {qtype:12s}: {count} samples")

    if all_errors:
        print(f"\n{len(all_errors)} cảnh báo / lỗi:")
        for err in all_errors:
            print(f"   -> {err}")
    else:
        print("\nKhông có lỗi!")

    if validate_only:
        print("\n(Chế độ validate only - không lưu file)")
        return samples

    if not samples:
        print("\nKhông có sample hợp lệ nào để lưu!")
        return []

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Google Sheet CSV -> dataset.json")
    parser.add_argument(
        "csv_file",
        nargs='?',
        default="c:/Users/admin/courses_graphrag_eval_uate/[CoursesAI] GraphRAG GroundTruth - Trang tính2.csv",
        help="[CoursesAI] GraphRAG GroundTruth - Trang tính2.csv (default)"
    )
    parser.add_argument("--output", default="data/ground_truth/dataset.json")
    parser.add_argument("--validate", action="store_true", help="Chỉ validate, không lưu file")
    args = parser.parse_args()
    print(f"Using CSV file: {args.csv_file}")
    convert_csv_to_dataset(args.csv_file, args.output, args.validate)
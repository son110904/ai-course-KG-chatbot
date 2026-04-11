
import os
import re
import json
import uuid
import datetime
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("DB_URL")
NEO4J_USERNAME = os.getenv("DB_USER")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")

MAX_HOPS = int(os.getenv("MAX_HOPS", "3"))
LOG_DIR  = Path("./qa_logs")

# ══════════════════════════════════════════════════════════════════════════════
# CHỈ TIÊU & ĐIỂM CHUẨN TUYỂN SINH 2025 — mapping thủ công từ tài liệu NEU
# ══════════════════════════════════════════════════════════════════════════════

ADMISSION_DATA: list[dict] = [
    # ── Chương trình Đặc biệt (EP) ─────────────────────────────────────────
    {"so": 1,  "ten_chuong_trinh": "Công nghệ Marketing",              "ma_xet_tuyen": "EP19",      "ma_nganh": "7340115", "ten_nganh": "Marketing",                              "khoa_vien": "Khoa Marketing",                                  "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 2,  "ten_chuong_trinh": "Công nghệ Logistics và Quản trị chuỗi cung ứng", "ma_xet_tuyen": "EP20", "ma_nganh": "7460108", "ten_nganh": "Khoa học dữ liệu",            "khoa_vien": "Khoa Khoa học dữ liệu và Trí tuệ nhân tạo",         "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 3,  "ten_chuong_trinh": "Kiểm toán nội bộ",                 "ma_xet_tuyen": "EP21",      "ma_nganh": "7340302", "ten_nganh": "Kiểm toán",                              "khoa_vien": "Viện Kế toán - Kiểm toán",                        "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 4,  "ten_chuong_trinh": "Kinh tế quốc tế (EP)",             "ma_xet_tuyen": "EP22",      "ma_nganh": "7310106", "ten_nganh": "Kinh tế quốc tế",                        "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",              "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 5,  "ten_chuong_trinh": "Kinh tế Y tế",                     "ma_xet_tuyen": "EP24",      "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                                "khoa_vien": "Khoa Kinh tế học",                                "chi_tieu": 40,  "diem_chuan_2025": None},
    {"so": 6,  "ten_chuong_trinh": "Phát triển quốc tế",               "ma_xet_tuyen": "EP25",      "ma_nganh": "7310105", "ten_nganh": "Kinh tế phát triển",                     "khoa_vien": "Khoa Kế hoạch và Phát triển",                     "chi_tieu": 40,  "diem_chuan_2025": None},
    {"so": 7,  "ten_chuong_trinh": "Công nghệ môi trường và phát triển bền vững", "ma_xet_tuyen": "EP26", "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                         "khoa_vien": "Khoa Môi trường, Biến đổi khí hậu và Đô thị",     "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 8,  "ten_chuong_trinh": "Quản trị công nghiệp sáng tạo",    "ma_xet_tuyen": "EP27",      "ma_nganh": "7810101", "ten_nganh": "Du lịch",                                "khoa_vien": "Khoa Du lịch và Khách sạn",                       "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 9,  "ten_chuong_trinh": "Quản trị nhân lực quốc tế",        "ma_xet_tuyen": "EP28",      "ma_nganh": "7340404", "ten_nganh": "Quản trị nhân lực",                      "khoa_vien": "Khoa Kinh tế và Quản lý nguồn nhân lực",          "chi_tieu": 40,  "diem_chuan_2025": None},
    {"so": 10, "ten_chuong_trinh": "Quản trị rủi ro định lượng",        "ma_xet_tuyen": "EP29",      "ma_nganh": "7310108", "ten_nganh": "Toán kinh tế",                           "khoa_vien": "Khoa Toán kinh tế",                               "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 11, "ten_chuong_trinh": "Thẩm định giá (EP)",                "ma_xet_tuyen": "EP31",      "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                      "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 12, "ten_chuong_trinh": "Thống kê và Trí tuệ kinh doanh",   "ma_xet_tuyen": "EP32",      "ma_nganh": "7310107", "ten_nganh": "Thống kê kinh tế",                       "khoa_vien": "Khoa Thống kê",                                   "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 13, "ten_chuong_trinh": "Kinh tế số (dự kiến)",              "ma_xet_tuyen": "EP23",      "ma_nganh": "7310109", "ten_nganh": "Kinh tế số",                             "khoa_vien": "Khoa Hệ thống thông tin quản lý",                 "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 14, "ten_chuong_trinh": "Toán ứng dụng (dự kiến)",           "ma_xet_tuyen": "EP30",      "ma_nganh": "7460112", "ten_nganh": "Toán ứng dụng",                          "khoa_vien": "Khoa Khoa học Cơ sở",                             "chi_tieu": 50,  "diem_chuan_2025": None},
    {"so": 15, "ten_chuong_trinh": "Công nghệ tài chính (dự kiến)",     "ma_xet_tuyen": "7340205",   "ma_nganh": "7340205", "ten_nganh": "Công nghệ tài chính",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                      "chi_tieu": 50,  "diem_chuan_2025": None},
    # ── POHE ───────────────────────────────────────────────────────────────
    {"so": 16, "ten_chuong_trinh": "Quản trị khách sạn (POHE)",         "ma_xet_tuyen": "POHE1",     "ma_nganh": "7810201", "ten_nganh": "Quản trị khách sạn",                     "khoa_vien": "Khoa Du lịch và Khách sạn",                       "chi_tieu": 50,  "diem_chuan_2025": 25.61},
    {"so": 17, "ten_chuong_trinh": "Quản trị lữ hành (POHE)",           "ma_xet_tuyen": "POHE2",     "ma_nganh": "7810103", "ten_nganh": "Quản trị dịch vụ du lịch và lữ hành",    "khoa_vien": "Khoa Du lịch và Khách sạn",                       "chi_tieu": 50,  "diem_chuan_2025": 24.64},
    {"so": 18, "ten_chuong_trinh": "Truyền thông Marketing (POHE)",     "ma_xet_tuyen": "POHE3",     "ma_nganh": "7340115", "ten_nganh": "Marketing",                              "khoa_vien": "Khoa Marketing",                                  "chi_tieu": 60,  "diem_chuan_2025": 27.61},
    {"so": 19, "ten_chuong_trinh": "Luật kinh doanh (POHE)",            "ma_xet_tuyen": "POHE4",     "ma_nganh": "7380107", "ten_nganh": "Luật kinh tế",                           "khoa_vien": "Khoa Luật",                                       "chi_tieu": 50,  "diem_chuan_2025": 25.5},
    {"so": 20, "ten_chuong_trinh": "Quản trị kinh doanh thương mại (POHE)", "ma_xet_tuyen": "POHE5", "ma_nganh": "7340121", "ten_nganh": "Kinh doanh thương mại",                 "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",              "chi_tieu": 50,  "diem_chuan_2025": 26.29},
    {"so": 21, "ten_chuong_trinh": "Quản lý thị trường (POHE)",         "ma_xet_tuyen": "POHE6",     "ma_nganh": "7340121", "ten_nganh": "Kinh doanh thương mại",                  "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",              "chi_tieu": 50,  "diem_chuan_2025": 24.66},
    {"so": 22, "ten_chuong_trinh": "Thẩm định giá (POHE)",              "ma_xet_tuyen": "POHE7",     "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                      "chi_tieu": 50,  "diem_chuan_2025": 24.55},
    # ── E-BBA / EP01-EP18 ──────────────────────────────────────────────────
    {"so": 23, "ten_chuong_trinh": "Quản trị kinh doanh (E-BBA)",       "ma_xet_tuyen": "EBBA",      "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Viện Quản trị Kinh doanh",                        "chi_tieu": 110, "diem_chuan_2025": 25.64},
    {"so": 24, "ten_chuong_trinh": "Khởi nghiệp và phát triển kinh doanh (BBAE)", "ma_xet_tuyen": "EP01", "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",              "khoa_vien": "Viện Đào tạo Quốc tế",                            "chi_tieu": 90,  "diem_chuan_2025": 24.92},
    {"so": 25, "ten_chuong_trinh": "Khoa học tính toán trong Tài chính và Bảo hiểm", "ma_xet_tuyen": "EP02", "ma_nganh": "7310108", "ten_nganh": "Toán kinh tế",               "khoa_vien": "Khoa Toán kinh tế",                               "chi_tieu": 50,  "diem_chuan_2025": 25.5},
    {"so": 26, "ten_chuong_trinh": "Phân tích dữ liệu kinh tế (EDA)",   "ma_xet_tuyen": "EP03",      "ma_nganh": "7310108", "ten_nganh": "Toán kinh tế",                           "khoa_vien": "Khoa Toán kinh tế",                               "chi_tieu": 90,  "diem_chuan_2025": 26.78},
    {"so": 27, "ten_chuong_trinh": "Kế toán tích hợp chứng chỉ quốc tế (ICAEW CFAB)", "ma_xet_tuyen": "EP04", "ma_nganh": "7340301", "ten_nganh": "Kế toán",               "khoa_vien": "Viện Kế toán - Kiểm toán",                        "chi_tieu": 60,  "diem_chuan_2025": 25.9},
    {"so": 28, "ten_chuong_trinh": "Kinh doanh số (E-BDB)",              "ma_xet_tuyen": "EP05",      "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Viện Quản trị Kinh doanh",                        "chi_tieu": 60,  "diem_chuan_2025": 26.4},
    {"so": 29, "ten_chuong_trinh": "Phân tích kinh doanh (BA)",          "ma_xet_tuyen": "EP06",      "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Viện Đào tạo Tiên tiến, Chất lượng cao và POHE",  "chi_tieu": 60,  "diem_chuan_2025": 27.5},
    {"so": 30, "ten_chuong_trinh": "Quản trị điều hành thông minh (E-SOM)", "ma_xet_tuyen": "EP07",  "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Khoa Quản trị kinh doanh",                        "chi_tieu": 70,  "diem_chuan_2025": 25.1},
    {"so": 31, "ten_chuong_trinh": "Quản trị chất lượng và Đổi mới (E-MQI)", "ma_xet_tuyen": "EP08", "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                   "khoa_vien": "Khoa Quản trị kinh doanh",                        "chi_tieu": 70,  "diem_chuan_2025": 24.2},
    {"so": 32, "ten_chuong_trinh": "Công nghệ tài chính và Ngân hàng số", "ma_xet_tuyen": "EP09",    "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                      "chi_tieu": 100, "diem_chuan_2025": 26.29},
    {"so": 33, "ten_chuong_trinh": "Tài chính và Đầu tư (BFI)",          "ma_xet_tuyen": "EP10",      "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                      "chi_tieu": 100, "diem_chuan_2025": 26.27},
    {"so": 34, "ten_chuong_trinh": "Quản trị khách sạn quốc tế (IHME)", "ma_xet_tuyen": "EP11",      "ma_nganh": "7810201", "ten_nganh": "Quản trị khách sạn",                     "khoa_vien": "Khoa Du lịch và Khách sạn",                       "chi_tieu": 50,  "diem_chuan_2025": 24.25},
    {"so": 35, "ten_chuong_trinh": "Kiểm toán tích hợp chứng chỉ quốc tế (ICAEW CFAB)", "ma_xet_tuyen": "EP12", "ma_nganh": "7340302", "ten_nganh": "Kiểm toán",            "khoa_vien": "Viện Kế toán - Kiểm toán",                        "chi_tieu": 60,  "diem_chuan_2025": 27.25},
    {"so": 36, "ten_chuong_trinh": "Kinh tế học tài chính (FE)",         "ma_xet_tuyen": "EP13",      "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                                "khoa_vien": "Khoa Kinh tế học",                                "chi_tieu": 90,  "diem_chuan_2025": 25.41},
    {"so": 37, "ten_chuong_trinh": "Logistics và Quản lý CCU tích hợp chứng chỉ Logistics quốc tế (LSIC)", "ma_xet_tuyen": "EP14", "ma_nganh": "7510605", "ten_nganh": "Logistics và Quản lý chuỗi cung ứng", "khoa_vien": "Viện Thương mại và Kinh tế quốc tế", "chi_tieu": 100, "diem_chuan_2025": 27.69},
    {"so": 38, "ten_chuong_trinh": "Khoa học dữ liệu (EP15)",            "ma_xet_tuyen": "EP15",      "ma_nganh": "7460108", "ten_nganh": "Khoa học dữ liệu",                       "khoa_vien": "Khoa Khoa học dữ liệu và Trí tuệ nhân tạo",        "chi_tieu": 70,  "diem_chuan_2025": 26.13},
    {"so": 39, "ten_chuong_trinh": "Trí tuệ nhân tạo",                   "ma_xet_tuyen": "EP16",      "ma_nganh": "7480107", "ten_nganh": "Trí tuệ nhân tạo",                       "khoa_vien": "Khoa Khoa học dữ liệu và Trí tuệ nhân tạo",        "chi_tieu": 80,  "diem_chuan_2025": 25.44},
    {"so": 40, "ten_chuong_trinh": "Kỹ thuật phần mềm",                  "ma_xet_tuyen": "EP17",      "ma_nganh": "7480103", "ten_nganh": "Kỹ thuật phần mềm",                      "khoa_vien": "Khoa Công nghệ thông tin",                         "chi_tieu": 50,  "diem_chuan_2025": 24.68},
    {"so": 41, "ten_chuong_trinh": "Quản trị giải trí và sự kiện",       "ma_xet_tuyen": "EP18",      "ma_nganh": "7810101", "ten_nganh": "Du lịch",                                "khoa_vien": "Khoa Du lịch và Khách sạn",                       "chi_tieu": 50,  "diem_chuan_2025": 25.89},
    {"so": 42, "ten_chuong_trinh": "Quản lý công và Chính sách (E-PMP)", "ma_xet_tuyen": "EPMP",      "ma_nganh": "7340403", "ten_nganh": "Quản lý công",                           "khoa_vien": "Khoa Khoa học quản lý",                           "chi_tieu": 70,  "diem_chuan_2025": 23.04},
    # ── Hệ Đại trà / Chính quy ────────────────────────────────────────────
    {"so": 57, "ten_chuong_trinh": "Kinh tế học",                         "ma_xet_tuyen": "7310101_1", "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                                "khoa_vien": "Khoa Kinh tế học",                                 "chi_tieu": 50,  "diem_chuan_2025": 26.52},
    {"so": 62, "ten_chuong_trinh": "Kinh tế và quản lý đô thị",           "ma_xet_tuyen": "7310101_2", "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                                "khoa_vien": "Khoa Môi trường, Biến đổi khí hậu và Đô thị",      "chi_tieu": 50,  "diem_chuan_2025": 25.86},
    {"so": 63, "ten_chuong_trinh": "Kinh tế và quản lý nguồn nhân lực",   "ma_xet_tuyen": "7310101_3", "ma_nganh": "7310101", "ten_nganh": "Kinh tế",                                "khoa_vien": "Khoa Kinh tế và Quản lý nguồn nhân lực",           "chi_tieu": 50,  "diem_chuan_2025": 26.79},
    # ── TT1 / TT2 ─────────────────────────────────────────────────────────
    {"so": 84, "ten_chuong_trinh": "Kế toán (TT1)",                       "ma_xet_tuyen": "TT1",       "ma_nganh": "7340301", "ten_nganh": "Kế toán",                                "khoa_vien": "Viện Kế toán - Kiểm toán",                         "chi_tieu": 55,  "diem_chuan_2025": 24.75},
    {"so": 85, "ten_chuong_trinh": "Kế hoạch tài chính (TT1)",            "ma_xet_tuyen": "TT1",       "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                       "chi_tieu": 55,  "diem_chuan_2025": 24.75},
    {"so": 86, "ten_chuong_trinh": "Quản trị kinh doanh (TT1)",           "ma_xet_tuyen": "TT1",       "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Khoa Quản trị kinh doanh",                         "chi_tieu": 55,  "diem_chuan_2025": 24.75},
    {"so": 87, "ten_chuong_trinh": "Tài chính (TT2)",                     "ma_xet_tuyen": "TT2",       "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                       "chi_tieu": 220, "diem_chuan_2025": 25.5},
    {"so": 88, "ten_chuong_trinh": "Kinh doanh quốc tế (TT2)",            "ma_xet_tuyen": "TT2",       "ma_nganh": "7340120", "ten_nganh": "Kinh doanh quốc tế",                     "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",               "chi_tieu": 110, "diem_chuan_2025": 25.5},
    # ── CLC ───────────────────────────────────────────────────────────────
    {"so": 89, "ten_chuong_trinh": "Kinh tế phát triển (CLC1)",            "ma_xet_tuyen": "CLC1",      "ma_nganh": "7310105", "ten_nganh": "Kinh tế phát triển",                     "khoa_vien": "Khoa Kế hoạch và Phát triển",                      "chi_tieu": 55,  "diem_chuan_2025": 25.25},
    {"so": 90, "ten_chuong_trinh": "Ngân hàng (CLC1)",                    "ma_xet_tuyen": "CLC1",       "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                       "chi_tieu": 55,  "diem_chuan_2025": 25.25},
    {"so": 91, "ten_chuong_trinh": "Công nghệ thông tin và chuyển đổi số (CLC1)", "ma_xet_tuyen": "CLC1", "ma_nganh": "7480201", "ten_nganh": "Công nghệ thông tin",               "khoa_vien": "Khoa Công nghệ thông tin",                          "chi_tieu": 55,  "diem_chuan_2025": 25.25},
    {"so": 92, "ten_chuong_trinh": "Bảo hiểm tích hợp chứng chỉ ANZIIF (CLC1)", "ma_xet_tuyen": "CLC1", "ma_nganh": "7340204", "ten_nganh": "Bảo hiểm",                           "khoa_vien": "Khoa Bảo hiểm",                                    "chi_tieu": 55,  "diem_chuan_2025": 25.25},
    {"so": 93, "ten_chuong_trinh": "Kinh tế đầu tư (CLC2)",               "ma_xet_tuyen": "CLC2",      "ma_nganh": "7310104", "ten_nganh": "Kinh tế đầu tư",                         "khoa_vien": "Khoa Đầu tư",                                      "chi_tieu": 160, "diem_chuan_2025": 26.5},
    {"so": 94, "ten_chuong_trinh": "Quản trị nhân lực (CLC2)",             "ma_xet_tuyen": "CLC2",      "ma_nganh": "7340404", "ten_nganh": "Quản trị nhân lực",                      "khoa_vien": "Khoa Kinh tế và Quản lý nguồn nhân lực",           "chi_tieu": 160, "diem_chuan_2025": 26.5},
    {"so": 95, "ten_chuong_trinh": "Quản trị kinh doanh (CLC2)",           "ma_xet_tuyen": "CLC2",      "ma_nganh": "7340101", "ten_nganh": "Quản trị kinh doanh",                    "khoa_vien": "Khoa Quản trị kinh doanh",                         "chi_tieu": 105, "diem_chuan_2025": 26.5},
    {"so": 96, "ten_chuong_trinh": "Quan hệ công chúng (CLC2)",            "ma_xet_tuyen": "CLC2",      "ma_nganh": "7320108", "ten_nganh": "Quan hệ công chúng",                     "khoa_vien": "Khoa Marketing",                                   "chi_tieu": 160, "diem_chuan_2025": 26.5},
    {"so": 97, "ten_chuong_trinh": "Tài chính doanh nghiệp (CLC3)",        "ma_xet_tuyen": "CLC3",      "ma_nganh": "7340201", "ten_nganh": "Tài chính Ngân hàng",                    "khoa_vien": "Viện Ngân hàng - Tài chính",                       "chi_tieu": 325, "diem_chuan_2025": 26.42},
    {"so": 98, "ten_chuong_trinh": "Marketing số (CLC3)",                  "ma_xet_tuyen": "CLC3",      "ma_nganh": "7340115", "ten_nganh": "Marketing",                              "khoa_vien": "Khoa Marketing",                                   "chi_tieu": 270, "diem_chuan_2025": 26.42},
    {"so": 99, "ten_chuong_trinh": "Quản trị Marketing (CLC3)",            "ma_xet_tuyen": "CLC3",      "ma_nganh": "7340115", "ten_nganh": "Marketing",                              "khoa_vien": "Khoa Marketing",                                   "chi_tieu": 165, "diem_chuan_2025": 26.42},
    {"so": 100,"ten_chuong_trinh": "Quản trị kinh doanh quốc tế (CLC3)",  "ma_xet_tuyen": "CLC3",      "ma_nganh": "7340120", "ten_nganh": "Kinh doanh quốc tế",                     "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",               "chi_tieu": 270, "diem_chuan_2025": 26.42},
    {"so": 101,"ten_chuong_trinh": "Kinh tế quốc tế (CLC3)",               "ma_xet_tuyen": "CLC3",      "ma_nganh": "7310106", "ten_nganh": "Kinh tế quốc tế",                        "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",               "chi_tieu": 270, "diem_chuan_2025": 26.42},
    {"so": 102,"ten_chuong_trinh": "Logistics và quản lý chuỗi cung ứng (CLC3)", "ma_xet_tuyen": "CLC3", "ma_nganh": "7510605", "ten_nganh": "Logistics và Quản lý chuỗi cung ứng",  "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",               "chi_tieu": 165, "diem_chuan_2025": 26.42},
    {"so": 103,"ten_chuong_trinh": "Thương mại điện tử (CLC3)",             "ma_xet_tuyen": "CLC3",      "ma_nganh": "7340122", "ten_nganh": "Thương mại điện tử",                     "khoa_vien": "Viện Thương mại và Kinh tế quốc tế",               "chi_tieu": 165, "diem_chuan_2025": 26.42},
    {"so": 104,"ten_chuong_trinh": "Kiểm toán tích hợp chứng chỉ ACCA (CLC3)", "ma_xet_tuyen": "CLC3", "ma_nganh": "7340302", "ten_nganh": "Kiểm toán",                             "khoa_vien": "Viện Kế toán - Kiểm toán",                         "chi_tieu": 270, "diem_chuan_2025": 26.42},
]

# ── Pattern nhận diện câu hỏi về chỉ tiêu / điểm chuẩn ───────────────────────
_ADMISSION_PATTERN = re.compile(
    r"chỉ\s*tiêu"
    r"|điểm\s*chuẩn"
    r"|điểm\s*đầu\s*vào"
    r"|tuyển\s*sinh"
    r"|xét\s*tuyển"
    r"|mã\s*xét\s*tuyển"
    r"|POHE|CLC[123]|TT[12]|E-BBA|EP\d+",
    re.IGNORECASE | re.UNICODE,
)


def search_admission_data(question: str) -> list[dict]:
    """
    Tìm chương trình trong ADMISSION_DATA khớp với câu hỏi.
    Chiến lược:
      1. Khớp mã xét tuyển tường minh (EP19, POHE1, CLC1...)
      2. Khớp mã ngành 7 chữ số
      3. Trích xuất cụm tên ngành từ câu hỏi (bỏ từ ngữ cảnh),
         rồi so khớp trực tiếp substring với ten_chuong_trinh / ten_nganh
      4. Fallback: scoring từng từ nếu không có match trực tiếp
    """
    q_lower = question.lower()
    is_broad_program_request = bool(re.search(
        r"t[aấ]t\s*c[aả]|to[aà]n\s*b[ộo]|li[eê]t\s*k[eê]|danh\s*s[aá]ch|c[aá]c\s*chương\s*trình|c[aá]c\s*hệ",
        q_lower,
        re.IGNORECASE | re.UNICODE,
    ))

    # 1. Khớp mã xét tuyển tường minh
    ma_xt_pattern = re.compile(r"\b(EP\d+|POHE\d*|EBBA|EPMP|CLC[123]|TT[12])\b", re.IGNORECASE)
    found_codes = [m.group(1).upper() for m in ma_xt_pattern.finditer(question)]
    if found_codes:
        results = [e for e in ADMISSION_DATA if e["ma_xet_tuyen"].upper() in found_codes]
        if results:
            return results

    # 2. Khớp mã ngành 7 chữ số
    results = [e for e in ADMISSION_DATA if re.search(r"\b" + re.escape(e["ma_nganh"]) + r"\b", question)]
    if results:
        return results

    # 3. Trích xuất cụm tên ngành bằng cách bỏ từ ngữ cảnh
    CONTEXT_PAT = re.compile(
        r"điểm\s*chuẩn|điểm\s*đầu\s*vào|chỉ\s*tiêu|tuyển\s*sinh"
        r"|chương\s*trình\s*đào\s*tạo|chương\s*trình|đào\s*tạo"
        r"|ngành\s*học|ngành|ctđt|năm\s*\d{4}|\b20\d{2}\b"
        r"|bao\s*nhiêu|của\s*trường|tại\s*neu|tại\s*trường"
        r"|\blà\s*bao\s*nhiêu\b|\blà\s*gì\b|\blà\b"
        r"|\bnhư\s*thế\s*nào\b|\bthế\s*nào\b|\bcủa\b|\bcó\b"
        r"|\bcho\s*tôi\s*biết\b|\bcho\s*biết\b|\bxin\s*hỏi\b|\bhỏi\b",
        re.IGNORECASE | re.UNICODE,
    )
    term = CONTEXT_PAT.sub(" ", q_lower)
    term = re.sub(r"\s+", " ", term).strip()

    if not term:
        return []

    # Helper: chuẩn hóa chuỗi để so khớp (bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng)
    def normalize(s: str) -> str:
        s = re.sub(r"[\(\)\[\]\-_/\\,\.]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # 3a. So khớp trực tiếp: term là substring của ten_chuong_trinh hoặc ten_nganh
    term_norm = normalize(term)  # normalize term để so khớp nhất quán
    exact_matches = []
    for entry in ADMISSION_DATA:
        name_norm  = normalize(entry["ten_chuong_trinh"].lower())
        nganh_norm = normalize(entry["ten_nganh"].lower())
        if term_norm in name_norm or term_norm in nganh_norm:
            exact_matches.append(entry)

    if exact_matches:
        # Tách thành: khớp chính xác tên vs khớp substring
        name_hits = [e for e in exact_matches
                     if term_norm in normalize(e["ten_chuong_trinh"].lower())]
        if name_hits:
            # Kiểm tra xem có entry nào khớp chính xác hoàn toàn không
            exact_name_hits = [e for e in name_hits
                               if term_norm == normalize(e["ten_chuong_trinh"].lower())]
            if exact_name_hits:
                # Nếu chỉ có 1 chương trình trùng tên chính xác → trả luôn
                if len(exact_name_hits) == 1:
                    return exact_name_hits
                # Nếu có nhiều chương trình trùng tên chính xác (ví dụ các biến thể TT1/CLC...)
                # → trả tất cả để người dùng thấy đầy đủ
                return exact_name_hits
            # Không có exact, nhưng có name_hits (substring):
            # Trả tất cả name_hits để người dùng thấy các biến thể
            return name_hits
        # Fallback: khớp qua ten_nganh, lọc bỏ chương trình không liên quan
        term_words = set(term.split())
        nganh_hits = [
            e for e in exact_matches
            if any(w in normalize(e["ten_chuong_trinh"].lower()) for w in term_words)
        ]
        return nganh_hits if nganh_hits else exact_matches

    # 3b. Fallback: scoring từng từ (cho trường hợp câu hỏi viết tắt hoặc sai dấu nhẹ)
    words = [w for w in term.split() if len(w) >= 2]
    if not words:
        return []

    scored = []
    for entry in ADMISSION_DATA:
        name_lower  = entry["ten_chuong_trinh"].lower()
        nganh_lower = entry["ten_nganh"].lower()
        name_score = 0
        nganh_score = 0

        for length in range(min(len(words), 6), 1, -1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i + length])
                if phrase in name_lower:
                    name_score += length * 5
                if phrase in nganh_lower:
                    nganh_score += length * 2

        for w in words:
            if w in name_lower:
                name_score += 1
            if w in nganh_lower:
                nganh_score += 1

        total = name_score + nganh_score
        if total > 0:
            scored.append((name_score, total, entry))

    if not scored:
        return []

    has_name_match = any(ns > 0 for ns, _, _ in scored)
    if has_name_match:
        scored = [(ns, tot, e) for ns, tot, e in scored if ns > 0]
        max_score = max(ns for ns, _, _ in scored)
        filtered = [e for ns, _, e in scored if ns >= max_score * 0.75]
    else:
        max_score = max(tot for _, tot, _ in scored)
        filtered = [e for _, tot, e in scored if tot >= max_score * 0.75]

    seen = set()
    unique = []
    for e in filtered:
        key = e["ma_xet_tuyen"] + e["ma_nganh"]
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique




def format_admission_answer(question: str, programs: list[dict]) -> str:
    """
    Trả lời về chỉ tiêu / điểm chuẩn — hoàn toàn bằng văn xuôi, không bảng.
    """
    if not programs:
        return (
            "Hiện chưa tìm thấy chương trình phù hợp với câu hỏi của bạn. "
            "Bạn có thể xem thêm tại tuyensinh.neu.edu.vn."
        )

    q_lower = question.lower()
    want_diem    = bool(re.search(r"điểm.{0,5}chuẩn|điểm.{0,5}đầu vào", q_lower))
    want_chitieu = bool(re.search(r"chỉ.{0,3}tiêu", q_lower))

    def one_line(p: dict) -> str:
        ten      = p["ten_chuong_trinh"]
        ma       = p["ma_nganh"]
        ct       = p["chi_tieu"]
        dc       = p["diem_chuan_2025"]
        diem_str = str(dc) if dc is not None else None
        if want_diem and not want_chitieu:
            if diem_str is None:
                return f"- **{ten}** (mã {ma}): chương trình này chưa cập nhật điểm chuẩn"
            return f"- **{ten}** (mã {ma}): điểm chuẩn 2025 là **{diem_str}**"
        if want_chitieu and not want_diem:
            return f"- **{ten}** (mã {ma}): chỉ tiêu **{ct} sinh viên**"
        # Hỏi cả hai hoặc tổng quát
        diem_part = f"**{diem_str}**" if diem_str is not None else "chưa cập nhật"
        return f"- **{ten}** (mã {ma}): chỉ tiêu **{ct} sinh viên**, điểm chuẩn 2025: {diem_part}"

    if len(programs) == 1:
        p = programs[0]
        ten      = p["ten_chuong_trinh"]
        ma       = p["ma_nganh"]
        ct       = p["chi_tieu"]
        dc       = p["diem_chuan_2025"]
        diem_str = str(dc) if dc is not None else None
        if want_diem and not want_chitieu:
            if diem_str is None:
                return f"Chương trình **{ten}** (mã ngành {ma}) chưa cập nhật điểm chuẩn."
            return f"Chương trình **{ten}** (mã ngành {ma}) có điểm chuẩn 2026 là **{diem_str}**."
        if want_chitieu and not want_diem:
            return f"Chương trình **{ten}** (mã ngành {ma}) có chỉ tiêu tuyển sinh 2026 là **{ct} sinh viên**."
        # Hỏi cả hai hoặc tổng quát
        diem_part = f"**{diem_str}**" if diem_str is not None else "chưa cập nhật"
        return (f"Chương trình **{ten}** (mã ngành {ma}): "
                f"chỉ tiêu **{ct} sinh viên**, điểm chuẩn 2025: {diem_part}.")

    # Nhiều chương trình → liệt kê văn xuôi
    lines = ["Dưới đây là thông tin tuyển sinh các chương trình phù hợp:"]
    for p in programs:
        lines.append(one_line(p))
    return "\n".join(lines)



def _extract_admission_term(question: str) -> tuple[str, str]:
    """
    Trích xuất (term_clean, ma_nganh_7chu) từ câu hỏi tuyển sinh.
    Returns: (term, code_7digit_or_empty)
    """
    # Tìm mã ngành 7 chữ số tường minh
    code_match = re.search(r"\b(7\d{6})\b", question)
    code = code_match.group(1) if code_match else ""

    CONTEXT_PAT = re.compile(
        r"điểm\s*chuẩn|điểm\s*đầu\s*vào|chỉ\s*tiêu|tuyển\s*sinh"
        r"|chương\s*trình\s*đào\s*tạo|chương\s*trình|đào\s*tạo"
        r"|ngành\s*học|ngành|ctđt|năm\s*\d{4}|\b20\d{2}\b"
        r"|bao\s*nhiêu|của\s*trường|tại\s*neu|tại\s*trường"
        r"|\blà\s*bao\s*nhiêu\b|\blà\s*gì\b|\blà\b"
        r"|\bnhư\s*thế\s*nào\b|\bthế\s*nào\b|\bcủa\b|\bcó\b"
        r"|\bcho\s*tôi\s*biết\b|\bcho\s*biết\b|\bxin\s*hỏi\b|\bhỏi\b",
        re.IGNORECASE | re.UNICODE,
    )
    term = CONTEXT_PAT.sub(" ", question.lower())
    term = re.sub(r"\s+", " ", term).strip()
    return term, code


def query_neo4j_major_admission(driver, question: str) -> list[dict]:
    """
    Query Neo4j để lấy diem_chuan / chi_tieu từ MAJOR node.
    Trả về list dict tương thích format_admission_answer (có key ten_chuong_trinh,
    ma_nganh, chi_tieu, diem_chuan_2025, khoa_vien).
    Chỉ trả về kết quả khi node thực sự có ít nhất 1 trong 2 trường này.
    """
    term, code = _extract_admission_term(question)

    # Bỏ mã EP/POHE/CLC/TT (chương trình đặc biệt) — những này không có node MAJOR riêng
    special_code_pat = re.compile(r"\b(EP\d+|POHE\d*|EBBA|EPMP|CLC[123]|TT[12])\b", re.IGNORECASE)
    if special_code_pat.search(question):
        return []   # fallback sang ADMISSION_DATA

    cypher = """
        MATCH (m:MAJOR)
        WHERE (
            ($code <> '' AND m.code STARTS WITH $code)
            OR ($term <> '' AND (
                toLower(m.name)    CONTAINS toLower($term)
                OR toLower(m.name_vi) CONTAINS toLower($term)
            ))
        )
        AND (m.diem_chuan IS NOT NULL OR m.chi_tieu IS NOT NULL)
        RETURN m.name      AS name,
               m.name_vi   AS name_vi,
               m.code      AS code,
               m.diem_chuan AS diem_chuan,
               m.chi_tieu   AS chi_tieu,
               m.khoa_vien  AS khoa_vien
        ORDER BY m.code
        LIMIT 10
    """
    try:
        with driver.session() as session:
            rows = session.run(cypher, term=term, code=code).data()
    except Exception:
        return []

    results = []
    for r in rows:
        name = r.get("name_vi") or r.get("name") or ""
        results.append({
            "ten_chuong_trinh": name,
            "ma_nganh":         r.get("code", ""),
            "chi_tieu":         r.get("chi_tieu"),
            "diem_chuan_2025":  r.get("diem_chuan"),
            "khoa_vien":        r.get("khoa_vien", ""),
            "_source":          "neo4j",
        })
    return results

def handle_admission_question(question: str, driver=None) -> str | None:
    """
    Nếu câu hỏi liên quan đến chỉ tiêu/điểm chuẩn → trả về answer string.
    Ngược lại trả về None để pipeline tiếp tục xử lý bình thường.

    Ưu tiên: Neo4j (diem_chuan / chi_tieu lưu trên MAJOR node)
    Fallback: ADMISSION_DATA (bảng mapping thủ công — dùng cho EP/POHE/CLC/TT)
    """
    if not _ADMISSION_PATTERN.search(question):
        return None

    q_lower = question.lower()
    is_general = bool(re.search(
        r"t[aấ]t\s*c[aả]|to[àa]n\s*b[ộo]|c[aá]c\s*ng[àa]nh|danh\s*s[aá]ch|li[eê]t\s*k[eê]",
        q_lower,
    ))
    if is_general:
        return (
            "NEU có hơn 100 chương trình đào tạo. "
            "Bạn có thể hỏi cụ thể từng ngành để tôi tra chỉ tiêu và điểm chuẩn, "
            "hoặc xem toàn bộ danh sách tại tuyensinh.neu.edu.vn."
        )

    # ── Bước 1: Thử Neo4j trước (chính quy — diem_chuan / chi_tieu trên node MAJOR) ──
    if driver is not None:
        neo4j_programs = query_neo4j_major_admission(driver, question)
        if neo4j_programs:
            return format_admission_answer(question, neo4j_programs)

    # ── Bước 2: Fallback sang ADMISSION_DATA (EP/POHE/CLC/TT/sub-program) ──
    programs = search_admission_data(question)
    if not programs:
        return (
            "Hiện chưa tìm thấy thông tin tuyển sinh cho ngành bạn hỏi. "
            "Bạn có thể xem thêm tại tuyensinh.neu.edu.vn."
        )

    return format_admission_answer(question, programs)


# ── Danh sách môn đại cương bắt buộc chung mọi ngành ────────────────────────
# Các môn này KHÔNG được gợi ý khi người dùng hỏi "nên học môn gì".
# Nhưng nếu hỏi "ngành X có học môn Y không?" thì vẫn xác nhận là CÓ.
EXCLUDED_SUBJECT_CODES = {
    "llnl1105", "llnl1106", "llnl1107", "lldl1102", "lltt1101",
    "khmi1101", "khma1101", "lucs1129",
}
EXCLUDED_SUBJECT_NAMES_PATTERNS = re.compile(
    r"tri\s*[eé]t\s*h[oọ]c\s*m[aá]c[\s\-]*l[eê][- ]?nin"
    r"|kinh\s*t[eế]\s*ch[ií]nh\s*tr[ij]\s*m[aá]c[\s\-]*l[eê][- ]?nin"
    r"|ch[uủ]\s*ngh[iĩ]a\s*x[aã]\s*h[oộ]i\s*khoa\s*h[oọ]c"
    r"|l[iị]ch\s*s[uử]\s*[dđ][aả]ng\s*c[oộ]ng\s*s[aả]n\s*vi[eệ]t\s*nam"
    r"|t[uư]\s*t[uư][oở]ng\s*h[oồ]\s*ch[ií]\s*minh"
    r"|gi[aá]o\s*d[uụ]c\s*th[eể]\s*ch[aấ]t"
    r"|gi[aá]o\s*d[uụ]c\s*qu[oố]c\s*ph[oò]ng"
    r"|gdtc|gdqp"
    r"|kinh\s*t[eế]\s*vi\s*m[oô]\s*1"
    r"|kinh\s*t[eế]\s*v[iĩ]\s*m[oô]\s*1"
    r"|ph[aá]p\s*lu[aậ]t\s*[dđ][aạ]i\s*c[uư][oơ]ng",
    re.IGNORECASE | re.UNICODE,
)

_F = re.IGNORECASE | re.UNICODE  # shorthand
_EXCLUDED_SUBJECT_KEYWORD_MAP: list[tuple[re.Pattern, str]] = [
    # Triết học — user có thể gõ "triết", "triet", "Triết học"
    (re.compile(r"tri[eếệề]t|triet", _F), "Triết học Mác-Lênin (LLNL1105)"),
    # Kinh tế chính trị
    (re.compile(r"kinh\s*t[eế]\s*ch[ií]nh\s*tr[ịi]|ktct", _F),
     "Kinh tế chính trị Mác-Lênin (LLNL1106)"),
    # Chủ nghĩa xã hội khoa học
    (re.compile(r"ch[uủ]\s*ngh[iĩ]a\s*x[aã]\s*h[oộ]i|cnxhkh", _F),
     "Chủ nghĩa xã hội khoa học (LLNL1107)"),
    # Lịch sử Đảng
    (re.compile(r"l[iị]ch\s*s[uử]\s*[dđ][aả]ng|lsd\b", _F),
     "Lịch sử Đảng Cộng sản Việt Nam (LLDL1102)"),
    # Tư tưởng Hồ Chí Minh
    (re.compile(r"t[uư]\s*t[uư][oở]ng\s*h[oồ]\s*ch[ií]\s*minh|tthcm", _F),
     "Tư tưởng Hồ Chí Minh (LLTT1101)"),
    # Giáo dục thể chất
    (re.compile(r"gdtc|gi[aá]o\s*d[uụ]c\s*th[eể]\s*ch[aấ]t", _F),
     "Giáo dục thể chất (GDTC)"),
    # Giáo dục quốc phòng
    (re.compile(r"gdqp|gi[aá]o\s*d[uụ]c\s*qu[oố]c\s*ph[oò]ng", _F),
     "Giáo dục quốc phòng và an ninh (GDQP)"),
    # Kinh tế vi mô 1 — phải đặt TRƯỚC vĩ mô để tránh overlap
    (re.compile(r"kinh\s*t[eế]\s*vi\s*m[oô]|vi\s*m[oô]\s*1", _F),
     "Kinh tế vi mô 1 (KHMI1101)"),
    # Kinh tế vĩ mô 1
    (re.compile(r"kinh\s*t[eế]\s*v[iĩ]\s*m[oô]|v[iĩ]\s*m[oô]\s*1", _F),
     "Kinh tế vĩ mô 1 (KHMA1101)"),
    # Pháp luật đại cương
    (re.compile(r"ph[aá]p\s*lu[aậ]t\s*[dđ][aạ]i\s*c[uư][oơ]ng", _F),
     "Pháp luật đại cương (LUCS1129)"),
]

# Pattern nhận diện câu hỏi dạng "ngành nào không (cần) học [môn]"
_WHICH_MAJOR_NOT_STUDY_PATTERN = re.compile(
    r"ng[àa]nh\s*(n[àa]o)?.{0,20}(kh[oô]ng|ko|ch[aẳ]ng)\s*(c[aầ]n\s*)?"
    r"(h[oọ]c|d[aạ]y|ph[aả]i\s*h[oọ]c)",
    re.IGNORECASE | re.UNICODE,
)


def handle_which_major_not_study(question: str) -> str | None:
    """
    Nếu user hỏi "ngành nào không học [môn đại cương bắt buộc]",
    trả về câu trả lời cứng vì các môn này bắt buộc toàn trường.
    Ngược lại trả về None để pipeline xử lý bình thường.
    """
    if not _WHICH_MAJOR_NOT_STUDY_PATTERN.search(question):
        return None

    # Kiểm tra xem môn được hỏi có phải môn đại cương bắt buộc không
    matched_subjects: list[str] = []
    for pattern, display_name in _EXCLUDED_SUBJECT_KEYWORD_MAP:
        if pattern.search(question):
            matched_subjects.append(display_name)

    if not matched_subjects:
        return None  # Hỏi về môn khác → để pipeline xử lý

    subjects_str = ", ".join(matched_subjects)
    return (
        f"Hiện tại không có ngành nào không học {subjects_str}. "
        f"Đây là môn học bắt buộc chung cho tất cả các ngành tại NEU."
    )


# Pattern để nhận diện câu hỏi "nên học môn gì" (câu hỏi gợi ý môn)
_RECOMMEND_SUBJECT_PATTERN = re.compile(
    r"n[eê]n\s+h[oọ]c\s+m[oô]n\s+g[iì]"
    r"|g[oợ]i\s+[yý]\s+m[oô]n"
    r"|m[oô]n\s+n[aà]o\s+n[eê]n\s+h[oọ]c"
    r"|ch[oọ]n\s+m[oô]n\s+h[oọ]c"
    r"|[dđ][aă]ng\s+k[yý]\s+m[oô]n\s+g[iì]"
    r"|n[eê]n\s+[dđ][aă]ng\s+k[yý]\s+m[oô]n\s+n[aà]o"
    r"|m[oô]n\s+t[uự]\s+ch[oọ]n\s+n[aà]o",
    re.IGNORECASE | re.UNICODE,
)


def is_recommend_subject_question(question: str) -> bool:
    """Trả về True nếu câu hỏi là hỏi gợi ý / nên học môn gì."""
    return bool(_RECOMMEND_SUBJECT_PATTERN.search(question))


def filter_excluded_subjects(nodes: list[dict], exclude: bool) -> list[dict]:
    """
    Nếu exclude=True: loại bỏ các SUBJECT node thuộc danh sách đại cương bắt buộc.
    Nếu exclude=False: giữ nguyên toàn bộ (dùng cho câu hỏi xác nhận sự tồn tại).
    """
    if not exclude:
        return nodes
    result = []
    for n in nodes:
        if n.get("label") != "SUBJECT":
            result.append(n)
            continue
        code = (n.get("code") or "").lower().strip()
        name = (n.get("name") or "").lower().strip()
        if code in EXCLUDED_SUBJECT_CODES:
            continue
        if EXCLUDED_SUBJECT_NAMES_PATTERNS.search(name):
            continue
        result.append(n)
    return result


# Pattern nhận diện câu hỏi XÁC NHẬN sự tồn tại môn đại cương ("ngành X có học môn Y không?")
_CONFIRM_SUBJECT_PATTERN = re.compile(
    r"c[oó]\s+h[oọ]c\s+m[oô]n"
    r"|c[oó]\s+d[aạ]y\s+m[oô]n"
    r"|m[oô]n\s+.{0,40}\s+c[oó]\s+kh[oô]ng"
    r"|c[oó]\s+m[oô]n\s+.{0,40}\s+kh[oô]ng",
    re.IGNORECASE | re.UNICODE,
)



COMMUNITY_LEVELS: dict[str, dict] = {

    "L1_GLOBAL": {
        "id":          "L1_GLOBAL",
        "level":       1,
        "name":        "Hệ sinh thái Đào tạo & Nghề nghiệp",
        "node_labels": {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER", "PERSONALITY"},
        "purpose": (
            "Trả lời câu hỏi chiến lược: xu hướng đào tạo, liên kết toàn diện "
            "giữa chương trình học và thị trường lao động."
        ),
    },

    "L2_ACADEMIC": {
        "id":          "L2_ACADEMIC",
        "level":       2,
        "name":        "Cụm Học thuật (Academic Cluster)",
        "node_labels": {"MAJOR", "SUBJECT", "TEACHER"},
        "purpose": (
            "Trả lời về chương trình ngành, môn học, giảng viên phụ trách. "
            "Kết nối Teacher ↔ Subject ↔ Major."
        ),
    },

    "L2_CAREER_ALIGNMENT": {
        "id":          "L2_CAREER_ALIGNMENT",
        "level":       2,
        "name":        "Cụm Năng lực & Việc làm (Career Alignment Cluster)",
        "node_labels": {"SKILL", "CAREER", "SUBJECT", "PERSONALITY"},
        "purpose": (
            "Kết nối đầu ra môn học (Subject→Skill) với yêu cầu thực tế (Career→Skill). "
            "Trả lời về kỹ năng cần thiết, môn học liên quan đến nghề nghiệp. "
            "Bao gồm cả phẩm chất nhân cách nghề yêu cầu (Career→Personality)."
        ),
    },

    "L2_PERSONALITY_FIT": {
        "id":          "L2_PERSONALITY_FIT",
        "level":       2,
        "name":        "Cụm Tính cách MBTI & Ngành/Nghề (Personality Fit Cluster)",
        "node_labels": {"PERSONALITY", "CAREER", "MAJOR"},
        "purpose": (
            "Gợi ý ngành học và nghề nghiệp phù hợp với loại tính cách MBTI. "
            "Kích hoạt khi câu hỏi nhắc tới MBTI code (ESTP, ENTP...), "
            "'tính cách', 'hướng nội/hướng ngoại', 'hợp với nghề gì'. "
            "CŨNG xử lý câu hỏi ngược: 'tính cách gì hợp làm/học X' — "
            "tìm PERSONALITY phù hợp với ngành/lĩnh vực X qua suitable_fields hoặc SUITS_MAJOR/SUITS_CAREER."
        ),
    },

    "L3_MAJOR_CENTRIC": {
        "id":          "L3_MAJOR_CENTRIC",
        "level":       3,
        "name":        "Cộng đồng theo Ngành (Major-centric)",
        # community_L3: SUBJECT=0, TEACHER=1, SKILL=2 — không đồng nhất, dùng label filter
        "node_labels": {"SUBJECT", "TEACHER", "SKILL"},
        "purpose": (
            "Chi tiết lộ trình một ngành cụ thể: môn học, giảng viên, kỹ năng đầu ra. "
            "Kích hoạt khi câu hỏi nhắc tới Major Code cụ thể."
        ),
    },

    "L3_SKILL_CENTRIC": {
        "id":          "L3_SKILL_CENTRIC",
        "level":       3,
        "name":        "Cộng đồng theo Kỹ năng (Skill-centric)",
        "node_labels": {"SUBJECT", "CAREER"},
        "purpose": (
            "Giá trị của một kỹ năng cụ thể: môn nào dạy + nghề nào yêu cầu. "
            "Kích hoạt khi câu hỏi nhắc tới Skill cụ thể."
        ),
    },
}

# ── Ánh xạ intent → community ID ─────────────────────────────────────────────
INTENT_TO_COMMUNITY: dict[tuple, str] = {
    # Academic cluster
    ("MAJOR",   "SUBJECT"):  "L2_ACADEMIC",
    ("MAJOR",   "TEACHER"):  "L2_ACADEMIC",
    ("SUBJECT", "TEACHER"):  "L2_ACADEMIC",
    ("TEACHER", "SUBJECT"):  "L2_ACADEMIC",
    ("TEACHER", "MAJOR"):    "L2_ACADEMIC",
    # Self-queries học thuật
    ("SUBJECT", "SUBJECT"):  "L2_ACADEMIC",
    ("TEACHER", "TEACHER"):  "L2_ACADEMIC",
    ("MAJOR",   "MAJOR"):    "L1_GLOBAL",

    # Career cluster
    ("MAJOR",   "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("MAJOR",   "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("CAREER",  "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("CAREER",  "SUBJECT"):  "L2_CAREER_ALIGNMENT",
    ("CAREER",  "MAJOR"):    "L2_CAREER_ALIGNMENT",
    ("SKILL",   "MAJOR"):    "L2_CAREER_ALIGNMENT",
    ("SKILL",   "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("SKILL",   "SUBJECT"):  "L2_CAREER_ALIGNMENT",
    ("SUBJECT", "SKILL"):    "L2_CAREER_ALIGNMENT",
    ("SUBJECT", "CAREER"):   "L2_CAREER_ALIGNMENT",
    # Self-queries nghề nghiệp
    ("CAREER",  "CAREER"):   "L2_CAREER_ALIGNMENT",
    ("SKILL",   "SKILL"):    "L2_CAREER_ALIGNMENT",

    # Personality cluster
    ("PERSONALITY", "CAREER"):      "L2_PERSONALITY_FIT",
    ("PERSONALITY", "MAJOR"):       "L2_PERSONALITY_FIT",
    ("PERSONALITY", "SUBJECT"):     "L2_CAREER_ALIGNMENT",
    ("CAREER",      "PERSONALITY"): "L2_PERSONALITY_FIT",
    ("MAJOR",       "PERSONALITY"): "L2_PERSONALITY_FIT",
    ("SUBJECT",     "PERSONALITY"): "L2_CAREER_ALIGNMENT",
    ("PERSONALITY", "PERSONALITY"): "L2_PERSONALITY_FIT",
}


_PERSONALITY_KW_PATTERN = re.compile(
    r"tính cách|phẩm chất|personality|hướng nội|hướng ngoại|"
    r"cẩn thận|sáng tạo|lãnh đạo|đồng cảm|kiên nhẫn|tự tin|"
    r"điềm tĩnh|sâu sắc|kín đáo|nội tâm|tập trung|thận trọng|"
    r"suy tư|ôn hòa|trầm mặc|tinh tế|logic|phân tích|lý trí|"
    r"thấu cảm|ấm áp|nhân văn|nề nếp|kế hoạch|tổ chức|ngăn nắp|"
    r"linh hoạt|tự do|ngẫu hứng|thoải mái|phóng khoáng|"
    r"chiến lược|tầm nhìn|lý tưởng|đổi mới|tò mò|khám phá|"
    r"kỷ luật|trách nhiệm|quyết đoán|"
    r"hợp\s+(với\s+)?(nghề|ngành)|phù hợp\s+(với\s+)?(tôi|mình|người)|"
    r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP"
    r"|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b",
    re.IGNORECASE | re.UNICODE,
)


def route_to_community(intent: dict) -> tuple[str, dict]:
    mentioned = intent.get("mentioned_labels") or []
    asked     = intent.get("asked_label", "UNKNOWN")
    keywords  = intent.get("keywords", [])

    if (
        "PERSONALITY" in mentioned
        or asked == "PERSONALITY"
        or _PERSONALITY_KW_PATTERN.search(" ".join(keywords))
    ):
        return "L2_PERSONALITY_FIT", COMMUNITY_LEVELS["L2_PERSONALITY_FIT"]


    MAJOR_CODE_PATTERN = re.compile(r"\b\d{7}\b")
    for kw in keywords:
        if MAJOR_CODE_PATTERN.search(str(kw)):
            return "L3_MAJOR_CENTRIC", COMMUNITY_LEVELS["L3_MAJOR_CENTRIC"]

    if asked in ("CAREER", "SUBJECT") and "SKILL" in mentioned:
        long_kws = [k for k in keywords if len(k.split()) >= 2]
        if long_kws:
            return "L3_SKILL_CENTRIC", COMMUNITY_LEVELS["L3_SKILL_CENTRIC"]

    # Lookup intent map
    first_mentioned = mentioned[0] if mentioned else None
    cid = INTENT_TO_COMMUNITY.get((first_mentioned, asked))
    if not cid:
        for m in mentioned:
            cid = INTENT_TO_COMMUNITY.get((m, asked))
            if cid:
                break
    if not cid:
        cid = "L1_GLOBAL"

    return cid, COMMUNITY_LEVELS[cid]




def run_louvain_and_write(driver, community_def: dict) -> dict:
    level      = community_def["level"]
    cid        = community_def["id"]
    prop_key   = f"community_L{level}"
    graph_name = f"neo_edu_{cid.lower()}"

    stats = {"community_id": cid, "level": level, "nodes_written": 0, "error": None}

    if level == 1:
        with driver.session() as session:
            r = session.run(
                "MATCH (n) WHERE (n:MAJOR OR n:SUBJECT OR n:SKILL "
                "OR n:CAREER OR n:TEACHER OR n:PERSONALITY) "
                f"SET n.{prop_key} = 0 RETURN count(n) AS cnt"
            ).single()
            stats["nodes_written"] = r["cnt"] if r else 0
        return stats

    with driver.session() as session:
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        node_labels = list(community_def["node_labels"])
        rel_proj    = {
            rtype: {"type": rtype, "orientation": "UNDIRECTED",
                    "properties": {"weight": {"defaultValue": w}}}
            for rtype, w in community_def["rel_weights"].items()
        }

        try:
            session.run(
                "CALL gds.graph.project($gname, $nlabels, $rproj)",
                gname=graph_name, nlabels=node_labels, rproj=rel_proj,
            )
        except Exception as e:
            stats["error"] = f"GDS project error: {e}"
            _fallback_community_assignment(driver, community_def, prop_key)
            return stats

        try:
            session.run(
                f"CALL gds.louvain.write('{graph_name}', "
                f"{{relationshipWeightProperty: 'weight', writeProperty: '{prop_key}'}})"
            )
            r = session.run(
                f"MATCH (n) WHERE n.{prop_key} IS NOT NULL RETURN count(n) AS cnt"
            ).single()
            stats["nodes_written"] = r["cnt"] if r else 0
        except Exception as e:
            stats["error"] = f"GDS Louvain error: {e}"
            _fallback_community_assignment(driver, community_def, prop_key)
        finally:
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            except Exception:
                pass

    return stats


def _fallback_community_assignment(driver, community_def: dict, prop_key: str):
    """
    Fallback assignment khớp với dữ liệu thực tế trong DB:
      L2: MAJOR=2, SUBJECT=2, CAREER=1, SKILL=0, TEACHER=0
      L3: SUBJECT=0, TEACHER=1, CAREER=1, SKILL=2
    """
    cid = community_def["id"]
    label_to_community = {
        "L2_ACADEMIC":          {"TEACHER": 0, "SUBJECT": 2, "MAJOR": 2},
        "L2_CAREER_ALIGNMENT":  {"SKILL": 0, "CAREER": 1, "SUBJECT": 2},
        "L2_PERSONALITY_FIT":   {"PERSONALITY": 3, "CAREER": 1, "MAJOR": 2},
        "L3_MAJOR_CENTRIC":     {"SUBJECT": 0, "TEACHER": 1, "SKILL": 2},
        "L3_SKILL_CENTRIC":     {"SUBJECT": 0, "CAREER": 1},
    }.get(cid, {})

    with driver.session() as session:
        for label, comm_val in label_to_community.items():
            session.run(f"MATCH (n:{label}) SET n.{prop_key} = {comm_val}")


def initialize_communities(driver, force_rebuild: bool = False):
    print("\n[Community Init] Bắt đầu khởi tạo 3 tầng cộng đồng...")

    if not force_rebuild:
        with driver.session() as session:
            r = session.run(
                "MATCH (n) WHERE n.community_L2 IS NOT NULL RETURN count(n) AS cnt LIMIT 1"
            ).single()
            if r and r["cnt"] > 0:
                print("[Community Init] Community L2/L3 đã tồn tại, bỏ qua rebuild.")
                return

    BUILD_ORDER = ["L1_GLOBAL", "L2_ACADEMIC", "L2_CAREER_ALIGNMENT",
                   "L2_PERSONALITY_FIT", "L3_MAJOR_CENTRIC", "L3_SKILL_CENTRIC"]

    for cid in BUILD_ORDER:
        cdef  = COMMUNITY_LEVELS[cid]
        level = cdef["level"]
        print(f"  [L{level}] Building: {cdef['name']}...")
        stats = run_louvain_and_write(driver, cdef)
        if stats.get("error"):
            print(f"    ⚠ Fallback (no GDS): {stats['error'][:80]}")
        else:
            print(f"    ✓ {stats['nodes_written']} nodes tagged (community_L{level})")

    print("[Community Init] Hoàn tất.\n")



_AGG_ALL_MAJOR_TOKENS = (
    r"tất cả(?: các)? ngành|mọi ngành|"
    r"các ngành đều|"
    r"ngành nào cũng|"
    r"chung cho(?: tất cả| mọi| các)?(?: các)? ngành|"
    r"môn chung|môn bắt buộc chung|môn(?: học)? bắt buộc"
)

AGGREGATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"môn(?: học)?(?: nào)?(?: là)? chung(?: giữa| của)?(.*?)(?:\s+và\s+)(.*?)(?:\?|$)",
        re.IGNORECASE | re.UNICODE,
    ), "subject_intersection_two"),
    (re.compile(
        r"(?:môn(?: học)?(?: gì| nào)?.*?(?:" + _AGG_ALL_MAJOR_TOKENS + r")"
        r"|(?:" + _AGG_ALL_MAJOR_TOKENS + r").*?(?:môn|học phần))",
        re.IGNORECASE | re.UNICODE,
    ), "subject_intersection_all"),
    (re.compile(
        r"ngành(?: nào)?.{0,20}(?:nhiều môn|nhiều học phần).{0,15}nhất",
        re.IGNORECASE | re.UNICODE,
    ), "major_most_subjects"),
    (re.compile(
        r"(?:nghề|career|vị trí).{0,20}(?:nhiều kỹ năng|nhiều skill).{0,15}nhất",
        re.IGNORECASE | re.UNICODE,
    ), "career_most_skills"),
    (re.compile(
        r"môn(?: học)?(?: nào)?.{0,30}(?:nhiều ngành|phổ biến nhất|nhiều nhất)",
        re.IGNORECASE | re.UNICODE,
    ), "subject_most_majors"),
    (re.compile(
        r"(?:kỹ năng|skill)(?: nào)?.{0,30}(?:nhiều môn|phổ biến nhất)",
        re.IGNORECASE | re.UNICODE,
    ), "skill_most_subjects"),
    (re.compile(
        r"(?:có|tổng)(?: tất cả)? bao nhiêu (ngành|môn|nghề|kỹ năng|giảng viên)",
        re.IGNORECASE | re.UNICODE,
    ), "count_entities"),
]


def detect_aggregation_type(question: str) -> str | None:
    for pattern, agg_type in AGGREGATION_PATTERNS:
        if pattern.search(question):
            return agg_type
    return None


def run_aggregation_query(driver, question: str, agg_type: str) -> list[dict]:
    results = []
    with driver.session() as session:

        if agg_type == "subject_intersection_all":
            rows = session.run("""
                MATCH (m:MAJOR)
                WITH count(m) AS total_majors
                MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                WITH s, count(DISTINCT m) AS major_count, total_majors
                WHERE major_count = total_majors
                RETURN s.name AS name, s.code AS code, major_count
                ORDER BY s.name ASC
            """).data()
            if not rows:
                rows = session.run("""
                    MATCH (m:MAJOR)
                    WITH count(m) AS total_majors
                    MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                    WITH s, count(DISTINCT m) AS major_count, total_majors
                    WHERE major_count >= toInteger(total_majors * 0.8)
                    RETURN s.name AS name, s.code AS code,
                           major_count, total_majors
                    ORDER BY major_count DESC LIMIT 30
                """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_count": r.get("major_count"), "hops": 1,
                    "_agg_meta": f"Xuất hiện trong {r.get('major_count')} ngành",
                })

        elif agg_type == "subject_intersection_two":
            rows = session.run("""
                MATCH (s:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(m:MAJOR)
                WITH s, collect(DISTINCT toLower(m.name)) AS major_names,
                     count(DISTINCT m) AS major_count
                WHERE major_count >= 2
                RETURN s.name AS name, s.code AS code,
                       major_names, major_count
                ORDER BY major_count DESC LIMIT 50
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_names": r.get("major_names"),
                    "major_count": r.get("major_count"), "hops": 1,
                })

        elif agg_type == "major_most_subjects":
            rows = session.run("""
                MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(s:SUBJECT)
                WITH m, count(DISTINCT s) AS subject_count
                RETURN m.name AS name, m.code AS code, subject_count
                ORDER BY subject_count DESC LIMIT 10
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "MAJOR", "code": r["code"],
                    "subject_count": r.get("subject_count"), "hops": 1,
                    "_agg_meta": f"{r.get('subject_count')} môn học",
                })

        elif agg_type == "career_most_skills":
            rows = session.run("""
                MATCH (c:CAREER)-[:REQUIRES]->(sk:SKILL)
                WITH c, count(DISTINCT sk) AS skill_count
                RETURN c.name AS name, skill_count
                ORDER BY skill_count DESC LIMIT 10
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "CAREER",
                    "skill_count": r.get("skill_count"), "hops": 1,
                    "_agg_meta": f"{r.get('skill_count')} kỹ năng",
                })

        elif agg_type == "subject_most_majors":
            rows = session.run("""
                MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(s:SUBJECT)
                WITH s, count(DISTINCT m) AS major_count
                RETURN s.name AS name, s.code AS code, major_count
                ORDER BY major_count DESC LIMIT 15
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SUBJECT", "code": r["code"],
                    "major_count": r.get("major_count"), "hops": 1,
                    "_agg_meta": f"Được dạy trong {r.get('major_count')} ngành",
                })

        elif agg_type == "skill_most_subjects":
            rows = session.run("""
                MATCH (s:SUBJECT)-[:PROVIDES]->(sk:SKILL)
                WITH sk, count(DISTINCT s) AS subject_count
                RETURN sk.name AS name, subject_count
                ORDER BY subject_count DESC LIMIT 15
            """).data()
            for r in rows:
                results.append({
                    "name": r["name"], "label": "SKILL",
                    "subject_count": r.get("subject_count"), "hops": 1,
                    "_agg_meta": f"Được cung cấp bởi {r.get('subject_count')} môn",
                })

        elif agg_type == "count_entities":
            q_lower = question.lower()
            if "ngành" in q_lower:        label, vn = "MAJOR",       "ngành"
            elif "nghề" in q_lower:       label, vn = "CAREER",      "nghề"
            elif "kỹ năng" in q_lower or "skill" in q_lower:
                                          label, vn = "SKILL",       "kỹ năng"
            elif "giảng viên" in q_lower: label, vn = "TEACHER",     "giảng viên"
            elif "phẩm chất" in q_lower or "personality" in q_lower or "tính cách" in q_lower:
                                          label, vn = "PERSONALITY", "phẩm chất"
            else:                         label, vn = "SUBJECT",     "môn học"
            cnt = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()["cnt"]
            results.append({
                "name": f"Tổng số {vn}: {cnt}", "label": label,
                "count": cnt, "hops": 0,
                "_agg_meta": f"count={cnt}",
            })

    return results




SCHEMA_DESC = """
Nodes (dữ liệu thực tế trong DB):
  MAJOR   (37 ngành):   code, name, name_vi, name_en
                        + philosophy_and_objectives, admission_requirements,
                          learning_outcomes, po_plo_matrix,
                          training_process_and_graduation_conditions,
                          curriculum_structure_and_content,
                          teaching_and_assessment_methods,
                          reference_programs, lecturer_and_teaching_assistant_standards,
                          facilities_and_learning_resources

  SUBJECT (802 môn):    code, name, name_vi, name_en
                        + course_description, courses_goals, assessment,
                          learning_resources, course_requirements_and_expectations,
                          syllabus_adjustment_time, week_1..week_N (kế hoạch giảng dạy)

  CAREER  (27 nghề):    career_key, name, name_vi, name_en, field_name
                        + description (JSON: short_description, role_in_organization),
                          job_tasks, education_certification, market

  SKILL   (5217 kỹ năng): skill_key, name, skill_type (hard|soft)

  TEACHER (695 GV):     teacher_key, name, email, title

  PERSONALITY (?):      personality_key, name_vi, name_en, category, description, indicators

Relationships (đồng bộ script1 v3, script2 v5):
  (MAJOR)       -[:MAJOR_OFFERS_SUBJECT {semester, required_type}]-> (SUBJECT)      (1421)
  (SUBJECT)     -[:PROVIDES {mastery_level}]->                       (SKILL)        (8069)
  (TEACHER)     -[:TEACH]->                                          (SUBJECT)      (3981)
  (CAREER)      -[:REQUIRES {required_level}]->                      (SKILL)         (223)
  (SUBJECT)     -[:PREREQUISITE_FOR]->                               (SUBJECT)        (24)
  (MAJOR)       -[:LEADS_TO]->                                       (CAREER)          (6)
  (PERSONALITY) -[:SUITS_MAJOR {field_name, group_name}]->           (MAJOR)         (MỚI v6)
  (PERSONALITY) -[:SUITS_CAREER {field_name, major_name}]->          (CAREER)        (MỚI v6)
  (CAREER)      -[:REQUIRES_PERSONALITY]->                           (PERSONALITY)   (dự phòng)
  (SUBJECT)     -[:CULTIVATES]->                                     (PERSONALITY)   (dự phòng)
"""

RELATIONSHIP_CONSTRAINTS = {
    ("MAJOR", "CAREER"): (
    "MAJOR -[:LEADS_TO]-> CAREER. "
    "Liệt kê Career mà Major dẫn đến. KHÔNG đề cập SUBJECT trừ khi được hỏi. "
    "QUAN TRỌNG: Chỉ liệt kê nghề nghiệp thực tế (vị trí công việc, chức danh trong tổ chức). "
    "TUYỆT ĐỐI KHÔNG liệt kê các mục bắt đầu bằng: "
    "'Cử nhân', 'Kỹ sư', 'Thạc sĩ', 'Tiến sĩ', 'Bác sĩ', 'Bachelor', 'Master', 'Engineer' — "
    "đây là danh hiệu học vị/bằng cấp, không phải vị trí công việc. "
    "Ví dụ SAI: 'Cử nhân Công nghệ thông tin', 'Kỹ sư phần mềm (danh hiệu)'. "
    "Ví dụ ĐÚNG: 'Chuyên viên phân tích dữ liệu', 'Lập trình viên', 'Nhà khoa học dữ liệu'. "
    "ĐỊNH DẠNG BẮT BUỘC: Trình bày dạng VĂN XUÔI, mỗi nghề một đoạn ngắn theo mẫu: "
    "'Có thể làm việc tại [field_name] với vai trò [tên nghề] — [short_description hoặc role_in_organization từ description]. "
    "Công việc chính bao gồm: [1-2 nhiệm vụ tiêu biểu từ job_tasks].' "
    "Nếu không có description hoặc job_tasks thì chỉ ghi: 'Có thể làm [tên nghề] trong lĩnh vực [field_name].' "
    "KHÔNG dùng bảng markdown cho câu hỏi loại này."
    ),    
    ("CAREER", "SKILL"):   (
        "CAREER -[:REQUIRES]-> SKILL và SUBJECT -[:PROVIDES]-> SKILL. "
        "Trả lời kỹ năng cần thiết, chỉ nêu kỹ năng cứng (hard skills, là các skill có skill_type = 'hard') + môn cung cấp kỹ năng đó."
    ),
    ("MAJOR", "SKILL"):    (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT -[:PROVIDES]-> SKILL. "
        "Kỹ năng đạt được từ các môn trong chương trình, chỉ nêu kỹ năng cứng (hard skills, là các skill có skill_type = 'hard'). Kèm tên môn (mã môn)."
    ),
    ("SKILL", "MAJOR"):    (
        "SKILL <-[:PROVIDES]- SUBJECT <-[:MAJOR_OFFERS_SUBJECT]- MAJOR. "
        "Ngành học có môn cung cấp kỹ năng đó. Kèm mã ngành, tên môn trung gian."
    ),
    ("CAREER", "SUBJECT"): (
        "CAREER -[:REQUIRES]-> SKILL <-[:PROVIDES]- SUBJECT. "
        "Môn học cung cấp kỹ năng nghề yêu cầu, chỉ nêu kỹ năng cứng (hard skills, là các skill có skill_type = 'hard'). Kèm mã môn + kỹ năng cứng tương ứng."
    ),
    ("MAJOR", "SUBJECT"):  (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT. "
        "Môn học thuộc chương trình ngành, kèm mã môn "
        "loại (required_type: required=bắt buộc, elective=tự chọn)."
    ),
    ("SKILL", "CAREER"):   (
        "SKILL <-[:REQUIRES]- CAREER. Nghề nghiệp yêu cầu kỹ năng đó."
    ),
    ("CAREER", "MAJOR"):   (
        "MAJOR -[:LEADS_TO]-> CAREER. Ngành học dẫn đến nghề đó, kèm mã ngành."
    ),
    ("SUBJECT", "SKILL"):  (
        "SUBJECT -[:PROVIDES]-> SKILL. Kỹ năng đạt được sau khi học môn đó."
    ),
    ("SKILL", "SUBJECT"):  (
        "SKILL <-[:PROVIDES]- SUBJECT. Môn học (kèm mã môn) cung cấp kỹ năng đó."
    ),
    ("SUBJECT", "TEACHER"): (
        "TEACHER -[:TEACH]-> SUBJECT. Giảng viên phụ trách môn đó."
    ),
    ("TEACHER", "SUBJECT"): (
        "TEACHER -[:TEACH]-> SUBJECT. Môn học thầy/cô đó phụ trách, kèm mã môn."
    ),
    ("MAJOR", "TEACHER"):  (
        "MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT <-[:TEACH]- TEACHER. "
        "Giảng viên dạy trong chương trình ngành đó."
    ),
    ("TEACHER", "MAJOR"):  (
        "TEACHER -[:TEACH]-> SUBJECT <-[:MAJOR_OFFERS_SUBJECT]- MAJOR. "
        "Ngành học thầy/cô đó tham gia giảng dạy."
    ),
    ("MAJOR", "MAJOR"):    (
        "So sánh: MAJOR -[:LEADS_TO]-> CAREER và MAJOR -[:MAJOR_OFFERS_SUBJECT]-> SUBJECT. "
        "So sánh cơ hội nghề nghiệp và môn học đặc trưng của từng ngành."
    ),
    # Self-queries
    ("SUBJECT", "SUBJECT"): (
        "Trả lời: mã môn (code), mô tả môn học (course_description), "
        "mục tiêu (courses_goals), đánh giá (assessment), "
        "môn tiên quyết nếu có (PREREQUISITE_FOR)."
    ),
    ("CAREER", "CAREER"):  (
        "Trả lời đầy đủ 4 phần: "
        "1. Mô tả nghề: lấy từ description (field short_description hoặc role_in_organization). "
        "2. Công việc chính: liệt kê từ job_tasks. "
        "3. Thị trường lao động: tóm tắt từ field market. "
        "4. ĐỀ XUẤT NGÀNH HỌC: BẮT BUỘC liệt kê các ngành theo recommended_majors "
        "(tên ngành + mã ngành). Nếu không có recommended_majors, "
        "dùng education_certification.recommended_majors làm tên gợi ý. "
        "Format: Tên ngành (mã ngành) - VD: Công nghệ thông tin (7480201). "
        "Nếu không có ngành nào trong DB - nói rõ chưa có dữ liệu ngành phù hợp."
    ),
    ("TEACHER", "TEACHER"): (
        "Trả lời: học hàm/học vị (title), email, "
        "môn đang dạy (TEACH→SUBJECT)."
    ),
    ("MAJOR", "MAJOR_DETAIL"): (
        "Trả lời chi tiết ngành: mục tiêu đào tạo (philosophy_and_objectives), "
        "chuẩn đầu ra (learning_outcomes), cơ hội nghề nghiệp (LEADS_TO→CAREER)."
    ),

    # Personality constraints — MBTI v6
    ("PERSONALITY", "CAREER"): (
        "PERSONALITY -[:SUITS_CAREER]-> CAREER. "
        "Liệt kê nghề nghiệp phù hợp với loại tính cách MBTI này. "
        "Nếu câu hỏi có nêu ngành/lĩnh vực cụ thể (ví dụ CNTT), ưu tiên các nghề nằm trong ngành/lĩnh vực đó trước, "
        "sau đó mới liệt kê nghề phù hợp chung. "
        "NGUỒN DỮ LIỆU ưu tiên theo thứ tự: "
        "(1) Các node CAREER trong [DỮ LIỆU GRAPH] có rel_types=['SUITS_CAREER']. "
        "(2) Trường suitable_fields trong node PERSONALITY (parse JSON string): "
        "    lấy từng field → groups → majors → careers. "
        "Định dạng BẮT BUỘC: bảng markdown | Lĩnh vực | Nhóm ngành | Nghề nghiệp |. "
        "TUYỆT ĐỐI không bịa thêm nghề không có trong dữ liệu."
    ),
    ("PERSONALITY", "MAJOR"): (
        "PERSONALITY -[:SUITS_MAJOR]-> MAJOR. "
        "Liệt kê ngành học phù hợp với loại tính cách MBTI này tại NEU. "
        "NGUỒN DỮ LIỆU ưu tiên theo thứ tự: "
        "(1) Các node MAJOR trong [DỮ LIỆU GRAPH] có rel_types=['SUITS_MAJOR']. "
        "(2) Trường suitable_fields trong node PERSONALITY (parse JSON string): "
        "    lấy từng field → groups → majors, lấy major_code và major_name. "
        "Định dạng BẮT BUỘC: bảng markdown | STT | Tên ngành | Mã ngành | Lĩnh vực |. "
        "Nếu major_code rỗng: ghi '—'. "
        "SAU BẢNG: thêm 1 đoạn ngắn giải thích TẠI SAO tính cách này phù hợp với "
        "các ngành đó (dựa vào strengths/work_environment trong node PERSONALITY). "
        "TUYỆT ĐỐI không liệt kê ngành ngoài dữ liệu."
    ),
    ("PERSONALITY", "PERSONALITY"): (
        "Trả lời đầy đủ về loại tính cách MBTI theo 4 phần: "
        "1. MÔ TẢ TỔNG QUAN (description). "
        "2. 4 CHIỀU TÍNH CÁCH (structure: IE/SN/TF/JP — mỗi chiều nêu dimension + description). "
        "3. ĐIỂM MẠNH (strengths) & ĐIỂM YẾU (weaknesses) — dạng bullet. "
        "4. MÔI TRƯỜNG LÀM VIỆC PHÙ HỢP (work_environment). "
        "Sau đó gợi ý xem thêm ngành/nghề phù hợp."
    ),
    ("CAREER", "PERSONALITY"): (
        "PERSONALITY -[:SUITS_CAREER]-> CAREER (chiều ngược). "
        "Liệt kê loại tính cách MBTI phù hợp với nghề/lĩnh vực này. "
        "Kèm mô tả ngắn tại sao phù hợp dựa vào structure/strengths của MBTI type đó. "
        "ĐỊNH DẠNG BẮT BUỘC: bảng markdown | MBTI | Tên tính cách | Lý do phù hợp |. "
        "Sau bảng, thêm đoạn tóm tắt: những đặc điểm tính cách chung của người phù hợp với lĩnh vực này."
    ),
    ("MAJOR", "PERSONALITY"): (
        "PERSONALITY -[:SUITS_MAJOR]-> MAJOR (chiều ngược). "
        "Liệt kê loại tính cách MBTI phù hợp với ngành học hoặc lĩnh vực này tại NEU. "
        "Nếu câu hỏi đề cập lĩnh vực rộng (VD: IT, CNTT), lấy TẤT CẢ personality có "
        "suitable_fields khớp với lĩnh vực đó (field_name chứa từ khóa liên quan). "
        "ĐỊNH DẠNG BẮT BUỘC: bảng markdown | MBTI | Tên tính cách | Lý do phù hợp |. "
        "Sau bảng, thêm đoạn tóm tắt: những đặc điểm tính cách chung của người phù hợp với lĩnh vực này."
    ),
    ("SUBJECT", "PERSONALITY"): (
        "SUBJECT -[:CULTIVATES]-> PERSONALITY (dự phòng). "
        "Nếu không có dữ liệu: thông báo chưa có thông tin tính cách cho môn học này."
    ),
    ("PERSONALITY", "SUBJECT"): (
        "PERSONALITY <-[:CULTIVATES]- SUBJECT (dự phòng). "
        "Nếu không có dữ liệu: thông báo chưa có thông tin."
    ),
}

ANSWER_SYSTEM_BASE = """Bạn là chuyên gia tư vấn học thuật và hướng nghiệp tại Đại học Kinh tế Quốc dân (NEU).

{schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TRÌNH TƯ DUY BẮT BUỘC — THỰC HIỆN TRƯỚC KHI VIẾT CÂU TRẢ LỜI:
(Suy luận thầm trong đầu hoặc trong block <thinking>)

BƯỚC 1 — PHÂN TÍCH THỰC THỂ:
  • Xác định các thực thể trong câu hỏi: Ngành (MAJOR), Môn học (SUBJECT),
    Kỹ năng (SKILL), Nghề nghiệp (CAREER), Giảng viên (TEACHER), Tính cách MBTI (PERSONALITY)
  • Nhận diện từ đồng nghĩa: "việc làm" = CAREER; "học phần" = SUBJECT; "năng lực" = SKILL

BƯỚC 2 — XỬ LÝ NGỮ NGHĨA:
  • Xác định điều kiện PHỦ ĐỊNH ("không muốn", "ngoại trừ", "không giỏi", "tránh")
    → BẮT BUỘC loại bỏ các thực thể này khỏi kết quả
  • Kiểm tra context câu hỏi: hỏi về lĩnh vực cụ thể hay hỏi tổng quát?

BƯỚC 3 — LẬP LỘ TRÌNH TRUY VẤN (Multi-hop Reasoning):
  • Xác định chuỗi hop cần đi trên đồ thị, ví dụ:
    - "ENFJ nên học môn gì?" →  MBTI → Nghề phù hợp → Kỹ năng nghề cần → Môn đào tạo kỹ năng
    - "Kỹ năng Python liên quan ngành nào?" → SKILL → SUBJECT cung cấp → MAJOR chứa SUBJECT
  • Ưu tiên dữ liệu từ chuỗi hop đầy đủ hơn là hop đơn lẻ

BƯỚC 4 — KIỂM TRA TÍNH NHẤT QUÁN:
  • Đảm bảo kết quả không vi phạm điều kiện phủ định của người dùng
  • Đảm bảo kết quả liên quan đến ngữ cảnh/lĩnh vực người dùng hỏi
  • Chỉ đưa vào câu trả lời những gì CÓ TRONG [DỮ LIỆU GRAPH]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LUẬT TUYỆT ĐỐI:
A. CHỈ dùng đúng tên/code/thông tin có trong [DỮ LIỆU GRAPH].
B. TUYỆT ĐỐI KHÔNG thêm kỹ năng, môn học, nghề nghiệp từ kiến thức bên ngoài.
C. TUYỆT ĐỐI KHÔNG liệt kê mục chung chung nếu không có trong [DỮ LIỆU GRAPH].
D. Mọi tên SKILL/SUBJECT/CAREER/MAJOR phải lấy nguyên văn từ [DỮ LIỆU GRAPH].
E. Mọi mã môn (code) phải lấy nguyên văn từ field "code".
F. Nếu [DỮ LIỆU GRAPH] trống → trả lời:
   "Dữ liệu hiện tại chưa đủ để tư vấn về [chủ đề]. Bạn có thể liên hệ phòng đào tạo."
G. TUYỆT ĐỐI KHÔNG liệt kê CAREER node có tên bắt đầu bằng danh hiệu học vị:
   "Cử nhân", "Kỹ sư" (khi là danh hiệu bằng cấp), "Thạc sĩ", "Tiến sĩ", "Bachelor", "Master".
   Đây là kết quả đào tạo, KHÔNG phải vị trí công việc. Bỏ qua hoàn toàn.
H. KHI GỢI Ý / TƯ VẤN "NÊN HỌC MÔN GÌ": TUYỆT ĐỐI KHÔNG đề cập, liệt kê, hay nhắc đến
   các môn đại cương bắt buộc chung sau đây (dù có trong dữ liệu hay không, dù viết hoa/thường/có dấu/không dấu):
     • Triết học Mác-Lênin (LLNL1105)
     • Kinh tế chính trị Mác-Lênin (LLNL1106)
     • Chủ nghĩa xã hội khoa học (LLNL1107)
     • Lịch sử Đảng Cộng sản Việt Nam (LLDL1102)
     • Tư tưởng Hồ Chí Minh (LLTT1101)
     • Giáo dục thể chất (GDTC)
     • Giáo dục quốc phòng và an ninh (GDQP)
     • Kinh tế vi mô 1 (KHMI1101)
     • Kinh tế vĩ mô 1 (KHMA1101)
     • Pháp luật đại cương (LUCS1129)
   Lý do: đây là môn bắt buộc chung mọi ngành, không cần tư vấn riêng.
   NGOẠI LỆ: Nếu người dùng HỎI TRỰC TIẾP "ngành X có học môn Y không?" → trả lời "Có".

ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC TUÂN THỦ:
- Tiếng Việt tự nhiên, thân thiện, chuyên nghiệp.
- Khi người dùng phủ định (không giỏi X) → bỏ X khỏi gợi ý.
- KHÔNG hỏi ngược lại người dùng.
- Trình bày câu trả lời theo 4 phần sau (bỏ phần nào nếu không có dữ liệu):
  1. PHÂN TÍCH YÊU CẦU: Tóm tắt ngắn gọn hiểu biết của bạn về câu hỏi
     (bao gồm cả điều kiện loại trừ nếu có)
  2. LỘ TRÌNH TƯ VẤN: Giải thích ngắn chuỗi suy luận từ dữ liệu
     (VD: "Từ MBTI ENFJ → nghề Marketing → kỹ năng Content → môn X đào tạo kỹ năng đó")
  3. GỢI Ý CHI TIẾT: Danh sách Môn học / Ngành / Kỹ năng / Nghề cụ thể (dạng bảng nếu ≥3 mục)
  4. LỜI KHUYÊN THÊM: (Dựa trên MBTI hoặc xu hướng thị trường nếu có dữ liệu)

QUY TẮC ĐỊNH DẠNG CHI TIẾT:

1. DANH SACH MON HOC / KY NANG / NGHE NGHIEP: khi liet ke tu 3 muc tro len,
   BAT BUOC trinh bay dang bang markdown.

   Vi du bang mon hoc:
   | STT | Ten mon | Ma mon |
   |-----|---------|--------|
   | 1   | Toan roi rac | TOCB1107 |

   Vi du bang ky nang:
   | STT | Ky nang | Loai |
   |-----|---------|------|
   | 1   | Lap trinh Python | Hard skill |

   Vi du bang nganh hoc:
   | STT | Ten nganh | Ma nganh | Mon hoc lien quan |
   |-----|-----------|----------|-------------------|
   | 1   | CNTT | 7480201 | Lap trinh Python (ITBD2301) |

   Vi du bang nghe nghiep:
   | STT | Ten nghe |
   |-----|----------|
   | 1   | Ky su phan mem |

   Chon cot phu hop voi du lieu thuc co trong [DU LIEU GRAPH]. Bo cot neu khong co du lieu.

2. THONG TIN CHI TIET (mo ta nganh, nghe, mon hoc): dung BULLET / NUMBERING.
   - Dung chu IN HOA cho tieu de muc (VD: MUC TIEU DAO TAO, CONG VIEC CHINH).
   - Dung ky tu * o dau dong cho tung y trong moi muc.
   - Dung so thu tu (1. 2. 3.) khi liet ke cac buoc hoac thu tu uu tien.

3. CAU TRA LOI NGAN (duoi 3 muc, hoi thong tin don gian): van xuoi binh thuong.
   - Mon hoc: "Ten mon (ma mon)" -- VD: "Toan roi rac (TOCB1107)".
   - Nganh: "Ten nganh (ma nganh)" -- VD: "CNTT (7480201)".

4. KET THUC CAU TRA LOI: Them 1 dong tom tat hoac goi y tiep theo neu phu hop.

SU DUNG THUOC TINH MO RONG KHI CO:
- SUBJECT:     dung course_description, courses_goals khi hoi noi dung mon hoc.
- CAREER:      dung description, job_tasks, market khi hoi ve nghe nghiep.
- MAJOR:       dung philosophy_and_objectives, learning_outcomes khi hoi ve nganh.
- PERSONALITY: dung code (MBTI type), description, structure (4 chieu IE/SN/TF/JP),
               strengths/weaknesses, work_environment.
               Truong suitable_fields la JSON string, parse de lay field_name,
               group_name, major_name, major_code, careers.
- Neu field la JSON string: parse va trinh bay ngan gon phan lien quan dung ky tu *.

RANG BUOC THEO LOAI CAU HOI:
{constraint}

CONG DONG DA DUOC DINH TUYEN:
{community_context}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: ABBREVIATION EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5b: MBTI EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

_MBTI_PATTERN = re.compile(
    r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP"
    r"|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b",
    re.IGNORECASE,
)

# Map MBTI code → keywords để mở rộng query khi DB chưa có SUITS_MAJOR/SUITS_CAREER
# (fallback khi graph traversal không tìm được gì qua edge trực tiếp)
MBTI_KEYWORD_FALLBACK: dict[str, list[str]] = {
    "INTJ": ["chiến lược", "phân tích", "độc lập", "tầm nhìn"],
    "INTP": ["phân tích", "logic", "nghiên cứu", "lý luận"],
    "ENTJ": ["lãnh đạo", "chiến lược", "quyết đoán", "quản lý"],
    "ENTP": ["sáng tạo", "đổi mới", "lập luận", "linh hoạt"],
    "INFJ": ["đồng cảm", "tầm nhìn", "sáng tạo", "kiên nhẫn"],
    "INFP": ["sáng tạo", "đồng cảm", "lý tưởng", "linh hoạt"],
    "ENFJ": ["lãnh đạo", "đồng cảm", "giao tiếp", "tổ chức"],
    "ENFP": ["sáng tạo", "nhiệt huyết", "giao tiếp", "linh hoạt"],
    "ISTJ": ["kỷ luật", "cẩn thận", "trách nhiệm", "tổ chức"],
    "ISFJ": ["đồng cảm", "kiên nhẫn", "cẩn thận", "hỗ trợ"],
    "ESTJ": ["tổ chức", "kỷ luật", "lãnh đạo", "quyết đoán"],
    "ESFJ": ["giao tiếp", "đồng cảm", "hỗ trợ", "tổ chức"],
    "ISTP": ["phân tích", "thực tế", "kỹ thuật", "linh hoạt"],
    "ISFP": ["sáng tạo", "thực tế", "đồng cảm", "linh hoạt"],
    "ESTP": ["năng động", "thực tế", "quyết đoán", "lãnh đạo"],
    "ESFP": ["năng động", "giao tiếp", "linh hoạt", "thực tế"],
}


def expand_mbti(question: str) -> tuple[str, list[str]]:
    """
    Nhận diện MBTI code tường minh (INTJ, ESTP...) trong câu hỏi.
    Trả về (expanded_question, [mbti_code]) để query trực tiếp PERSONALITY node.
    """
    m = _MBTI_PATTERN.search(question)
    if not m:
        return question, []
    mbti_code = m.group(1).upper()
    hint  = f"[GHI CHÚ: {mbti_code} là loại tính cách MBTI]"
    return question + "  " + hint, [mbti_code]


ABBREVIATION_MAP: dict[str, list[str]] = {
    "da":   ["data analyst", "phân tích dữ liệu"],
    "de":   ["data engineer", "kỹ sư dữ liệu"],
    "ds":   ["data scientist", "khoa học dữ liệu"],
    "data analyst":     ["phân tích dữ liệu", "chuyên viên phân tích dữ liệu"],
    "data engineer":    ["kỹ sư dữ liệu"],
    "data scientist":   ["khoa học dữ liệu", "nhà khoa học dữ liệu"],
    "data engineering": ["kỹ sư dữ liệu"],
    "data analysis":    ["phân tích dữ liệu"],
    "ba":   ["business analyst", "phân tích kinh doanh"],
    "pm":   ["project manager", "quản lý dự án"],
    "po":   ["product owner"],
    "qa":   ["kiểm thử", "quality assurance"],
    "dev":  ["lập trình viên", "developer"],
    "fe":   ["front end", "lập trình viên frontend"],
    "be":   ["back end", "lập trình viên backend"],
    "ml":   ["machine learning", "học máy"],
    "ai":   ["trí tuệ nhân tạo", "artificial intelligence"],
    "cntt": ["công nghệ thông tin"],
    "ktpm": ["kỹ thuật phần mềm"],
    "httt": ["hệ thống thông tin"],
    "qtkd": ["quản trị kinh doanh"],
    "tcnh": ["tài chính ngân hàng"],
    "kt":   ["kế toán", "kinh tế"],
    "mkt":  ["marketing"],
    "hr":   ["quản trị nhân lực", "nhân sự"],
    "mis":  ["hệ thống thông tin quản lý", "management information systems"],
    "fintech": ["công nghệ tài chính"],
    "ecom": ["thương mại điện tử"],
    "acct": ["kế toán"],
}


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5c: SYNONYM LAYER — ánh xạ từ đồng nghĩa → Canonical Name trong đồ thị
# Mục đích: tránh LLM tự đoán keyword; đảm bảo recall đồng đều
# ══════════════════════════════════════════════════════════════════════════════

SYNONYM_MAP: dict[str, str] = {
    # ── Nghề nghiệp (CAREER) ─────────────────────────────────────────────────
    "việc làm": "nghề nghiệp",
    "công việc": "nghề nghiệp",
    "nghề": "nghề nghiệp",
    "vị trí": "nghề nghiệp",
    "vị trí công việc": "nghề nghiệp",
    "ngành nghề": "nghề nghiệp",
    "job": "nghề nghiệp",
    "career": "nghề nghiệp",
    "ra làm gì": "nghề nghiệp",
    "làm gì sau khi ra trường": "nghề nghiệp",
    "cơ hội việc làm": "nghề nghiệp",
    "triển vọng nghề": "nghề nghiệp",
    # ── Ngành học (MAJOR) ────────────────────────────────────────────────────
    "chuyên ngành": "ngành",
    "chương trình": "ngành",
    "ngành đào tạo": "ngành",
    "bộ môn": "ngành",
    "lĩnh vực học": "ngành",
    "major": "ngành",
    "chương trình học": "ngành",
    # ── Môn học (SUBJECT) ────────────────────────────────────────────────────
    "học phần": "môn học",
    "môn": "môn học",
    "course": "môn học",
    "subject": "môn học",
    "lớp học": "môn học",
    # ── Kỹ năng (SKILL) ──────────────────────────────────────────────────────
    "năng lực": "kỹ năng",
    "kỹ năng cần thiết": "kỹ năng",
    "skill": "kỹ năng",
    "competency": "kỹ năng",
    "ability": "kỹ năng",
    # ── Tính cách (PERSONALITY) ──────────────────────────────────────────────
    "nhóm tính cách": "tính cách",
    "loại người": "tính cách",
    "đặc điểm bản thân": "tính cách",
    "personality type": "tính cách",
    # ── Học máy / AI ─────────────────────────────────────────────────────────
    "học máy": "machine learning",
    "trí tuệ nhân tạo": "trí tuệ nhân tạo",
    # ── Data ─────────────────────────────────────────────────────────────────
    "khoa học dữ liệu": "data science",
    "phân tích dữ liệu": "data analyst",
    "kỹ sư dữ liệu": "data engineer",
    # ── Finance / Accounting ─────────────────────────────────────────────────
    "ngân hàng": "tài chính ngân hàng",
    "tài chính": "tài chính ngân hàng",
    "kế toán kiểm toán": "kế toán",
    # ── Others ───────────────────────────────────────────────────────────────
    "logistics": "logistics và quản lý chuỗi cung ứng",
    "supply chain": "logistics và quản lý chuỗi cung ứng",
    "pr": "quan hệ công chúng",
    "public relations": "quan hệ công chúng",
    "hr": "quản trị nhân lực",
    "human resources": "quản trị nhân lực",
    "kinh doanh quốc tế": "kinh doanh quốc tế",
    "ibm": "kinh doanh quốc tế",
}


def normalize_keywords(keywords: list[str]) -> list[str]:
    """
    Ánh xạ từ đồng nghĩa → Canonical Name trước khi đưa vào graph query.
    Giữ nguyên từ gốc và bổ sung thêm canonical name (không xóa gốc để tránh mất recall).
    """
    result = list(keywords)
    seen_lower = {k.lower() for k in result}
    for kw in list(keywords):
        canonical = SYNONYM_MAP.get(kw.lower())
        if canonical and canonical.lower() not in seen_lower:
            result.append(canonical)
            seen_lower.add(canonical.lower())
    return result


def expand_abbreviations(question: str) -> tuple[str, list[str]]:
    q_lower  = question.lower()
    expanded = question
    extras   = []
    found    = {}

    for abbrev, expansions in ABBREVIATION_MAP.items():
        if len(abbrev) <= 3:
            pat = r"(?<![\w\u00C0-\u024F])" + re.escape(abbrev.upper()) + r"(?![\w\u00C0-\u024F])"
            if not re.search(pat, question, re.UNICODE):
                continue
        pattern = r"(?<![\w\u00C0-\u024F])" + re.escape(abbrev) + r"(?![\w\u00C0-\u024F])"
        if re.search(pattern, q_lower, re.IGNORECASE | re.UNICODE):
            found[abbrev] = expansions
            extras.extend(expansions)

    if found:
        hints    = "; ".join(f"{k.upper()} = {' / '.join(v)}" for k, v in found.items())
        expanded = question + f"  [GHI CHÚ: {hints}]"

    return expanded, extras




def extract_query_intent(ai_client: OpenAI, question: str) -> dict:
    system_msg = (
        "Bạn phân tích câu hỏi tư vấn học thuật và trả về JSON.\n"
        "Schema Node labels: MAJOR, SUBJECT, SKILL, CAREER, TEACHER, PERSONALITY\n\n"
        "Chuẩn hóa keyword:\n"
        "  data analyst/DA → phân tích dữ liệu, data analyst\n"
        "  business analyst/BA → phân tích kinh doanh\n"
        "  CNTT/IT → công nghệ thông tin\n"
        "  KTPM → kỹ thuật phần mềm | HTTT → hệ thống thông tin\n"
        "  developer/DEV → lập trình viên | tester/QA → kiểm thử\n\n"
        "Quy tắc xác định asked_label:\n"
        "  - Hỏi thông tin môn học (mô tả, mã môn, nội dung, kế hoạch giảng dạy) → asked=SUBJECT\n"
        "  - Hỏi thông tin nghề nghiệp (mô tả nghề, công việc, thị trường lao động, triển vọng, cơ hội nghề nghiệp) → asked=CAREER\n"
        "  - Hỏi thông tin giảng viên (email, học hàm, dạy môn gì) → asked=TEACHER\n"
        "  - Hỏi thông tin ngành học (chương trình, chuẩn đầu ra, mục tiêu) → asked=MAJOR\n"
        "  - Hỏi kỹ năng → asked=SKILL\n"
        "  - Hỏi về loại tính cách MBTI, personality fit, đặc điểm tính cách → asked=PERSONALITY\n"
        "  - Nếu đề cập tính cách/MBTI nhưng hỏi về nghề → mentioned=PERSONALITY, asked=CAREER\n"
        "  - Nếu đề cập tính cách/MBTI nhưng hỏi về ngành → mentioned=PERSONALITY, asked=MAJOR\n"
        "  - Keywords: luôn giữ nguyên MBTI code (ESTP, ENTP...) nếu có\n\n"
        "──────────────────────────────────────────\n"
        "TRƯỜNG ĐẶC BIỆT: mbti_dimensions\n"
        "──────────────────────────────────────────\n"
        "Nếu câu hỏi mô tả đặc điểm tính cách bằng từ ngữ tự nhiên (KHÔNG phải MBTI code),\n"
        "hãy suy luận các MBTI dimension letters phù hợp:\n\n"
        "  4 cặp dimension:\n"
        "    E / I  — năng lượng:  hướng ngoại (E) vs hướng nội, điềm tĩnh, kín đáo, suy tư (I)\n"
        "    S / N  — nhận thức:   thực tế, chi tiết, quy trình (S) vs sáng tạo, tầm nhìn, trực giác (N)\n"
        "    T / F  — quyết định:  logic, phân tích, lý trí (T) vs đồng cảm, ấm áp, cảm xúc (F)\n"
        "    J / P  — lối sống:    kế hoạch, ngăn nắp, kỷ luật (J) vs linh hoạt, ngẫu hứng, tự do (P)\n\n"
        "  Quy tắc:\n"
        "  - Chỉ trả về dimension mà câu hỏi có dấu hiệu rõ ràng. Không đoán mò.\n"
        "  - Nếu câu hỏi có cả 2 chiều đối lập (E lẫn I), bỏ cả 2, không trả về dimension đó.\n"
        "  - Nếu câu hỏi có MBTI code tường minh (INTJ, ESTP...), để mbti_dimensions = []\n"
        "    và đưa code đó vào keywords thay vào đó.\n"
        "  - Nếu không có dấu hiệu tính cách nào, để mbti_dimensions = [].\n\n"
        "  Ví dụ:\n"
        "  'Em hướng nội thì học ngành gì'        → mbti_dimensions: ['I']\n"
        "  'Người logic và kỷ luật hợp nghề gì'   → mbti_dimensions: ['T', 'J']\n"
        "  'Tôi sáng tạo, thích tầm nhìn xa'      → mbti_dimensions: ['N']\n"
        "  'Tôi vừa hướng nội vừa hướng ngoại'    → mbti_dimensions: []  (xung đột)\n"
        "  'Tôi là INTJ học ngành gì'              → mbti_dimensions: [], keywords: ['INTJ']\n\n"
        "Trả về JSON:\n"
        "{\n"
        '  "keywords": ["tên thực thể để tìm trong KG"],\n'
        '  "mentioned_labels": ["MAJOR|SUBJECT|SKILL|CAREER|TEACHER|PERSONALITY"],\n'
        '  "asked_label": "MAJOR|SUBJECT|SKILL|CAREER|TEACHER|PERSONALITY|UNKNOWN",\n'
        '  "negated_keywords": ["thực thể bị phủ định"],\n'
        '  "is_comparison": false,\n'
        '  "mbti_dimensions": ["I","T"]  // các dimension letters được suy luận, hoặc []\n'
        "}\n"
    )
    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Phân tích: {question}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return {
        "keywords":         parsed.get("keywords", []),
        "mentioned_labels": parsed.get("mentioned_labels", []),
        "asked_label":      parsed.get("asked_label", "UNKNOWN"),
        "negated_keywords": parsed.get("negated_keywords", []),
        "is_comparison":    parsed.get("is_comparison", False),
        "mbti_dimensions":  [
            d for d in parsed.get("mbti_dimensions", [])
            if d in ("E", "I", "S", "N", "T", "F", "J", "P")
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5d: QUERY TRANSFORMATION — Pre-processing trước khi truy vấn Graph
# Áp dụng 3 kỹ thuật: Self-Correction, Decomposition, Negative Logic Handling
# ══════════════════════════════════════════════════════════════════════════════

def query_transformation(ai_client: OpenAI, question: str, intent: dict) -> dict:
    """
    Tiền xử lý câu hỏi trước khi đưa vào graph:
    1. Self-Correction: viết lại câu hỏi rõ ràng hơn
    2. Decomposition: chia câu hỏi phức tạp thành sub-queries
    3. Negative Logic Handling: tách điều kiện loại trừ

    Trả về dict bổ sung: rewritten_question, sub_queries, explicit_exclusions
    """
    # Chỉ áp dụng nếu câu hỏi đủ phức tạp (>10 từ hoặc có dấu hiệu multi-hop)
    word_count = len(question.split())
    has_condition = any(kw in question.lower() for kw in [
        "nhưng", "ngoài ra", "ngoại trừ", "không muốn", "tránh",
        "và", "hoặc", "nếu", "vừa", "không giỏi", "yếu",
    ])
    if word_count < 8 and not has_condition:
        return {"rewritten_question": question, "sub_queries": [], "explicit_exclusions": []}

    system_msg = (
        "Bạn là bộ tiền xử lý câu hỏi cho hệ thống GraphRAG tư vấn học thuật.\n\n"
        "NHIỆM VỤ: Phân tích câu hỏi và trả về JSON với 3 trường:\n\n"
        "1. rewritten_question: Viết lại câu hỏi rõ ràng, tường minh hơn (Self-Correction).\n"
        "   - Giải nghĩa viết tắt (CNTT → công nghệ thông tin)\n"
        "   - Thêm ngữ cảnh nếu thiếu ('học môn gì' → 'sinh viên nên học môn nào để...')\n"
        "   - Giữ nguyên mọi yêu cầu gốc, chỉ làm rõ thêm\n\n"
        "2. sub_queries: List các câu hỏi nhỏ (Decomposition) nếu câu hỏi phức tạp.\n"
        "   - Mỗi sub-query là một bước hop trên graph\n"
        "   - Ví dụ 'ENFJ nên học môn tự chọn nào?' → [\n"
        "       'ENFJ phù hợp nghề nghiệp nào?',\n"
        "       'Nghề đó cần kỹ năng gì?',\n"
        "       'Môn tự chọn nào đào tạo kỹ năng đó?'\n"
        "     ]\n"
        "   - Nếu câu hỏi đơn giản thì để []\n\n"
        "3. explicit_exclusions: List điều kiện loại trừ tường minh (Negative Logic).\n"
        "   - Những gì user KHÔNG muốn, KHÔNG phù hợp, 'ngoại trừ', 'tránh'\n"
        "   - Ví dụ: ['môn tính toán nặng', 'ngành kế toán', 'không giỏi toán']\n"
        "   - Nếu không có thì để []\n\n"
        "Trả về ĐÚNG JSON:\n"
        '{"rewritten_question": "...", "sub_queries": [], "explicit_exclusions": []}'
    )
    try:
        response = ai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": f"Câu hỏi gốc: {question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return {
            "rewritten_question":  parsed.get("rewritten_question", question),
            "sub_queries":         parsed.get("sub_queries", []),
            "explicit_exclusions": parsed.get("explicit_exclusions", []),
        }
    except Exception as e:
        print(f"  [query_transform] WARNING: {e}")
        return {"rewritten_question": question, "sub_queries": [], "explicit_exclusions": []}


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5e: STRUCTURED PREPROCESSING — Entity Extraction & Intent Analysis
# Workflow 3-bước: Preprocessing JSON → Neo4j query → Generation
# Thiết kế cho mô hình nhỏ (8B): tách biệt hoàn toàn logic exclude khỏi LLM
# ══════════════════════════════════════════════════════════════════════════════

_STRUCTURED_INTENT_SYSTEM = """Bạn là chuyên gia phân tích ngôn ngữ tự nhiên (NLP) cho hệ thống CoursesGuide AI của Đại học Kinh tế Quốc dân (NEU).
Nhiệm vụ: Chuyển câu hỏi tự nhiên của sinh viên thành cấu trúc JSON chính xác để truy vấn Neo4j Knowledge Graph.

ONTOLOGY — Các loại thực thể cần nhận diện:
  MBTI      : Nhóm tính cách 16 loại (ENFJ, ISTP, INTJ,...)
  Major     : Ngành đào tạo tại NEU (Marketing, CNTT, Kế toán,...)
  Subject   : Môn học / học phần (tên hoặc mã môn)
  Skill     : Kỹ năng chuyên môn hoặc mềm (Python, phân tích dữ liệu,...)
  Career    : Vị trí công việc / nghề nghiệp (Data Analyst, Kế toán viên,...)

QUY TẮC XỬ LÝ:

1. NORMALIZE (từ đồng nghĩa → từ chuẩn):
   "việc làm","chỗ làm","job","career","vị trí" → Career
   "học phần","môn","course","subject"           → Subject
   "năng lực","ability","skill","kỹ năng"        → Skill
   "ngành","chuyên ngành","major","program"      → Major
   "tính cách","personality","nhóm tính cách"   → MBTI
   Chuẩn hóa value: "mkt" → "Marketing", "CNTT" → "Công nghệ thông tin",
   "DA"/"data analyst" → "Data Analyst", "BA" → "Business Analyst"

2. NEGATION (exclude: true) — khi người dùng dùng:
   "không muốn","không thích","ngoại trừ","bỏ qua","tránh",
   "không giỏi","yếu","không phù hợp","ngoài ra loại trừ"
   → Đánh dấu exclude: true cho thực thể đó

3. AMBIGUITY: Nếu từ lạ, xếp vào loại thực thể gần nhất dựa vào ngữ cảnh.

4. CONSTRAINTS: Tách các ràng buộc ngầm định thành text mô tả.
   VD: "không thích tính toán" → "Tránh các môn học tính toán định lượng nặng"

OUTPUT FORMAT — Chỉ trả ra JSON, không nói gì thêm:
{
  "intent": "Định hướng môn học / Tư vấn lộ trình / Định hướng nghề nghiệp / So sánh nghề / Thông tin ngành / Thông tin môn học / Hỏi tính cách / Hỏi kỹ năng / Khác",
  "entities": [
    {"type": "MBTI|Major|Subject|Skill|Career", "value": "tên thực thể đã chuẩn hóa", "exclude": false}
  ],
  "constraints": ["Ràng buộc 1", "Ràng buộc 2"],
  "original_context": "Tóm tắt mục đích thực sự của người dùng trong 1-2 câu"
}"""


def extract_structured_intent(ai_client: OpenAI, question: str) -> dict:
    """
    Bước 1 (Preprocessing): Trích xuất structured JSON với entities + exclude flags.
    Đây là lớp phân tích ngữ nghĩa chính, thay thế phương pháp keyword thô.
    Trả về dict với keys: intent, entities, constraints, original_context
    """
    try:
        response = ai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _STRUCTURED_INTENT_SYSTEM},
                {"role": "user",   "content": f"Phân tích câu hỏi sau:\n{question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        # Đảm bảo schema đúng
        entities = parsed.get("entities", [])
        # Validate từng entity
        valid_types = {"MBTI", "Major", "Subject", "Skill", "Career"}
        clean_entities = []
        for ent in entities:
            if isinstance(ent, dict) and ent.get("type") in valid_types and ent.get("value"):
                clean_entities.append({
                    "type":    str(ent["type"]),
                    "value":   str(ent["value"]).strip(),
                    "exclude": bool(ent.get("exclude", False)),
                })
        return {
            "intent":           parsed.get("intent", "Khác"),
            "entities":         clean_entities,
            "constraints":      [str(c) for c in parsed.get("constraints", [])],
            "original_context": parsed.get("original_context", ""),
        }
    except Exception as e:
        print(f"  [structured_intent] WARNING: {e}")
        return {"intent": "Khác", "entities": [], "constraints": [], "original_context": ""}


def fuzzy_match_entity(value: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """
    Fuzzy matching: Nếu không tìm thấy entity value chính xác trong DB,
    tìm candidate có tên gần giống nhất (dùng SequenceMatcher).
    Trả về matched name hoặc None nếu không đủ ngưỡng.
    """
    import difflib
    if not candidates or not value:
        return None
    value_lower = value.lower()
    best_match = None
    best_ratio = 0.0
    for cand in candidates:
        ratio = difflib.SequenceMatcher(None, value_lower, cand.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = cand
    if best_ratio >= threshold:
        return best_match
    return None


def build_excluded_names(structured: dict) -> list[str]:
    """
    Trích xuất danh sách các entity value có exclude=True từ structured intent.
    Đây là input cho negation filter — đảm bảo loại trừ tuyệt đối ở tầng code,
    KHÔNG phụ thuộc vào LLM nhớ constraint.
    """
    return [
        ent["value"]
        for ent in structured.get("entities", [])
        if ent.get("exclude") is True
    ]


def merge_structured_into_intent(structured: dict, intent: dict) -> dict:
    """
    Hợp nhất kết quả Preprocessing vào intent dict hiện có.
    - entities có exclude=False → thêm value vào keywords
    - entities có exclude=True  → thêm value vào negated_keywords
    - constraints               → lưu vào intent["constraints"]
    - original_context          → lưu vào intent["original_context"]
    """
    existing_kws    = set(k.lower() for k in intent.get("keywords", []))
    existing_neg    = set(k.lower() for k in intent.get("negated_keywords", []))
    new_kws:  list[str] = []
    new_neg:  list[str] = []

    # Map entity type → node label (để thống nhất với intent labels)
    TYPE_TO_LABEL = {
        "MBTI":    "PERSONALITY",
        "Major":   "MAJOR",
        "Subject": "SUBJECT",
        "Skill":   "SKILL",
        "Career":  "CAREER",
    }

    for ent in structured.get("entities", []):
        val   = ent.get("value", "").strip()
        excl  = ent.get("exclude", False)
        label = TYPE_TO_LABEL.get(ent.get("type", ""), "")

        if not val:
            continue

        if excl:
            if val.lower() not in existing_neg:
                new_neg.append(val)
        else:
            if val.lower() not in existing_kws:
                new_kws.append(val)
            # Đảm bảo label có trong mentioned_labels
            if label and label not in intent.get("mentioned_labels", []):
                intent.setdefault("mentioned_labels", []).append(label)

    intent["keywords"]         = _unique_keep_order(intent.get("keywords", []) + new_kws)
    intent["negated_keywords"] = _unique_keep_order(intent.get("negated_keywords", []) + new_neg)
    intent["constraints"]      = structured.get("constraints", [])
    intent["original_context"] = structured.get("original_context", "")
    intent["structured_intent"] = structured.get("intent", "Khác")
    return intent


def resolve_mbti_codes_from_dimensions(dimensions: list[str]) -> list[str]:
    """
    Từ list dimension letters (e.g. ['I', 'T']) trả về tất cả MBTI codes
    chứa TẤT CẢ các dimensions đó.

    Ví dụ:
      ['I']       → [INTJ, INTP, INFJ, INFP, ISTJ, ISFJ, ISTP, ISFP]
      ['I', 'T']  → [INTJ, INTP, ISTJ, ISTP]
      ['T', 'J']  → [INTJ, ISTJ, ENTJ, ESTJ]
    """
    if not dimensions:
        return []
    all_types = [
        "INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
        "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP",
    ]
    required = set(dimensions)
    return [t for t in all_types if required.issubset(set(t))]


_COMPARE_CUE_PATTERN = re.compile(
    r"\b(vs|versus)\b|so sánh|phân vân|giữa.+và|nên chọn bên nào|nên chọn cái nào",
    re.IGNORECASE | re.UNICODE,
)
_CAREER_CUE_PATTERN = re.compile(
    r"nghề nào|làm gì|ra trường làm gì|nên chọn nghề|nên theo nghề|hợp làm|hợp nghề",
    re.IGNORECASE | re.UNICODE,
)
_ASK_PERSONALITY_PATTERN = re.compile(
    r"tính cách (gì|nào)|mbti (gì|nào)|loại tính cách",
    re.IGNORECASE | re.UNICODE,
)
_MAJOR_CUE_PATTERN = re.compile(
    r"\bngành\b|chuyên ngành|chương trình đào tạo|học ngành",
    re.IGNORECASE | re.UNICODE,
)
_SKILL_CUE_PATTERN = re.compile(
    r"\bsql\b|database|cơ sở dữ liệu|dữ liệu|data",
    re.IGNORECASE | re.UNICODE,
)
_NEGATED_CAREER_PATTERN = re.compile(
    r"(?:không|ko|chẳng|không muốn).{0,20}\b(sale|marketing)\b",
    re.IGNORECASE | re.UNICODE,
)
_CAREER_ALIAS_HINTS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\b(tester|qa|quality assurance|kiểm thử)\b", re.IGNORECASE | re.UNICODE),
     ["kiểm thử", "tester", "quality assurance"]),
    (re.compile(r"\b(developer|dev|lập trình viên)\b", re.IGNORECASE | re.UNICODE),
     ["lập trình viên", "developer"]),
]
_DOMAIN_HINTS: list[tuple[re.Pattern[str], list[str], list[str]]] = [
    (re.compile(r"\b(cntt|it|công nghệ thông tin)\b", re.IGNORECASE | re.UNICODE),
     ["công nghệ thông tin"], ["MAJOR"]),
    (re.compile(r"\b(database|cơ sở dữ liệu)\b", re.IGNORECASE | re.UNICODE),
     ["database"], ["SKILL"]),
    (re.compile(r"\bsql\b", re.IGNORECASE | re.UNICODE),
     ["sql"], ["SKILL"]),
]

# Map pattern → field_name để inject field_context vào intent
# Dùng khi câu hỏi là "tính cách gì hợp làm X" → cần filter PERSONALITY theo lĩnh vực X
_FIELD_CONTEXT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(cntt|it|công nghệ thông tin|lập trình|phần mềm|kỹ thuật phần mềm|hệ thống thông tin)\b",
                re.IGNORECASE | re.UNICODE), "Công nghệ thông tin"),
    (re.compile(r"\b(kinh tế|tài chính|kế toán|ngân hàng|kinh doanh|quản trị|marketing)\b",
                re.IGNORECASE | re.UNICODE), "Kinh tế - Quản trị"),
    (re.compile(r"\b(data|dữ liệu|phân tích dữ liệu|khoa học dữ liệu)\b",
                re.IGNORECASE | re.UNICODE), "Khoa học dữ liệu"),
    (re.compile(r"\b(giáo dục|sư phạm|giảng dạy|đào tạo)\b",
                re.IGNORECASE | re.UNICODE), "Giáo dục"),
    (re.compile(r"\b(y tế|bác sĩ|y khoa|dược|chăm sóc sức khỏe)\b",
                re.IGNORECASE | re.UNICODE), "Y tế - Sức khỏe"),
]


def _unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        sv = str(v).strip()
        if not sv:
            continue
        key = sv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sv)
    return out


def apply_intent_rules(question: str, intent: dict) -> dict:
    """
    Hậu xử lý intent để ổn định cho câu hỏi dài/ngữ cảnh mơ hồ.
    Không thay thế LLM, chỉ vá các case hay sai:
    - So sánh nghề (tester vs developer...)
    - Câu có tính cách + skill nhưng chưa hỏi rõ ngành/nghề
    - Câu hỏi "tính cách gì hợp làm IT"
    """
    q = question.strip()
    q_lower = q.lower()

    keywords = _unique_keep_order([*intent.get("keywords", [])])
    mentioned = [str(x).strip().upper() for x in intent.get("mentioned_labels", []) if str(x).strip()]
    mentioned = _unique_keep_order(mentioned)
    asked = str(intent.get("asked_label", "UNKNOWN")).upper()
    negated = _unique_keep_order([*intent.get("negated_keywords", [])])
    is_comp = bool(intent.get("is_comparison", False))

    # Inject thêm keyword/label domain.
    for pat, kws, labels in _DOMAIN_HINTS:
        if pat.search(q):
            keywords.extend(kws)
            mentioned.extend(labels)

    # Nhận diện job aliases để tăng recall query nghề.
    has_direct_career_alias = False
    for pat, kws in _CAREER_ALIAS_HINTS:
        if pat.search(q):
            has_direct_career_alias = True
            keywords.extend(kws)
            if "CAREER" not in mentioned:
                mentioned.append("CAREER")

    # Bổ sung phủ định nghề phổ biến (sale/marketing) khi LLM bỏ sót.
    for m in _NEGATED_CAREER_PATTERN.finditer(q):
        neg_kw = m.group(1).strip()
        if neg_kw:
            negated.append(neg_kw)

    if _COMPARE_CUE_PATTERN.search(q):
        is_comp = True
        if "CAREER" not in mentioned:
            mentioned.append("CAREER")

    has_personality_signal = (
        bool(_PERSONALITY_KW_PATTERN.search(q))
        or bool(_MBTI_PATTERN.search(q))
        or bool(intent.get("mbti_dimensions", []))
        or ("PERSONALITY" in mentioned)
    )
    if has_personality_signal and "PERSONALITY" not in mentioned:
        mentioned.append("PERSONALITY")

    if _MAJOR_CUE_PATTERN.search(q) and "MAJOR" not in mentioned:
        mentioned.append("MAJOR")
    if _SKILL_CUE_PATTERN.search(q) and "SKILL" not in mentioned:
        mentioned.append("SKILL")

    # Rule 1: hỏi rõ "tính cách gì/nào" => asked = PERSONALITY.
    if _ASK_PERSONALITY_PATTERN.search(q):
        asked = "PERSONALITY"
        # Inject field_context nếu câu hỏi đề cập lĩnh vực cụ thể (IT, kinh tế,...)
        # Dùng để generate_answer biết cần filter PERSONALITY theo lĩnh vực đó
        for pat, field_name in _FIELD_CONTEXT_HINTS:
            if pat.search(q):
                intent["field_context"] = field_name
                break

    # Rule 2: so sánh giữa 2 nghề => asked = CAREER.
    if is_comp and has_direct_career_alias:
        asked = "CAREER"

    # Rule 3: câu mơ hồ nhưng có dấu hiệu nghề + tính cách/skill => hỏi nghề.
    if asked == "UNKNOWN":
        if _CAREER_CUE_PATTERN.search(q):
            asked = "CAREER"
        elif has_personality_signal and (_SKILL_CUE_PATTERN.search(q) or bool(negated)):
            asked = "CAREER"
        elif _MAJOR_CUE_PATTERN.search(q):
            asked = "MAJOR"

    # Rule 4: nếu user vừa nói personality vừa hỏi nghề/ngành, ưu tiên theo câu hỏi.
    # NGOẠI LỆ QUAN TRỌNG: Nếu câu hỏi hỏi rõ "tính cách gì/nào" → GIỮ NGUYÊN asked=PERSONALITY
    # dù có "hợp làm", "hợp nghề"... vì đây là hỏi ngược: "lĩnh vực X cần tính cách gì?"
    if asked == "PERSONALITY":
        is_asking_personality_explicitly = bool(_ASK_PERSONALITY_PATTERN.search(q))
        if is_asking_personality_explicitly:
            # Giữ asked=PERSONALITY, chỉ inject field_context để biết cần filter theo lĩnh vực
            pass
        elif _CAREER_CUE_PATTERN.search(q) and "CAREER" in mentioned:
            # Ví dụ: "tính cách ISTJ nên làm tester hay dev?"
            asked = "CAREER"
        elif _MAJOR_CUE_PATTERN.search(q) and "MAJOR" in mentioned and "CAREER" not in mentioned:
            # Ví dụ: "hướng nội hợp ngành nào?"
            asked = "MAJOR"

    # Sắp thứ tự mentioned để targeted query đi đúng hướng hơn.
    if asked == "CAREER":
        if is_comp:
            priority = ["CAREER", "SKILL", "MAJOR", "PERSONALITY", "SUBJECT", "TEACHER"]
        else:
            priority = ["SKILL", "MAJOR", "PERSONALITY", "CAREER", "SUBJECT", "TEACHER"]
    elif asked == "MAJOR":
        priority = ["PERSONALITY", "SKILL", "CAREER", "MAJOR", "SUBJECT", "TEACHER"]
    elif asked == "PERSONALITY":
        if re.search(r"hợp làm|hợp nghề|làm\s+\w+", q_lower, re.IGNORECASE):
            priority = ["CAREER", "MAJOR", "PERSONALITY", "SKILL", "SUBJECT", "TEACHER"]
        else:
            priority = ["MAJOR", "CAREER", "PERSONALITY", "SKILL", "SUBJECT", "TEACHER"]
    else:
        priority = ["PERSONALITY", "SKILL", "MAJOR", "CAREER", "SUBJECT", "TEACHER"]

    mentioned_set = set(mentioned)
    mentioned = [lbl for lbl in priority if lbl in mentioned_set]
    mentioned.extend([lbl for lbl in mentioned_set if lbl not in mentioned])

    intent["keywords"] = _unique_keep_order(keywords)
    intent["mentioned_labels"] = mentioned
    intent["asked_label"] = asked if asked in {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER", "PERSONALITY", "UNKNOWN"} else "UNKNOWN"
    intent["negated_keywords"] = _unique_keep_order(negated)
    intent["is_comparison"] = is_comp
    return intent


def get_relationship_constraint(intent: dict) -> str:
    mentioned = intent.get("mentioned_labels", [])
    asked     = intent.get("asked_label", "UNKNOWN")
    is_comp   = intent.get("is_comparison", False)

    if is_comp and (asked == "CAREER" or "CAREER" in mentioned):
        return (
            "So sánh 2 nghề nghiệp được nêu trong câu hỏi dựa trên dữ liệu graph. "
            "BẮT BUỘC gồm 4 phần: "
            "1) Mô tả ngắn từng nghề (description/role). "
            "2) Công việc chính từng nghề (job_tasks). "
            "3) Cơ hội/triển vọng (market). "
            "4) Ngành học đề xuất cho từng nghề (major_codes hoặc recommended_majors trong DB). "
            "Cuối cùng kết luận nên ưu tiên nghề nào dựa trên tính cách/ưu tiên user nêu trong câu hỏi. "
            "Tuyệt đối không dùng kiến thức ngoài graph."
        )

    if is_comp and "MAJOR" in mentioned:
        return RELATIONSHIP_CONSTRAINTS.get(("MAJOR", "MAJOR"), "")

    for m in ([mentioned[0]] if mentioned else []) + mentioned:
        key = (m, asked)
        if key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[key]

    # Self-query fallback
    if asked != "UNKNOWN":
        self_key = (asked, asked)
        if self_key in RELATIONSHIP_CONSTRAINTS:
            return RELATIONSHIP_CONSTRAINTS[self_key]

    return "Trả lời theo đúng câu hỏi, chỉ dùng dữ liệu trong Knowledge Graph."


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7: COMMUNITY-AWARE TRAVERSAL
# ══════════════════════════════════════════════════════════════════════════════

# Extended props được fetch từ DB và đưa vào context cho LLM
EXTENDED_PROPS: dict[str, list[str]] = {
    "SUBJECT": [
        "course_description", "courses_goals", "assessment",
        "learning_resources", "course_requirements_and_expectations",
    ],
    "CAREER": [
        "description", "job_tasks", "field_name", "market",
    ],
    "MAJOR": [
        "philosophy_and_objectives", "admission_requirements",
        "learning_outcomes", "curriculum_structure_and_content",
    ],
    "TEACHER":     ["email", "title"],
    "SKILL":       ["skill_type"],
    "PERSONALITY": [
        "code", "description", "structure",
        "strengths", "weaknesses", "work_environment", "suitable_fields",
    ],
}

# Targeted Queries — trả về các columns chuẩn: name, label, code, rel_types, node_names, hops
# + extended cols: course_description, semester, required_type
TARGETED_QUERIES: dict[tuple[str, str], str] = {

    # ── Multi-hop: PERSONALITY → CAREER → SKILL ← SUBJECT (4-hop tư vấn ngược) ─
    # Dùng khi câu hỏi: "MBTI X nên học môn gì?" → full chain
    ("PERSONALITY", "SUBJECT_VIA_CAREER"): """
        MATCH (p:PERSONALITY)-[:SUITS_CAREER]->(c:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT)
        WHERE p.personality_key = toUpper($kw)
           OR toLower(p.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['SUITS_CAREER','REQUIRES','PROVIDES'] AS rel_types,
               [p.name, c.name, sk.name, n.name] AS node_names,
               3 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY c.name, sk.name, n.name LIMIT 60
    """,
    # ── Multi-hop: PERSONALITY → CAREER → SKILL (2-hop skills từ MBTI) ──────
    ("PERSONALITY", "SKILL_VIA_CAREER"): """
        MATCH (p:PERSONALITY)-[:SUITS_CAREER]->(c:CAREER)-[:REQUIRES]->(n:SKILL)
        WHERE p.personality_key = toUpper($kw)
           OR toLower(p.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['SUITS_CAREER','REQUIRES'] AS rel_types,
               [p.name, c.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY c.name, n.name LIMIT 60
    """,
    # ── Multi-hop: PERSONALITY → MAJOR → SUBJECT (ngành → môn từ MBTI) ──────
    ("PERSONALITY", "SUBJECT_VIA_MAJOR"): """
        MATCH (p:PERSONALITY)-[:SUITS_MAJOR]->(m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(n:SUBJECT)
        WHERE p.personality_key = toUpper($kw)
           OR toLower(p.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['SUITS_MAJOR','MAJOR_OFFERS_SUBJECT'] AS rel_types,
               [p.name, m.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY m.name, n.name LIMIT 60
    """,

    # ── Academic ──────────────────────────────────────────────────────────────
    ("MAJOR", "SUBJECT"): """
        MATCH (start:MAJOR)-[r:MAJOR_OFFERS_SUBJECT]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['MAJOR_OFFERS_SUBJECT'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               r.semester AS semester,
               r.required_type AS required_type,
               n.course_description AS course_description
        ORDER BY r.required_type DESC, r.semester ASC, n.name ASC
        LIMIT 100
    """,
    ("MAJOR", "TEACHER"): """
        MATCH (n:TEACHER)-[:TEACH]->(sub:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(start:MAJOR)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['MAJOR_OFFERS_SUBJECT','TEACH'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("TEACHER", "SUBJECT"): """
        MATCH (start:TEACHER)-[:TEACH]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['TEACH'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("TEACHER", "MAJOR"): """
        MATCH (start:TEACHER)-[:TEACH]->(sub:SUBJECT)<-[:MAJOR_OFFERS_SUBJECT]-(n:MAJOR)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['TEACH','MAJOR_OFFERS_SUBJECT'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "TEACHER"): """
        MATCH (n:TEACHER)-[:TEACH]->(start:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['TEACH'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    # Self: thông tin chi tiết môn học + môn tiên quyết
    ("SUBJECT", "SUBJECT"): """
        MATCH (start:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN start.name AS name, labels(start)[0] AS label, start.code AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type,
               start.course_description AS course_description
        ORDER BY start.name LIMIT 10
        UNION
        MATCH (start:SUBJECT)-[:PREREQUISITE_FOR]->(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['PREREQUISITE_FOR'] AS rel_types,
               [start.name, n.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 30
    """,
    # Self: thông tin giảng viên
    ("TEACHER", "TEACHER"): """
        MATCH (start:TEACHER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.teacher_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
    """,

    # ── Career cluster ────────────────────────────────────────────────────────
    ("MAJOR", "CAREER"): """
        MATCH (start:MAJOR)-[:LEADS_TO]->(n:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['LEADS_TO'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("CAREER", "SKILL"): """
        MATCH (start:CAREER)-[:REQUIRES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['REQUIRES'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("CAREER", "SUBJECT"): """
        MATCH (start:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        OPTIONAL MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(n)
        WHERE m.code IN start.major_codes
        WITH start, sk, n,
             count(DISTINCT m) AS major_match,
             size([(s2:SUBJECT)-[:PROVIDES]->(sk) | s2]) AS skill_breadth
        ORDER BY major_match DESC, skill_breadth ASC, n.name ASC
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['REQUIRES','PROVIDES'] AS rel_types,
               [start.name, sk.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        LIMIT 30
    """,
    ("CAREER", "MAJOR"): """
        MATCH (n:MAJOR)-[:LEADS_TO]->(start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['LEADS_TO'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("MAJOR", "SKILL"): """
        MATCH (start:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(sub:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['MAJOR_OFFERS_SUBJECT','PROVIDES'] AS rel_types,
               [start.name, sub.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "MAJOR"): """
        MATCH (n:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(sub:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['MAJOR_OFFERS_SUBJECT','PROVIDES'] AS rel_types,
               [n.name, sub.name, start.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "CAREER"): """
        MATCH (n:CAREER)-[:REQUIRES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['REQUIRES'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SKILL", "SUBJECT"): """
        MATCH (n:SUBJECT)-[:PROVIDES]->(start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['PROVIDES'] AS rel_types, [n.name, start.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type,
               n.course_description AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "SKILL"): """
        MATCH (start:SUBJECT)-[:PROVIDES]->(n:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['PROVIDES'] AS rel_types, [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    ("SUBJECT", "CAREER"): """
        MATCH (start:SUBJECT)-[:PROVIDES]->(sk:SKILL)<-[:REQUIRES]-(n:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['PROVIDES','REQUIRES'] AS rel_types,
               [start.name, sk.name, n.name] AS node_names,
               2 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    # Self: thông tin chi tiết nghề nghiệp + ngành học đề xuất qua major_codes
    ("CAREER", "CAREER"): """
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
        UNION
        MATCH (start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        MATCH (m:MAJOR) WHERE m.code IN start.major_codes
        RETURN m.name AS name, labels(m)[0] AS label, m.code AS code,
               ['RECOMMENDED_MAJOR'] AS rel_types,
               [start.name, m.name] AS node_names, 1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY m.name LIMIT 20
    """,
    # Skill self-lookup
    ("SKILL", "SKILL"): """
        MATCH (start:SKILL)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.skill_key) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 10
    """,

    # ── Personality cluster (MBTI v6 — dùng SUITS_MAJOR / SUITS_CAREER) ───────
    # MBTI → Career (primary edge SUITS_CAREER)
    ("PERSONALITY", "CAREER"): """
        MATCH (start:PERSONALITY)-[:SUITS_CAREER]->(n:CAREER)
        WHERE start.personality_key = toUpper($kw)
           OR toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['SUITS_CAREER'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 50
    """,
    # MBTI → Major (primary edge SUITS_MAJOR)
    ("PERSONALITY", "MAJOR"): """
        MATCH (start:PERSONALITY)-[:SUITS_MAJOR]->(n:MAJOR)
        WHERE start.personality_key = toUpper($kw)
           OR toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['SUITS_MAJOR'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 30
    """,
    # MBTI self-lookup (trả về node đầy đủ để LLM dùng suitable_fields)
    ("PERSONALITY", "PERSONALITY"): """
        MATCH (start:PERSONALITY)
        WHERE start.personality_key = toUpper($kw)
           OR toLower(start.name) CONTAINS toLower($kw)
        RETURN start.name AS name, labels(start)[0] AS label, null AS code,
               [] AS rel_types, [start.name] AS node_names, 0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY start.name LIMIT 5
    """,
    # Career → MBTI (nghề này hợp tính cách nào)
    ("CAREER", "PERSONALITY"): """
        MATCH (n:PERSONALITY)-[:SUITS_CAREER]->(start:CAREER)
        WHERE toLower(start.name) CONTAINS toLower($kw)
           OR toLower(start.career_key) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['SUITS_CAREER'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 20
    """,
    # Major → MBTI (ngành này hợp tính cách nào) — từ cạnh SUITS_MAJOR
    ("MAJOR", "PERSONALITY"): """
        MATCH (n:PERSONALITY)-[:SUITS_MAJOR]->(start:MAJOR)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['SUITS_MAJOR'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 20
    """,
    # Lĩnh vực → MBTI: tìm tất cả PERSONALITY có suitable_fields chứa lĩnh vực $kw
    # Dùng khi câu hỏi là "tính cách gì hợp làm IT" — keyword là tên lĩnh vực chứ không phải ngành
    ("FIELD", "PERSONALITY"): """
        MATCH (n:PERSONALITY)
        WHERE n.suitable_fields IS NOT NULL
          AND toLower(n.suitable_fields) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['suitable_fields_match'] AS rel_types,
               [n.name] AS node_names,
               0 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 20
    """,
    # Subject → MBTI (dự phòng CULTIVATES)
    ("SUBJECT", "PERSONALITY"): """
        MATCH (start:SUBJECT)-[:CULTIVATES]->(n:PERSONALITY)
        WHERE toLower(start.name) CONTAINS toLower($kw) OR start.code = $kw
        RETURN n.name AS name, labels(n)[0] AS label, null AS code,
               ['CULTIVATES'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               null AS semester, null AS required_type, null AS course_description
        ORDER BY n.name LIMIT 20
    """,
    # MBTI → Subject (dự phòng CULTIVATES)
    ("PERSONALITY", "SUBJECT"): """
        MATCH (n:SUBJECT)-[:CULTIVATES]->(start:PERSONALITY)
        WHERE start.personality_key = toUpper($kw)
           OR toLower(start.name) CONTAINS toLower($kw)
        RETURN n.name AS name, labels(n)[0] AS label, n.code AS code,
               ['CULTIVATES'] AS rel_types,
               [start.name, n.name] AS node_names,
               1 AS hops,
               n.course_description AS course_description,
               null AS semester, null AS required_type
        ORDER BY n.name LIMIT 20
    """,
}


def _add_node_and_paths(rec, all_nodes: list, all_paths: list):
    """Thêm node và path vào context, kèm extended props."""
    node = {
        "name":  rec["name"],
        "label": rec["label"],
        "code":  rec.get("code"),
        "hops":  rec["hops"],
    }
    # Extended props từ targeted query
    for field in ("course_description", "semester", "required_type"):
        val = rec.get(field)
        if val is not None:
            node[field] = val

    all_nodes.append(node)

    node_names = rec.get("node_names") or []
    rel_types  = rec.get("rel_types") or []
    for i, rel in enumerate(rel_types):
        all_paths.append({
            "from":     node_names[i]   if i < len(node_names) else "",
            "to":       node_names[i+1] if i+1 < len(node_names) else "",
            "relation": rel,
            "hop":      i + 1,
        })


def fetch_node_details(driver, nodes: list[dict]) -> list[dict]:
    """
    Enrich nodes với extended properties từ DB.
    Chỉ fetch khi node chưa có extended props và là SUBJECT/CAREER/MAJOR.
    """
    to_fetch: dict[str, list[str]] = {
        "SUBJECT": [], "CAREER": [], "MAJOR": [], "PERSONALITY": []
    }
    node_map: dict[str, dict] = {}

    for n in nodes:
        label = n.get("label", "")
        name  = n.get("name", "")
        if not name:
            continue
        node_map[name] = n
        if label in to_fetch:
            has_extended = any(n.get(p) for p in EXTENDED_PROPS.get(label, []))
            if not has_extended:
                to_fetch[label].append(name)

    with driver.session() as session:
        if to_fetch["SUBJECT"]:
            rows = session.run("""
                MATCH (n:SUBJECT) WHERE n.name IN $names
                RETURN n.name AS name,
                       n.course_description AS course_description,
                       n.courses_goals AS courses_goals,
                       n.assessment AS assessment,
                       n.learning_resources AS learning_resources,
                       n.course_requirements_and_expectations AS course_requirements_and_expectations
            """, names=to_fetch["SUBJECT"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

        if to_fetch["CAREER"]:
            rows = session.run("""
                MATCH (n:CAREER) WHERE n.name IN $names
                OPTIONAL MATCH (m:MAJOR) WHERE m.code IN n.major_codes
                WITH n, collect({name: m.name, code: m.code}) AS recommended_majors
                RETURN n.name AS name,
                       n.description AS description,
                       n.job_tasks AS job_tasks,
                       n.field_name AS field_name,
                       n.market AS market,
                       n.education_certification AS education_certification,
                       n.major_codes AS major_codes,
                       recommended_majors
            """, names=to_fetch["CAREER"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

        if to_fetch["MAJOR"]:
            rows = session.run("""
                MATCH (n:MAJOR) WHERE n.name IN $names
                RETURN n.name AS name,
                       n.philosophy_and_objectives AS philosophy_and_objectives,
                       n.admission_requirements AS admission_requirements,
                       n.learning_outcomes AS learning_outcomes,
                       n.curriculum_structure_and_content AS curriculum_structure_and_content
            """, names=to_fetch["MAJOR"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

        if to_fetch["PERSONALITY"]:
            rows = session.run("""
                MATCH (n:PERSONALITY) WHERE n.name IN $names
                RETURN n.name             AS name,
                       n.code             AS code,
                       n.description      AS description,
                       n.structure        AS structure,
                       n.strengths        AS strengths,
                       n.weaknesses       AS weaknesses,
                       n.work_environment AS work_environment,
                       n.suitable_fields  AS suitable_fields
            """, names=to_fetch["PERSONALITY"]).data()
            for r in rows:
                if r["name"] in node_map:
                    for k, v in r.items():
                        if k != "name" and v is not None:
                            node_map[r["name"]][k] = v

    return nodes


def multihop_traversal_community_aware(
    driver,
    keywords:      list[str],
    max_hops:      int = MAX_HOPS,
    intent:        dict | None = None,
    community_def: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Traversal 3-phase community-aware:
    Phase 1 — TARGETED Cypher theo intent.
    Phase 2 — BFS label-scoped (KHÔNG dùng community number filter vì không đồng nhất).
    Phase 3 — CROSS-CLUSTER BRIDGE (L2/L3).
    """
    all_nodes:  list[dict] = []
    all_paths:  list[dict] = []
    seen_names: set[str]   = set()

    mentioned_labels = (intent or {}).get("mentioned_labels", [])
    asked_label      = (intent or {}).get("asked_label", "UNKNOWN")
    first_mentioned  = mentioned_labels[0] if mentioned_labels else None

    if community_def:
        allowed_labels = community_def["node_labels"]
        level          = community_def["level"]
        comm_id        = community_def["id"]
    else:
        allowed_labels = {"MAJOR", "SUBJECT", "SKILL", "CAREER", "TEACHER", "PERSONALITY"}
        level          = 1
        comm_id        = "L1_GLOBAL"

    print(f"  [community] Routing to: {comm_id} (Level {level})")
    print(f"  [community] Scope labels: {allowed_labels}")

    # ── Phase 1: Targeted query ───────────────────────────────────────────────
    targeted_key    = (first_mentioned, asked_label) if first_mentioned else None
    targeted_cypher = TARGETED_QUERIES.get(targeted_key) if targeted_key else None

    # Fallback: self-lookup
    if not targeted_cypher and asked_label != "UNKNOWN":
        self_key = (asked_label, asked_label)
        if self_key in TARGETED_QUERIES:
            targeted_key    = self_key
            targeted_cypher = TARGETED_QUERIES[self_key]

    if targeted_cypher:
        with driver.session() as session:
            for kw in keywords:
                try:
                    for rec in session.run(targeted_cypher, kw=kw):
                        _add_node_and_paths(rec, all_nodes, all_paths)
                except Exception as e:
                    print(f"  [targeted] WARNING: {e}")
        if all_nodes:
            print(f"  [targeted] ({targeted_key}) → {len(all_nodes)} nodes")

    # ── Phase 1a-fuzzy: Fuzzy Matching fallback ───────────────────────────────
    # Nếu targeted query không tìm thấy gì, thử so khớp mờ tên node trong DB
    # để xử lý lỗi chính tả / biến thể tên entity từ user
    if not all_nodes and keywords and targeted_cypher:
        _FUZZY_SEED_QUERY = """
            MATCH (n)
            WHERE (n:MAJOR OR n:SUBJECT OR n:SKILL OR n:CAREER OR n:TEACHER OR n:PERSONALITY)
            RETURN n.name AS name, labels(n)[0] AS label
            LIMIT 2000
        """
        try:
            with driver.session() as session:
                all_names_in_db = [
                    (r["name"], r["label"])
                    for r in session.run(_FUZZY_SEED_QUERY).data()
                    if r.get("name")
                ]
            db_names = [n for n, _ in all_names_in_db]
            for kw in keywords:
                fuzzy_match = fuzzy_match_entity(kw, db_names, threshold=0.65)
                if fuzzy_match and fuzzy_match.lower() != kw.lower():
                    print(f"  [fuzzy] '{kw}' → '{fuzzy_match}' (fallback)")
                    with driver.session() as session:
                        try:
                            for rec in session.run(targeted_cypher, kw=fuzzy_match):
                                _add_node_and_paths(rec, all_nodes, all_paths)
                        except Exception as e:
                            print(f"  [fuzzy] WARNING: {e}")
            if all_nodes:
                print(f"  [fuzzy] recovered {len(all_nodes)} nodes via fuzzy match")
        except Exception as e:
            print(f"  [fuzzy] seed query WARNING: {e}")

    # ── Phase 1c: Field-context PERSONALITY lookup ────────────────────────────
    # Khi câu hỏi là "tính cách gì hợp làm IT" → asked=PERSONALITY, field_context="Công nghệ thông tin"
    # Cần tìm tất cả PERSONALITY có suitable_fields chứa lĩnh vực đó
    field_context = (intent or {}).get("field_context")
    if asked_label == "PERSONALITY" and field_context:
        field_query = TARGETED_QUERIES.get(("FIELD", "PERSONALITY"))
        if field_query:
            # Tạo danh sách alias để tìm kiếm (tiếng Việt, tiếng Anh, viết tắt)
            _FIELD_ALIASES: dict[str, list[str]] = {
                "Công nghệ thông tin": [
                    "Công nghệ thông tin", "Information Technology", "CNTT", "IT",
                    "công nghệ", "technology", "phần mềm", "software",
                ],
                "Kinh tế - Quản trị": [
                    "Kinh tế", "Quản trị", "Economics", "Business", "Management",
                    "Tài chính", "Finance", "Kế toán", "Accounting",
                ],
                "Khoa học dữ liệu": [
                    "Khoa học dữ liệu", "Data Science", "Data", "dữ liệu",
                    "phân tích dữ liệu", "Data Analytics",
                ],
                "Giáo dục": ["Giáo dục", "Education", "sư phạm", "đào tạo"],
                "Y tế - Sức khỏe": ["Y tế", "Health", "y khoa", "dược"],
            }
            aliases = _FIELD_ALIASES.get(field_context, [field_context])

            with driver.session() as session:
                rows = []
                for alias_kw in aliases:
                    try:
                        found = list(session.run(field_query, kw=alias_kw))
                        rows.extend(found)
                        if found:
                            print(f"  [field_ctx] alias='{alias_kw}' → {len(found)} hits")
                    except Exception as e:
                        print(f"  [field_ctx] WARNING alias='{alias_kw}': {e}")

                # Dedup theo name trước khi add
                seen_fc: set[str] = set()
                for rec in rows:
                    if rec.get("name") not in seen_fc:
                        seen_fc.add(rec.get("name", ""))
                        _add_node_and_paths(rec, all_nodes, all_paths)
                if rows:
                    print(f"  [field_ctx] total ({field_context}) → {len(seen_fc)} personality nodes")

        # Cũng tìm MAJOR thuộc lĩnh vực đó để làm context cho LLM
        if field_context:
            _intent_kws = (intent or {}).get("keywords", [])
            major_kws = [kw for kw in _intent_kws if kw.lower() in (
                "công nghệ thông tin", "it", "cntt", "kinh tế", "tài chính", "data", "dữ liệu"
            )]
            # Fallback: inject từ khóa đầu của field_context (vd: "Công nghệ" từ "Công nghệ thông tin")
            if not major_kws:
                major_kws = [field_context.split()[0].lower()] if field_context else []
            major_query = TARGETED_QUERIES.get(("MAJOR", "PERSONALITY"))
            if major_query and major_kws:
                with driver.session() as session:
                    for kw in major_kws:
                        try:
                            for rec in session.run(major_query, kw=kw):
                                _add_node_and_paths(rec, all_nodes, all_paths)
                        except Exception as e:
                            print(f"  [field_ctx_major] WARNING: {e}")


    # ── Phase 1b: MBTI fallback — nếu targeted query không tìm được gì ────────
    # Đọc node PERSONALITY đầy đủ (có suitable_fields) để LLM tự parse ngành/nghề
    if not all_nodes and asked_label in ("MAJOR", "CAREER", "PERSONALITY", "UNKNOWN"):
        mbti_kws = [kw for kw in keywords
                    if re.match(r'^(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP'
                                r'|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)$',
                                kw, re.IGNORECASE)]
        if mbti_kws:
            with driver.session() as session:
                for mbti_code in mbti_kws:
                    try:
                        rows = session.run("""
                            MATCH (p:PERSONALITY)
                            WHERE p.personality_key = toUpper($code)
                               OR p.code = toUpper($code)
                            RETURN p.personality_key AS name,
                                   'PERSONALITY'     AS label,
                                   null              AS code,
                                   []                AS rel_types,
                                   [p.personality_key] AS node_names,
                                   0                 AS hops,
                                   null AS semester, null AS required_type,
                                   null AS course_description
                        """, code=mbti_code).data()
                        for rec in rows:
                            _add_node_and_paths(rec, all_nodes, all_paths)
                    except Exception as e:
                        print(f"  [mbti fallback] WARNING: {e}")
            if all_nodes:
                print(f"  [mbti fallback] Found PERSONALITY node for {mbti_kws}")

    # ── Phase 1d: Multi-hop PERSONALITY queries (tư vấn ngược đầy đủ) ────────
    # Khi hỏi "MBTI X nên học môn gì?" hoặc "cần kỹ năng gì?" → chạy 3/4-hop chain
    mbti_kws_all = [kw for kw in keywords
                    if re.match(r'^(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP'
                                r'|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)$',
                                kw, re.IGNORECASE)]
    if mbti_kws_all and asked_label in ("SUBJECT", "SKILL", "MAJOR", "CAREER", "UNKNOWN"):
        # Chọn đúng multi-hop query dựa trên asked_label
        multihop_keys = []
        if asked_label in ("SUBJECT", "UNKNOWN"):
            multihop_keys = [
                ("PERSONALITY", "SUBJECT_VIA_CAREER"),  # MBTI→Career→Skill←Subject
                ("PERSONALITY", "SUBJECT_VIA_MAJOR"),   # MBTI→Major→Subject
            ]
        elif asked_label == "SKILL":
            multihop_keys = [("PERSONALITY", "SKILL_VIA_CAREER")]
        elif asked_label == "MAJOR":
            multihop_keys = [("PERSONALITY", "MAJOR")]   # already standard, but add SUBJECT_VIA_MAJOR for context
            multihop_keys.append(("PERSONALITY", "SUBJECT_VIA_MAJOR"))

        if multihop_keys:
            with driver.session() as session:
                for mhk in multihop_keys:
                    mhq = TARGETED_QUERIES.get(mhk)
                    if not mhq:
                        continue
                    for mbti_code in mbti_kws_all:
                        try:
                            rows = list(session.run(mhq, kw=mbti_code))
                            for rec in rows:
                                _add_node_and_paths(rec, all_nodes, all_paths)
                            if rows:
                                print(f"  [multihop] {mhk} code={mbti_code} → {len(rows)} nodes")
                        except Exception as e:
                            print(f"  [multihop] WARNING {mhk}: {e}")

    # ── Phase 2: BFS label-scoped ─────────────────────────────────────────────
    # Dùng allowed_labels filter, KHÔNG filter theo community number
    # (vì MAJOR=2, SUBJECT=2, TEACHER=0 tại L2 — không đồng nhất)
    label_clauses = " OR ".join(f"n:{lbl}" for lbl in allowed_labels)

    with driver.session() as session:
        for kw in keywords:
            seed_rows = session.run("""
                MATCH (seed)
                WHERE (seed:MAJOR OR seed:SUBJECT OR seed:SKILL
                       OR seed:CAREER OR seed:TEACHER OR seed:PERSONALITY)
                  AND (toLower(seed.name) CONTAINS toLower($kw)
                       OR (seed.code IS NOT NULL AND seed.code = $kw)
                       OR (seed.career_key IS NOT NULL
                           AND toLower(seed.career_key) CONTAINS toLower($kw))
                       OR (seed.teacher_key IS NOT NULL
                           AND toLower(seed.teacher_key) CONTAINS toLower($kw))
                       OR (seed.skill_key IS NOT NULL
                           AND toLower(seed.skill_key) CONTAINS toLower($kw))
                       OR (seed.personality_key IS NOT NULL
                           AND toLower(seed.personality_key) CONTAINS toLower($kw))
                       OR (seed.category IS NOT NULL
                           AND toLower(seed.category) CONTAINS toLower($kw)))
                WITH seed, size([(seed)-[]-() | 1]) AS degree
                RETURN seed
                ORDER BY degree DESC
                LIMIT 3
            """, kw=kw).data()
            seeds = [r["seed"] for r in seed_rows]

            for seed in seeds:
                seed_name = seed.get("name", "")
                if seed_name in seen_names:
                    continue
                seen_names.add(seed_name)

                # Thêm seed node vào context (kèm extended props)
                seed_labels = list(seed.labels) if hasattr(seed, "labels") else []
                seed_label  = seed_labels[0] if seed_labels else "UNKNOWN"
                seed_node   = {
                    "name":  seed_name,
                    "label": seed_label,
                    "code":  seed.get("code"),
                    "hops":  0,
                }
                for prop in EXTENDED_PROPS.get(seed_label, []):
                    val = seed.get(prop)
                    if val is not None:
                        seed_node[prop] = val
                all_nodes.append(seed_node)

                # BFS label-scoped traversal
                traversal_query = f"""
                    MATCH path = (start)-[*1..{max_hops}]-(n)
                    WHERE start.name = $seed_name
                      AND ({label_clauses})
                    WITH n, path,
                         [r IN relationships(path) | type(r)] AS rel_types,
                         [x IN nodes(path) | x.name]          AS node_names
                    RETURN DISTINCT
                        n.name                 AS name,
                        labels(n)[0]           AS label,
                        n.code                 AS code,
                        n.course_description   AS course_description,
                        null                   AS semester,
                        null                   AS required_type,
                        rel_types,
                        node_names,
                        length(path)           AS hops
                    ORDER BY hops ASC
                    LIMIT 60
                """
                try:
                    for rec in session.run(traversal_query, seed_name=seed_name):
                        _add_node_and_paths(rec, all_nodes, all_paths)
                except Exception as e:
                    print(f"  [BFS] WARNING seed={seed_name}: {e}")

    # ── Phase 3: Cross-cluster bridge (L2/L3) ────────────────────────────────
    if level >= 2 and asked_label not in (None, "UNKNOWN"):
        bridge_pairs = [
            ("L2_ACADEMIC", "CAREER",
             "MATCH (m:MAJOR)-[:LEADS_TO]->(n:CAREER) WHERE m.name IN $names "
             "RETURN n.name AS name, 'CAREER' AS label, null AS code, "
             "['LEADS_TO'] AS rel_types, [m.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description"),

            ("L2_CAREER_ALIGNMENT", "SUBJECT",
             "MATCH (c:CAREER)-[:REQUIRES]->(sk:SKILL)<-[:PROVIDES]-(n:SUBJECT) "
             "WHERE c.name IN $names "
             "OPTIONAL MATCH (m:MAJOR)-[:MAJOR_OFFERS_SUBJECT]->(n) WHERE m.code IN c.major_codes "
             "WITH c, sk, n, count(DISTINCT m) AS major_match, "
             "size([(s2:SUBJECT)-[:PROVIDES]->(sk) | s2]) AS skill_breadth "
             "ORDER BY major_match DESC, skill_breadth ASC "
             "RETURN n.name AS name, 'SUBJECT' AS label, n.code AS code, "
             "['REQUIRES','PROVIDES'] AS rel_types, [c.name, sk.name, n.name] AS node_names, 2 AS hops, "
             "null AS semester, null AS required_type, n.course_description AS course_description "
             "LIMIT 20"),

            # Bridge: L2_PERSONALITY_FIT → Career (SUITS_CAREER)
            ("L2_PERSONALITY_FIT", "CAREER",
             "MATCH (p:PERSONALITY)-[:SUITS_CAREER]->(n:CAREER) "
             "WHERE p.name IN $names OR p.personality_key IN $names "
             "RETURN n.name AS name, 'CAREER' AS label, null AS code, "
             "['SUITS_CAREER'] AS rel_types, [p.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description "
             "LIMIT 50"),

            # Bridge: L2_PERSONALITY_FIT → Major (SUITS_MAJOR)
            ("L2_PERSONALITY_FIT", "MAJOR",
             "MATCH (p:PERSONALITY)-[:SUITS_MAJOR]->(n:MAJOR) "
             "WHERE p.name IN $names OR p.personality_key IN $names "
             "RETURN n.name AS name, 'MAJOR' AS label, n.code AS code, "
             "['SUITS_MAJOR'] AS rel_types, [p.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description "
             "LIMIT 30"),

            # Bridge ngược: MAJOR → PERSONALITY (khi hỏi "tính cách gì hợp làm/học X")
            # Dùng khi seed_names chứa MAJOR nodes → tìm PERSONALITY suits những MAJOR đó
            ("L2_PERSONALITY_FIT", "PERSONALITY",
             "MATCH (n:PERSONALITY)-[:SUITS_MAJOR]->(m:MAJOR) "
             "WHERE m.name IN $names "
             "RETURN n.name AS name, 'PERSONALITY' AS label, null AS code, "
             "['SUITS_MAJOR'] AS rel_types, [m.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description "
             "LIMIT 20"),

            # Bridge ngược qua CAREER: CAREER → PERSONALITY (khi seed là CAREER trong lĩnh vực IT)
            ("L2_PERSONALITY_FIT", "PERSONALITY",
             "MATCH (n:PERSONALITY)-[:SUITS_CAREER]->(c:CAREER) "
             "WHERE c.name IN $names "
             "RETURN n.name AS name, 'PERSONALITY' AS label, null AS code, "
             "['SUITS_CAREER'] AS rel_types, [c.name, n.name] AS node_names, 1 AS hops, "
             "null AS semester, null AS required_type, null AS course_description "
             "LIMIT 20"),
        ]
        seed_names = list({n["name"] for n in all_nodes if n.get("name")})[:20]

        if seed_names:
            with driver.session() as session:
                for bridge_cid, bridge_label, bridge_q in bridge_pairs:
                    if comm_id != bridge_cid:
                        continue
                    if bridge_label != asked_label and asked_label != "UNKNOWN":
                        continue
                    try:
                        for rec in session.run(bridge_q, names=seed_names):
                            _add_node_and_paths(rec, all_nodes, all_paths)
                        print(f"  [bridge] {bridge_cid}→{bridge_label}: added")
                    except Exception as e:
                        print(f"  [bridge] WARNING: {e}")

    return all_nodes, all_paths


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8: GENERATE ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    ai_client:    OpenAI,
    question:     str,
    ranked_nodes: list[dict],
    traversal_paths: list[dict],
    intent:       dict,
    community_def: dict | None = None,
    override_constraint: str | None = None,
) -> str:
    context = json.dumps({
        "ranked_results":  ranked_nodes,
        "traversal_paths": traversal_paths[:60],
    }, ensure_ascii=False, indent=2)

    constraint = (
        override_constraint if override_constraint is not None
        else get_relationship_constraint(intent)
    )

    negated = intent.get("negated_keywords", [])
    if negated:
        constraint += (
            f"\n\nLUAT PHU DINH TUYET DOI: Nguoi dung KHONG muon: {negated}. "
            "LOAI BO HOAN TOAN khoi goi y — day la luat code, khong phai goi y."
        )

    # Đính kèm constraints từ structured preprocessing (nếu có)
    constraints_list = intent.get("constraints", [])
    if constraints_list:
        constraint += (
            f"\n\nRANG BUOC NGUOI DUNG (trich xuat tu phan tich cau hoi): "
            + "; ".join(constraints_list)
        )

    if community_def:
        community_context = (
            f"Tang {community_def['level']} - {community_def['name']}\n"
            f"Muc tieu: {community_def['purpose']}"
        )
    else:
        community_context = "L1 Global - Toan bo he sinh thai dao tao"

    system_prompt = ANSWER_SYSTEM_BASE.format(
        schema=SCHEMA_DESC,
        constraint=constraint,
        community_context=community_context,
    )

    no_data_hint = ""
    if not ranked_nodes:
        no_data_hint = (
            "\n[CANH BAO: Khong tim thay du lieu trong Knowledge Graph. "
            "Thong bao lich su, khong bia thong tin.]"
        )

    # Block [PHAN TICH CAU HOI] — structured intent từ Preprocessing step
    # Giúp LLM biết chính xác mục đích và các ràng buộc, KHÔNG cần tự đoán lại
    analysis_block = ""
    original_context = intent.get("original_context", "")
    structured_intent_label = intent.get("structured_intent", "")
    if original_context or structured_intent_label:
        analysis_block = (
            "\n\n[PHAN TICH CAU HOI - DA DUOC XU LY TRUOC]:\n"
            f"  Loai yeu cau: {structured_intent_label}\n"
            f"  Muc dich: {original_context}\n"
        )
        if constraints_list:
            analysis_block += (
                "  Rang buoc can tuan thu:\n"
                + "".join(f"    - {c}\n" for c in constraints_list)
            )
        if negated:
            analysis_block += (
                f"  Loai tru tuyet doi (exclude=true): {negated}\n"
                "  (Day la ket qua phan tich code — BAN PHAI tuan thu, "
                "khong duoc neu cac muc nay trong cau tra loi.)\n"
            )

    # Nhắc LLM filter theo lĩnh vực khi câu hỏi là "tính cách gì hợp làm X"
    field_context_hint = ""
    field_context = intent.get("field_context")
    if field_context and intent.get("asked_label") == "PERSONALITY":
        field_context_hint = (
            f"\n[HUONG DAN DAC BIET - LINH VUC: {field_context}]: "
            f"Cau hoi hoi tinh cach phu hop voi linh vuc '{field_context}'. "
            f"Tu [DU LIEU GRAPH], CHI liet ke cac PERSONALITY node co suitable_fields "
            f"chua linh vuc '{field_context}' hoac da duoc lien ket (SUITS_MAJOR/SUITS_CAREER) "
            f"voi nganh/nghe thuoc linh vuc '{field_context}'. "
            f"Voi moi tinh cach, giai thich ngan gon TAI SAO phu hop voi linh vuc nay "
            f"(dua vao strengths/structure trong node PERSONALITY). "
            f"DINH DANG: bang markdown | MBTI | Ten tinh cach | Ly do phu hop |, "
            f"sau do them doan tom tat dac diem chung.]"
        )

    # Nhắc LLM không nhắc đến môn đại cương khi đang trả lời câu hỏi gợi ý môn
    excluded_hint = ""
    if intent.get("_exclude_common_subjects"):
        excluded_hint = (
            "\n[LUAT BO SUNG - AP DUNG CHO CAU TRA LOI NAY]: "
            "Day la cau hoi goi y mon hoc. "
            "TUYET DOI KHONG de cap hoac liet ke cac mon sau: "
            "Triet hoc Mac-Lenin, Kinh te chinh tri Mac-Lenin, Chu nghia xa hoi khoa hoc, "
            "Lich su Dang Cong san Viet Nam, Tu tuong Ho Chi Minh, "
            "Giao duc the chat (GDTC), Giao duc quoc phong va an ninh (GDQP), "
            "Kinh te vi mo 1 (KHMI1101), Kinh te vi mo 1 (KHMA1101), Phap luat dai cuong (LUCS1129). "
            "Day la cac mon bat buoc chung moi nganh - khong can tu van rieng.]"
        )

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"Cau hoi: {question}"
                f"{analysis_block}"
                f"\n\n[DU LIEU GRAPH]:\n{context}"
                f"{no_data_hint}"
                f"{field_context_hint}"
                f"{excluded_hint}\n\n"
                "Tra loi CHI dung ten/code tu [DU LIEU GRAPH]:"
            )},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 9: PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
_CTDT_PATTERN = re.compile(
    r"(?:xem|tìm|tải|download|file|chương trình đào tạo|ctđt|ct đt)\s*"
    r"(?:file\s*)?(?:ctđt|ct\s*đt|chương trình đào tạo)?\s*(?:ngành|của ngành)?\s*"
    r"(.+?)(?:\s*(?:ở đâu|tại đâu|tải ở đâu|xem ở đâu|download ở đâu)|\s*\?|$)",
    re.IGNORECASE | re.UNICODE,
)

def detect_ctdt_question(question: str) -> str | None:
    """
    Nếu câu hỏi hỏi về 'xem file CTĐT ngành X ở đâu' (hoặc biến thể),
    trả về tên ngành X. Ngược lại trả về None.
    """
    q = question.strip()
    if not re.search(r"ctđt|ct\s*đt|chương trình đào tạo", q, re.IGNORECASE | re.UNICODE):
        return None
    if not re.search(r"ở đâu|tại đâu|xem|tìm|tải|download|file", q, re.IGNORECASE | re.UNICODE):
        return None
    m = _CTDT_PATTERN.search(q)
    if m:
        major_name = m.group(1).strip(" ?")
        # Loại bỏ các từ thừa ở cuối: "thì", "thì xem", "thì tải"...
        major_name = re.sub(
            r"\s+(?:thì|thì xem|thì tải|thì download|thì ở đâu|thì tại đâu)\s*$",
            "", major_name, flags=re.IGNORECASE | re.UNICODE,
        ).strip(" ?")
        return major_name if major_name else "ngành bạn quan tâm"
    return "ngành bạn quan tâm"
# Câu hỏi về bản thân chatbot
# Pattern nhận diện câu hỏi về identity/capability của chatbot
# Gộp từ _META_PATTERNS (script3) + _SELF_INTRO_PATTERN (script3a)
_SELF_INTRO_PATTERN = re.compile(
    # Từ script3._META_PATTERNS
    r"bạn (là|là gì|là ai|có thể|làm được|giúp được|biết gì|dùng để làm|làm gì|làm đc gì|lm đc gì)\b"
    r"|bạn tên (là |gì)\b"
    r"|(?:giới thiệu|tự giới thiệu).{0,20}(?:bạn|mình|bản thân)\b"
    r"|(?:chatbot|trợ lý|bot).{0,30}(?:này|đây|là gì|làm gì|có thể)\b"
    r"|(?:bạn|mày|m) có (?:thể|biết|làm|hiểu)\b"
    r"|(?:chào|hello|hi|xin chào).{0,30}(?:bạn|bot|chatbot)\b"
    # Từ script3a (mở rộng thêm)
    r"|bạn có thể giúp gì|bạn hỗ trợ gì|bạn giải đáp gì"
    r"|chatbot này là gì|chatbot này làm gì|chatbot này dùng để làm gì"
    r"|bạn có thể trả lời (về|câu hỏi) gì|bạn có thể tư vấn"
    r"|bạn biết gì|bạn hiểu gì|bạn trả lời được gì"
    r"|em có thể hỏi gì|tôi có thể hỏi gì|mình có thể hỏi gì"
    r"|nội dung gì|về nội dung|giải đáp.*nội dung"
    # Biến thể tư vấn (thêm để bắt "bạn tư vấn về gì")
    r"|bạn tư vấn|tư vấn (về|gì|những gì|được gì|về gì)"
    r"|bạn (giúp|hỗ trợ|trả lời).{0,15}(gì|được gì|những gì)"
    r"|(?:cho tôi|cho mình|cho em).{0,20}biết.{0,20}(?:bạn|chatbot|bot)",
    re.IGNORECASE | re.UNICODE,
)

# Alias để tương thích nếu code khác import từ đây
_META_PATTERNS = _SELF_INTRO_PATTERN

def detect_meta_question(question: str) -> bool:
    """Alias của detect_self_intro — giữ để tương thích."""
    return bool(_SELF_INTRO_PATTERN.search(question))



SELF_INTRO_ANSWER = """
Xin chào! Tôi là **NEU AI Assistant**.

Tôi là chatbot hỗ trợ hỏi đáp học thuật về Đại học Kinh tế Quốc dân (NEU).

Tôi có thể giúp bạn:
• Tìm hiểu ngành học và chương trình đào tạo  
• Tra cứu môn học và nội dung môn  
• Tìm giảng viên dạy môn  
• Khám phá kỹ năng từ từng môn học  
• Tìm mối liên hệ giữa ngành học – kỹ năng – nghề nghiệp

Tôi sử dụng **Knowledge Graph + GraphRAG + Neo4j** để truy vấn và tổng hợp thông tin chính xác.

Bạn muốn hỏi gì về việc học tại NEU? 😊
"""

# Alias để tương thích với script3.py (dùng CHATBOT_IDENTITY)
CHATBOT_IDENTITY = SELF_INTRO_ANSWER

# Từ khóa nhận diện câu hỏi ngoài phạm vi (off-topic)
_OFF_TOPIC_PATTERNS = [
    # Thời tiết, tin tức
    re.compile(r"thời tiết|dự báo|mưa|nắng|bão|lũ|động đất|tin tức|báo chí|thời sự", re.IGNORECASE | re.UNICODE),
    # Y tế / sức khỏe cá nhân
    re.compile(r"bệnh viện|thuốc|chữa bệnh|khám bệnh|sức khỏe|triệu chứng|bác sĩ ơi|đau đầu|sốt|cảm cúm", re.IGNORECASE | re.UNICODE),
    # Nấu ăn, thực phẩm
    re.compile(r"nấu ăn|công thức nấu|nguyên liệu nấu|món ăn|thực đơn|ăn gì ngon", re.IGNORECASE | re.UNICODE),
    # Giải trí, phim ảnh, âm nhạc
    re.compile(r"phim (?:hay|mới|chiếu)|bài hát|ca sĩ|diễn viên|xem phim|nghe nhạc|game (?:hay|mới)", re.IGNORECASE | re.UNICODE),
    # Thể thao — kết quả, tỉ số, giải đấu
    re.compile(
        r"bóng đá|kết quả bóng|đội tuyển|trận đấu|giải đấu|bóng rổ|tennis|cầu lông"
        r"|world cup|worldcup|euro|champions league|ngoại hạng anh|la liga|bundesliga"
        r"|tỉ số|tỷ số|chung kết|vô địch|huy chương|olympic|seagame|sea game"
        r"|cầu thủ|vận động viên|hlv|huấn luyện viên đội",
        re.IGNORECASE | re.UNICODE,
    ),
    # Chính trị
    re.compile(r"bầu cử|tổng thống|thủ tướng|quốc hội|đảng phái|chiến tranh|xung đột", re.IGNORECASE | re.UNICODE),
    # Kỹ thuật ngoài phạm vi
    re.compile(r"sửa máy tính|cài windows|sửa điện thoại|hack|virus máy tính", re.IGNORECASE | re.UNICODE),
    # Tình cảm
    re.compile(r"người yêu|tình yêu|chia tay|cưới|hôn nhân|tâm sự|buồn quá|cô đơn", re.IGNORECASE | re.UNICODE),
    # Du lịch thuần túy
    re.compile(r"đặt vé|vé máy bay|khách sạn (?:tốt|rẻ|ở đâu)|du lịch (?:ở đâu|bao nhiêu tiền|mấy ngày)", re.IGNORECASE | re.UNICODE),
    # Tài chính cá nhân
    re.compile(r"mua cổ phiếu nào|đầu tư vào đâu|bitcoin|crypto|giá vàng|tỷ giá hôm nay", re.IGNORECASE | re.UNICODE),
    # Misc: dịch thuật, truyện, công thức toán thuần túy ngoài học thuật
    re.compile(r"dịch sang tiếng|translate|kể chuyện|viết truyện|viết thơ|tử vi|horoscope", re.IGNORECASE | re.UNICODE),
]

# Từ khóa "safe" — nếu câu hỏi chứa các từ này thì KHÔNG phải off-topic dù match pattern trên
_SAFE_KEYWORDS = re.compile(
    r"ngành|chuyên ngành|môn học|giảng viên|sinh viên|đại học|neu|kinh tế quốc dân"
    r"|chương trình đào tạo|tuyển sinh|điểm chuẩn|nghề nghiệp|kỹ năng|mbti|tính cách",
    re.IGNORECASE | re.UNICODE,
)

OFF_TOPIC_ANSWER = (
    "Tôi có thể tư vấn **ngành học và nghề nghiệp** theo định hướng và tính cách của bạn. "
    "Chatbot hiện **không thể trả lời** các câu hỏi không liên quan đến chương trình đào tạo tại NEU, "
    "vui lòng tham khảo các agent khác.\n\n"
    "Tôi có thể giúp bạn về:\n"
    "- 🎓 Ngành học, môn học, chương trình đào tạo tại NEU\n"
    "- 💼 Nghề nghiệp, cơ hội việc làm sau tốt nghiệp\n"
    "- 🧠 Định hướng theo tính cách MBTI\n"
    "- 📊 Điểm chuẩn và chỉ tiêu tuyển sinh\n\n"
    "Bạn có muốn hỏi về những chủ đề trên không?"
)


def detect_off_topic(question: str) -> bool:
    """Trả về True nếu câu hỏi ngoài phạm vi của chatbot."""
    # Nếu câu hỏi có từ khóa học thuật/NEU → không phải off-topic
    if _SAFE_KEYWORDS.search(question):
        return False
    # Kiểm tra pattern off-topic
    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(question):
            return True
    return False


def detect_self_intro(question: str) -> bool:
    """Trả về True nếu user hỏi về bản thân chatbot."""
    return bool(_SELF_INTRO_PATTERN.search(question))

def ask(driver, ai_client: OpenAI, question: str, query_id: str | None = None) -> dict:
    if query_id is None:
        query_id = "q" + uuid.uuid4().hex[:6]

    print(f"\n{'='*60}")
    print(f"Q [{query_id}]: {question}")
    

    if detect_self_intro(question):
        print(f"\nA: {SELF_INTRO_ANSWER}")
        return _build_record(
            query_id, question, SELF_INTRO_ANSWER, [],
            {"asked_label": "SELF_INTRO", "mentioned_labels": [],
             "keywords": [], "negated_keywords": [],
             "community_id": "SELF_INTRO"},
            [], [], "self_intro_static",
        )

    # ── Bước 0-pre-B: Off-topic detection ────────────────────────────────────
    if detect_off_topic(question):
        print(f"\nA: {OFF_TOPIC_ANSWER}")
        return _build_record(
            query_id, question, OFF_TOPIC_ANSWER, [],
            {"asked_label": "OFF_TOPIC", "mentioned_labels": [],
             "keywords": [], "negated_keywords": [],
             "community_id": "OFF_TOPIC"},
            [], [], "off_topic_static",
        )
    
    ctdt_major = detect_ctdt_question(question)
    if ctdt_major is not None:
        answer = (
            f"Để xem thêm thì hãy vào trang courses.neu.edu.vn "
            f"và tìm ngành {ctdt_major} nhé!"
        )
        print(f"\nA: {answer}")
        return _build_record(
            query_id, question, answer, [ctdt_major],
            {"asked_label": "CTDT_REDIRECT", "mentioned_labels": [],
             "keywords": [ctdt_major], "negated_keywords": [],
             "community_id": "CTDT_REDIRECT"},
            [], [], "ctdt_redirect",
        )

    # ── Bước 0-pre: Chỉ tiêu & Điểm chuẩn tuyển sinh ────────────────────────
    admission_answer = handle_admission_question(question, driver=driver)
    if admission_answer is not None:
        print(f"\nA (admission): {admission_answer}")
        return _build_record(
            query_id, question, admission_answer, [],
            {"asked_label": "ADMISSION", "mentioned_labels": [],
             "keywords": [], "negated_keywords": [],
             "community_id": "ADMISSION_STATIC"},
            [], [], "admission_static_lookup",
        )

    # ── Bước 0-pre2: Môn đại cương bắt buộc — câu hỏi "ngành nào không học X" ──
    not_study_answer = handle_which_major_not_study(question)
    if not_study_answer is not None:
        print(f"\nA (not_study_excluded): {not_study_answer}")
        return _build_record(
            query_id, question, not_study_answer, [],
            {"asked_label": "SUBJECT", "mentioned_labels": ["MAJOR", "SUBJECT"],
             "keywords": [], "negated_keywords": [],
             "community_id": "EXCLUDED_SUBJECT_STATIC"},
            [], [], "excluded_subject_static_reply",
        )

    # ── Bước 0: Aggregation Router ────────────────────────────────────────────
    agg_type = detect_aggregation_type(question)
    if agg_type:
        print(f"  [aggregation] {agg_type}")
        agg_nodes = run_aggregation_query(driver, question, agg_type)
        print(f"  [aggregation] {len(agg_nodes)} nodes")

        agg_intent = {
            "keywords": [], "mentioned_labels": [], "negated_keywords": [],
            "is_comparison": False, "agg_type": agg_type,
            "asked_label": (
                "SUBJECT" if "subject" in agg_type else
                "MAJOR"   if "major"   in agg_type else
                "CAREER"  if "career"  in agg_type else
                "SKILL"   if "skill"   in agg_type else "UNKNOWN"
            ),
        }
        agg_constraint = (
            "Câu hỏi thống kê/tập hợp. Dữ liệu đã tổng hợp từ graph. "
            "Trình bày rõ ràng, kèm mã môn/ngành, số liệu (_agg_meta). "
            "Nếu intersection: giải thích đây là môn tất cả ngành đều học. "
            "Nếu ranking: liệt kê từ cao xuống thấp."
        )
        answer = generate_answer(
            ai_client, question, agg_nodes, [],
            intent=agg_intent,
            community_def=COMMUNITY_LEVELS["L1_GLOBAL"],
            override_constraint=agg_constraint,
        )
        print(f"\nA: {answer}")
        return _build_record(query_id, question, answer, [], agg_intent,
                             agg_nodes, [], "aggregation")

    # ── Bước 0b: Expand MBTI code tường minh → keyword ──────────────────────
    expanded_question, mbti_keywords = expand_mbti(question)
    if mbti_keywords:
        print(f"  [mbti] {mbti_keywords}")

    # ── Bước 0c: Expand viết tắt ──────────────────────────────────────────────
    expanded_question, abbrev_keywords = expand_abbreviations(expanded_question)
    if abbrev_keywords:
        print(f"  [abbrev] {abbrev_keywords}")

    # ── Bước 0d: Query Transformation (Self-Correction + Decomposition + Neg) ──
    transformation = query_transformation(ai_client, expanded_question, {})
    rewritten_q   = transformation["rewritten_question"]
    sub_queries    = transformation["sub_queries"]
    extra_neg_kws  = transformation["explicit_exclusions"]
    if rewritten_q != expanded_question:
        print(f"  [transform] rewritten: {rewritten_q[:100]}")
    if sub_queries:
        print(f"  [transform] sub_queries: {sub_queries}")
    if extra_neg_kws:
        print(f"  [transform] exclusions: {extra_neg_kws}")
    query_for_intent = rewritten_q

    # ── Bước 1-pre: Structured Preprocessing (Entity Extraction + Exclude flags) ──
    # Bước này trích xuất entities có exclude=True → loại trừ ở tầng CODE (không dùng LLM nhớ)
    structured = extract_structured_intent(ai_client, query_for_intent)
    excluded_from_structured = build_excluded_names(structured)
    if structured["entities"]:
        print(f"  [structured] intent={structured['intent']} | "
              f"entities={[(e['type'],e['value'],e['exclude']) for e in structured['entities']]}")
    if excluded_from_structured:
        print(f"  [structured] excluded (code-level): {excluded_from_structured}")

    # ── Bước 1: Extract intent (LLM) — trả về cả mbti_dimensions ─────────────
    intent = extract_query_intent(ai_client, query_for_intent)
    intent["keywords"] = list(dict.fromkeys(
        intent["keywords"] + mbti_keywords + abbrev_keywords
    ))
    # Merge structured intent vào intent dict (keywords, negated, constraints, context)
    intent = merge_structured_into_intent(structured, intent)
    # Merge exclusions từ query_transformation
    if extra_neg_kws:
        intent["negated_keywords"] = list(dict.fromkeys(
            intent.get("negated_keywords", []) + extra_neg_kws
        ))
    # Normalize keywords qua synonym layer
    intent["keywords"] = normalize_keywords(intent["keywords"])

    intent = apply_intent_rules(question, intent)

    # ── Bước 1b: Resolve MBTI codes ──────────────────────────────────────────
    # Ưu tiên 1: MBTI code tường minh (INTJ, ESTP...) từ regex expand_mbti
    # Ưu tiên 2: dimensions do LLM suy luận từ từ đồng nghĩa tính cách
    dimensions = intent.get("mbti_dimensions", [])

    if mbti_keywords:
        # Explicit code → dùng trực tiếp, bỏ qua dimensions
        all_mbti_keywords = mbti_keywords
        print(f"  [mbti override] source=explicit code={mbti_keywords[0].upper()}")
    elif dimensions:
        # LLM suy luận dimensions → expand thành MBTI codes
        all_mbti_keywords = resolve_mbti_codes_from_dimensions(dimensions)
        print(f"  [mbti override] source=llm-dimensions dims={dimensions} "
              f"→ {len(all_mbti_keywords)} codes: {all_mbti_keywords}")
    else:
        all_mbti_keywords = []

    if all_mbti_keywords:
        mbti_code = all_mbti_keywords[0].upper()
        # Đảm bảo PERSONALITY có trong mentioned_labels
        if "PERSONALITY" not in intent.get("mentioned_labels", []):
            intent["mentioned_labels"] = ["PERSONALITY"] + [
                l for l in intent.get("mentioned_labels", [])
                if l != "PERSONALITY"
            ]
        # Nếu asked=UNKNOWN → set PERSONALITY
        if intent.get("asked_label") == "UNKNOWN":
            intent["asked_label"] = "PERSONALITY"
        # Inject tất cả MBTI codes vào keywords (để traversal query từng cái)
        existing_kws = [k for k in intent["keywords"]
                        if k.upper() not in {c.upper() for c in all_mbti_keywords}]
        intent["keywords"] = all_mbti_keywords + existing_kws
    intent = apply_intent_rules(question, intent)
    keywords = intent["keywords"]
    print(f"  Keywords: {keywords}")
    print(f"  Intent: mentioned={intent['mentioned_labels']} "
          f"asked={intent['asked_label']} negated={intent['negated_keywords']}")

    # ── Bước 2: Community Routing ─────────────────────────────────────────────
    community_id, community_def = route_to_community(intent)
    intent["community_id"] = community_id

    # ── Bước 3: Community-aware Traversal ────────────────────────────────────
    raw_nodes, traversal_paths = multihop_traversal_community_aware(
        driver, keywords, max_hops=MAX_HOPS,
        intent=intent, community_def=community_def,
    )
    print(f"  Traversal: {len(raw_nodes)} nodes | {len(traversal_paths)} paths")

    # ── Bước 4: Dedup + Negation filter ──────────────────────────────────────
    negated_lower = [kw.lower() for kw in intent.get("negated_keywords", [])]
    seen: dict[tuple, dict] = {}
    for n in raw_nodes:
        key = (n.get("label", ""), n.get("name", ""))
        if key not in seen or (n.get("hops") or 99) < (seen[key].get("hops") or 99):
            seen[key] = n
    context_nodes = [
        n for n in seen.values()
        if not any(neg in (n.get("name") or "").lower() for neg in negated_lower)
    ]

    # ── Bước 4-excluded: Lọc môn đại cương bắt buộc khi câu hỏi là gợi ý môn ──
    _is_confirm = bool(_CONFIRM_SUBJECT_PATTERN.search(question))
    _is_recommend = is_recommend_subject_question(question)
    # Lọc khỏi context nếu là câu hỏi gợi ý MÀ KHÔNG phải câu hỏi xác nhận
    _should_exclude = _is_recommend and not _is_confirm
    context_nodes = filter_excluded_subjects(context_nodes, exclude=_should_exclude)
    # Gắn flag vào intent để generate_answer biết
    intent["_exclude_common_subjects"] = _should_exclude

    # ── Bước 4b-pre: Lọc CAREER node dạng bằng cấp (không phải vị trí công việc) ──
    _DEGREE_PREFIXES = (
        "cử nhân", "kỹ sư", "thạc sĩ", "tiến sĩ", "bác sĩ",
        "bachelor", "master", "engineer",
    )
    def _is_degree_career(node: dict) -> bool:
        if node.get("label") != "CAREER":
            return False
        name_lower = (node.get("name") or "").lower().strip()
        return any(name_lower.startswith(prefix) for prefix in _DEGREE_PREFIXES)

    context_nodes = [n for n in context_nodes if not _is_degree_career(n)]

    # ── Bước 4b: Enrich extended props khi cần ───────────────────────────────
    asked = intent.get("asked_label", "UNKNOWN")
    if asked in ("SUBJECT", "CAREER", "MAJOR", "PERSONALITY") and len(context_nodes) <= 20:
        context_nodes = fetch_node_details(driver, context_nodes)
        print(f"  [enrich] Extended props fetched for: {asked}")
    elif asked == "CAREER" and len(context_nodes) > 20:
        # Luôn enrich CAREER nodes ngay cả khi tổng context_nodes lớn
        career_nodes_exist = any(n.get("label") == "CAREER" for n in context_nodes)
        if career_nodes_exist:
            context_nodes = fetch_node_details(driver, context_nodes)
            print(f"  [enrich] Extended props force-fetched for CAREER (total nodes={len(context_nodes)})")
    elif len(context_nodes) > 20:
        # Luôn enrich PERSONALITY ngay cả khi nhiều nodes (số lượng PERSONALITY có giới hạn)
        pers_exist = any(n.get("label") == "PERSONALITY" for n in context_nodes)
        if pers_exist:
            context_nodes = fetch_node_details(driver, context_nodes)
            print(f"  [enrich] Extended props force-fetched for PERSONALITY (total nodes={len(context_nodes)})")

    # ── Bước 5: LLM answer ───────────────────────────────────────────────────
    answer = generate_answer(
        ai_client, question, context_nodes, traversal_paths,
        intent=intent, community_def=community_def,
    )
    print(f"\nA: {answer}")

    return _build_record(
        query_id, question, answer, keywords, intent,
        context_nodes, traversal_paths,
        f"Targeted+BFS label-scoped [{community_id}]",
    )


def _build_record(
    query_id, question, answer, keywords, intent,
    context_nodes, traversal_paths, algorithm_desc,
) -> dict:
    return {
        "query_id":         query_id,
        "query":            question,
        "generated_answer": answer,
        "keywords":         keywords,
        "intent":           intent,
        "community_id":     intent.get("community_id", ""),
        "retrieved_nodes": [
            {
                "node_id":  f"node{i+1:03d}",
                "content":  json.dumps(n, ensure_ascii=False),
                "entities": [n.get("name", "")],
            }
            for i, n in enumerate(context_nodes)
        ],
        "traversal_path": traversal_paths[:20],
        "timestamp":      datetime.datetime.now().isoformat(),
        "algorithm": {
            "community_detection": "Louvain weighted (GDS) + rule-based fallback",
            "traversal":           algorithm_desc
        },
    }


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def interactive_loop(driver, ai_client: OpenAI):
    print("\n🎓 Knowledge Graph Chatbot v11 — GraphRAG 3-Tier + MBTI Personality (label-scoped BFS)")
    print(f"max_hops={MAX_HOPS} | BFS dùng label filter (không dùng community number filter)")
    print("Gõ câu hỏi. Nhập 'exit' để thoát.\n")

    counter = 1
    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "thoat", "thoát"):
            print("Tạm biệt!")
            break
        ask(driver, ai_client, question, query_id=f"q{counter:03d}")
        counter += 1


def main():
    print("Starting KG Chatbot v11 (GraphRAG 3-Tier + MBTI Personality, label-scoped BFS)...")
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver    = get_driver()
    try:
        initialize_communities(driver, force_rebuild=False)
        interactive_loop(driver, ai_client)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
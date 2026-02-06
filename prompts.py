

# Prompt chính cho entity và relationship extraction
DEFAULT_ENTITY_EXTRACTION_PROMPT = """
-Goal-
Given a text document about courses, subjects, and educational programs, identify all entities and relationships.

-Steps-
1. Identify all entities. For each entity, extract:
- entity_name: Name of the entity (capitalized)
- entity_type: One of these types: [{entity_types}]
- entity_description: Comprehensive description

Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Identify all relationships between entities:
- source_entity: Source entity name
- target_entity: Target entity name  
- relationship_description: Why they are related
- relationship_strength: Numeric score (1-10)

Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in Vietnamese/English as needed, using **{record_delimiter}** as delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################

Example 1 (Course Prerequisites):
Entity_types: COURSE,SUBJECT,PREREQUISITE
Text:
Môn học Hệ điều hành (CS301) yêu cầu sinh viên phải hoàn thành môn Cấu trúc dữ liệu và giải thuật (CS201) trước khi đăng ký. Hệ điều hành là môn học cốt lõi của ngành Khoa học máy tính, tập trung vào quản lý tài nguyên hệ thống.

Output:
("entity"{tuple_delimiter}HỆ ĐIỀU HÀNH{tuple_delimiter}SUBJECT{tuple_delimiter}Môn học cốt lõi của ngành Khoa học máy tính về quản lý tài nguyên hệ thống, mã CS301)
{record_delimiter}
("entity"{tuple_delimiter}CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT{tuple_delimiter}SUBJECT{tuple_delimiter}Môn học tiên quyết cho Hệ điều hành, mã CS201)
{record_delimiter}
("relationship"{tuple_delimiter}HỆ ĐIỀU HÀNH{tuple_delimiter}CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT{tuple_delimiter}Cấu trúc dữ liệu và giải thuật là môn tiên quyết của Hệ điều hành{tuple_delimiter}9)
{completion_delimiter}

Example 2 (Course Program):
Entity_types: COURSE,PROGRAM,DEPARTMENT,SKILL
Text:
Chương trình Cử nhân Khoa học máy tính của Khoa Công nghệ thông tin bao gồm 140 tín chỉ. Sinh viên sẽ học các kỹ năng lập trình, thiết kế hệ thống, và phát triển phần mềm.

Output:
("entity"{tuple_delimiter}CỬ NHÂN KHOA HỌC MÁY TÍNH{tuple_delimiter}PROGRAM{tuple_delimiter}Chương trình đào tạo 140 tín chỉ về khoa học máy tính)
{record_delimiter}
("entity"{tuple_delimiter}KHOA CÔNG NGHỆ THÔNG TIN{tuple_delimiter}DEPARTMENT{tuple_delimiter}Khoa quản lý chương trình Cử nhân Khoa học máy tính)
{record_delimiter}
("entity"{tuple_delimiter}LẬP TRÌNH{tuple_delimiter}SKILL{tuple_delimiter}Kỹ năng cơ bản được đào tạo trong chương trình)
{record_delimiter}
("entity"{tuple_delimiter}THIẾT KẾ HỆ THỐNG{tuple_delimiter}SKILL{tuple_delimiter}Kỹ năng về thiết kế hệ thống phần mềm)
{record_delimiter}
("entity"{tuple_delimiter}PHÁT TRIỂN PHẦN MỀM{tuple_delimiter}SKILL{tuple_delimiter}Kỹ năng phát triển ứng dụng phần mềm)
{record_delimiter}
("relationship"{tuple_delimiter}CỬ NHÂN KHOA HỌC MÁY TÍNH{tuple_delimiter}KHOA CÔNG NGHỆ THÔNG TIN{tuple_delimiter}Chương trình được quản lý bởi khoa CNTT{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}CỬ NHÂN KHOA HỌC MÁY TÍNH{tuple_delimiter}LẬP TRÌNH{tuple_delimiter}Chương trình đào tạo kỹ năng lập trình{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}CỬ NHÂN KHOA HỌC MÁY TÍNH{tuple_delimiter}THIẾT KẾ HỆ THỐNG{tuple_delimiter}Chương trình đào tạo kỹ năng thiết kế hệ thống{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}CỬ NHÂN KHOA HỌC MÁY TÍNH{tuple_delimiter}PHÁT TRIỂN PHẦN MỀM{tuple_delimiter}Chương trình đào tạo kỹ năng phát triển phần mềm{tuple_delimiter}8)
{completion_delimiter}

Example 3 (Course Details):
Entity_types: SUBJECT,TOPIC,SEMESTER,INSTRUCTOR
Text:
Môn Trí tuệ nhân tạo (AI401) được giảng dạy bởi GS. Nguyễn Văn A vào học kỳ 1. Môn học bao gồm các chủ đề: Machine Learning, Deep Learning, và Natural Language Processing.

Output:
("entity"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}SUBJECT{tuple_delimiter}Môn học nâng cao về AI, mã AI401)
{record_delimiter}
("entity"{tuple_delimiter}GS. NGUYỄN VĂN A{tuple_delimiter}INSTRUCTOR{tuple_delimiter}Giảng viên dạy môn Trí tuệ nhân tạo)
{record_delimiter}
("entity"{tuple_delimiter}HỌC KỲ 1{tuple_delimiter}SEMESTER{tuple_delimiter}Học kỳ giảng dạy môn AI)
{record_delimiter}
("entity"{tuple_delimiter}MACHINE LEARNING{tuple_delimiter}TOPIC{tuple_delimiter}Chủ đề trong môn Trí tuệ nhân tạo)
{record_delimiter}
("entity"{tuple_delimiter}DEEP LEARNING{tuple_delimiter}TOPIC{tuple_delimiter}Chủ đề trong môn Trí tuệ nhân tạo)
{record_delimiter}
("entity"{tuple_delimiter}NATURAL LANGUAGE PROCESSING{tuple_delimiter}TOPIC{tuple_delimiter}Chủ đề trong môn Trí tuệ nhân tạo)
{record_delimiter}
("relationship"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}GS. NGUYỄN VĂN A{tuple_delimiter}GS. Nguyễn Văn A giảng dạy môn Trí tuệ nhân tạo{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}HỌC KỲ 1{tuple_delimiter}Môn học được giảng dạy vào học kỳ 1{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}MACHINE LEARNING{tuple_delimiter}Machine Learning là chủ đề chính của môn học{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}DEEP LEARNING{tuple_delimiter}Deep Learning là chủ đề chính của môn học{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}TRÍ TUỆ NHÂN TẠO{tuple_delimiter}NATURAL LANGUAGE PROCESSING{tuple_delimiter}NLP là chủ đề chính của môn học{tuple_delimiter}8)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


# Prompt cho summarization (merge descriptions)
DESCRIPTION_SUMMARIZATION_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of entity/relationship descriptions.

Given one entity/relationship and a list of descriptions, please:
1. Concatenate all descriptions into a single, comprehensive description
2. Remove contradictions and redundancies  
3. Keep all important information
4. Write in third person
5. Include entity names for full context

#######
-Data-
Entity/Relationship: {entity_name}
Description List: 
{description_list}
#######

Generate a single comprehensive description (2-3 sentences):
"""


# Prompt cho validation (kiểm tra extracted entities)
ENTITY_VALIDATION_PROMPT = """
Review the following extracted entities and relationships for accuracy and completeness.

Entities:
{entities}

Relationships:
{relationships}

Source Text:
{source_text}

Questions:
1. Are there any important entities missing?
2. Are there any incorrect entities?
3. Are the relationships accurate?
4. Suggest improvements:

Response (JSON format):
{{
    "missing_entities": [...],
    "incorrect_entities": [...],
    "relationship_issues": [...],
    "suggestions": "..."
}}
"""
# career_advisor.py
"""
Career Advisor System - AI Chatbot for Career & Curriculum Guidance
Dựa trên career descriptions và curriculum từ MinIO
"""

from openai import OpenAI
from typing import List, Dict, Any, Optional
from graph_database import GraphDatabaseConnection
from graph_manager_v3 import GraphManagerV3
from logger import Logger
import unicodedata
import json


class CareerAdvisor:
    """
    AI Career Advisor Chatbot
    - Tư vấn ngành học dựa trên nghề nghiệp mong muốn
    - Tư vấn nghề nghiệp dựa trên môn học giỏi
    - Đề xuất lộ trình học tập
    - Phân tích skills và requirements
    """
    
    logger = Logger("CareerAdvisor").get_logger()
    
    def __init__(
        self,
        graph_manager: GraphManagerV3,
        client: OpenAI,
        model: str = "gpt-4o-mini"
    ):
        """Initialize career advisor."""
        self.graph_manager = graph_manager
        self.client = client
        self.model = model
        self.db = graph_manager.db
        
        self.logger.info("Career Advisor initialized")
    
    # =========================================================
    # MAIN ADVISORY FUNCTIONS
    # =========================================================
    
    def advise_career_to_major(
        self,
        desired_career: str,
        student_strengths: Optional[List[str]] = None,
        interests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tư vấn ngành học dựa trên nghề nghiệp mong muốn.
        
        Args:
            desired_career: Nghề nghiệp mong muốn (VD: "kỹ sư phần mềm")
            student_strengths: Môn học giỏi (VD: ["toán", "lý"])
            interests: Sở thích (VD: ["lập trình", "thiết kế"])
        
        Returns:
            Recommendations với majors, skills, learning_path, COURSES
        """
        self.logger.info(f"Career-to-Major advisory: {desired_career}")
        
        # Step 1: Tìm career entities liên quan
        career_entities = self._find_career_entities(desired_career)
        
        if not career_entities:
            return self._generate_fallback_career_advice(desired_career)
        
        # Step 2: Tìm majors/programs liên quan
        related_majors = self._find_related_majors(career_entities)
        
        # Step 3: Phân tích skills requirements
        required_skills = self._analyze_skills_requirements(career_entities)
        
        # Step 4: ⭐ MỚI - Tìm các học phần cụ thể liên quan đến skills
        recommended_courses = self._find_courses_for_skills(required_skills, related_majors)
        
        # Step 5: Match với student strengths
        matching_score = self._calculate_student_match(
            required_skills,
            student_strengths,
            interests
        )
        
        # Step 6: Generate advice với LLM (bao gồm courses)
        advice = self._generate_career_advice_with_courses(
            desired_career=desired_career,
            career_info=career_entities,
            majors=related_majors,
            skills=required_skills,
            courses=recommended_courses,
            student_strengths=student_strengths,
            interests=interests,
            matching_score=matching_score
        )
        
        return {
            'career': desired_career,
            'recommended_majors': related_majors,
            'required_skills': required_skills,
            'recommended_courses': recommended_courses,  # ⭐ NEW
            'matching_score': matching_score,
            'advice': advice,
            'learning_path': self._create_learning_path_with_courses(related_majors, recommended_courses)
        }
    
    def advise_major_to_career(
        self,
        strong_subjects: List[str],
        interests: Optional[List[str]] = None,
        personality_traits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tư vấn ngành học và nghề nghiệp dựa trên môn học giỏi.
        
        Args:
            strong_subjects: Môn học giỏi (VD: ["toán", "lý", "hóa"])
            interests: Sở thích (VD: ["công nghệ", "nghiên cứu"])
            personality_traits: Tính cách (VD: ["logic", "tỉ mỉ"])
        
        Returns:
            Recommendations với suitable_majors, careers, paths
        """
        self.logger.info(f"Major-to-Career advisory: {strong_subjects}")
        
        # Step 1: Map subjects to relevant majors
        suitable_majors = self._map_subjects_to_majors(strong_subjects)
        
        # Step 2: Tìm career paths từ majors
        career_paths = self._find_career_paths_from_majors(suitable_majors)
        
        # Step 3: Filter theo interests và personality
        filtered_careers = self._filter_by_interests_personality(
            career_paths,
            interests,
            personality_traits
        )
        
        # Step 4: Generate comprehensive advice
        advice = self._generate_major_advice(
            strong_subjects=strong_subjects,
            majors=suitable_majors,
            careers=filtered_careers,
            interests=interests,
            personality=personality_traits
        )
        
        return {
            'strong_subjects': strong_subjects,
            'suitable_majors': suitable_majors,
            'career_options': filtered_careers,
            'advice': advice,
            'next_steps': self._suggest_next_steps(suitable_majors)
        }
    
    def compare_majors(
        self,
        major_names: List[str]
    ) -> Dict[str, Any]:
        """
        So sánh các ngành học.
        
        Args:
            major_names: Danh sách tên ngành (VD: ["CNTT", "Kỹ thuật phần mềm"])
        
        Returns:
            Comparison với career_paths, skills, difficulty, job_market
        """
        self.logger.info(f"Comparing majors: {major_names}")
        
        # Lấy thông tin chi tiết cho mỗi ngành
        majors_info = []
        for major in major_names:
            info = self._get_major_details(major)
            if info:
                majors_info.append(info)
        
        # So sánh và generate advice
        comparison = self._generate_major_comparison(majors_info)
        
        return comparison
    
    def get_learning_roadmap(
        self,
        major_name: str,
        current_grade: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tạo lộ trình học tập chi tiết cho một ngành.
        
        Args:
            major_name: Tên ngành
            current_grade: Lớp hiện tại (10, 11, 12)
        
        Returns:
            Roadmap với courses, timeline, prerequisites
        """
        self.logger.info(f"Creating roadmap for: {major_name}")
        
        # Lấy curriculum details
        curriculum = self._get_curriculum_details(major_name)
        
        # Tạo roadmap
        roadmap = self._create_detailed_roadmap(curriculum, current_grade)
        
        return roadmap
    
    # =========================================================
    # CAREER ENTITY SEARCH
    # =========================================================
    
    def _find_career_entities(self, career_term: str) -> List[Dict[str, Any]]:
        """Tìm career entities từ graph."""
        
        # Normalize
        career_norm = self._normalize_text(career_term)
        
        with self.db.get_session() as session:
            # Tìm career description entities
            results = session.run("""
                MATCH (e:Entity)
                WHERE e.type IN ['nghề_nghiệp', 'career', 'vị_trí_công_việc']
                  AND (
                    toLower(e.name) CONTAINS $term
                    OR e.name_normalized CONTAINS $term_norm
                  )
                RETURN e.name as name,
                       e.type as type,
                       properties(e) as props
                LIMIT 5
            """, term=career_term.lower(), term_norm=career_norm).data()
            
            return results
    
    def _find_related_majors(
        self,
        career_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tìm các ngành học liên quan đến career."""
        
        if not career_entities:
            return []
        
        career_names = [e['name'] for e in career_entities]
        
        with self.db.get_session() as session:
            # Tìm majors có relationship với careers
            results = session.run("""
                MATCH (career:Entity)-[r]-(major:Entity)
                WHERE career.name IN $careers
                  AND major.type IN ['ngành_học', 'chương_trình_đào_tạo', 'major']
                RETURN DISTINCT major.name as name,
                       major.type as type,
                       type(r) as relationship,
                       properties(major) as props
            """, careers=career_names).data()
            
            # Nếu không tìm thấy direct relationship, tìm qua skills
            if not results:
                results = self._find_majors_via_skills(career_entities)
            
            return results
    
    def _find_majors_via_skills(
        self,
        career_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tìm majors thông qua shared skills."""
        
        career_names = [e['name'] for e in career_entities]
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (career:Entity)-[:YÊU_CẦU_KỸ_NĂNG|CẦN_KỸ_NĂNG]-(skill:Entity)
                WHERE career.name IN $careers
                  AND skill.type IN ['kỹ_năng', 'skill']
                
                MATCH (skill)-[:ĐƯỢC_HỌC_TỪ|PHÁT_TRIỂN_QUA]-(major:Entity)
                WHERE major.type IN ['ngành_học', 'chương_trình_đào_tạo']
                
                RETURN major.name as name,
                       major.type as type,
                       collect(DISTINCT skill.name) as shared_skills,
                       properties(major) as props
                ORDER BY size(shared_skills) DESC
                LIMIT 5
            """, careers=career_names).data()
            
            return results
    
    # =========================================================
    # COURSE RECOMMENDATION (NEW)
    # =========================================================
    
    def _find_courses_for_skills(
        self,
        required_skills: Dict[str, List[str]],
        majors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Tìm các học phần cụ thể từ Trường Kinh tế Quốc dân 
        liên quan đến skills cần thiết.
        
        Returns:
            List of courses với mã_học_phần, tên, số_tín_chỉ, skills_covered
        """
        self.logger.info("Finding specific courses for required skills...")
        
        # Flatten all skills
        all_skills = []
        for skill_list in required_skills.values():
            all_skills.extend(skill_list)
        
        if not all_skills and not majors:
            return []
        
        # Query 1: Tìm courses trực tiếp qua skills
        courses_via_skills = self._query_courses_by_skills(all_skills)
        
        # Query 2: Tìm courses qua majors
        courses_via_majors = self._query_courses_by_majors(majors)
        
        # Query 3: Tìm courses có tên liên quan đến skills
        courses_via_names = self._query_courses_by_keyword_match(all_skills)
        
        # Combine và deduplicate
        all_courses = courses_via_skills + courses_via_majors + courses_via_names
        
        # Deduplicate by mã_học_phần or name
        unique_courses = {}
        for course in all_courses:
            code = course.get('mã_học_phần') or course.get('name')
            if code and code not in unique_courses:
                unique_courses[code] = course
        
        # ⭐ FILTER - Remove obviously irrelevant courses
        filtered_courses = []
        for course in unique_courses.values():
            course_name = course.get('name', '').lower()
            
            # Skip courses that are clearly not relevant
            skip_keywords = [
                'lập trình',      # Programming (unless skills mention it)
                'tiếng anh',      # English (too generic)
                'kỹ năng bổ trợ', # Too generic
                'thể dục',        # PE
                'quân sự',        # Military
            ]
            
            # Check if course should be skipped
            should_skip = False
            for keyword in skip_keywords:
                if keyword in course_name:
                    # Only skip if this keyword is NOT in required skills
                    if not any(keyword in skill.lower() for skill in all_skills):
                        should_skip = True
                        break
            
            if not should_skip:
                filtered_courses.append(course)
        
        # Sort by relevance (courses with more skills covered first)
        sorted_courses = sorted(
            filtered_courses,
            key=lambda c: len(c.get('skills_covered', [])),
            reverse=True
        )
        
        return sorted_courses[:10]  # Top 10 most relevant
    
    def _query_courses_by_skills(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Tìm courses qua relationships với skills."""
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (course:Entity {type: 'học_phần'})-[r]-(skill:Entity)
                WHERE skill.type IN ['kỹ_năng', 'skill']
                  AND ANY(s IN $skills WHERE toLower(skill.name) CONTAINS toLower(s))
                RETURN DISTINCT 
                    course.name as name,
                    course.mã_học_phần as mã_học_phần,
                    course.số_tín_chỉ as số_tín_chỉ,
                    course.số_giờ_trên_lớp as số_giờ,
                    collect(DISTINCT skill.name) as skills_covered,
                    properties(course) as props
                ORDER BY size(skills_covered) DESC
                LIMIT 10
            """, skills=skills).data()
            
            return results
    
    def _query_courses_by_majors(self, majors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tìm courses thuộc các majors được recommend."""
        
        if not majors:
            return []
        
        major_names = [m['name'] for m in majors[:3]]  # Top 3 majors
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (major:Entity)-[r]-(course:Entity)
                WHERE major.name IN $majors
                  AND course.type = 'học_phần'
                  AND major.type IN ['ngành_học', 'chương_trình_đào_tạo']
                RETURN DISTINCT
                    course.name as name,
                    course.mã_học_phần as mã_học_phần,
                    course.số_tín_chỉ as số_tín_chỉ,
                    course.số_giờ_trên_lớp as số_giờ,
                    major.name as from_major,
                    [] as skills_covered,
                    properties(course) as props
                LIMIT 10
            """, majors=major_names).data()
            
            return results
    
    def _query_courses_by_keyword_match(self, skills: List[str]) -> List[Dict[str, Any]]:
        """
        Tìm courses có tên chứa keywords liên quan đến skills.
        VD: skill "lập trình" → course "Lập trình Java"
        """
        
        # Extract keywords từ skills
        keywords = []
        for skill in skills:
            # Lấy từ đầu tiên (thường là keyword chính)
            words = skill.lower().split()
            if words:
                keywords.append(words[0])
        
        keywords = list(set(keywords))[:10]  # Unique, max 10
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (course:Entity {type: 'học_phần'})
                WHERE ANY(kw IN $keywords WHERE toLower(course.name) CONTAINS kw)
                RETURN 
                    course.name as name,
                    course.mã_học_phần as mã_học_phần,
                    course.số_tín_chỉ as số_tín_chỉ,
                    course.số_giờ_trên_lớp as số_giờ,
                    [] as skills_covered,
                    properties(course) as props
                LIMIT 10
            """, keywords=keywords).data()
            
            return results
    
    # =========================================================
    # MAJOR TO CAREER MAPPING
    # =========================================================
    
    def _map_subjects_to_majors(
        self,
        subjects: List[str]
    ) -> List[Dict[str, Any]]:
        """Map môn học sang các ngành phù hợp."""
        
        # Subject to major mapping
        subject_mapping = {
            'toán': ['công_nghệ_thông_tin', 'kỹ_thuật', 'kinh_tế', 'tài_chính'],
            'lý': ['kỹ_thuật', 'công_nghệ_thông_tin', 'vật_lý'],
            'hóa': ['hóa_học', 'y_dược', 'công_nghệ_sinh_học'],
            'sinh': ['y_dược', 'sinh_học', 'công_nghệ_sinh_học'],
            'văn': ['ngôn_ngữ', 'báo_chí', 'marketing'],
            'sử': ['lịch_sử', 'giáo_dục', 'nhân_văn'],
            'địa': ['địa_lý', 'du_lịch', 'môi_trường'],
            'anh': ['ngoại_ngữ', 'kinh_doanh_quốc_tế', 'du_lịch']
        }
        
        # Tìm majors trong graph
        majors = []
        
        for subject in subjects:
            subject_norm = self._normalize_text(subject)
            
            # Tìm direct matches
            with self.db.get_session() as session:
                results = session.run("""
                    MATCH (major:Entity)
                    WHERE major.type IN ['ngành_học', 'chương_trình_đào_tạo']
                      AND (
                        major.môn_học_chính CONTAINS $subject
                        OR major.môn_học_liên_quan CONTAINS $subject
                      )
                    RETURN major.name as name,
                           properties(major) as props
                    LIMIT 5
                """, subject=subject_norm).data()
                
                majors.extend(results)
        
        # Deduplicate
        seen = set()
        unique_majors = []
        for major in majors:
            if major['name'] not in seen:
                seen.add(major['name'])
                unique_majors.append(major)
        
        return unique_majors[:10]
    
    def _find_career_paths_from_majors(
        self,
        majors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tìm career paths từ majors."""
        
        if not majors:
            return []
        
        major_names = [m['name'] for m in majors]
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (major:Entity)-[r]-(career:Entity)
                WHERE major.name IN $majors
                  AND career.type IN ['nghề_nghiệp', 'career', 'vị_trí_công_việc']
                RETURN DISTINCT career.name as name,
                       career.type as type,
                       major.name as from_major,
                       properties(career) as props
                LIMIT 20
            """, majors=major_names).data()
            
            return results
    
    # =========================================================
    # SKILLS ANALYSIS
    # =========================================================
    
    def _analyze_skills_requirements(
        self,
        career_entities: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Phân tích skills requirements cho careers."""
        
        career_names = [e['name'] for e in career_entities]
        
        with self.db.get_session() as session:
            results = session.run("""
                MATCH (career:Entity)-[r]-(skill:Entity)
                WHERE career.name IN $careers
                  AND skill.type IN ['kỹ_năng', 'skill', 'năng_lực']
                  AND type(r) IN ['YÊU_CẦU_KỸ_NĂNG', 'CẦN_KỸ_NĂNG', 'CẦN_NĂNG_LỰC']
                RETURN skill.name as skill_name,
                       skill.loại as skill_type,
                       r.mức_độ as proficiency_level
            """, careers=career_names).data()
        
        # Categorize skills
        skills_by_category = {
            'kỹ_năng_chuyên_môn': [],
            'kỹ_năng_mềm': [],
            'ngoại_ngữ': [],
            'khác': []
        }
        
        for skill in results:
            skill_type = skill.get('skill_type', 'khác')
            skills_by_category.get(skill_type, skills_by_category['khác']).append(
                skill['skill_name']
            )
        
        return skills_by_category
    
    def _calculate_student_match(
        self,
        required_skills: Dict[str, List[str]],
        student_strengths: Optional[List[str]],
        interests: Optional[List[str]]
    ) -> float:
        """Tính matching score giữa student và career."""
        
        if not student_strengths and not interests:
            return 0.5  # Neutral
        
        # Simple matching algorithm
        total_required = sum(len(skills) for skills in required_skills.values())
        if total_required == 0:
            return 0.5
        
        matched = 0
        
        # Check strengths
        if student_strengths:
            for strength in student_strengths:
                strength_norm = self._normalize_text(strength)
                for skills in required_skills.values():
                    for skill in skills:
                        if strength_norm in self._normalize_text(skill):
                            matched += 1
        
        # Check interests
        if interests:
            for interest in interests:
                interest_norm = self._normalize_text(interest)
                for skills in required_skills.values():
                    for skill in skills:
                        if interest_norm in self._normalize_text(skill):
                            matched += 0.5
        
        score = min(matched / total_required, 1.0)
        return round(score, 2)
    
    # =========================================================
    # FILTERING & MATCHING
    # =========================================================
    
    def _filter_by_interests_personality(
        self,
        careers: List[Dict[str, Any]],
        interests: Optional[List[str]],
        personality_traits: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Filter careers theo interests và personality."""
        
        if not interests and not personality_traits:
            return careers
        
        # Score each career
        scored_careers = []
        
        for career in careers:
            score = 0
            career_text = json.dumps(career).lower()
            
            # Match interests
            if interests:
                for interest in interests:
                    if self._normalize_text(interest) in career_text:
                        score += 2
            
            # Match personality
            if personality_traits:
                for trait in personality_traits:
                    if self._normalize_text(trait) in career_text:
                        score += 1
            
            scored_careers.append({
                'career': career,
                'score': score
            })
        
        # Sort by score
        scored_careers.sort(key=lambda x: x['score'], reverse=True)
        
        return [sc['career'] for sc in scored_careers[:10]]
    
    # =========================================================
    # ADVICE GENERATION WITH LLM
    # =========================================================
    
    def _generate_career_advice_with_courses(
        self,
        desired_career: str,
        career_info: List[Dict[str, Any]],
        majors: List[Dict[str, Any]],
        skills: Dict[str, List[str]],
        courses: List[Dict[str, Any]],
        student_strengths: Optional[List[str]],
        interests: Optional[List[str]],
        matching_score: float
    ) -> str:
        """Generate comprehensive career advice INCLUDING specific courses."""
        
        # Build context
        context = self._build_career_context_with_courses(
            career_info, majors, skills, courses
        )
        
        # Build prompt
        prompt = f"""Bạn là chuyên gia tư vấn nghề nghiệp và hướng nghiệp của Trường Kinh tế Quốc dân.

THÔNG TIN HỌC SINH:
- Nghề nghiệp mong muốn: {desired_career}
- Điểm mạnh: {', '.join(student_strengths) if student_strengths else 'Chưa cung cấp'}
- Sở thích: {', '.join(interests) if interests else 'Chưa cung cấp'}
- Độ phù hợp: {matching_score * 100:.0f}%

THÔNG TIN NGHỀ NGHIỆP, NGÀNH HỌC VÀ HỌC PHẦN:
{context}

YÊU CẦU TƯ VẤN (NGẮN GỌN):

1. Đánh giá độ phù hợp (2-3 câu)

2. Ngành học đề xuất (TOP 2-3 ngành):
   - Tên ngành và lý do ngắn gọn

3. ⭐ Học phần cần học (CHỈ liệt kê, KHÔNG làm bảng):
   - Mỗi học phần: Tên - Mã (nếu có) - Tại sao quan trọng (1 câu)
   - Chỉ liệt kê học phần THỰC SỰ liên quan đến nghề
   - KHÔNG liệt kê các môn quá chung chung như "Tiếng Anh", "Kỹ năng bổ trợ"
   
4. Kỹ năng cần tự học (ngắn gọn):
   - 3-5 kỹ năng quan trọng nhất
   - Cách học (1 câu mỗi kỹ năng)

5. Lộ trình gợi ý (ngắn gọn):
   - THPT: Môn nào cần học tốt
   - Đại học năm 1-2: Focus gì
   - Đại học năm 3-4: Focus gì

FORMAT:
- Sử dụng bullet points, KHÔNG dùng bảng
- Ngắn gọn, súc tích
- Mỗi phần 3-5 dòng
- Tổng độ dài: ~300-400 từ

Trả lời bằng tiếng Việt, thân thiện.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia tư vấn hướng nghiệp của Trường Kinh tế Quốc dân. Hãy trả lời NGẮN GỌN, KHÔNG dùng bảng, chỉ bullet points."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000,  # Reduced from 2000
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _generate_major_advice(
        self,
        strong_subjects: List[str],
        majors: List[Dict[str, Any]],
        careers: List[Dict[str, Any]],
        interests: Optional[List[str]],
        personality: Optional[List[str]]
    ) -> str:
        """Generate advice for major selection based on strengths."""
        
        # Build context
        majors_text = "\n".join([
            f"- {m['name']}: {m.get('props', {}).get('mô_tả', 'N/A')}"
            for m in majors[:5]
        ])
        
        careers_text = "\n".join([
            f"- {c['name']} (từ ngành {c.get('from_major', 'N/A')})"
            for c in careers[:10]
        ])
        
        prompt = f"""Bạn là chuyên gia tư vấn hướng nghiệp cho học sinh.

THÔNG TIN HỌC SINH:
- Môn học giỏi: {', '.join(strong_subjects)}
- Sở thích: {', '.join(interests) if interests else 'Chưa cung cấp'}
- Tính cách: {', '.join(personality) if personality else 'Chưa cung cấp'}

CÁC NGÀNH HỌC PHÙ HỢP:
{majors_text}

CƠ HỘI NGHỀ NGHIỆP:
{careers_text}

YÊU CẦU:
1. Phân tích điểm mạnh của học sinh
2. Đề xuất top 3 ngành học phù hợp nhất
3. Giải thích tại sao mỗi ngành phù hợp với môn học giỏi
4. Nêu cơ hội nghề nghiệp từ mỗi ngành
5. So sánh ưu nhược điểm của các ngành
6. Gợi ý cách khám phá thêm về các ngành (trải nghiệm, tìm hiểu)

Trả lời thân thiện, động viên và chi tiết. Sử dụng bullet points khi cần.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia tư vấn hướng nghiệp."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _generate_major_comparison(
        self,
        majors_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """So sánh các ngành học."""
        
        # Build comparison context
        context = ""
        for info in majors_info:
            context += f"\nNGÀNH: {info['name']}\n"
            context += f"- Mô tả: {info.get('mô_tả', 'N/A')}\n"
            context += f"- Thời gian đào tạo: {info.get('thời_gian', 'N/A')}\n"
            context += f"- Cơ hội việc làm: {info.get('cơ_hội_việc_làm', 'N/A')}\n"
        
        prompt = f"""So sánh các ngành học sau:

{context}

YÊU CẦU:
1. So sánh về nội dung đào tạo
2. So sánh về cơ hội nghề nghiệp
3. So sánh về độ khó
4. So sánh về thu nhập tiềm năng
5. Đề xuất ngành phù hợp cho từng profile học sinh khác nhau

Trả lời chi tiết, khách quan.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân tích ngành học."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return {
            'majors': [m['name'] for m in majors_info],
            'comparison': response.choices[0].message.content
        }
    
    # =========================================================
    # LEARNING PATH CREATION
    # =========================================================
    
    def _create_learning_path_with_courses(
        self,
        majors: List[Dict[str, Any]],
        courses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tạo lộ trình học tập chi tiết với courses cụ thể."""
        
        if not majors:
            return {}
        
        # Get top major
        top_major = majors[0]
        
        # Categorize courses by year (estimate)
        foundation_courses = []
        advanced_courses = []
        
        for course in courses[:10]:
            course_name = course['name'].lower()
            # Simple heuristic: courses with "cơ sở", "nhập môn" = foundation
            if any(kw in course_name for kw in ['cơ sở', 'nhập môn', 'căn bản', 'fundamental']):
                foundation_courses.append(course)
            else:
                advanced_courses.append(course)
        
        # If heuristic didn't work, split by half
        if not foundation_courses and not advanced_courses:
            mid = len(courses) // 2
            foundation_courses = courses[:mid]
            advanced_courses = courses[mid:]
        
        # Create path
        path = {
            'major': top_major['name'],
            'preparation': {
                'grade_10': [
                    'Củng cố kiến thức toán, lý cơ bản',
                    'Tìm hiểu về ngành học qua sách, video',
                    'Tham gia câu lạc bộ liên quan'
                ],
                'grade_11': [
                    'Học sâu các môn chính liên quan đến ngành',
                    'Tham gia các cuộc thi học sinh giỏi',
                    'Trải nghiệm thực tế qua summer camp'
                ],
                'grade_12': [
                    'Ôn thi đại học tập trung',
                    'Tìm hiểu Trường Kinh tế Quốc dân',
                    'Chuẩn bị hồ sơ xét tuyển'
                ]
            },
            'university': {
                'year_1-2': {
                    'description': 'Kiến thức nền tảng',
                    'courses': [
                        f"{c['name']} ({c.get('mã_học_phần', 'N/A')})"
                        for c in foundation_courses[:5]
                    ] if foundation_courses else ['Các học phần nền tảng của ngành']
                },
                'year_3-4': {
                    'description': 'Chuyên môn sâu + Thực tập',
                    'courses': [
                        f"{c['name']} ({c.get('mã_học_phần', 'N/A')})"
                        for c in advanced_courses[:5]
                    ] if advanced_courses else ['Các học phần chuyên sâu']
                }
            }
        }
        
        return path
    
    def _create_learning_path(
        self,
        majors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tạo lộ trình học tập."""
        
        if not majors:
            return {}
        
        # Get top major
        top_major = majors[0]
        
        # Create path
        path = {
            'major': top_major['name'],
            'preparation': {
                'grade_10': [
                    'Củng cố kiến thức toán, lý cơ bản',
                    'Tìm hiểu về ngành học qua sách, video',
                    'Tham gia câu lạc bộ liên quan'
                ],
                'grade_11': [
                    'Học sâu các môn chính liên quan',
                    'Tham gia các cuộc thi học sinh giỏi',
                    'Trải nghiệm thực tế qua summer camp'
                ],
                'grade_12': [
                    'Ôn thi đại học tập trung',
                    'Tìm hiểu các trường đào tạo tốt',
                    'Chuẩn bị hồ sơ xét tuyển'
                ]
            },
            'university': {
                'year_1-2': 'Kiến thức nền tảng',
                'year_3-4': 'Chuyên môn sâu + Thực tập'
            }
        }
        
        return path
    
    def _suggest_next_steps(
        self,
        majors: List[Dict[str, Any]]
    ) -> List[str]:
        """Gợi ý các bước tiếp theo."""
        
        steps = [
            "📚 Tìm hiểu chi tiết về các ngành đề xuất",
            "🎓 Tham quan ngày hội tuyển sinh các trường đại học",
            "💼 Tìm hiểu về cơ hội việc làm sau khi tốt nghiệp",
            "👥 Trao đổi với sinh viên đang học các ngành này",
            "🔍 Đánh giá lại sở thích và điểm mạnh của bản thân",
            "📝 Lập kế hoạch học tập từ bây giờ đến khi thi đại học"
        ]
        
        return steps
    
    # =========================================================
    # HELPERS
    # =========================================================
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text."""
        if not text:
            return ""
        text = unicodedata.normalize('NFC', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    def _build_career_context_with_courses(
        self,
        career_info: List[Dict[str, Any]],
        majors: List[Dict[str, Any]],
        skills: Dict[str, List[str]],
        courses: List[Dict[str, Any]]
    ) -> str:
        """Build context string for LLM including courses."""
        
        context = "=== THÔNG TIN NGHỀ NGHIỆP ===\n"
        for career in career_info:
            context += f"\nNghề: {career['name']}\n"
            context += f"Loại: {career['type']}\n"
            props = career.get('props', {})
            for key, val in props.items():
                if key not in ['name', 'type', 'embedding', 'name_normalized']:
                    context += f"  {key}: {val}\n"
        
        context += "\n=== NGÀNH HỌC LIÊN QUAN TẠI TRƯỜNG KINH TẾ QUỐC DÂN ===\n"
        for major in majors[:5]:
            context += f"- {major['name']}\n"
        
        context += "\n=== KỸ NĂNG CẦN THIẾT ===\n"
        for category, skill_list in skills.items():
            if skill_list:
                context += f"{category}:\n"
                for skill in skill_list[:5]:
                    context += f"  - {skill}\n"
        
        # ⭐ NEW - Courses section
        context += "\n=== HỌC PHẦN CỤ THỂ TẠI TRƯỜNG KINH TẾ QUỐC DÂN ===\n"
        if courses:
            context += "Các học phần được đề xuất để phát triển skills cần thiết:\n\n"
            for i, course in enumerate(courses[:10], 1):
                context += f"{i}. {course['name']}\n"
                if course.get('mã_học_phần'):
                    context += f"   - Mã: {course['mã_học_phần']}\n"
                if course.get('số_tín_chỉ'):
                    context += f"   - Tín chỉ: {course['số_tín_chỉ']}\n"
                if course.get('skills_covered'):
                    skills_str = ', '.join(course['skills_covered'][:3])
                    context += f"   - Skills: {skills_str}\n"
                context += "\n"
        else:
            context += "(Chưa có thông tin chi tiết về học phần - đề xuất dựa trên ngành học)\n"
        
        return context
    
    def _build_career_context(
        self,
        career_info: List[Dict[str, Any]],
        majors: List[Dict[str, Any]],
        skills: Dict[str, List[str]]
    ) -> str:
        """Build context string for LLM."""
        
        context = "=== THÔNG TIN NGHỀ NGHIỆP ===\n"
        for career in career_info:
            context += f"\nNghề: {career['name']}\n"
            context += f"Loại: {career['type']}\n"
            props = career.get('props', {})
            for key, val in props.items():
                if key not in ['name', 'type', 'embedding', 'name_normalized']:
                    context += f"  {key}: {val}\n"
        
        context += "\n=== NGÀNH HỌC LIÊN QUAN ===\n"
        for major in majors[:5]:
            context += f"- {major['name']}\n"
        
        context += "\n=== KỸ NĂNG CẦN THIẾT ===\n"
        for category, skill_list in skills.items():
            if skill_list:
                context += f"{category}:\n"
                for skill in skill_list[:5]:
                    context += f"  - {skill}\n"
        
        return context
    
    def _get_major_details(self, major_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a major."""
        
        major_norm = self._normalize_text(major_name)
        
        with self.db.get_session() as session:
            result = session.run("""
                MATCH (m:Entity)
                WHERE m.type IN ['ngành_học', 'chương_trình_đào_tạo']
                  AND (
                    toLower(m.name) = $term
                    OR m.name_normalized = $term_norm
                  )
                RETURN m.name as name, properties(m) as props
                LIMIT 1
            """, term=major_name.lower(), term_norm=major_norm).single()
            
            if result:
                return {
                    'name': result['name'],
                    **result['props']
                }
            
            return None
    
    def _get_curriculum_details(self, major_name: str) -> Dict[str, Any]:
        """Get curriculum details for a major."""
        
        major_norm = self._normalize_text(major_name)
        
        with self.db.get_session() as session:
            # Get courses related to major
            courses = session.run("""
                MATCH (major:Entity)-[r]-(course:Entity)
                WHERE major.name_normalized = $major_norm
                  AND course.type = 'học_phần'
                RETURN course.name as course_name,
                       course.số_tín_chỉ as credits,
                       type(r) as relation
                LIMIT 20
            """, major_norm=major_norm).data()
            
            return {
                'major': major_name,
                'courses': courses
            }
    
    def _create_detailed_roadmap(
        self,
        curriculum: Dict[str, Any],
        current_grade: Optional[int]
    ) -> Dict[str, Any]:
        """Create detailed learning roadmap."""
        
        roadmap = {
            'major': curriculum['major'],
            'timeline': {}
        }
        
        # High school preparation
        if current_grade and current_grade <= 12:
            roadmap['timeline']['high_school'] = {
                'now_to_grade_12': 'Chuẩn bị thi đại học',
                'focus_subjects': 'Các môn liên quan đến ngành'
            }
        
        # University
        courses = curriculum.get('courses', [])
        if courses:
            roadmap['timeline']['university'] = {
                'year_1': [c['course_name'] for c in courses[:5]],
                'year_2': [c['course_name'] for c in courses[5:10]],
                'year_3-4': 'Chuyên môn sâu + Thực tập'
            }
        
        return roadmap
    
    def _generate_fallback_career_advice(self, career: str) -> Dict[str, Any]:
        """Generate fallback advice when no data found."""
        
        return {
            'career': career,
            'recommended_majors': [],
            'required_skills': {},
            'matching_score': 0,
            'advice': f"""⚠️ Hiện tại chưa có đủ thông tin chi tiết về nghề "{career}" trong hệ thống.

Tuy nhiên, bạn có thể:

1. **Tìm hiểu thêm**: 
   - Tìm kiếm thông tin trực tuyến về nghề này
   - Trao đổi với người đang làm nghề này

2. **Xác định skills cần thiết**:
   - Kỹ năng chuyên môn gì?
   - Kỹ năng mềm nào quan trọng?

3. **Tìm ngành học phù hợp**:
   - Ngành nào đào tạo skills này?
   - Trường nào có chương trình tốt?

💡 Hãy thử tìm kiếm với từ khóa khác hoặc hỏi về ngành học cụ thể!
""",
            'learning_path': {}
        }
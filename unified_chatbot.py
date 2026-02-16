# unified_chatbot.py
"""
UNIFIED EDUCATIONAL AI CHATBOT
Kết hợp cả GraphRAG Query và Career Advisor trong 1 chatbot thông minh
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import re

from graph_database import GraphDatabaseConnection
from graph_manager_v3 import GraphManagerV3
from query_handler_v3 import QueryHandlerV3
from career_advisor import CareerAdvisor
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()
logger = Logger("UnifiedChatbot").get_logger()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

# Database
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not OPENAI_API_KEY or not DB_URL or not DB_PASSWORD:
    print("❌ Error: Missing configuration in .env file")
    sys.exit(1)

# =========================================================
# UNIFIED CHATBOT CLASS
# =========================================================

class UnifiedEducationalChatbot:
    """
    Chatbot thống nhất với 2 chức năng:
    1. GraphRAG Query - Trả lời về học phần, giảng viên, tài liệu
    2. Career Advisor - Tư vấn ngành học, nghề nghiệp
    
    Tự động phát hiện intent và route đến handler phù hợp.
    """
    
    def __init__(
        self,
        db_connection: GraphDatabaseConnection,
        graph_manager: GraphManagerV3,
        client: OpenAI
    ):
        """Initialize unified chatbot."""
        self.db = db_connection
        self.graph_manager = graph_manager
        self.client = client
        
        # Initialize both handlers
        self.query_handler = QueryHandlerV3(
            graph_manager=graph_manager,
            client=client,
            model=MODEL
        )
        
        self.career_advisor = CareerAdvisor(
            graph_manager=graph_manager,
            client=client,
            model=MODEL
        )
        
        # Conversation history
        self.conversation_history = []
        
        logger.info("Unified Chatbot initialized")
    
    # =========================================================
    # MAIN CHAT FUNCTION
    # =========================================================
    
    def chat(self, user_input: str) -> str:
        """
        Main chat function - routes to appropriate handler.
        
        Args:
            user_input: User's question or statement
            
        Returns:
            Response string
        """
        # Detect intent
        intent = self._detect_intent(user_input)
        
        logger.info(f"User input: {user_input}")
        logger.info(f"Detected intent: {intent}")
        
        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'intent': intent
        })
        
        # Route to appropriate handler
        if intent == 'career_advice':
            response = self._handle_career_advice(user_input)
        
        elif intent == 'course_query':
            response = self._handle_course_query(user_input)
        
        elif intent == 'compare_majors':
            response = self._handle_compare_majors(user_input)
        
        elif intent == 'learning_path':
            response = self._handle_learning_path(user_input)
        
        elif intent == 'general_info':
            response = self._handle_general_info(user_input)
        
        else:
            # Fallback - ask for clarification
            response = self._ask_clarification()
        
        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    # =========================================================
    # INTENT DETECTION
    # =========================================================
    
    def _detect_intent(self, user_input: str) -> str:
        """
        Detect user intent from input.
        
        Returns:
            Intent type: career_advice, course_query, compare_majors, 
                        learning_path, general_info
        """
        user_lower = user_input.lower()
        
        # Career advice patterns
        career_patterns = [
            r'muốn làm',
            r'muốn trở thành',
            r'nghề',
            r'giỏi.*nên học',
            r'học ngành gì',
            r'nên chọn ngành',
            r'tư vấn.*ngành',
            r'làm.*nghề gì',
            r'em giỏi'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in career_patterns):
            return 'career_advice'
        
        # Compare majors patterns
        compare_patterns = [
            r'so sánh',
            r'khác nhau',
            r'hay',
            r'vs',
            r'versus',
            r'chọn.*hay'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in compare_patterns):
            # Check if comparing majors
            if any(word in user_lower for word in ['ngành', 'chương trình', 'cntt', 'kỹ thuật']):
                return 'compare_majors'
        
        # Learning path patterns
        path_patterns = [
            r'lộ trình',
            r'học.*như thế nào',
            r'cần học gì',
            r'chuẩn bị',
            r'roadmap'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in path_patterns):
            return 'learning_path'
        
        # Course query patterns (GraphRAG)
        course_patterns = [
            r'giảng viên',
            r'thầy',
            r'cô',
            r'môn',
            r'học phần',
            r'tín chỉ',
            r'mã học phần',
            r'email',
            r'tài liệu',
            r'sách',
            r'giáo trình',
            r'tiên quyết',
            r'điều kiện',
            r'số giờ',
            r'chuẩn đầu ra',
            r'mục tiêu',
            r'đánh giá'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in course_patterns):
            return 'course_query'
        
        # General info
        return 'general_info'
    
    # =========================================================
    # INTENT HANDLERS
    # =========================================================
    
    def _handle_career_advice(self, user_input: str) -> str:
        """Handle career advisory queries."""
        
        # Extract information from input
        info = self._extract_career_info(user_input)
        
        # Check what type of career advice
        if info['desired_career']:
            # Career → Major advice
            result = self.career_advisor.advise_career_to_major(
                desired_career=info['desired_career'],
                student_strengths=info['strengths'],
                interests=info['interests']
            )
            
            response = self._format_career_to_major_response(result)
        
        elif info['strengths']:
            # Subject → Career advice
            result = self.career_advisor.advise_major_to_career(
                strong_subjects=info['strengths'],
                interests=info['interests'],
                personality_traits=info['personality']
            )
            
            response = self._format_subject_to_career_response(result)
        
        else:
            # Need more info
            response = """Để tư vấn tốt hơn, em cho anh/chị biết thêm:

📌 Nếu em đã biết nghề muốn làm:
   → "Em muốn làm [nghề gì]"
   
📌 Nếu em chưa biết chọn gì:
   → "Em giỏi [môn nào]"
   → "Em thích [gì]"

Ví dụ:
• "Em muốn làm kỹ sư phần mềm, em giỏi toán lý"
• "Em giỏi toán hóa, thích nghiên cứu"
"""
        
        return response
    
    def _handle_course_query(self, user_input: str) -> str:
        """Handle course information queries using GraphRAG."""
        
        # Use GraphRAG query handler
        response = self.query_handler.ask_question(
            query=user_input,
            k=2,
            top_k_seeds=5,
            max_nodes=80,
            use_embeddings=True
        )
        
        return response
    
    def _handle_compare_majors(self, user_input: str) -> str:
        """Handle major comparison queries."""
        
        # Extract major names
        majors = self._extract_major_names(user_input)
        
        if len(majors) < 2:
            return """Để so sánh, em cần cung cấp ít nhất 2 ngành học.

Ví dụ:
• "So sánh Công nghệ thông tin và Kỹ thuật phần mềm"
• "CNTT hay Kỹ thuật điện tử?"
• "Khác nhau giữa CNTT và An toàn thông tin"
"""
        
        # Get comparison
        result = self.career_advisor.compare_majors(majors)
        
        response = f"""📊 SO SÁNH: {' vs '.join(majors)}

{result['comparison']}

---
💡 Tip: Hãy hỏi thêm nếu muốn biết chi tiết về ngành nào!
"""
        
        return response
    
    def _handle_learning_path(self, user_input: str) -> str:
        """Handle learning path queries."""
        
        # Extract major name
        major = self._extract_major_from_input(user_input)
        
        if not major:
            return """Để xem lộ trình học tập, em cho anh/chị biết ngành em quan tâm.

Ví dụ:
• "Lộ trình học Công nghệ thông tin"
• "Cần chuẩn bị gì để học CNTT?"
• "Học Kỹ thuật phần mềm như thế nào?"
"""
        
        # Get roadmap
        roadmap = self.career_advisor.get_learning_roadmap(major)
        
        response = self._format_learning_path_response(roadmap)
        
        return response
    
    def _handle_general_info(self, user_input: str) -> str:
        """Handle general information queries."""
        
        # Use LLM with context from both systems
        system_prompt = """Bạn là trợ lý AI giáo dục, hỗ trợ:
1. Trả lời câu hỏi về học phần, giảng viên, chương trình đào tạo
2. Tư vấn ngành học và nghề nghiệp cho học sinh

Khi trả lời:
- Thân thiện, nhiệt tình
- Gợi ý cách hỏi cụ thể hơn nếu câu hỏi chưa rõ
- Đưa ra ví dụ minh họa
"""
        
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _ask_clarification(self) -> str:
        """Ask for clarification when intent unclear."""
        
        return """Em muốn hỏi về:

1️⃣  Thông tin học phần (môn học, giảng viên, tài liệu)
    Ví dụ: "Giảng viên nào dạy môn PTTKHT?"

2️⃣  Tư vấn ngành học và nghề nghiệp
    Ví dụ: "Em muốn làm kỹ sư phần mềm, nên học gì?"

3️⃣  So sánh các ngành học
    Ví dụ: "So sánh CNTT và Kỹ thuật điện tử"

4️⃣  Lộ trình học tập
    Ví dụ: "Lộ trình học CNTT từ lớp 10"

Hãy hỏi cụ thể hơn để anh/chị hỗ trợ tốt nhất nhé! 😊
"""
    
    # =========================================================
    # INFORMATION EXTRACTION
    # =========================================================
    
    def _extract_career_info(self, text: str) -> dict:
        """Extract career-related information from text."""
        
        info = {
            'desired_career': None,
            'strengths': [],
            'interests': [],
            'personality': []
        }
        
        text_lower = text.lower()
        
        # Extract desired career - IMPROVED
        career_patterns = [
            # Pattern 1: "muốn làm X, ..." - STOP at comma
            r'muốn\s+làm\s+([^,\?]+?)(?:\s*,|\s+thì|\s+nên|\s+nhưng|\s+\?|$)',
            # Pattern 2: "làm X thì..." - STOP at "thì"  
            r'(?:^|\s)làm\s+([^,\?]+?)(?:\s+thì|\s+nên|\s*,|\s+nhưng|\s+\?|$)',
            # Pattern 3: "muốn trở thành X"
            r'muốn\s+trở\s+thành\s+([^,\?]+?)(?:\s*,|\s+thì|\s+nên|\s+nhưng|\s+\?|$)',
            # Pattern 4: "nghề X"
            r'nghề\s+([^,\?]+?)(?:\s+thì|\s+nên|\s*,|\s+\?|$)',
        ]
        
        for pattern in career_patterns:
            match = re.search(pattern, text_lower)
            if match:
                career = match.group(1).strip()
                
                # Clean up - remove question words and conjunctions
                cleanup_patterns = [
                    r'\s+thì.*$',           # " thì học ngành gì"
                    r'\s+nên.*$',           # " nên học gì"
                    r'\s+nhưng.*$',         # " nhưng không biết"
                    r'\s+học.*$',           # " học gì"
                    r'\s+gì.*$',            # " gì"
                    r'\s+nào.*$',           # " nào"
                    r'\s+như\s+thế\s+nào.*$'  # " như thế nào"
                ]
                
                for cleanup in cleanup_patterns:
                    career = re.sub(cleanup, '', career)
                
                career = career.strip()
                
                # Validate: must be reasonable length and not just stopwords
                if len(career) > 2 and career not in ['em', 'tôi', 'mình']:
                    info['desired_career'] = career
                    break
        
        # Extract subjects
        subject_keywords = ['toán', 'lý', 'hóa', 'sinh', 'văn', 'sử', 'địa', 'anh']
        for subject in subject_keywords:
            if subject in text_lower:
                info['strengths'].append(subject)
        
        # Extract interests
        interest_patterns = [
            r'thích\s+([^,\.]+?)(?:\s*,|\s*\.|\s*$)',
            r'đam\s+mê\s+([^,\.]+?)(?:\s*,|\s*\.|\s*$)',
            r'yêu\s+thích\s+([^,\.]+?)(?:\s*,|\s*\.|\s*$)'
        ]
        
        for pattern in interest_patterns:
            match = re.search(pattern, text_lower)
            if match:
                interest = match.group(1).strip()
                if len(interest) > 2:
                    info['interests'].append(interest)
        
        # Extract personality traits
        personality_keywords = ['logic', 'sáng tạo', 'tỉ mỉ', 'kiên nhẫn', 'năng động']
        for trait in personality_keywords:
            if trait in text_lower:
                info['personality'].append(trait)
        
        return info
    
    def _extract_major_names(self, text: str) -> list:
        """Extract major names from comparison query."""
        
        # Common separators
        separators = [' và ', ' vs ', ' hay ', ' hoặc ', ',']
        
        text_lower = text.lower()
        
        # Remove common words
        text_lower = re.sub(r'(so sánh|khác nhau|nên chọn|giữa)', '', text_lower)
        
        # Split by separators
        majors = [text_lower]
        for sep in separators:
            if sep in majors[0]:
                majors = majors[0].split(sep)
                break
        
        # Clean and filter
        majors = [m.strip() for m in majors]
        majors = [m for m in majors if len(m) > 2]
        
        return majors[:3]  # Max 3 majors
    
    def _extract_major_from_input(self, text: str) -> str:
        """Extract major name from learning path query."""
        
        text_lower = text.lower()
        
        # Remove noise words
        text_lower = re.sub(r'(lộ trình|học|chuẩn bị|như thế nào|thế nào)', '', text_lower)
        text_lower = text_lower.strip()
        
        # Common major keywords
        major_keywords = ['cntt', 'công nghệ thông tin', 'kỹ thuật', 'phần mềm']
        
        for keyword in major_keywords:
            if keyword in text_lower:
                return keyword
        
        # Return cleaned text if not empty
        if len(text_lower) > 3:
            return text_lower
        
        return None
    
    # =========================================================
    # RESPONSE FORMATTING
    # =========================================================
    
    def _format_career_to_major_response(self, result: dict) -> str:
        """Format career-to-major advisory response."""
        
        response = f"🎯 TƯ VẤN NGÀNH HỌC\n"
        response += "=" * 60 + "\n\n"
        
        if result['recommended_majors']:
            response += f"✅ Độ phù hợp: {result['matching_score'] * 100:.0f}%\n\n"
            
            response += "📚 NGÀNH HỌC ĐỀ XUẤT (Trường Kinh tế Quốc dân):\n"
            for i, major in enumerate(result['recommended_majors'][:3], 1):
                response += f"  {i}. {major['name']}\n"
            
            # ⭐ SIMPLIFIED - Chỉ show nếu có courses relevant
            if result.get('recommended_courses') and len(result['recommended_courses']) > 0:
                # Filter out generic courses
                relevant_courses = [
                    c for c in result['recommended_courses'][:8]
                    if c.get('mã_học_phần') and c.get('mã_học_phần') != 'N/A'
                ]
                
                if relevant_courses:
                    response += "\n📖 HỌC PHẦN CHUYÊN MÔN CỤ THỂ:\n"
                    for i, course in enumerate(relevant_courses[:5], 1):  # Max 5
                        course_line = f"  {i}. {course['name']}"
                        if course.get('mã_học_phần'):
                            course_line += f" ({course['mã_học_phần']})"
                        response += course_line + "\n"
            
            if result['required_skills']:
                response += "\n💪 KỸ NĂNG CẦN PHÁT TRIỂN:\n"
                for category, skills in result['required_skills'].items():
                    if skills and category != 'khác':  # Skip generic category
                        response += f"  • {', '.join(skills[:3])}\n"
        
        response += f"\n{'-' * 60}\n"
        response += result['advice']
        
        return response
    
    def _format_subject_to_career_response(self, result: dict) -> str:
        """Format subject-to-career advisory response."""
        
        response = f"🎓 TƯ VẤN NGHỀ NGHIỆP & NGÀNH HỌC\n"
        response += "=" * 60 + "\n\n"
        
        if result['suitable_majors']:
            response += "📚 NGÀNH HỌC PHÙ HỢP:\n"
            for i, major in enumerate(result['suitable_majors'][:3], 1):
                response += f"  {i}. {major['name']}\n"
        
        if result['career_options']:
            response += "\n💼 CƠ HỘI NGHỀ NGHIỆP:\n"
            for i, career in enumerate(result['career_options'][:5], 1):
                response += f"  {i}. {career['name']}\n"
        
        response += f"\n{'-' * 60}\n"
        response += result['advice']
        
        return response
    
    def _format_learning_path_response(self, roadmap: dict) -> str:
        """Format learning path response."""
        
        if not roadmap:
            return "⚠️  Chưa có thông tin lộ trình chi tiết cho ngành này."
        
        response = f"📚 LỘ TRÌNH HỌC TẬP\n"
        response += "=" * 60 + "\n\n"
        
        if 'major' in roadmap:
            response += f"Ngành: {roadmap['major']}\n\n"
        
        if 'preparation' in roadmap:
            response += "🎒 GIAI ĐOẠN PHỔ THÔNG:\n"
            for grade, tasks in roadmap['preparation'].items():
                response += f"\n{grade.upper().replace('_', ' ')}:\n"
                if isinstance(tasks, list):
                    for task in tasks:
                        response += f"  • {task}\n"
                else:
                    response += f"  • {tasks}\n"
        
        if 'university' in roadmap:
            response += "\n🎓 GIAI ĐOẠN ĐẠI HỌC:\n"
            for year, content in roadmap['university'].items():
                response += f"\n{year.upper().replace('_', ' ')}:\n"
                if isinstance(content, list):
                    for item in content:
                        response += f"  • {item}\n"
                else:
                    response += f"  • {content}\n"
        
        return response
    
    # =========================================================
    # CONVERSATION MANAGEMENT
    # =========================================================
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation."""
        
        if not self.conversation_history:
            return "Chưa có lịch sử hội thoại."
        
        summary = "📝 LỊCH SỬ HỘI THOẠI\n"
        summary += "=" * 60 + "\n\n"
        
        for i, msg in enumerate(self.conversation_history, 1):
            if msg['role'] == 'user':
                summary += f"👤 Bạn: {msg['content']}\n"
                if 'intent' in msg:
                    summary += f"   (Intent: {msg['intent']})\n"
            else:
                preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                summary += f"🤖 Bot: {preview}\n"
            summary += "\n"
        
        return summary
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# =========================================================
# INITIALIZATION
# =========================================================

def initialize_chatbot():
    """Initialize unified chatbot."""
    try:
        print("🔌 Connecting to knowledge graph...")
        
        db_connection = GraphDatabaseConnection(
            uri=DB_URL,
            user=DB_USERNAME,
            password=DB_PASSWORD
        )
        
        stats = db_connection.get_database_stats()
        
        if stats['nodes'] == 0:
            print("❌ Database is empty!")
            print("Please run 'python build_graph_complete.py' first")
            db_connection.close()
            return None, None
        
        print(f"✅ Connected to knowledge graph")
        print(f"   Nodes: {stats['nodes']}")
        print(f"   Relationships: {stats['relationships']}")
        print()
        
        # Initialize components
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        graph_manager = GraphManagerV3(
            db_connection=db_connection,
            auto_clear=False,
            openai_client=client
        )
        
        chatbot = UnifiedEducationalChatbot(
            db_connection=db_connection,
            graph_manager=graph_manager,
            client=client
        )
        
        return db_connection, chatbot
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return None, None


# =========================================================
# INTERACTIVE MODE
# =========================================================

def interactive_mode(chatbot: UnifiedEducationalChatbot):
    """Interactive chat mode."""
    
    print("\n" + "=" * 80)
    print("🤖 CHATBOT GIÁO DỤC THÔNG MINH")
    print("=" * 80)
    print("\n👋 Xin chào! Tôi có thể giúp bạn:")
    print("   🔍 Tra cứu thông tin học phần, giảng viên, tài liệu")
    print("   🎓 Tư vấn ngành học và nghề nghiệp")
    print("   📊 So sánh các ngành học")
    print("   📚 Lộ trình học tập")
    print("\n💡 Commands:")
    print("   'examples' - Xem ví dụ câu hỏi")
    print("   'history' - Xem lịch sử hội thoại")
    print("   'clear' - Xóa lịch sử")
    print("   'quit' hoặc 'exit' - Thoát")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("💬 Bạn: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Tạm biệt! Chúc bạn học tốt và thành công!")
                break
            
            elif user_input.lower() == 'examples':
                show_examples()
                continue
            
            elif user_input.lower() == 'history':
                print("\n" + chatbot.get_conversation_summary())
                continue
            
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("✅ Đã xóa lịch sử hội thoại")
                continue
            
            elif user_input.lower() == 'help':
                show_help()
                continue
            
            # Get response
            print("\n🤔 Đang suy nghĩ...\n")
            response = chatbot.chat(user_input)
            
            print("🤖 Bot:\n")
            print(response)
            print("\n" + "-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            logger.error(f"Chat error: {e}", exc_info=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def show_examples():
    """Show example questions."""
    
    print("\n" + "=" * 80)
    print("📚 VÍ DỤ CÂU HỎI")
    print("=" * 80)
    
    examples = {
        "🔍 TRA CỨU HỌC PHẦN": [
            "Giảng viên nào dạy môn Phân tích và thiết kế hệ thống?",
            "Mã học phần của môn Phân tích thiết kế hệ thống là gì?",
            "Tài liệu tham khảo cho môn này?",
            "Email của giảng viên Trần Thị Mỹ Diệp?"
        ],
        "🎓 TƯ VẤN NGÀNH HỌC": [
            "Em muốn làm kỹ sư phần mềm, nên học ngành gì?",
            "Em giỏi toán và lý, thích công nghệ",
            "Em thích nghiên cứu và làm việc với máy tính"
        ],
        "📊 SO SÁNH NGÀNH": [
            "So sánh Công nghệ thông tin và Kỹ thuật phần mềm",
            "CNTT hay Kỹ thuật điện tử tốt hơn?",
            "Khác nhau giữa CNTT và An toàn thông tin"
        ],
        "📚 LỘ TRÌNH HỌC TẬP": [
            "Lộ trình học Công nghệ thông tin từ lớp 10",
            "Cần chuẩn bị gì để học CNTT?",
            "Học Kỹ thuật phần mềm như thế nào?"
        ]
    }
    
    for category, questions in examples.items():
        print(f"\n{category}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print("\n" + "=" * 80 + "\n")

def show_help():
    """Show help information."""
    
    print("\n" + "=" * 80)
    print("📖 HƯỚNG DẪN SỬ DỤNG")
    print("=" * 80)
    print("""
CHATBOT NÀY HỖ TRỢ:

1. 🔍 TRA CỨU THÔNG TIN HỌC PHẦN
   - Hỏi về giảng viên, email, chức danh
   - Hỏi về môn học: tín chỉ, mã học phần, giờ học
   - Hỏi về tài liệu, sách giáo trình
   - Hỏi về học phần tiên quyết, điều kiện
   
   Ví dụ: "Giảng viên nào dạy PTTKHT?"

2. 🎓 TƯ VẤN NGÀNH HỌC & NGHỀ NGHIỆP
   - Tư vấn ngành học từ nghề nghiệp mong muốn
   - Tư vấn nghề nghiệp từ môn học giỏi
   - Phân tích điểm mạnh, sở thích
   
   Ví dụ: "Em muốn làm kỹ sư phần mềm, em giỏi toán lý"

3. 📊 SO SÁNH CÁC NGÀNH HỌC
   - So sánh nội dung, cơ hội việc làm
   - Phân tích ưu nhược điểm
   
   Ví dụ: "So sánh CNTT và Kỹ thuật điện tử"

4. 📚 LỘ TRÌNH HỌC TẬP
   - Xem roadmap từ THPT đến Đại học
   - Kế hoạch học tập chi tiết
   
   Ví dụ: "Lộ trình học CNTT"

COMMANDS:
  'examples' - Xem câu hỏi mẫu
  'history' - Xem lịch sử chat
  'clear' - Xóa lịch sử
  'help' - Hiện hướng dẫn này
  'quit' - Thoát

TIPS:
  • Hỏi cụ thể để được kết quả tốt nhất
  • Cung cấp nhiều thông tin (môn giỏi, sở thích)
  • Dùng tên đầy đủ của môn học
""")
    print("=" * 80 + "\n")


# =========================================================
# MAIN
# =========================================================

def main():
    """Main entry point."""
    
    # Initialize
    db_connection, chatbot = initialize_chatbot()
    
    if not chatbot:
        sys.exit(1)
    
    try:
        # Interactive mode
        interactive_mode(chatbot)
    
    finally:
        # Cleanup
        if db_connection:
            db_connection.close()
            print("\n🔌 Đã đóng kết nối")


if __name__ == "__main__":
    main()
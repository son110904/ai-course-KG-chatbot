# query_handler_v3.py
"""
ENHANCED Query Handler V3
- Improved Vietnamese query processing
- Better fallback mechanisms
- Context-aware responses
- Enhanced entity matching
"""

from graph_manager_v3 import GraphManagerV3
from logger import Logger
from typing import List, Dict, Optional
import re
import unicodedata


class QueryHandlerV3:
    """
    Enhanced Query Handler with:
    - Smart query term extraction
    - Multiple search strategies
    - Intelligent fallback responses
    - Vietnamese text normalization
    """

    logger = Logger("QueryHandlerV3").get_logger()

    def __init__(self, graph_manager: GraphManagerV3, client, model: str):
        self.graph_manager = graph_manager
        self.client = client
        self.model = model
        self.logger.info(f"Initialized QueryHandlerV3 with model={model}")

    # =========================================================
    # MAIN QUERY HANDLER
    # =========================================================
    
    def ask_question(
        self,
        query: str,
        k: int = 2,
        top_k_seeds: int = 5,
        max_nodes: int = 80,
        use_embeddings: bool = True
    ) -> str:
        """
        Answer question using k-hop retrieval with intelligent fallback.
        
        Args:
            query: User question
            k: K-hop depth (1-3)
            top_k_seeds: Number of seed entities (3-10)
            max_nodes: Max nodes in subgraph (50-200)
            use_embeddings: Use semantic search
            
        Returns:
            Answer string
        """
        # Normalize query
        query = unicodedata.normalize('NFC', query)
        
        self.logger.info(f"Processing query: {query}")
        self.logger.info(f"  Settings: k={k}, seeds={top_k_seeds}, max_nodes={max_nodes}, embeddings={use_embeddings}")

        # Step 1: Extract query terms and entities
        query_analysis = self._analyze_query(query)
        
        self.logger.info(f"  Query analysis:")
        self.logger.info(f"    - Terms: {query_analysis['terms']}")
        self.logger.info(f"    - Entities: {query_analysis['entities']}")
        self.logger.info(f"    - Question type: {query_analysis['question_type']}")
        
        # Step 2: Find seed entities
        seed_entities = self.graph_manager.find_relevant_entities(
            query_terms=query_analysis['all_keywords'],
            top_k=top_k_seeds,
            use_embeddings=use_embeddings
        )

        # FALLBACK 1: No entities found
        if not seed_entities:
            return self._generate_no_entities_response(query, query_analysis)

        self.logger.info(f"  Found {len(seed_entities)} seed entities: {seed_entities}")

        # Step 3: Get subgraph
        subgraph = self.graph_manager.get_k_hop_subgraph(
            seed_entities=seed_entities,
            k=k,
            max_nodes=max_nodes
        )

        # FALLBACK 2: Empty subgraph
        if not subgraph or not subgraph.get('nodes'):
            return self._generate_empty_subgraph_response(query, seed_entities)

        # Step 4: Check subgraph relevance
        relevance_check = self._check_subgraph_relevance(
            subgraph, query, query_analysis
        )
        
        self.logger.info(f"  Relevance check: {relevance_check['has_relevant_data']}")
        if relevance_check['missing_info']:
            self.logger.info(f"    Missing: {relevance_check['missing_info']}")

        # Step 5: Format context
        context = self.graph_manager.format_subgraph_for_context(subgraph)

        # Step 6: Generate response with LLM
        system_prompt = self._build_system_prompt(
            query_analysis,
            relevance_check
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        f"Câu hỏi: {query}\n\n"
                        f"Seed entities: {', '.join(seed_entities)}\n\n"
                        f"Knowledge graph context:\n{context}\n\n"
                        f"Hãy trả lời chính xác dựa trên thông tin trong graph."
                    )
                }
            ],
            max_tokens=1000,
            temperature=0
        )

        answer = response.choices[0].message.content

        # Step 7: Enhance answer if needed
        if relevance_check['missing_info']:
            answer = self._enhance_answer_with_missing_info(
                answer,
                relevance_check['missing_info']
            )

        return answer

    # =========================================================
    # QUERY ANALYSIS
    # =========================================================
    
    def _analyze_query(self, query: str) -> Dict:
        """
        Analyze query to extract:
        - Main entities mentioned
        - Query terms
        - Question type
        """
        query_lower = query.lower()
        
        # Detect question type
        question_type = self._detect_question_type(query_lower)
        
        # Extract entities mentioned in query
        entities = self._extract_mentioned_entities(query)
        
        # Extract query terms
        terms = self._extract_query_terms(query)
        
        # Combine all keywords
        all_keywords = list(set(entities + terms))
        
        return {
            'question_type': question_type,
            'entities': entities,
            'terms': terms,
            'all_keywords': all_keywords
        }
    
    def _detect_question_type(self, query_lower: str) -> str:
        """Detect what kind of question is being asked."""
        
        patterns = {
            'instructor': ['giảng viên', 'giáo viên', 'thầy', 'cô', 'ai giảng', 'ai dạy'],
            'email': ['email', 'mail', 'liên hệ'],
            'credits': ['tín chỉ', 'số tín', 'credit'],
            'code': ['mã học phần', 'mã môn', 'code'],
            'prerequisite': ['tiên quyết', 'điều kiện', 'prerequisite', 'học trước'],
            'materials': ['tài liệu', 'sách', 'giáo trình', 'reference'],
            'software': ['phần mềm', 'software', 'công cụ', 'tool'],
            'objectives': ['mục tiêu', 'objective', 'goal'],
            'outcomes': ['chuẩn đầu ra', 'clo', 'outcome', 'learning outcome'],
            'description': ['mô tả', 'description', 'nội dung', 'về gì'],
            'assessment': ['đánh giá', 'assessment', 'thi', 'kiểm tra'],
            'hours': ['giờ', 'hour', 'thời gian'],
            'department': ['khoa', 'viện', 'department', 'faculty']
        }
        
        for q_type, keywords in patterns.items():
            if any(kw in query_lower for kw in keywords):
                return q_type
        
        return 'general'
    
    def _extract_mentioned_entities(self, query: str) -> List[str]:
        """
        Extract entity names mentioned in query.
        Enhanced for Vietnamese course/instructor names.
        """
        entities = []
        
        # Method 1: Look for quoted strings (explicit mentions)
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Method 2: Look for capitalized phrases
        # Vietnamese course names often have caps
        words = query.split()
        current_entity = []
        
        for word in words:
            # Check if word starts with capital or is all caps
            # But exclude common question words even if capitalized
            if word and (word[0].isupper() or word.isupper()):
                # Skip if it's a question word
                if word.lower() not in ['gì', 'nào', 'ai', 'đâu', 'sao']:
                    current_entity.append(word)
            else:
                if current_entity:
                    entity_name = ' '.join(current_entity)
                    if len(entity_name) > 3:  # Filter out short caps
                        entities.append(entity_name)
                    current_entity = []
        
        # Add last entity
        if current_entity:
            entity_name = ' '.join(current_entity)
            if len(entity_name) > 3:
                entities.append(entity_name)
        
        # Method 3: Look for common Vietnamese name patterns
        # Teacher names: "ThS. ...", "TS. ...", "PGS. ...", "GS. ..."
        name_patterns = [
            r'((?:ThS|TS|PGS|GS)\.?\s+[A-ZĐĂÂÊÔƠƯ][a-zđăâêôơư]+(?:\s+[A-ZĐĂÂÊÔƠƯ][a-zđăâêôơư]+)+)',
            r'((?:Thạc sĩ|Tiến sĩ|Phó Giáo sư|Giáo sư)\s+[A-ZĐĂÂÊÔƠƯ][a-zđăâêôơư]+(?:\s+[A-ZĐĂÂÊÔƠƯ][a-zđăâêôơư]+)+)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Deduplicate
        entities = list(dict.fromkeys(entities))  # Preserve order
        
        return entities
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract meaningful terms from query.
        Filter out stopwords and common question words.
        """
        # Normalize
        query = unicodedata.normalize('NFC', query)
        
        # Extract words
        terms = re.findall(r'\w+', query.lower())

        # Vietnamese stopwords - MINIMAL set
        # Only remove truly meaningless words, keep domain-specific terms
        stopwords = {
            # Question words
            'là', 'bao', 'nhiêu', 'có', 'mấy', 'gì', 'nào', 'thế', 
            'ai', 'khi', 'nào', 'đâu', 'sao', 'thì', 'này', 'đó',
            
            # Conjunctions & prepositions
            'và', 'với', 'trong', 'về', 'cho', 'của', 'để', 'từ', 
            'hay', 'hoặc', 'nhưng', 'mà',
            
            # Verb helpers
            'không', 'chưa', 'đã', 'sẽ', 'đang', 'vẫn', 'được',
            
            # Articles/determiners  
            'các', 'những', 'một', 'cái',
            
            # Common but meaningless
            'như', 'em', 'tôi', 'bạn', 'ạ'
        }
        
        # NOTE: We do NOT remove domain terms like:
        # - 'môn', 'học', 'phần' (course-related)
        # - 'cho' when part of entity name
        # These may be part of entity names or important context

        # Filter
        meaningful_terms = [
            t for t in terms 
            if t not in stopwords and len(t) > 2
        ]
        
        return meaningful_terms

    # =========================================================
    # RELEVANCE CHECKING
    # =========================================================
    
    def _check_subgraph_relevance(
        self,
        subgraph: Dict,
        query: str,
        query_analysis: Dict
    ) -> Dict:
        """
        Check if subgraph contains relevant information to answer query.
        
        Returns:
            {
                'has_relevant_data': bool,
                'missing_info': List[str],
                'found_info': List[str]
            }
        """
        question_type = query_analysis['question_type']
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        found_info = []
        missing_info = []
        
        # Check based on question type
        if question_type == 'instructor':
            has_instructor = any(n.get('type') == 'giảng_viên' for n in nodes)
            has_teaching_rel = any('GIẢNG' in e.get('type', '').upper() for e in edges)
            
            if has_instructor or has_teaching_rel:
                found_info.append('thông tin giảng viên')
            else:
                missing_info.append('thông tin giảng viên')
        
        elif question_type == 'email':
            has_email = any(n.get('email') for n in nodes)
            
            if has_email:
                found_info.append('email')
            else:
                missing_info.append('email')
        
        elif question_type == 'credits':
            has_credits = any(
                n.get('số_tín_chỉ') or n.get('tín_chỉ')
                for n in nodes
            )
            
            if has_credits:
                found_info.append('số tín chỉ')
            else:
                missing_info.append('số tín chỉ')
        
        elif question_type == 'code':
            has_code = any(n.get('mã_học_phần') for n in nodes)
            
            if has_code:
                found_info.append('mã học phần')
            else:
                missing_info.append('mã học phần')
        
        elif question_type == 'prerequisite':
            has_prereq = any('TIÊN_QUYẾT' in e.get('type', '').upper() for e in edges)
            
            if has_prereq:
                found_info.append('học phần tiên quyết')
            else:
                missing_info.append('học phần tiên quyết')
        
        elif question_type == 'materials':
            has_materials = any(n.get('type') == 'tài_liệu' for n in nodes)
            
            if has_materials:
                found_info.append('tài liệu')
            else:
                missing_info.append('tài liệu')
        
        elif question_type == 'software':
            has_software = any(n.get('type') == 'phần_mềm' for n in nodes)
            
            if has_software:
                found_info.append('phần mềm')
            else:
                missing_info.append('phần mềm')
        
        has_relevant_data = len(found_info) > 0
        
        return {
            'has_relevant_data': has_relevant_data,
            'found_info': found_info,
            'missing_info': missing_info
        }

    # =========================================================
    # SYSTEM PROMPT BUILDING
    # =========================================================
    
    def _build_system_prompt(
        self,
        query_analysis: Dict,
        relevance_check: Dict
    ) -> str:
        """Build context-aware system prompt."""
        
        question_type = query_analysis['question_type']
        has_data = relevance_check['has_relevant_data']
        missing_info = relevance_check['missing_info']
        
        base_prompt = """Bạn là hệ thống trả lời câu hỏi về chương trình đào tạo dựa trên Knowledge Graph.

QUY TẮC CHUNG:
- CHỈ trả lời dựa trên thông tin có trong graph context
- Trả lời bằng tiếng Việt tự nhiên, rõ ràng
- KHÔNG suy đoán hoặc bịa thông tin không có trong graph
- Nếu thiếu thông tin, nêu rõ phần nào THIẾU
"""
        
        # Add question-type specific instructions
        if question_type == 'instructor':
            base_prompt += """
LOẠI CÂU HỎI: Giảng viên
- Liệt kê TẤT CẢ giảng viên tìm thấy
- Bao gồm email nếu có
- Bao gồm chức danh nếu có
"""
        
        elif question_type == 'email':
            base_prompt += """
LOẠI CÂU HỎI: Email/Liên hệ
- Cung cấp email chính xác
- Nếu có nhiều người, liệt kê tất cả
"""
        
        elif question_type == 'credits' or question_type == 'code':
            base_prompt += """
LOẠI CÂU HỎI: Thông tin học phần
- Trả lời chính xác số liệu
- Nêu rõ đơn vị (tín chỉ, giờ, v.v.)
"""
        
        elif question_type == 'materials':
            base_prompt += """
LOẠI CÂU HỎI: Tài liệu tham khảo
- Liệt kê đầy đủ tài liệu
- Phân loại: Giáo trình / Tài liệu tham khảo
- Bao gồm tác giả, năm xuất bản nếu có
"""
        
        # Add data availability notice
        if not has_data and missing_info:
            base_prompt += f"""
⚠️ CHÚ Ý: Graph THIẾU thông tin về: {', '.join(missing_info)}

Khi trả lời:
1. Nêu rõ thông tin NÀO có trong graph
2. Nêu rõ thông tin NÀO thiếu
3. Đề xuất cách tìm thông tin thiếu (hỏi cụ thể hơn, hoặc liên hệ khoa)
"""
        else:
            base_prompt += """
- Trình bày thông tin đầy đủ và có tổ chức
- Sử dụng bullet points khi cần thiết
"""
        
        return base_prompt

    # =========================================================
    # FALLBACK RESPONSES
    # =========================================================
    
    def _generate_no_entities_response(
        self,
        query: str,
        query_analysis: Dict
    ) -> str:
        """Generate response when no entities found."""
        
        self.logger.warning(f"No entities found for query: {query}")
        
        # Try to suggest similar entities
        suggestions = []
        for term in query_analysis['all_keywords']:
            similar = self.graph_manager.search_entities(term, limit=3)
            suggestions.extend(similar)
        
        suggestions = list(set(suggestions))[:5]
        
        response = "⚠️ Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu.\n\n"
        
        if suggestions:
            response += "Có thể bạn đang tìm kiếm:\n"
            for s in suggestions:
                response += f"  • {s}\n"
            response += "\n💡 Hãy thử hỏi lại với các tên này."
        else:
            response += "💡 Gợi ý:\n"
            response += "  • Hỏi về các học phần cụ thể (ví dụ: 'Phân tích và thiết kế hệ thống')\n"
            response += "  • Hỏi về giảng viên, tài liệu, hoặc chương trình đào tạo\n"
            response += "  • Sử dụng tên đầy đủ của học phần\n"
        
        return response
    
    def _generate_empty_subgraph_response(
        self,
        query: str,
        seed_entities: List[str]
    ) -> str:
        """Generate response when subgraph is empty."""
        
        self.logger.warning(f"Empty subgraph for entities: {seed_entities}")
        
        response = f"✓ Tìm thấy: {', '.join(seed_entities)}\n\n"
        response += "⚠️ Tuy nhiên, không có thông tin bổ sung hoặc mối quan hệ nào được ghi nhận.\n\n"
        response += "Điều này có thể do:\n"
        response += "  • Thông tin chưa được cập nhật đầy đủ vào hệ thống\n"
        response += "  • Entity này chưa có liên kết với các thông tin khác\n"
        response += "  • Cần mở rộng tìm kiếm (tăng k-hop hoặc số seed entities)\n"
        
        return response
    
    def _enhance_answer_with_missing_info(
        self,
        answer: str,
        missing_info: List[str]
    ) -> str:
        """Enhance answer with notice about missing information."""
        
        # Don't modify if answer already mentions missing info
        if any(word in answer.lower() for word in ['không có', 'thiếu', 'chưa có']):
            return answer
        
        enhancement = "\n\n---\n"
        enhancement += f"ℹ️ Lưu ý: Thông tin về **{', '.join(missing_info)}** chưa có trong cơ sở dữ liệu.\n"
        enhancement += "Bạn có thể:\n"
        enhancement += "  • Hỏi cụ thể hơn về thông tin có sẵn\n"
        enhancement += "  • Liên hệ khoa/viện quản lý để biết thêm chi tiết\n"
        
        return answer + enhancement

    # =========================================================
    # COMPATIBILITY
    # =========================================================
    
    def ask_question_with_khop(self, query: str, **kwargs) -> str:
        """Alias for compatibility."""
        return self.ask_question(query, **kwargs)
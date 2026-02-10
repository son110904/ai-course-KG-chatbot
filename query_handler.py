# query_handler.py
from graph_manager import GraphManager
from logger import Logger
from typing import List, Dict
import re
import unicodedata


class QueryHandler:
    """
    GraphRAG Query Handler with:
    - K-hop retrieval
    - Embedding support
    - Fallback responses for missing data ✅ NEW
    - Vietnamese text normalization ✅ NEW
    """

    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client, model: str):
        self.graph_manager = graph_manager
        self.client = client
        self.model = model
        self.logger.info(f"Initialized with model={model}")

    # =========================================================
    # MAIN K-HOP WITH FALLBACK
    # =========================================================
    def ask_question_with_khop(
        self,
        query: str,
        k: int = 2,
        top_k_seeds: int = 5,
        max_nodes: int = 80,
        use_embeddings: bool = True
    ) -> str:
        """
        Answer question with k-hop retrieval and intelligent fallback.
        IMPROVED: Better handling of missing data scenarios.
        """
        # Normalize query
        query = unicodedata.normalize('NFC', query)
        
        self.logger.info(f"K-hop query: {query} (embeddings={use_embeddings})")

        # Extract query terms
        query_terms = self._extract_query_terms(query)
        
        # Find seed entities
        seed_entities = self.graph_manager.find_relevant_entities(
            query_terms=query_terms,
            top_k=top_k_seeds,
            use_embeddings=use_embeddings
        )

        # FALLBACK 1: No entities found
        if not seed_entities:
            return self._generate_no_entities_response(query, query_terms)

        # Get subgraph
        subgraph = self.graph_manager.get_k_hop_subgraph(
            seed_entities=seed_entities,
            k=k,
            max_nodes=max_nodes
        )

        # FALLBACK 2: Empty subgraph
        if not subgraph or not subgraph.get('nodes'):
            return self._generate_empty_subgraph_response(query, seed_entities)

        # Check if subgraph has relevant information
        has_data, missing_fields = self._check_subgraph_completeness(
            subgraph, query, query_terms
        )

        # Format context
        context = self.graph_manager.format_subgraph_for_context(subgraph)

        # Generate response with appropriate system prompt
        system_prompt = self._build_system_prompt(has_data, missing_fields)
        
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
                        f"Graph context:\n{context}\n\n"
                        f"Hãy trả lời chính xác:"
                    )
                }
            ],
            max_tokens=800
        )

        answer = response.choices[0].message.content

        # FALLBACK 3: Enhance answer if missing fields detected
        if missing_fields and not has_data:
            answer = self._enhance_answer_with_suggestions(
                answer, missing_fields, seed_entities, subgraph
            )

        return answer

    # =========================================================
    # FALLBACK RESPONSE GENERATORS
    # =========================================================
    
    def _generate_no_entities_response(self, query: str, query_terms: List[str]) -> str:
        """Generate response when no entities found."""
        self.logger.warning(f"No entities found for query: {query}")
        
        # Try to suggest similar entities
        suggestions = []
        for term in query_terms:
            similar = self.graph_manager.search_entities(term, limit=3)
            suggestions.extend(similar)
        
        suggestions = list(set(suggestions))[:5]
        
        response = "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."
        
        if suggestions:
            response += "\n\nCó thể bạn đang tìm kiếm:\n"
            for s in suggestions:
                response += f"  - {s}\n"
            response += "\nHãy thử hỏi lại với các tên này."
        else:
            response += "\n\nGợi ý: Hãy thử:\n"
            response += "  - Hỏi về các học phần cụ thể (ví dụ: 'An toàn và bảo mật thông tin')\n"
            response += "  - Hỏi về giảng viên, tài liệu, hoặc chương trình đào tạo\n"
        
        return response
    
    def _generate_empty_subgraph_response(self, query: str, seed_entities: List[str]) -> str:
        """Generate response when subgraph is empty."""
        self.logger.warning(f"Empty subgraph for entities: {seed_entities}")
        
        response = f"Tìm thấy: {', '.join(seed_entities)}\n\n"
        response += "Tuy nhiên, không có thông tin bổ sung hoặc mối quan hệ nào được ghi nhận trong cơ sở dữ liệu.\n\n"
        response += "Điều này có thể do:\n"
        response += "  - Thông tin chưa được cập nhật đầy đủ\n"
        response += "  - Entity này chưa có liên kết với các thông tin khác\n"
        
        return response
    
    def _check_subgraph_completeness(
        self, 
        subgraph: Dict, 
        query: str, 
        query_terms: List[str]
    ) -> tuple:
        """
        Check if subgraph has information to answer query.
        
        Returns:
            (has_data: bool, missing_fields: List[str])
        """
        missing_fields = []
        
        # Check for specific query patterns
        query_lower = query.lower()
        
        # Pattern 1: Asking about giảng viên
        if any(term in query_lower for term in ['giảng viên', 'giáo viên', 'thầy', 'cô']):
            # Check if we have giảng_viên entities or GIẢNG_DẠY relationships
            has_instructor = any(
                node.get('type') == 'giảng_viên' 
                for node in subgraph.get('nodes', [])
            )
            has_teaching_rel = any(
                'GIẢNG' in edge.get('type', '').upper() or 
                'DẠY' in edge.get('type', '').upper()
                for edge in subgraph.get('edges', [])
            )
            
            if not (has_instructor or has_teaching_rel):
                missing_fields.append('thông tin giảng viên')
        
        # Pattern 2: Asking about tín chỉ
        if any(term in query_lower for term in ['tín chỉ', 'số tín chỉ', 'credit']):
            has_credit_info = any(
                'credit' in str(node).lower() or 'tín chỉ' in str(node).lower()
                for node in subgraph.get('nodes', [])
            )
            
            if not has_credit_info:
                missing_fields.append('số tín chỉ')
        
        # Pattern 3: Asking about tài liệu
        if any(term in query_lower for term in ['tài liệu', 'sách', 'giáo trình']):
            has_document = any(
                node.get('type') == 'tài_liệu'
                for node in subgraph.get('nodes', [])
            )
            
            if not has_document:
                missing_fields.append('tài liệu tham khảo')
        
        # Pattern 4: Asking about môn tiên quyết
        if any(term in query_lower for term in ['tiên quyết', 'điều kiện', 'prerequisite']):
            has_prerequisite = any(
                'TIÊN' in edge.get('type', '').upper() or
                'QUYẾT' in edge.get('type', '').upper()
                for edge in subgraph.get('edges', [])
            )
            
            if not has_prerequisite:
                missing_fields.append('học phần tiên quyết')
        
        has_data = len(missing_fields) == 0
        
        if missing_fields:
            self.logger.info(f"Missing fields detected: {missing_fields}")
        
        return has_data, missing_fields
    
    def _build_system_prompt(self, has_data: bool, missing_fields: List[str]) -> str:
        """Build appropriate system prompt based on data availability."""
        
        base_prompt = """Bạn là hệ thống QA dựa trên GraphRAG cho chương trình đào tạo.

QUY TẮC:
- Chỉ trả lời dựa trên dữ liệu trong graph context
- Trả lời bằng tiếng Việt tự nhiên
- KHÔNG suy đoán hoặc bịa thông tin
"""
        
        if not has_data and missing_fields:
            base_prompt += f"""
⚠️ CHÚ Ý: Graph thiếu thông tin về: {', '.join(missing_fields)}

Khi trả lời:
1. Nêu rõ thông tin nào CÓ trong graph
2. Nêu rõ thông tin nào THIẾU
"""
        else:
            base_prompt += """
- Trình bày thông tin một cách đầy đủ và rõ ràng
- Nếu có nhiều giảng viên, liệt kê tất cả
- Nếu có email hoặc thông tin liên hệ, bao gồm chúng
"""
        
        return base_prompt
    
    def _enhance_answer_with_suggestions(
        self, 
        answer: str, 
        missing_fields: List[str],
        seed_entities: List[str],
        subgraph: Dict
    ) -> str:
        """Enhance answer with helpful suggestions."""
        
        # Don't modify if answer already mentions missing info
        if 'không có' in answer.lower() or 'thiếu' in answer.lower():
            return answer
        
        # Add missing fields notice
        enhancement = "\n\n---\n"
        enhancement += f"⚠️ Lưu ý: Thông tin về {', '.join(missing_fields)} chưa có trong cơ sở dữ liệu.\n\n"
        
        # Generate suggestions based on available data
        available_info = []
        
        # Check what we DO have
        node_types = set(node.get('type') for node in subgraph.get('nodes', []))
        edge_types = set(edge.get('type') for edge in subgraph.get('edges', []))
        
        if 'học_phần' in node_types:
            available_info.append("thông tin học phần")
        if 'khoa' in node_types:
            available_info.append("khoa quản lý")
        if any('THUỘC' in t for t in edge_types if t):
            available_info.append("cơ cấu tổ chức")
        
        if available_info:
            enhancement += "Thông tin hiện có:\n"
            for info in available_info:
                enhancement += f"  ✓ {info}\n"
            enhancement += "\nBạn có thể hỏi thêm về các thông tin này."
        
        return answer + enhancement

    # =========================================================
    # QUERY TERM EXTRACTION
    # =========================================================
    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract meaningful terms from query.
        IMPROVED: Better Vietnamese stopword handling.
        """
        # Normalize
        query = unicodedata.normalize('NFC', query)
        
        # Extract words
        terms = re.findall(r'\w+', query.lower())

        # Vietnamese stopwords
        stop = {
            "là", "bao", "nhiêu", "có", "mấy", "cho", "của",
            "môn", "học", "phần", "gì", "nào", "thế", "như",
            "và", "với", "trong", "về", "các", "những", "được"
        }

        # Filter
        meaningful_terms = [t for t in terms if t not in stop and len(t) > 2]
        
        self.logger.debug(f"Extracted terms: {meaningful_terms}")
        
        return meaningful_terms

    # =========================================================
    # ENTRY POINT
    # =========================================================
    def ask_question(self, query: str, method="khop", **kwargs):
        """
        Main entry point for asking questions.
        
        Args:
            query: User question
            method: Always "khop" (for compatibility)
            **kwargs: Additional parameters for k-hop retrieval
            
        Returns:
            Answer string
        """
        return self.ask_question_with_khop(query, **kwargs)
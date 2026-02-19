# minio_loader_v3.py
"""
ENHANCED MinIO Loader V3
- Optimized for structured JSON from MinIO
- Better table and paragraph handling
- Improved entity extraction from both sources
"""

from minio import Minio
from openai import OpenAI
import json
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from logger import Logger
import unicodedata


class MinioLoaderV3:
    """
    Enhanced MinIO document loader with:
    - Direct JSON structure parsing
    - Smart table-paragraph integration
    - Type-aware entity extraction
    - Parallel processing support
    """
    
    logger = Logger("MinioLoaderV3").get_logger()
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_workers: int = 10,
        secure: bool = False
    ):
        """Initialize MinIO loader with configuration."""
        self.minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.client = client
        self.model = model
        self.max_workers = max_workers
        
        self.logger.info(f"Initialized MinioLoaderV3: {endpoint}/{bucket_name}")
    
    # =========================================================
    # DOCUMENT LOADING FROM MINIO
    # =========================================================
    
    def load_documents_from_folders(
        self, 
        folders: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Load all JSON documents from specified MinIO folders.
        
        Args:
            folders: List of folder prefixes (e.g., ["courses-processed/curriculum/"])
            
        Returns:
            List of structured documents with metadata
        """
        all_documents = []
        
        for folder in folders:
            self.logger.info(f"Loading from folder: {folder}")
            
            try:
                # List all objects in folder
                objects = self.minio_client.list_objects(
                    self.bucket_name,
                    prefix=folder,
                    recursive=True
                )
                
                folder_docs = []
                for obj in objects:
                    if obj.object_name.lower().endswith('.json'):
                        doc = self._load_json_document(obj.object_name)
                        if doc:
                            folder_docs.append(doc)
                
                self.logger.info(f"  ✓ Loaded {len(folder_docs)} documents from {folder}")
                all_documents.extend(folder_docs)
                
            except Exception as e:
                self.logger.error(f"Error loading folder {folder}: {e}")
        
        self.logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _load_json_document(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Load and parse a single JSON document from MinIO."""
        try:
            response = self.minio_client.get_object(self.bucket_name, object_name)
            content = response.read()
            data = json.loads(content.decode('utf-8'))
            
            # Add metadata
            data['minio_object_name'] = object_name
            data['minio_bucket'] = self.bucket_name
            
            # Normalize text
            if 'content' in data:
                if 'paragraphs' in data['content']:
                    data['content']['paragraphs'] = [
                        unicodedata.normalize('NFC', p) 
                        for p in data['content']['paragraphs']
                    ]
            
            response.close()
            response.release_conn()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading {object_name}: {e}")
            return None
    
    # =========================================================
    # SMART CHUNKING WITH TABLE-PARAGRAPH INTEGRATION
    # =========================================================
    
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 2000,
        overlap_size: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks from structured documents.
        
        Strategy:
        1. Keep related paragraphs together
        2. Keep tables with their context
        3. Create mixed chunks when beneficial
        
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self._create_document_chunks(doc, chunk_size, overlap_size)
            all_chunks.extend(doc_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _create_document_chunks(
        self,
        doc: Dict[str, Any],
        chunk_size: int,
        overlap_size: int
    ) -> List[Dict[str, Any]]:
        """Create chunks from a single document."""
        chunks = []
        
        content = doc.get('content', {})
        paragraphs = content.get('paragraphs', [])
        tables = content.get('tables', [])
        
        # Metadata for all chunks
        base_metadata = {
            'source_file': doc.get('source_file', 'unknown'),
            'document_type': doc.get('document_type', 'unknown'),
            'minio_path': doc.get('minio_object_name', 'unknown')
        }
        
        # Strategy 1: Process paragraphs into text chunks
        if paragraphs:
            para_chunks = self._chunk_paragraphs(
                paragraphs, 
                chunk_size, 
                overlap_size,
                base_metadata
            )
            chunks.extend(para_chunks)
        
        # Strategy 2: Process tables separately with context
        if tables:
            table_chunks = self._create_table_chunks(
                tables,
                paragraphs,
                base_metadata
            )
            chunks.extend(table_chunks)
        
        return chunks
    
    def _chunk_paragraphs(
        self,
        paragraphs: List[str],
        chunk_size: int,
        overlap_size: int,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create overlapping chunks from paragraphs."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk_size
            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_type': 'text',
                    'chunk_index': len(chunks),
                    'paragraph_range': [len(chunks) * 10, i],  # Approximate
                    **metadata
                })
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-overlap_size:] if len(chunk_text) > overlap_size else chunk_text
                current_chunk = [overlap_text, para]
                current_size = len(overlap_text) + para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_type': 'text',
                'chunk_index': len(chunks),
                'paragraph_range': [len(chunks) * 10, len(paragraphs)],
                **metadata
            })
        
        return chunks
    
    def _create_table_chunks(
        self,
        tables: List[Dict[str, Any]],
        paragraphs: List[str],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks for tables with surrounding context.
        
        Strategy:
        - Simple tables: Convert to text representation
        - Complex tables: Include structure info
        - Add context from nearby paragraphs
        """
        chunks = []
        
        for table in tables:
            table_index = table.get('table_index', 0)
            table_type = table.get('table_type', 'simple')
            
            # Find context paragraphs (paragraphs mentioning this table)
            context = self._find_table_context(table_index, paragraphs)
            
            # Convert table to text
            table_text = self._table_to_text(table)
            
            # Combine context + table
            if context:
                chunk_text = f"{context}\n\n{table_text}"
            else:
                chunk_text = table_text
            
            chunks.append({
                'text': chunk_text,
                'chunk_type': 'table',
                'table_index': table_index,
                'table_type': table_type,
                'chunk_index': len(chunks),
                **metadata
            })
        
        return chunks
    
    def _find_table_context(
        self,
        table_index: int,
        paragraphs: List[str],
        context_window: int = 3
    ) -> str:
        """Find paragraphs that provide context for a table."""
        # Look for paragraphs mentioning "Bảng X" or table-related keywords
        table_keywords = [
            f"bảng {table_index + 1}",
            f"table {table_index + 1}",
            "bảng",
            "table"
        ]
        
        relevant_paras = []
        for para in paragraphs:
            para_lower = para.lower()
            if any(kw in para_lower for kw in table_keywords):
                relevant_paras.append(para)
                if len(relevant_paras) >= context_window:
                    break
        
        return '\n'.join(relevant_paras[:context_window])
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table structure to readable text."""
        table_type = table.get('table_type', 'simple')
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        lines = [f"[TABLE {table.get('table_index', 0) + 1}]"]
        
        if table_type == 'simple':
            # Simple table with consistent structure
            if headers:
                lines.append("Headers: " + " | ".join(headers))
            
            for i, row in enumerate(rows, 1):
                if isinstance(row, dict):
                    row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    lines.append(f"Row {i}: {row_text}")
        
        else:  # complex table
            # More detailed representation for complex tables
            lines.append(f"Complex table with {len(rows)} rows")
            for i, row in enumerate(rows[:10], 1):  # Limit to 10 rows
                if isinstance(row, dict) and 'cells' in row:
                    cells = row['cells']
                    row_text = " | ".join(f"{k}: {v}" for k, v in cells.items() if v and str(v).strip())
                    if row_text:
                        lines.append(f"Row {i}: {row_text}")
        
        return "\n".join(lines)
    
    # =========================================================
    # ENTITY & RELATION EXTRACTION
    # =========================================================
    
    def extract_elements(
        self,
        chunks: List[Dict[str, Any]],
        use_parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities and relations from chunks using LLM.
        
        Args:
            chunks: List of document chunks
            use_parallel: Use parallel processing
            
        Returns:
            List of extracted elements (entities + relations)
        """
        if use_parallel:
            return self._extract_elements_parallel(chunks)
        else:
            return self._extract_elements_sequential(chunks)
    
    def _extract_elements_parallel(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract elements using parallel processing."""
        elements = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        elements.append(result)
                    
                    completed += 1
                    if completed % 10 == 0:
                        self.logger.info(f"  Progress: {completed}/{len(chunks)} chunks")
                        
                except Exception as e:
                    chunk = futures[future]
                    self.logger.error(f"Error extracting from chunk: {e}")
        
        return elements
    
    def _extract_elements_sequential(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract elements sequentially (for debugging)."""
        elements = []
        
        for i, chunk in enumerate(chunks, 1):
            try:
                result = self._extract_from_chunk(chunk)
                if result:
                    elements.append(result)
                
                if i % 10 == 0:
                    self.logger.info(f"  Progress: {i}/{len(chunks)} chunks")
                    
            except Exception as e:
                self.logger.error(f"Error extracting from chunk {i}: {e}")
        
        return elements
    
    def _extract_from_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract entities and relations from a single chunk using LLM."""
        
        chunk_text = chunk.get('text', '')
        chunk_type = chunk.get('chunk_type', 'text')
        source_file = chunk.get('source_file', 'unknown')
        
        # Build context-aware prompt
        prompt = self._build_extraction_prompt(chunk_text, chunk_type, source_file)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from Vietnamese educational documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            result = self._parse_extraction_result(result_text)
            
            if result:
                result['chunk_metadata'] = {
                    'chunk_type': chunk_type,
                    'source_file': source_file,
                    'chunk_index': chunk.get('chunk_index', 0)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM extraction error: {e}")
            return None
    
    def _build_extraction_prompt(
        self,
        text: str,
        chunk_type: str,
        source_file: str
    ) -> str:
        """Build enhanced extraction prompt based on chunk type and document type."""
        
        # Detect if this is career description
        is_career_desc = 'career' in source_file.lower() or 'nghề' in text.lower()
        
        base_prompt = f"""Trích xuất thông tin có cấu trúc từ văn bản sau đây.

NGUỒN: {source_file}
LOẠI CHUNK: {chunk_type}

VĂN BẢN:
{text}

YÊU CẦU:
Trích xuất các ENTITIES (thực thể) và RELATIONS (quan hệ) theo định dạng JSON.

ENTITIES có các loại sau:
"""
        
        if is_career_desc:
            # Career description specific entities
            base_prompt += """
⭐ ƯU TIÊN CHO CAREER DESCRIPTION:
- nghề_nghiệp: Tên nghề, vị trí công việc (VD: "Lập trình game", "Kỹ sư phần mềm", "Data Scientist")
  Properties: mô_tả, nhóm_nghề, vai_trò, mức_lương_junior, mức_lương_senior
  
- kỹ_năng: Kỹ năng cần thiết cho nghề
  Properties: loại (chuyên_môn/mềm/ngoại_ngữ), mức_độ_yêu_cầu, mô_tả
  
- ngành_học: Ngành học phù hợp (VD: "Công nghệ thông tin", "Công nghệ đa phương tiện")
  Properties: mô_tả, thời_gian_đào_tạo
  
- công_việc_chính: Công việc, nhiệm vụ của nghề
  Properties: mô_tả
  
- chứng_chỉ: Chứng chỉ liên quan
  Properties: tên, yêu_cầu

QUAN TRỌNG - Trích xuất từ Career Description:
- Tên nghề: Tìm "Tên nghề", "Career", "NGHỀ NGHIỆP"
- Nhóm nghề: Tìm "Nhóm nghề", "lĩnh vực"
- Kỹ năng: Tìm trong phần "Kỹ năng yêu cầu", "hard skills", "soft skills"
  + Kỹ năng chuyên môn: C++, C#, Java, Python, v.v.
  + Kỹ năng mềm: Giải quyết vấn đề, sáng tạo, v.v.
- Ngành học: Tìm "Ngành học phù hợp", "ngành nghề liên quan"
- Mức lương: Tìm "Mức lương", "Junior", "Senior"

VÍ DỤ:
Từ: "Lập trình game (Game Developer)...các ngành nghề liên quan đến công nghệ thông tin như Ngành công nghệ đa phương tiện"

Trích xuất:
- Entity: nghề_nghiệp "Lập trình game" với properties {mô_tả: "Game Developer..."}
- Entity: kỹ_năng "C++" với properties {loại: "chuyên_môn"}
- Entity: kỹ_năng "C#" với properties {loại: "chuyên_môn"}
- Entity: ngành_học "Công nghệ đa phương tiện"
- Relation: YÊU_CẦU_KỸ_NĂNG từ "Lập trình game" -> "C++"
- Relation: ĐÀO_TẠO_CHO_NGHỀ từ "Công nghệ đa phương tiện" -> "Lập trình game"
"""
        else:
            # Syllabus/curriculum entities
            base_prompt += """
- học_phần: Các môn học, học phần (VD: "Phân tích và thiết kế hệ thống", "Lập trình Java")
  Properties: mã_học_phần, số_tín_chỉ, số_giờ_trên_lớp
  
- giảng_viên: Giảng viên, giáo viên
  Properties: email, chức_danh, học_vị
  
- khoa: Khoa, viện quản lý
  Properties: tên_đầy_đủ, địa_chỉ
  
- tài_liệu: Sách, giáo trình, tài liệu tham khảo
  Properties: tác_giả, năm_xuất_bản
  
- phần_mềm: Phần mềm, công cụ sử dụng
  
- chương_trình: Chương trình đào tạo, ngành học
  
- mục_tiêu: Mục tiêu học phần (CG)
  
- chuẩn_đầu_ra: Chuẩn đầu ra (CLO)
"""

        base_prompt += """

RELATIONS có thể là:
"""
        
        if is_career_desc:
            base_prompt += """
⭐ ƯU TIÊN CHO CAREER:
- YÊU_CẦU_KỸ_NĂNG: nghề_nghiệp -> kỹ_năng
- ĐÀO_TẠO_CHO_NGHỀ: ngành_học -> nghề_nghiệp  
- PHÁT_TRIỂN_KỸ_NĂNG: ngành_học -> kỹ_năng
- CÓ_CÔNG_VIỆC: nghề_nghiệp -> công_việc_chính
- YÊU_CẦU_CHỨNG_CHỈ: nghề_nghiệp -> chứng_chỉ
"""
        else:
            base_prompt += """
- GIẢNG_DẠY: giảng_viên -> học_phần
- THUỘC_KHOA: học_phần -> khoa
- SỬ_DỤNG: học_phần -> tài_liệu, học_phần -> phần_mềm
- TIÊN_QUYẾT: học_phần -> học_phần
- THUỘC_CHƯƠNG_TRÌNH: học_phần -> chương_trình
"""

        if chunk_type == 'table':
            base_prompt += """

ĐẶC BIỆT - Đây là BẢNG:
- Trích xuất chính xác từng dòng
- Giữ nguyên cấu trúc thông tin
"""
        else:
            base_prompt += """

ĐẶC BIỆT - Đây là PARAGRAPH:
- Trích xuất tất cả entities được đề cập
- Tạo relations logic giữa các entities
"""

        base_prompt += """

⚠️ CRITICAL - FILTER TEST DATA:
NEVER extract test/example data:
- Tên người giả: "Nguyễn Văn A", "Trần Văn B", "Họ Tên" 
- Single letters: "A", "B", "C"
- Keywords: "test", "example", "sample", "demo"
- Placeholder text: "...", "xxx", "abc"

VALIDATION RULES:
✓ Extract: "ThS. Trần Thị Mỹ Diệp" (real name with title)
✗ Skip: "Nguyễn Văn A" (test pattern: Văn + single letter)
✓ Extract: "Lập trình Java" (real course)
✗ Skip: "Môn học A" (placeholder)

If unsure whether data is real or test → SKIP IT.
Only extract entities you are confident are REAL.

OUTPUT FORMAT (JSON):
{
  "entities": [
    {
      "name": "tên entity chính xác",
      "type": "loại entity",
      "properties": {
        "key": "value"
      }
    }
  ],
  "relations": [
    {
      "source": "tên entity nguồn",
      "target": "tên entity đích",
      "type": "loại quan hệ",
      "properties": {
        "description": "mô tả ngắn"
      }
    }
  ]
}

CHỈ trả về JSON, KHÔNG thêm giải thích hoặc markdown.
"""
        
        return base_prompt
    
    def _parse_extraction_result(self, result_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract JSON."""
        try:
            # Remove markdown code blocks if present
            result_text = result_text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result_text = result_text.strip()
            
            # Parse JSON
            result = json.loads(result_text)
            
            # Validate structure
            if 'entities' not in result or 'relations' not in result:
                self.logger.warning("Invalid extraction result: missing entities or relations")
                return None
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extraction result: {e}")
            self.logger.debug(f"Result text: {result_text[:500]}")
            return None
    
    # =========================================================
    # CACHING SUPPORT
    # =========================================================
    
    def load_or_process(
        self,
        cache_file: str,
        process_func,
        *args,
        **kwargs
    ):
        """Load from cache or process and save."""
        
        if os.path.exists(cache_file):
            self.logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        self.logger.info(f"Processing (no cache found)")
        result = process_func(*args, **kwargs)
        
        # Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        self.logger.info(f"Saved to cache: {cache_file}")
        
        return result
# minio_loader_v2.py
"""
IMPROVED VERSION: Better semantic context preservation for paragraphs + tables
Key improvements:
1. Smart table-paragraph linking with section headers
2. Metadata-enriched chunks
3. Enhanced table formatting with full context
4. Better LLM extraction prompts
"""

from minio import Minio
from minio.error import S3Error
from logger import Logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import json
import re
from typing import List, Dict, Tuple, Optional


class MinioLoaderV2:
    """
    Enhanced MinIO loader with improved semantic preservation.
    """

    def __init__(
        self,
        endpoint,
        access_key,
        secret_key,
        bucket_name,
        client,
        model="gpt-4o-mini",
        max_workers=10,
        secure=False
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        self.bucket_name = bucket_name
        self.openai_client = client
        self.model = model
        self.max_workers = max_workers
        self.logger = Logger("MinioLoaderV2").get_logger()

        self.logger.info(f"Initialized MinIO client for bucket: {bucket_name}")

    # =========================================================
    # FILE HANDLING
    # =========================================================

    def list_files(self, prefix):
        """List all .json files in a folder."""
        try:
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=True
            )

            files = []
            for obj in objects:
                if (
                    obj.object_name.lower().endswith(".json")
                    and not obj.object_name.split("/")[-1].startswith("~")
                ):
                    files.append(obj.object_name)

            self.logger.info(f"Found {len(files)} .json files in {prefix}")
            return files

        except S3Error as e:
            self.logger.error(f"Error listing files from {prefix}: {e}")
            return []

    def download_file(self, object_name):
        """Download a file from MinIO."""
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            content = response.read()
            response.close()
            response.release_conn()

            self.logger.debug(f"Downloaded: {object_name}")
            return content

        except S3Error as e:
            self.logger.error(f"Error downloading {object_name}: {e}")
            return None

    # =========================================================
    # SECTION DETECTION & STRUCTURE ANALYSIS
    # =========================================================

    def detect_section_header(self, text: str) -> Optional[Tuple[int, str]]:
        """
        Detect if a paragraph is a section header.
        Returns: (section_number, section_title) or None
        
        Examples:
        "1. THÔNG TIN TỔNG QUÁT" -> (1, "THÔNG TIN TỔNG QUÁT")
        "5. MỤC TIÊU HỌC PHẦN" -> (5, "MỤC TIÊU HỌC PHẦN")
        """
        pattern = r'^(\d+)\.\s+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]+)'
        match = re.match(pattern, text.strip())
        
        if match:
            section_num = int(match.group(1))
            section_title = match.group(2).strip()
            return (section_num, section_title)
        
        return None

    def detect_table_reference(self, text: str) -> Optional[int]:
        """
        Detect table reference in paragraph.
        
        Examples:
        "Bảng 1. Mục tiêu học phần" -> 1
        "Xem Bảng 2" -> 2
        """
        patterns = [
            r'Bảng\s+(\d+)',
            r'bảng\s+(\d+)',
            r'Table\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        return None

    # =========================================================
    # ENHANCED JSON → STRUCTURED DOCUMENT
    # =========================================================

    def parse_json_to_structured_doc(self, json_content: bytes) -> Dict:
        """
        Parse JSON to structured document with semantic sections.
        
        Returns:
        {
            'source_file': str,
            'document_type': str,
            'sections': [
                {
                    'section_number': int,
                    'section_title': str,
                    'content': [
                        {'type': 'paragraph', 'text': str},
                        {'type': 'table', 'data': dict, 'context': str}
                    ]
                }
            ]
        }
        """
        try:
            data = json.loads(json_content.decode("utf-8"))
            content = data.get("content", {})
            
            paragraphs = content.get("paragraphs", [])
            tables = content.get("tables", [])
            
            # Build structured document
            structured_doc = {
                'source_file': data.get('source_file', 'unknown'),
                'document_type': data.get('document_type', 'unknown'),
                'sections': []
            }
            
            current_section = None
            table_pointer = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if this is a section header
                section_info = self.detect_section_header(para)
                
                if section_info:
                    # Start new section
                    section_num, section_title = section_info
                    current_section = {
                        'section_number': section_num,
                        'section_title': section_title,
                        'content': []
                    }
                    structured_doc['sections'].append(current_section)
                    
                    # Add header as first paragraph
                    current_section['content'].append({
                        'type': 'paragraph',
                        'text': para
                    })
                else:
                    # Regular paragraph
                    if current_section is None:
                        # Create default section for content before first header
                        current_section = {
                            'section_number': 0,
                            'section_title': 'PREAMBLE',
                            'content': []
                        }
                        structured_doc['sections'].append(current_section)
                    
                    current_section['content'].append({
                        'type': 'paragraph',
                        'text': para
                    })
                    
                    # Check for table reference
                    table_ref = self.detect_table_reference(para)
                    
                    if table_ref and table_pointer < len(tables):
                        # Find matching table
                        matching_table = None
                        for idx, table in enumerate(tables[table_pointer:], start=table_pointer):
                            if table.get('table_index') == table_ref:
                                matching_table = table
                                table_pointer = idx + 1
                                break
                        
                        if matching_table:
                            current_section['content'].append({
                                'type': 'table',
                                'data': matching_table,
                                'context': f"Section {current_section['section_number']}: {current_section['section_title']}"
                            })
            
            # Add any remaining tables
            while table_pointer < len(tables):
                if current_section:
                    current_section['content'].append({
                        'type': 'table',
                        'data': tables[table_pointer],
                        'context': f"Section {current_section['section_number']}: {current_section['section_title']}"
                    })
                table_pointer += 1
            
            return structured_doc
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON to structured doc: {e}")
            return None

    # =========================================================
    # ENHANCED TABLE FORMATTING
    # =========================================================

    def format_table_with_context(self, table_data: Dict, context: str) -> str:
        """
        Format table with full semantic context.
        
        Improvements:
        - Clear table boundaries
        - Section context
        - Table index/title
        - Structured header-row format
        - Better cell labeling
        """
        lines = []
        
        # Header
        lines.append("\n" + "="*80)
        lines.append(f"TABLE {table_data.get('table_index', '?')}")
        lines.append(f"Context: {context}")
        lines.append("="*80)
        
        # Headers
        headers = table_data.get('headers', [])
        if headers:
            lines.append("\nCOLUMNS: " + " | ".join(headers))
            lines.append("-"*80)
        
        # Rows
        rows = table_data.get('rows', [])
        for row in rows:
            row_label = row.get('row_label', '')
            cells = row.get('cells', {})
            
            if row_label:
                lines.append(f"\n[ROW: {row_label}]")
            
            for col_name, cell_value in cells.items():
                lines.append(f"  • {col_name}: {cell_value}")
        
        lines.append("\n" + "="*80 + "\n")
        
        return "\n".join(lines)

    # =========================================================
    # SMART CHUNKING WITH METADATA
    # =========================================================

    def create_chunks_from_structured_doc(
        self,
        structured_doc: Dict,
        chunk_size: int = 2000,
        overlap_size: int = 300
    ) -> List[Dict]:
        """
        Create chunks from structured document with metadata.
        
        Each chunk includes:
        - text: the actual content
        - metadata: section info, document info
        - chunk_type: 'text', 'table', or 'mixed'
        """
        chunks = []
        
        source_file = structured_doc.get('source_file', 'unknown')
        doc_type = structured_doc.get('document_type', 'unknown')
        
        for section in structured_doc.get('sections', []):
            section_num = section.get('section_number', 0)
            section_title = section.get('section_title', 'UNKNOWN')
            
            current_text = f"# Section {section_num}: {section_title}\n\n"
            current_type = 'text'
            
            for item in section.get('content', []):
                item_type = item.get('type')
                
                if item_type == 'paragraph':
                    text = item.get('text', '')
                    
                    # Check if adding this would exceed chunk size
                    if len(current_text) + len(text) > chunk_size and len(current_text) > 0:
                        # Save current chunk
                        chunks.append({
                            'text': current_text,
                            'metadata': {
                                'source_file': source_file,
                                'document_type': doc_type,
                                'section_number': section_num,
                                'section_title': section_title
                            },
                            'chunk_type': current_type
                        })
                        
                        # Start new chunk with overlap
                        overlap_text = current_text[-overlap_size:] if len(current_text) > overlap_size else current_text
                        current_text = f"# Section {section_num}: {section_title}\n\n{overlap_text}"
                    
                    current_text += text + "\n\n"
                
                elif item_type == 'table':
                    table_text = self.format_table_with_context(
                        item.get('data', {}),
                        item.get('context', section_title)
                    )
                    
                    # Tables are important - give them their own chunk if needed
                    if len(current_text) + len(table_text) > chunk_size:
                        # Save current text chunk
                        if len(current_text.strip()) > 0:
                            chunks.append({
                                'text': current_text,
                                'metadata': {
                                    'source_file': source_file,
                                    'document_type': doc_type,
                                    'section_number': section_num,
                                    'section_title': section_title
                                },
                                'chunk_type': current_type
                            })
                        
                        # Create table chunk
                        chunks.append({
                            'text': f"# Section {section_num}: {section_title}\n\n{table_text}",
                            'metadata': {
                                'source_file': source_file,
                                'document_type': doc_type,
                                'section_number': section_num,
                                'section_title': section_title,
                                'table_index': item['data'].get('table_index')
                            },
                            'chunk_type': 'table'
                        })
                        
                        current_text = f"# Section {section_num}: {section_title}\n\n"
                        current_type = 'text'
                    else:
                        current_text += table_text
                        current_type = 'mixed'
            
            # Save remaining content
            if len(current_text.strip()) > len(f"# Section {section_num}: {section_title}"):
                chunks.append({
                    'text': current_text,
                    'metadata': {
                        'source_file': source_file,
                        'document_type': doc_type,
                        'section_number': section_num,
                        'section_title': section_title
                    },
                    'chunk_type': current_type
                })
        
        self.logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    # =========================================================
    # LOAD DOCUMENTS (UPDATED)
    # =========================================================

    def load_documents_from_folders(self, folders: List[str]) -> List[Dict]:
        """Load all .json documents from multiple folders as structured docs."""
        all_structured_docs = []

        for folder in folders:
            self.logger.info(f"Loading documents from folder: {folder}")
            files = self.list_files(folder)

            for file_path in files:
                content = self.download_file(file_path)

                if content:
                    structured_doc = self.parse_json_to_structured_doc(content)
                    if structured_doc:
                        all_structured_docs.append(structured_doc)
                        self.logger.debug(f"Loaded structured doc: {file_path}")

        self.logger.info(f"Total structured documents loaded: {len(all_structured_docs)}")
        return all_structured_docs

    # =========================================================
    # CHUNKING (UPDATED)
    # =========================================================

    def split_documents(
        self,
        structured_docs: List[Dict],
        chunk_size: int = 2000,
        overlap_size: int = 300
    ) -> List[Dict]:
        """Split structured documents into chunks with metadata."""
        all_chunks = []

        for doc in structured_docs:
            chunks = self.create_chunks_from_structured_doc(
                doc,
                chunk_size=chunk_size,
                overlap_size=overlap_size
            )
            all_chunks.extend(chunks)

        self.logger.info(f"Split {len(structured_docs)} documents into {len(all_chunks)} chunks")
        return all_chunks

    # =========================================================
    # ENHANCED LLM EXTRACTION
    # =========================================================

    def extract_elements(self, chunks: List[Dict], use_parallel: bool = True) -> List[str]:
        """
        Extract entities and relations using LLM with enhanced prompts.
        Now uses chunk metadata for better context.
        """
        self.logger.info(f"Extracting elements from {len(chunks)} chunks...")

        system_prompt = """Bạn là hệ thống trích xuất thông tin từ tài liệu học thuật tiếng Việt.

NHIỆM VỤ: Trích xuất ENTITIES và RELATIONSHIPS từ văn bản.

ĐỊNH DẠNG ĐẦU RA (KHÔNG giải thích thêm):

ENTITY: <tên entity>
TYPE: <loại entity>

RELATION: <entity_1> -> <quan hệ> -> <entity_2>

CÁC LOẠI ENTITY:
- học_phần: tên môn học, mã môn học
- giảng_viên: tên giảng viên, chức danh (TS., ThS., etc.)
- tài_liệu: sách, giáo trình, tài liệu tham khảo
- khoa: khoa/viện quản lý
- chương_trình_đào_tạo: tên chương trình đào tạo
- mục_tiêu: mục tiêu học phần (CG, CLO)
- phần_mềm: công cụ, IDE

CÁC LOẠI QUAN HỆ:
- GIẢNG_DẠY: giảng_viên -> GIẢNG_DẠY -> học_phần
- THUỘC_VỀ: học_phần -> THUỘC_VỀ -> khoa
- SỬ_DỤNG: học_phần -> SỬ_DỤNG -> tài_liệu
- SỬ_DỤNG: học_phần -> SỬ_DỤNG -> phần_mềm
- TIÊN_QUYẾT: học_phần -> TIÊN_QUYẾT -> học_phần
- CÓ_MỤC_TIÊU: học_phần -> CÓ_MỤC_TIÊU -> mục_tiêu
- THUỘC_CTĐT: học_phần -> THUỘC_CTĐT -> chương_trình_đào_tạo

LƯU Ý QUAN TRỌNG:
1. Với BẢNG: Trích xuất TẤT CẢ thông tin quan trọng (giảng viên, email, mục tiêu, CLO, etc.)
2. Chuẩn hóa tên entity (ví dụ: "TS. Phạm Xuân Lâm" -> "Phạm Xuân Lâm")
3. Trích xuất mã học phần chính xác (ví dụ: CNTT1153)
4. Với học phần tiên quyết, trích xuất cả tên và mã
5. Với email, lưu như metadata của entity giảng_viên

VÍ DỤ:

Input: "Giảng viên giảng dạy học phần: TS. Phạm Xuân Lâm (lampx@neu.edu.vn)"

Output:
ENTITY: Phạm Xuân Lâm
TYPE: giảng_viên
METADATA: email=lampx@neu.edu.vn, chức_danh=TS

RELATION: Phạm Xuân Lâm -> GIẢNG_DẠY -> Lập trình Java
"""

        if use_parallel:
            return self._batch_api_call_with_metadata(
                chunks,
                system_prompt,
                500
            )

        results = []
        for chunk in chunks:
            try:
                # Prepare chunk with metadata
                chunk_text = chunk['text']
                metadata = chunk.get('metadata', {})
                
                context = f"[Document: {metadata.get('source_file', 'unknown')}]\n"
                context += f"[Section: {metadata.get('section_title', 'unknown')}]\n"
                context += f"[Type: {chunk.get('chunk_type', 'text')}]\n\n"
                context += chunk_text[:2000]  # Limit to 2000 chars
                
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=500
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                self.logger.warning(f"Chunk extraction failed: {e}")
                results.append("")

        return [r for r in results if r]

    def _batch_api_call_with_metadata(
        self,
        chunks: List[Dict],
        system_prompt: str,
        max_tokens: int
    ) -> List[str]:
        """Parallel API call helper with metadata support."""
        results = [None] * len(chunks)

        def worker(index, chunk):
            try:
                chunk_text = chunk['text']
                metadata = chunk.get('metadata', {})
                
                context = f"[Document: {metadata.get('source_file', 'unknown')}]\n"
                context += f"[Section: {metadata.get('section_title', 'unknown')}]\n"
                context += f"[Type: {chunk.get('chunk_type', 'text')}]\n\n"
                context += chunk_text[:2000]
                
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=max_tokens
                )
                return index, response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"Worker {index} failed: {e}")
                return index, ""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(worker, i, chunk)
                for i, chunk in enumerate(chunks)
            ]

            for future in as_completed(futures):
                idx, res = future.result()
                results[idx] = res

        return [r for r in results if r]

    # =========================================================
    # CACHE
    # =========================================================

    def load_or_process(self, file_path, process_function, *args, **kwargs):
        """Load cached data or process and cache."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(file_path):
            self.logger.info(f"Loading cached data from {file_path}")
            with open(file_path, "rb") as f:
                return pickle.load(f)

        self.logger.info(f"Cache miss - processing {file_path}")
        data = process_function(*args, **kwargs)

        if data is not None:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

        return data
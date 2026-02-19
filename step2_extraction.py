"""
STEP 2: Entity & Relation Extraction
Input: List of text chunks
Output: List of extracted entities and relations
"""

from typing import List
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from config import MAX_WORKERS, USE_MINI_MODEL, EXTRACTION_MAX_TOKENS


class ExtractionOutput:
    """Output của bước extraction"""
    def __init__(self, extractions: List[str], stats: dict):
        self.extractions = extractions
        self.stats = stats
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("STEP 2: EXTRACTION - OUTPUT")
        print("=" * 80)
        print(f"📥 Số chunks đầu vào: {self.stats['num_chunks']}")
        print(f"📤 Số extraction results: {self.stats['num_extractions']}")
        print(f"⚡ Workers song song: {self.stats['max_workers']}")
        print(f"🤖 Model sử dụng: {self.stats['model']}")
        print(f"⏱️  Thời gian xử lý: {self.stats['processing_time']:.2f}s")
        print(f"\n📋 Sample extraction result đầu tiên:")
        print("-" * 80)
        print(self.extractions[0][:500] if self.extractions else "Không có kết quả")
        print("-" * 80)
        print("=" * 80)
    
    def save_to_file(self, output_dir: str = "pipeline_outputs"):
        """Lưu output ra file txt"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "step2_extraction_output.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 2: EXTRACTION - DETAILED OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            
            # Stats
            f.write("📊 THỐNG KÊ:\n")
            f.write(f"   - Số chunks đầu vào: {self.stats['num_chunks']}\n")
            f.write(f"   - Số extraction results: {self.stats['num_extractions']}\n")
            f.write(f"   - Workers song song: {self.stats['max_workers']}\n")
            f.write(f"   - Model sử dụng: {self.stats['model']}\n")
            f.write(f"   - Thời gian xử lý: {self.stats['processing_time']:.2f}s\n\n")
            
            # All extractions
            f.write("=" * 80 + "\n")
            f.write("📝 TẤT CẢ ENTITIES & RELATIONS:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, extraction in enumerate(self.extractions):
                f.write(f"--- EXTRACTION {i+1}/{len(self.extractions)} ---\n")
                f.write(extraction)
                f.write("\n\n" + "-" * 80 + "\n\n")
        
        print(f"💾 Đã lưu output vào: {filepath}")
        return filepath


class EntityRelationExtractor:
    """Class để extract entities và relations từ text chunks"""
    
    SYSTEM_PROMPT = """
You are an information extraction system.

Extract ENTITIES and RELATIONSHIPS from the text.

STRICT FORMAT (no explanation, no markdown):

ENTITY: <entity name>
RELATION: <entity_1> -> <relation> -> <entity_2>

Rules:
- Use '->' exactly for relations
- Entity names: max 5 words
- Use Vietnamese if the text is Vietnamese
- Do NOT invent relations not present in text

Example:
ENTITY: Hệ điều hành
ENTITY: Tiến trình
RELATION: Hệ điều hành -> quản lý -> Tiến trình
"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini" if USE_MINI_MODEL else "gpt-4o"
    
    def _process_single_chunk(self, item_data):
        """Xử lý một chunk"""
        index, chunk = item_data
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": chunk[:1500]}
                ],
                max_tokens=EXTRACTION_MAX_TOKENS
            )
            return index, response.choices[0].message.content
        except Exception as e:
            print(f"[WARN] Chunk {index} failed: {e}")
            return index, ""
    
    def extract(self, chunks: List[str]) -> ExtractionOutput:
        """
        Extract entities và relations từ tất cả chunks song song
        
        Args:
            chunks: List of text chunks
            
        Returns:
            ExtractionOutput: Object chứa extraction results và stats
        """
        import time
        start_time = time.time()
        
        results = [None] * len(chunks)
        
        # Batch processing với ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._process_single_chunk, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        # Lọc kết quả không rỗng
        extractions = [r for r in results if r]
        
        processing_time = time.time() - start_time
        
        stats = {
            'num_chunks': len(chunks),
            'num_extractions': len(extractions),
            'max_workers': MAX_WORKERS,
            'model': self.model,
            'processing_time': processing_time
        }
        
        return ExtractionOutput(extractions, stats)


if __name__ == "__main__":
    # Test extraction
    from dotenv import load_dotenv
    load_dotenv()
    
    test_chunks = [
        "Hệ điều hành quản lý tài nguyên hệ thống. CPU thực thi các tiến trình.",
        "Python là ngôn ngữ lập trình. Django là framework của Python."
    ]
    
    extractor = EntityRelationExtractor(api_key=os.getenv("OPENAI_API_KEY"))
    output = extractor.extract(test_chunks)
    output.print_summary()
"""
STEP 1: Document Chunking
Input: List of documents (strings)
Output: List of text chunks with overlap
"""

from typing import List
from config import CHUNK_SIZE, OVERLAP_SIZE


class ChunkingOutput:
    """Output của bước chunking"""
    def __init__(self, chunks: List[str], stats: dict):
        self.chunks = chunks
        self.stats = stats
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("STEP 1: CHUNKING - OUTPUT")
        print("=" * 80)
        print(f"📄 Số lượng documents gốc: {self.stats['num_documents']}")
        print(f"✂️  Số lượng chunks tạo ra: {self.stats['num_chunks']}")
        print(f"📏 Kích thước chunk: {self.stats['chunk_size']} ký tự")
        print(f"🔄 Overlap: {self.stats['overlap_size']} ký tự")
        print(f"📊 Độ dài trung bình mỗi chunk: {self.stats['avg_chunk_length']:.0f} ký tự")
        print(f"📋 Sample chunk đầu tiên (100 ký tự):")
        print(f"   {self.chunks[0][:100]}...")
        print("=" * 80)
    
    def save_to_file(self, output_dir: str = "pipeline_outputs"):
        """Lưu output ra file txt"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "step1_chunking_output.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 1: CHUNKING - DETAILED OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            
            # Stats
            f.write("📊 THỐNG KÊ:\n")
            f.write(f"   - Số documents gốc: {self.stats['num_documents']}\n")
            f.write(f"   - Số chunks tạo ra: {self.stats['num_chunks']}\n")
            f.write(f"   - Kích thước chunk: {self.stats['chunk_size']} ký tự\n")
            f.write(f"   - Overlap: {self.stats['overlap_size']} ký tự\n")
            f.write(f"   - Độ dài TB mỗi chunk: {self.stats['avg_chunk_length']:.0f} ký tự\n\n")
            
            # All chunks
            f.write("=" * 80 + "\n")
            f.write("📝 TẤT CẢ CHUNKS:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(self.chunks):
                f.write(f"--- CHUNK {i+1}/{len(self.chunks)} (độ dài: {len(chunk)} ký tự) ---\n")
                f.write(chunk)
                f.write("\n\n" + "-" * 80 + "\n\n")
        
        print(f"💾 Đã lưu output vào: {filepath}")
        return filepath


def chunk_documents(documents: List[str]) -> ChunkingOutput:
    """
    Chia nhỏ documents thành các chunks có overlap
    
    Args:
        documents: Danh sách các document (string)
        
    Returns:
        ChunkingOutput: Object chứa chunks và thống kê
    """
    chunks = []
    
    for doc in documents:
        # Chia document thành chunks với overlap
        for i in range(0, len(doc), CHUNK_SIZE - OVERLAP_SIZE):
            chunk = doc[i:i + CHUNK_SIZE]
            if chunk.strip():  # Chỉ thêm chunk không rỗng
                chunks.append(chunk)
    
    # Tính toán thống kê
    stats = {
        'num_documents': len(documents),
        'num_chunks': len(chunks),
        'chunk_size': CHUNK_SIZE,
        'overlap_size': OVERLAP_SIZE,
        'avg_chunk_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    }
    
    return ChunkingOutput(chunks, stats)


if __name__ == "__main__":
    # Test chunking
    test_docs = [
        "Đây là một văn bản test " * 200,  # ~4000 ký tự
        "Văn bản thứ hai " * 150
    ]
    
    output = chunk_documents(test_docs)
    output.print_summary()
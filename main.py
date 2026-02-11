"""
GraphRAG Pipeline - Main Orchestrator
Kết hợp tất cả các bước để tạo thành pipeline hoàn chỉnh
"""

import os
import time
from typing import List
from dotenv import load_dotenv

# Import các bước
from step1_chunking import chunk_documents
from step2_extraction import EntityRelationExtractor
from step3_graph_building import GraphBuilder
from step4_community_detection import CommunityDetector
from step5_answer_generation import AnswerGenerator
from docx_reader import read_docx_from_directory


class GraphRAGPipeline:
    """
    Pipeline hoàn chỉnh cho GraphRAG
    
    Flow:
    Documents → Chunks → Extractions → Graph → Communities → Answer
    """
    
    def __init__(self, api_key: str):
        """
        Khởi tạo pipeline
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.extractor = EntityRelationExtractor(api_key)
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector()
        self.answer_generator = AnswerGenerator(api_key)
    
    def run(self, documents: List[str], query: str) -> str:
        """
        Chạy toàn bộ pipeline
        
        Args:
            documents: List of document texts
            query: User's question
            
        Returns:
            str: Final answer
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("🚀 GRAPHRAG PIPELINE - BẮT ĐẦU")
        print("=" * 80)
        print(f"📚 Số documents: {len(documents)}")
        print(f"📏 Tổng số ký tự: {sum(len(d) for d in documents):,}")
        print(f"❓ Query: {query}")
        print("=" * 80)
        
        # ==================== STEP 1: CHUNKING ====================
        print("\n[STEP 1/5] Chunking documents...")
        step1_output = chunk_documents(documents)
        step1_output.print_summary()
        step1_output.save_to_file()
        
        # ==================== STEP 2: EXTRACTION ====================
        print("\n[STEP 2/5] Extracting entities & relations...")
        step2_output = self.extractor.extract(step1_output.chunks)
        step2_output.print_summary()
        step2_output.save_to_file()
        
        # ==================== STEP 3: GRAPH BUILDING ====================
        print("\n[STEP 3/5] Building knowledge graph...")
        step3_output = self.graph_builder.build(step2_output.extractions)
        step3_output.print_summary()
        step3_output.save_to_file()
        
        # ==================== STEP 4: COMMUNITY DETECTION ====================
        print("\n[STEP 4/5] Detecting communities...")
        step4_output = self.community_detector.detect(step3_output.graph)
        step4_output.print_summary()
        step4_output.save_to_file()
        
        # ==================== STEP 5: ANSWER GENERATION ====================
        print("\n[STEP 5/5] Generating answer...")
        step5_output = self.answer_generator.generate(
            step4_output.communities,
            query
        )
        step5_output.print_summary()
        step5_output.save_to_file()
        
        # ==================== SUMMARY ====================
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("✅ PIPELINE HOÀN THÀNH")
        print("=" * 80)
        print(f"⏱️  Tổng thời gian: {elapsed:.2f}s")
        print(f"📊 Pipeline stats:")
        print(f"   - Documents → Chunks: {len(documents)} → {step1_output.stats['num_chunks']}")
        print(f"   - Chunks → Extractions: {step1_output.stats['num_chunks']} → {step2_output.stats['num_extractions']}")
        print(f"   - Extractions → Graph: {step2_output.stats['num_extractions']} → {step3_output.stats['num_nodes']} nodes, {step3_output.stats['num_edges']} edges")
        print(f"   - Graph → Communities: {step3_output.stats['num_nodes']} nodes → {step4_output.stats['large_communities']} communities")
        print(f"   - Communities → Answer: {step4_output.stats['large_communities']} communities → 1 answer")
        print("=" * 80)
        print(f"\n📁 Tất cả output files đã được lưu trong thư mục: pipeline_outputs/")
        
        return step5_output.answer


def main():
    """Main entry point"""
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Lỗi: Không tìm thấy OPENAI_API_KEY trong file .env")
        return
    
    # Load documents
    print("\n📂 Đang load documents từ thư mục 'example_docx'...")
    documents = read_docx_from_directory("example_docx")
    
    if not documents:
        print("❌ Không tìm thấy file .docx trong thư mục 'example_docx'")
        return
    
    print(f"✅ Đã load {len(documents)} documents")
    
    # Get query from user
    query = input("\n❓ Nhập câu hỏi của bạn: ").strip()
    if not query:
        query = "Tổng hợp nội dung chính của các tài liệu"
        print(f"   (Sử dụng query mặc định: {query})")
    
    # Run pipeline
    pipeline = GraphRAGPipeline(api_key)
    answer = pipeline.run(documents, query)
    
    print("\n🎉 Hoàn thành!")


if __name__ == "__main__":
    main()
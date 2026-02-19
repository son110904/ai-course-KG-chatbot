# build_graph_complete.py
"""
COMPLETE KNOWLEDGE GRAPH BUILDER
Unified system for both GraphRAG Query and Career Advisor
Loads all data from MinIO: curriculum, syllabus, career descriptions
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from minio_loader_v3 import MinioLoaderV3
from graph_database import GraphDatabaseConnection
from graph_manager_v3 import GraphManagerV3
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()

logger = Logger("BuildCompleteGraph").get_logger()

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "203.113.132.48:8008")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "course2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "course2-s3-uiauia")
MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "syllabus")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# ALL folders - Complete system
MINIO_FOLDERS = [
    "courses-processed/curriculum/",         # Chương trình đào tạo (for both)
    "courses-processed/syllabus/",          # Đề cương học phần (for GraphRAG)
    "courses-processed/career_description/" # Mô tả nghề nghiệp (for Career Advisor)
]

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

# Neo4j Configuration
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_URL or not DB_PASSWORD:
    raise ValueError("Neo4j credentials must be set in .env file")

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "300"))

# =========================================================
# PROCESSING PIPELINE
# =========================================================

def build_complete_knowledge_graph(cache_prefix="complete_graph"):
    """
    Build complete knowledge graph for:
    1. GraphRAG - Query answering system
    2. Career Advisor - Career guidance system
    
    Loads all data from MinIO and creates unified graph.
    """
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("COMPLETE KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 80)
    logger.info("Building unified graph for:")
    logger.info("  ✓ GraphRAG Query System")
    logger.info("  ✓ Career Advisor Chatbot")
    logger.info("=" * 80)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Check Neo4j database status
    db_connection = GraphDatabaseConnection(
        uri=DB_URL,
        user=DB_USERNAME,
        password=DB_PASSWORD
    )
    
    stats = db_connection.get_database_stats()
    has_data = stats['nodes'] > 0 or stats['relationships'] > 0
    
    if has_data:
        print(f"\n⚠️  WARNING: Database already contains data!")
        print(f"   Nodes: {stats['nodes']}")
        print(f"   Relationships: {stats['relationships']}")
        print(f"\n   Options:")
        print(f"   1. Clear and rebuild (all existing data will be lost)")
        print(f"   2. Add to existing data (may create duplicates)")
        print(f"   3. Skip and use existing data")
        print(f"   4. Cancel")
        
        choice = input("\n   Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            logger.info("User chose to clear and rebuild database")
            auto_clear = True
        elif choice == "2":
            logger.info("User chose to add to existing data")
            auto_clear = False
        elif choice == "3":
            logger.info("User chose to skip and use existing data")
            db_connection.close()
            return {
                'documents': 0,
                'chunks': 0,
                'elements': 0,
                'graph': stats,
                'time': 0,
                'skipped': True
            }
        else:
            logger.info("User cancelled operation")
            db_connection.close()
            return None
    else:
        auto_clear = False
        logger.info("Database is empty, proceeding with data loading...")
    
    # Initialize MinIO loader V3
    minio_loader = MinioLoaderV3(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        client=client,
        model=MODEL,
        max_workers=MAX_WORKERS,
        secure=MINIO_SECURE
    )
    
    # Initialize graph manager V3
    graph_manager = GraphManagerV3(
        db_connection=db_connection,
        auto_clear=auto_clear,
        openai_client=client
    )
    
    # Step 1: Load documents from MinIO
    logger.info(f"[1/5] Loading ALL documents from MinIO...")
    logger.info(f"  Endpoint: {MINIO_ENDPOINT}")
    logger.info(f"  Bucket: {MINIO_BUCKET}")
    logger.info(f"  Folders:")
    for folder in MINIO_FOLDERS:
        logger.info(f"    - {folder}")
    
    documents = minio_loader.load_documents_from_folders(MINIO_FOLDERS)
    
    if not documents:
        logger.error("No documents loaded from MinIO")
        db_connection.close()
        return None
    
    logger.info(f"  ✓ Loaded {len(documents)} documents")
    
    # Analyze document types
    doc_types = {}
    for doc in documents:
        doc_type = doc.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    logger.info(f"  Document types breakdown:")
    for doc_type, count in doc_types.items():
        logger.info(f"    - {doc_type}: {count}")
    
    # Validate data completeness
    has_syllabus = any('syllabus' in dt.lower() for dt in doc_types.keys())
    has_curriculum = any('curriculum' in dt.lower() for dt in doc_types.keys())
    has_career = any('career' in dt.lower() for dt in doc_types.keys())
    
    logger.info(f"\n  Data completeness check:")
    logger.info(f"    {'✓' if has_syllabus else '✗'} Syllabus data (for GraphRAG queries)")
    logger.info(f"    {'✓' if has_curriculum else '✗'} Curriculum data (for both systems)")
    logger.info(f"    {'✓' if has_career else '✗'} Career descriptions (for Career Advisor)")
    
    if not has_career:
        logger.warning("\n  ⚠️  WARNING: No career description data found!")
        logger.warning("     Career Advisor will have limited functionality.")
        logger.warning("     Make sure 'career description' folder has JSON files.")
    
    # Step 2: Smart Chunking
    logger.info(f"\n[2/5] Creating intelligent chunks...")
    chunks = minio_loader.split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE
    )
    logger.info(f"  ✓ Created {len(chunks)} chunks")
    
    # Log chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    logger.info(f"  Chunk types:")
    for chunk_type, count in chunk_types.items():
        logger.info(f"    - {chunk_type}: {count}")
    
    # Step 3: Entity & Relation Extraction
    logger.info(f"\n[3/5] Extracting entities & relations with LLM...")
    logger.info(f"  This may take 10-30 minutes for {len(chunks)} chunks...")
    logger.info(f"  Using {MAX_WORKERS} parallel workers")
    
    elements = minio_loader.load_or_process(
        f"data/{cache_prefix}_elements.pkl",
        minio_loader.extract_elements,
        chunks,
        use_parallel=True
    )
    logger.info(f"  ✓ Extracted {len(elements)} element sets")
    
    # Log extraction stats
    total_entities = sum(len(e.get('entities', [])) for e in elements)
    total_relations = sum(len(e.get('relations', [])) for e in elements)
    logger.info(f"  Extraction statistics:")
    logger.info(f"    - Total entities: {total_entities}")
    logger.info(f"    - Total relations: {total_relations}")
    logger.info(f"    - Avg entities per chunk: {total_entities/len(chunks):.1f}")
    logger.info(f"    - Avg relations per chunk: {total_relations/len(chunks):.1f}")
    
    # Step 4: Build Graph
    logger.info(f"\n[4/5] Building unified knowledge graph...")
    graph_stats = graph_manager.build_graph_from_elements(elements)
    logger.info(f"  ✓ Graph built successfully!")
    logger.info(f"    - Nodes created: {graph_stats['nodes']}")
    logger.info(f"    - Edges created: {graph_stats['edges']}")
    
    # Step 5: Comprehensive Verification
    logger.info(f"\n[5/5] Verifying graph structure and content...")
    
    with db_connection.get_session() as session:
        # Entity type distribution
        type_stats = session.run("""
            MATCH (e:Entity)
            RETURN e.type AS type, count(*) AS count
            ORDER BY count DESC
        """).data()
        
        logger.info(f"\n  Entity type distribution (top 15):")
        for stat in type_stats[:15]:
            logger.info(f"    - {stat['type']}: {stat['count']}")
        
        # GraphRAG specific checks
        logger.info(f"\n  GraphRAG System Verification:")
        
        hoc_phan_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type = 'học_phần'
            RETURN count(*) as count
        """).single()['count']
        
        giang_vien_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type = 'giảng_viên'
            RETURN count(*) as count
        """).single()['count']
        
        tai_lieu_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type = 'tài_liệu'
            RETURN count(*) as count
        """).single()['count']
        
        logger.info(f"    ✓ Học phần entities: {hoc_phan_count}")
        logger.info(f"    ✓ Giảng viên entities: {giang_vien_count}")
        logger.info(f"    ✓ Tài liệu entities: {tai_lieu_count}")
        
        if hoc_phan_count < 5:
            logger.warning("    ⚠️  Few học phần entities - check syllabus data")
        
        # Career Advisor specific checks
        logger.info(f"\n  Career Advisor System Verification:")
        
        career_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type IN ['nghề_nghiệp', 'career', 'vị_trí_công_việc']
            RETURN count(*) as count
        """).single()['count']
        
        major_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type IN ['ngành_học', 'chương_trình_đào_tạo', 'major']
            RETURN count(*) as count
        """).single()['count']
        
        skill_count = session.run("""
            MATCH (e:Entity)
            WHERE e.type IN ['kỹ_năng', 'skill', 'năng_lực']
            RETURN count(*) as count
        """).single()['count']
        
        logger.info(f"    ✓ Nghề nghiệp entities: {career_count}")
        logger.info(f"    ✓ Ngành học entities: {major_count}")
        logger.info(f"    ✓ Kỹ năng entities: {skill_count}")
        
        # Key relationships for Career Advisor
        career_major_rels = session.run("""
            MATCH ()-[r]->()
            WHERE type(r) IN ['ĐÀO_TẠO_CHO_NGHỀ', 'TRAINS_FOR', 'YÊU_CẦU_NGÀNH']
            RETURN count(*) as count
        """).single()['count']
        
        skill_rels = session.run("""
            MATCH ()-[r]->()
            WHERE type(r) IN ['YÊU_CẦU_KỸ_NĂNG', 'PHÁT_TRIỂN_KỸ_NĂNG', 'CẦN_KỸ_NĂNG']
            RETURN count(*) as count
        """).single()['count']
        
        logger.info(f"    ✓ Career ↔ Major relationships: {career_major_rels}")
        logger.info(f"    ✓ Skill-related relationships: {skill_rels}")
        
        if career_count < 5:
            logger.warning("    ⚠️  Very few career entities!")
            logger.warning("       → Career Advisor will have limited functionality")
            logger.warning("       → Check 'career description' folder in MinIO")
        
        if career_major_rels < 3:
            logger.warning("    ⚠️  Few career-major relationships!")
            logger.warning("       → Career recommendations may be limited")
        
        # Sample entities for verification
        logger.info(f"\n  Sample Entities (for verification):")
        
        samples = session.run("""
            MATCH (e:Entity)
            WHERE e.type IN ['học_phần', 'nghề_nghiệp', 'ngành_học']
            RETURN e.type as type, e.name as name
            ORDER BY e.type
            LIMIT 15
        """).data()
        
        current_type = None
        for sample in samples:
            if sample['type'] != current_type:
                current_type = sample['type']
                logger.info(f"\n    {current_type.upper()}:")
            logger.info(f"      • {sample['name']}")
        
        # Embeddings status
        logger.info(f"\n  Embeddings Status:")
        embed_count = session.run("""
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN count(*) as count
        """).single()['count']
        
        total_nodes = graph_stats['nodes']
        embed_pct = (embed_count / total_nodes * 100) if total_nodes > 0 else 0
        
        logger.info(f"    Entities with embeddings: {embed_count}/{total_nodes} ({embed_pct:.1f}%)")
        
        if embed_pct > 80:
            logger.info(f"    ✓ Excellent embedding coverage")
        elif embed_pct > 50:
            logger.info(f"    ⚠️  Good embedding coverage")
        else:
            logger.warning(f"    ⚠️  Low embedding coverage - may affect search quality")
    
    elapsed = time.time() - start_time
    logger.info(f"\n✓ Complete graph building finished in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    db_connection.close()
    
    return {
        'documents': len(documents),
        'doc_types': doc_types,
        'chunks': len(chunks),
        'chunk_types': chunk_types,
        'elements': len(elements),
        'total_entities': total_entities,
        'total_relations': total_relations,
        'graph': graph_stats,
        'entity_types': type_stats,
        'graphrag_stats': {
            'học_phần': hoc_phan_count,
            'giảng_viên': giang_vien_count,
            'tài_liệu': tai_lieu_count
        },
        'career_stats': {
            'nghề_nghiệp': career_count,
            'ngành_học': major_count,
            'kỹ_năng': skill_count,
            'career_major_links': career_major_rels,
            'skill_links': skill_rels
        },
        'embeddings': {
            'count': embed_count,
            'total': total_nodes,
            'percentage': embed_pct
        },
        'time': elapsed,
        'skipped': False
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BUILD COMPLETE KNOWLEDGE GRAPH")
    print("=" * 80)
    print("\nThis unified system supports:")
    print("  🔍 GraphRAG Query System")
    print("     → Answer questions about courses, instructors, materials")
    print("     → Example: 'Giảng viên nào dạy môn Phân tích thiết kế hệ thống?'")
    print()
    print("  🎓 Career Advisor Chatbot")
    print("     → Career guidance based on interests and strengths")
    print("     → Example: 'Em muốn làm kỹ sư phần mềm, nên học ngành gì?'")
    print()
    print("=" * 80)
    print("\nData Sources (MinIO):")
    print(f"  Endpoint: {MINIO_ENDPOINT}")
    print(f"  Bucket: {MINIO_BUCKET}")
    print(f"  Folders:")
    for folder in MINIO_FOLDERS:
        print(f"    • {folder}")
    print()
    print("Configuration:")
    print(f"  Neo4j: {DB_URL}")
    print(f"  Model: {MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Overlap: {OVERLAP_SIZE}")
    print(f"  Workers: {MAX_WORKERS}")
    print()
    print("Features:")
    print(f"  ✓ Direct JSON structure parsing")
    print(f"  ✓ Smart table-paragraph integration")
    print(f"  ✓ Type-aware entity extraction")
    print(f"  ✓ Vietnamese text normalization")
    print(f"  ✓ Embedding-based semantic search")
    print(f"  ✓ Career-to-major relationship mapping")
    print()
    
    response = input("Continue building complete graph? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted")
        exit(0)
    
    try:
        results = build_complete_knowledge_graph()
        
        if results is None:
            print("\n⚠️  Operation cancelled")
            exit(0)
        
        if results.get('skipped'):
            print("\n" + "=" * 80)
            print("📊 USING EXISTING DATA")
            print("=" * 80)
            print(f"Nodes: {results['graph']['nodes']}")
            print(f"Edges: {results['graph']['relationships']}")
            print("=" * 80)
        else:
            # Comprehensive Summary
            print("\n" + "=" * 80)
            print("📊 COMPLETE GRAPH BUILD SUMMARY")
            print("=" * 80)
            
            print(f"\n📚 DATA LOADED:")
            print(f"  Documents: {results['documents']}")
            for doc_type, count in results['doc_types'].items():
                print(f"    • {doc_type}: {count}")
            
            print(f"\n🔪 CHUNKING:")
            print(f"  Total chunks: {results['chunks']}")
            for chunk_type, count in results['chunk_types'].items():
                print(f"    • {chunk_type}: {count}")
            
            print(f"\n🤖 EXTRACTION:")
            print(f"  Element sets: {results['elements']}")
            print(f"  Total entities extracted: {results['total_entities']}")
            print(f"  Total relations extracted: {results['total_relations']}")
            
            print(f"\n🕸️  GRAPH:")
            print(f"  Nodes: {results['graph']['nodes']}")
            print(f"  Edges: {results['graph']['edges']}")
            
            print(f"\n🔍 GRAPHRAG SYSTEM:")
            grs = results['graphrag_stats']
            print(f"  ✓ Học phần: {grs['học_phần']}")
            print(f"  ✓ Giảng viên: {grs['giảng_viên']}")
            print(f"  ✓ Tài liệu: {grs['tài_liệu']}")
            
            print(f"\n🎓 CAREER ADVISOR SYSTEM:")
            cs = results['career_stats']
            print(f"  ✓ Nghề nghiệp: {cs['nghề_nghiệp']}")
            print(f"  ✓ Ngành học: {cs['ngành_học']}")
            print(f"  ✓ Kỹ năng: {cs['kỹ_năng']}")
            print(f"  ✓ Career ↔ Major links: {cs['career_major_links']}")
            print(f"  ✓ Skill links: {cs['skill_links']}")
            
            print(f"\n🔮 EMBEDDINGS:")
            emb = results['embeddings']
            print(f"  Coverage: {emb['count']}/{emb['total']} ({emb['percentage']:.1f}%)")
            
            print(f"\n⏱️  TIME:")
            print(f"  Total: {results['time']:.1f}s ({results['time']/60:.1f} minutes)")
            
            print("\n" + "=" * 80)
            
            # Next Steps
            print("\n💡 NEXT STEPS:")
            print("\n  For GraphRAG Query System:")
            print("    1. python debug_graph_v3.py")
            print("       → Verify học phần, giảng viên data")
            print()
            print("    2. python query_cli_v3.py")
            print("       → Try: 'Giảng viên nào dạy PTTKHT?'")
            print("       → Try: 'Tài liệu tham khảo cho môn này?'")
            
            print("\n  For Career Advisor:")
            print("    1. python career_advisor_cli.py")
            print("       → Function 1: Career → Major advisory")
            print("       → Function 2: Subject → Career advisory")
            print()
            print("    2. python career_advisor_cli.py examples")
            print("       → See usage examples")
            
            print("\n" + "=" * 80)
            
            # Warnings & Recommendations
            if cs['nghề_nghiệp'] < 10:
                print("\n⚠️  WARNING: Few career entities detected!")
                print("   → Career Advisor functionality will be limited")
                print("   → Action: Check 'career description' folder in MinIO")
                print("   → Run: python check_minio_docs.py")
            
            if grs['học_phần'] < 10:
                print("\n⚠️  WARNING: Few học phần entities detected!")
                print("   → GraphRAG query system will have limited data")
                print("   → Action: Check 'syllabus' folder in MinIO")
            
            if emb['percentage'] < 50:
                print("\n⚠️  WARNING: Low embedding coverage!")
                print("   → Semantic search quality may be affected")
                print("   → This might be due to API rate limits")
                print("   → Re-run build to generate remaining embeddings")
            
            if cs['nghề_nghiệp'] >= 10 and grs['học_phần'] >= 10 and emb['percentage'] > 80:
                print("\n✅ EXCELLENT! Graph is ready for both systems!")
                print("   → GraphRAG: Ready for course queries")
                print("   → Career Advisor: Ready for career guidance")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error occurred: {e}")
        print("Check logs/ directory for details")
        raise
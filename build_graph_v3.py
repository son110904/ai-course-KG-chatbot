# build_graph_v3.py
"""
OPTIMIZED VERSION V3: Build knowledge graph from MinIO JSON files
Uses MinioLoaderV3 and GraphManagerV3 for best performance
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

logger = Logger("BuildGraphV3").get_logger()

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "203.113.132.48:8008")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "course2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "course2-s3-uiauia")
MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "syllabus")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Folders to load from
MINIO_FOLDERS = [
    "courses-processed/curriculum/",
    "courses-processed/syllabus/",
    "courses-processed/career description/"
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

def process_documents_from_minio(cache_prefix="minio_v3"):
    """
    Load documents from MinIO and build knowledge graph.
    OPTIMIZED V3: Best performance with structured JSON handling.
    """
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("OPTIMIZED MINIO DOCUMENT PROCESSING PIPELINE V3")
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
        print(f"   2. Skip and use existing data")
        print(f"   3. Cancel")
        
        choice = input("\n   Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            logger.info("User chose to clear and rebuild database")
            auto_clear = True
        elif choice == "2":
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
    logger.info(f"[1/4] Loading JSON documents from MinIO...")
    logger.info(f"  Endpoint: {MINIO_ENDPOINT}")
    logger.info(f"  Bucket: {MINIO_BUCKET}")
    logger.info(f"  Folders: {MINIO_FOLDERS}")
    
    documents = minio_loader.load_documents_from_folders(MINIO_FOLDERS)
    
    if not documents:
        logger.error("No documents loaded from MinIO")
        db_connection.close()
        return None
    
    logger.info(f"  ✓ Loaded {len(documents)} documents")
    
    # Log document types
    doc_types = {}
    for doc in documents:
        doc_type = doc.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    logger.info(f"  Document types: {doc_types}")
    
    # Step 2: Smart Chunking
    logger.info(f"[2/4] Creating intelligent chunks...")
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
    logger.info(f"  Chunk types: {chunk_types}")
    
    # Step 3: Entity & Relation Extraction
    logger.info(f"[3/4] Extracting entities & relations with LLM...")
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
    logger.info(f"  Total entities: {total_entities}")
    logger.info(f"  Total relations: {total_relations}")
    
    # Step 4: Build Graph
    logger.info(f"[4/4] Building knowledge graph...")
    graph_stats = graph_manager.build_graph_from_elements(elements)
    logger.info(f"  ✓ Graph built: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    
    # Step 5: Verify entity types
    logger.info(f"[5/5] Verifying entity types...")
    with db_connection.get_session() as session:
        type_stats = session.run("""
            MATCH (e:Entity)
            RETURN e.type AS type, count(*) AS count
            ORDER BY count DESC
        """).data()
        
        logger.info("Entity type distribution:")
        for stat in type_stats[:15]:
            logger.info(f"  - {stat['type']}: {stat['count']}")
    
    # Step 6: Sample entities
    logger.info(f"[6/6] Sample entities:")
    with db_connection.get_session() as session:
        samples = session.run("""
            MATCH (e:Entity)
            WHERE e.type IN ['học_phần', 'giảng_viên', 'tài_liệu']
            RETURN e.type as type, e.name as name, e.mã_học_phần as ma
            LIMIT 20
        """).data()
        
        for sample in samples:
            logger.info(f"  - [{sample['type']}] {sample['name']} {f'({sample['ma']})' if sample.get('ma') else ''}")
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Processing complete in {elapsed:.1f}s")
    
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
        'time': elapsed,
        'skipped': False
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BUILD OPTIMIZED KNOWLEDGE GRAPH FROM MINIO (V3)")
    print("=" * 80)
    print(f"\nMinIO Configuration:")
    print(f"  Endpoint: {MINIO_ENDPOINT}")
    print(f"  Bucket: {MINIO_BUCKET}")
    print(f"  Folders: {MINIO_FOLDERS}")
    print(f"\nNeo4j Configuration:")
    print(f"  URI: {DB_URL}")
    print(f"\nProcessing Configuration:")
    print(f"  Model: {MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Overlap: {OVERLAP_SIZE}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"\nV3 Optimizations:")
    print(f"  ✓ Direct JSON structure parsing")
    print(f"  ✓ Smart table-paragraph integration")
    print(f"  ✓ Type-aware entity extraction")
    print(f"  ✓ Enhanced property handling")
    print(f"  ✓ Vietnamese text normalization")
    print(f"  ✓ Embedding-based search support")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted")
        exit(0)
    
    try:
        results = process_documents_from_minio()
        
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
            # Summary
            print("\n" + "=" * 80)
            print("📊 PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Documents: {results['documents']}")
            print(f"  Document types: {results['doc_types']}")
            print(f"\nChunks: {results['chunks']}")
            print(f"  - Text chunks: {results['chunk_types'].get('text', 0)}")
            print(f"  - Table chunks: {results['chunk_types'].get('table', 0)}")
            print(f"\nExtraction:")
            print(f"  - Element sets: {results['elements']}")
            print(f"  - Total entities: {results['total_entities']}")
            print(f"  - Total relations: {results['total_relations']}")
            print(f"\nGraph:")
            print(f"  - Nodes: {results['graph']['nodes']}")
            print(f"  - Edges: {results['graph']['edges']}")
            print(f"\nTop Entity Types:")
            for stat in results.get('entity_types', [])[:10]:
                print(f"  - {stat['type']}: {stat['count']}")
            print(f"\nTime: {results['time']:.1f}s")
            print("=" * 80)
            
            # Suggestions
            print("\n💡 Next Steps:")
            print("  1. Run 'python debug_graph_v3.py' to inspect the graph")
            print("  2. Run 'python query_cli_v3.py' to query the graph")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
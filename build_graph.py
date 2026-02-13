# build_graph_v2.py
"""
IMPROVED VERSION: Build knowledge graph with enhanced semantic preservation
Uses MinioLoaderV2 and GraphManagerV2 for better context handling
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from minio_loader import MinioLoaderV2
from graph_database import GraphDatabaseConnection
from graph_manager import GraphManagerV2
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()


logger = Logger("BuildGraphV2").get_logger()

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
    "courses-processed/career_description/"
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

def process_documents_from_minio(cache_prefix="minio_v2"):
    """
    Load documents from MinIO and build knowledge graph.
    IMPROVED: Uses enhanced semantic preservation.
    """
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("ENHANCED MINIO DOCUMENT PROCESSING PIPELINE V2")
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
    
    # Initialize MinIO loader V2
    minio_loader = MinioLoaderV2(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        client=client,
        model=MODEL,
        max_workers=MAX_WORKERS,
        secure=MINIO_SECURE
    )
    
    # Initialize graph manager V2
    graph_manager = GraphManagerV2(
        db_connection=db_connection,
        auto_clear=auto_clear,
        openai_client=client
    )
    
    # Step 1: Load documents from MinIO as structured docs
    logger.info(f"[1/4] Loading structured documents from MinIO...")
    logger.info(f"  Endpoint: {MINIO_ENDPOINT}")
    logger.info(f"  Bucket: {MINIO_BUCKET}")
    logger.info(f"  Folders: {MINIO_FOLDERS}")
    
    structured_docs = minio_loader.load_documents_from_folders(MINIO_FOLDERS)
    
    if not structured_docs:
        logger.error("No documents loaded from MinIO")
        db_connection.close()
        return None
    
    logger.info(f"  ✓ Loaded {len(structured_docs)} structured documents")
    
    # Step 2: Smart Chunking with Metadata
    logger.info(f"[2/4] Creating chunks with metadata...")
    chunks = minio_loader.split_documents(
        structured_docs,
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
    
    # Step 3: Entity & Relation Extraction with Enhanced Prompts
    logger.info(f"[3/4] Extracting entities & relations with metadata...")
    elements = minio_loader.load_or_process(
        f"data/{cache_prefix}_elements.pkl",
        minio_loader.extract_elements,
        chunks,
        use_parallel=True
    )
    logger.info(f"  ✓ Extracted {len(elements)} element sets")
    
    # Step 4: Build Graph with Properties
    logger.info(f"[4/4] Building knowledge graph with enhanced properties...")
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
        for stat in type_stats[:10]:
            logger.info(f"  - {stat['type']}: {stat['count']}")
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Processing complete in {elapsed:.1f}s")
    
    db_connection.close()
    
    return {
        'documents': len(structured_docs),
        'chunks': len(chunks),
        'chunk_types': chunk_types,
        'elements': len(elements),
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
    print("BUILD ENHANCED KNOWLEDGE GRAPH FROM MINIO (V2)")
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
    print(f"\nEnhancements:")
    print(f"  ✓ Semantic section detection")
    print(f"  ✓ Smart table-paragraph linking")
    print(f"  ✓ Metadata-enriched chunks")
    print(f"  ✓ Enhanced entity properties")
    print(f"  ✓ Type-based entity labels")
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
            print(f"Chunks: {results['chunks']}")
            print(f"  - Text chunks: {results['chunk_types'].get('text', 0)}")
            print(f"  - Table chunks: {results['chunk_types'].get('table', 0)}")
            print(f"  - Mixed chunks: {results['chunk_types'].get('mixed', 0)}")
            print(f"Elements: {results['elements']}")
            print(f"Nodes: {results['graph']['nodes']}")
            print(f"Edges: {results['graph']['edges']}")
            print(f"\nTop Entity Types:")
            for stat in results.get('entity_types', [])[:5]:
                print(f"  - {stat['type']}: {stat['count']}")
            print(f"\nTime: {results['time']:.1f}s")
            print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
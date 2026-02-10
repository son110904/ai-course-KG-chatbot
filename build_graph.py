# build_graph.py
"""
Script để xây dựng knowledge graph từ documents.
Chạy script này TRƯỚC KHI sử dụng Streamlit interface.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from document_processor import DocumentProcessor, read_docx_from_directory
from graph_database import GraphDatabaseConnection
from graph_manager import GraphManager
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()

logger = Logger("BuildGraph").get_logger()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Validation
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")
if not DB_URL or not DB_PASSWORD:
    raise ValueError("Neo4j credentials (DB_URL, DB_PASSWORD) must be set in .env file")

# Processing configuration
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "300"))
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "example_docx")

# =========================================================
# INITIALIZE COMPONENTS
# =========================================================
logger.info("=" * 80)
logger.info("KNOWLEDGE GRAPH BUILDER")
logger.info("=" * 80)
logger.info(f"Model: {MODEL}")
logger.info(f"Neo4j URL: {DB_URL}")
logger.info(f"Document Directory: {DOCUMENT_DIR}")
logger.info(f"Chunk Size: {CHUNK_SIZE}, Overlap: {OVERLAP_SIZE}")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize document processor
document_processor = DocumentProcessor(
    client=client,
    model=MODEL,
    max_workers=MAX_WORKERS
)

# Initialize database connection
try:
    db_connection = GraphDatabaseConnection(
        uri=DB_URL,
        user=DB_USERNAME,
        password=DB_PASSWORD
    )
    logger.info("✓ Neo4j connection established")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    logger.error("Please ensure Neo4j is running and credentials are correct")
    raise

# Initialize graph manager WITH auto_clear=True (will clear database)
logger.warning("⚠️  This will CLEAR the existing database!")
response = input("Continue? (yes/no): ")
if response.lower() != 'yes':
    logger.info("Aborted by user")
    exit(0)

graph_manager = GraphManager(
    db_connection=db_connection,
    auto_clear=True,
    openai_client=client  # Pass OpenAI client for embeddings
)

# =========================================================
# PROCESSING PIPELINE
# =========================================================

def process_documents(documents, cache_prefix="default"):
    """
    Process documents through the full pipeline with caching.
    
    Args:
        documents: List of document texts
        cache_prefix: Prefix for cache files
        
    Returns:
        Dict with processing results
    """
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Chunking
    logger.info(f"[1/3] Chunking {len(documents)} documents...")
    chunks = document_processor.split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE
    )
    logger.info(f"  ✓ Created {len(chunks)} chunks")
    
    # Step 2: Entity & Relation Extraction (with caching)
    logger.info(f"[2/3] Extracting entities & relations...")
    elements = document_processor.load_or_process(
        f"data/{cache_prefix}_elements.pkl",
        document_processor.extract_elements,
        chunks,
        use_parallel=True
    )
    logger.info(f"  ✓ Extracted {len(elements)} element sets")
    
    # Step 3: Build Graph in Neo4j
    logger.info(f"[3/3] Building knowledge graph in Neo4j...")
    graph_stats = graph_manager.build_graph_from_elements(elements)
    logger.info(f"  ✓ Graph built: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✓ Processing complete in {elapsed:.1f}s")
    logger.info("=" * 80)
    
    return {
        'chunks': len(chunks),
        'elements': len(elements),
        'graph': graph_stats,
        'time': elapsed
    }


# =========================================================
# MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":
    try:
        # Load documents
        logger.info(f"Loading documents from {DOCUMENT_DIR}...")
        documents = read_docx_from_directory(DOCUMENT_DIR)
        
        if not documents:
            logger.error(f"No documents found in {DOCUMENT_DIR}")
            logger.info("Please add .docx files to the directory and try again")
            exit(1)
        
        total_chars = sum(len(d) for d in documents)
        logger.info(f"✓ Loaded {len(documents)} documents ({total_chars:,} characters)")
        
        # Process documents
        results = process_documents(documents, cache_prefix="neo4j_khop")
        
        # Show processing summary
        print("\n" + "=" * 80)
        print("📈 PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {results['chunks']}")
        print(f"Extracted Elements: {results['elements']}")
        print(f"Graph Nodes: {results['graph']['nodes']}")
        print(f"Graph Edges: {results['graph']['edges']}")
        print(f"Processing Time: {results['time']:.1f}s")
        print("=" * 80)
        
        # Get database stats
        db_stats = db_connection.get_database_stats()
        print("\n📊 Neo4j Database:")
        print(f"  Total Nodes: {db_stats['nodes']}")
        print(f"  Total Relationships: {db_stats['relationships']}")
        print(f"  Labels: {', '.join(db_stats['labels'])}")
        print("\n" + "=" * 80)
        print("✅ Knowledge graph successfully built!")
        print("=" * 80)
        print("\n🚀 Next step: Run Streamlit interface")
        print("   streamlit run streamlit_app.py")
        print("   or: python -m streamlit run streamlit_app.py")
        print()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if db_connection:
            db_connection.close()
        logger.info("Script completed")
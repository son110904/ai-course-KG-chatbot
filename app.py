from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from document_processor import DocumentProcessor
from graph_database import GraphDatabaseConnection
from graph_manager import GraphManager
from query_handler import QueryHandler
from document_processor import read_docx_from_directory
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()

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
logger = Logger("MainApp").get_logger()

logger.info("=" * 80)
logger.info("GRAPHRAG CHATBOT - NEO4J ONLY MODE")
logger.info("=" * 80)
logger.info(f"Model: {MODEL}")
logger.info(f"Neo4j URL: {DB_URL}")
logger.info(f"Max Workers: {MAX_WORKERS}")
logger.info(f"Chunk Size: {CHUNK_SIZE}")

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

# Initialize graph manager (Neo4j only)
graph_manager = GraphManager(db_connection=db_connection)

# Initialize query handler
query_handler = QueryHandler(
    graph_manager=graph_manager,
    client=client,
    model="gpt-4o"  # Use better model for final answers
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


def interactive_query_loop():
    """
    Run interactive query loop for user questions.
    """
    logger.info("\n" + "=" * 80)
    logger.info("INTERACTIVE QUERY MODE")
    logger.info("=" * 80)
    print("\nAvailable commands:")
    print("  - Type your question to get an answer")
    print("  - 'stats' - Show graph statistics")
    print("  - 'search <term>' - Search for entities")
    print("  - 'neighbors <entity>' - Show entity neighbors")
    print("  - 'method <communities|centrality>' - Change retrieval method")
    print("  - 'exit' - Quit\n")
    
    current_method = 'communities'
    
    while True:
        try:
            query = input("❓ Question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting query mode")
                break
            
            if query.lower() == 'stats':
                stats = query_handler.get_graph_stats()
                print("\n📊 Graph Statistics:")
                print(f"  Nodes: {stats['nodes']}")
                print(f"  Edges: {stats['edges']}")
                print(f"  Communities: {stats['communities']}")
                print(f"  Top 3 communities (by size):")
                for i, comm in enumerate(stats['top_communities'][:3], 1):
                    print(f"    {i}. {len(comm)} entities: {', '.join(comm[:5])}...")
                print()
                continue
            
            if query.lower().startswith('search '):
                search_term = query[7:].strip()
                results = graph_manager.search_entities(search_term, limit=20)
                print(f"\n🔍 Found {len(results)} entities matching '{search_term}':")
                for entity in results:
                    print(f"  - {entity}")
                print()
                continue
            
            if query.lower().startswith('neighbors '):
                entity_name = query[10:].strip()
                neighbors = graph_manager.get_entity_neighbors(entity_name, max_depth=2)
                print(f"\n🔗 Neighbors of '{entity_name}':")
                if neighbors:
                    for neighbor in neighbors[:20]:
                        print(f"  - {neighbor}")
                    if len(neighbors) > 20:
                        print(f"  ... and {len(neighbors) - 20} more")
                else:
                    print("  No neighbors found")
                print()
                continue
            
            if query.lower().startswith('method '):
                method = query[7:].strip().lower()
                if method in ['communities', 'centrality']:
                    current_method = method
                    print(f"\n✓ Retrieval method changed to: {current_method}\n")
                else:
                    print(f"\n❌ Invalid method. Use 'communities' or 'centrality'\n")
                continue
            
            logger.info(f"Processing query with method={current_method}: {query}")
            answer = query_handler.ask_question(query, method=current_method)
            
            print("\n" + "=" * 80)
            print("💡 ANSWER")
            print("=" * 80)
            print(answer)
            print("=" * 80 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\nExiting query mode")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}\n")


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
            exit(1)
        
        total_chars = sum(len(d) for d in documents)
        logger.info(f"✓ Loaded {len(documents)} documents ({total_chars:,} characters)")
        
        # Process documents
        results = process_documents(documents, cache_prefix="neo4j")
        
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
        
        # Run example query
        example_query = "Tổng hợp nội dung chính của các tài liệu"
        logger.info(f"\nExample query: {example_query}")
        answer = query_handler.ask_question(example_query, method='communities')
        
        print("\n" + "=" * 80)
        print("💡 EXAMPLE ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # Enter interactive mode
        interactive_query_loop()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if db_connection:
            db_connection.close()
        logger.info("Application shutdown complete")
# build_graph_detailed.py
"""
Enhanced version với detailed logging cho debugging
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json

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

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")
if not DB_URL or not DB_PASSWORD:
    raise ValueError("Neo4j credentials must be set in .env file")

MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "300"))
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "example_docx")

# =========================================================
# DETAILED LOGGING FUNCTIONS
# =========================================================

def log_extraction_details(elements, output_file="extraction_details.txt"):
    """Log chi tiết tất cả elements được extract"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTION DETAILS - ALL ELEMENTS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, elem in enumerate(elements):
            f.write(f"\n{'='*60}\n")
            f.write(f"ELEMENT SET {idx + 1}/{len(elements)}\n")
            f.write(f"{'='*60}\n")
            f.write(elem)
            f.write("\n")
            
            # Parse and count
            lines = elem.split('\n')
            entities = [l for l in lines if l.strip().startswith('ENTITY:')]
            types = [l for l in lines if l.strip().startswith('TYPE:')]
            relations = [l for l in lines if l.strip().startswith('RELATION:')]
            
            f.write(f"\nSummary:\n")
            f.write(f"  - Entities: {len(entities)}\n")
            f.write(f"  - Types: {len(types)}\n")
            f.write(f"  - Relations: {len(relations)}\n")
    
    print(f"✅ Detailed extraction logged to: {output_file}")


def log_graph_structure(db_connection, output_file="graph_structure.txt"):
    """Log chi tiết structure của graph trong Neo4j"""
    
    with db_connection.get_session() as session:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GRAPH STRUCTURE IN NEO4J\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. All nodes with types
            f.write("\n" + "=" * 60 + "\n")
            f.write("ALL NODES (grouped by type)\n")
            f.write("=" * 60 + "\n\n")
            
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, collect(n.name) as names
                ORDER BY type
            """)
            
            for record in result:
                node_type = record['type'] or 'unknown'
                names = record['names']
                f.write(f"\n[{node_type}] - {len(names)} nodes:\n")
                for name in names[:20]:  # First 20
                    f.write(f"  - {name}\n")
                if len(names) > 20:
                    f.write(f"  ... and {len(names) - 20} more\n")
            
            # 2. All relationships
            f.write("\n" + "=" * 60 + "\n")
            f.write("ALL RELATIONSHIPS\n")
            f.write("=" * 60 + "\n\n")
            
            result = session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN type(r) as rel_type, 
                       a.name as source, 
                       b.name as target,
                       a.type as source_type,
                       b.type as target_type
                ORDER BY rel_type, source
            """)
            
            relationships = result.data()
            
            # Group by relationship type
            from collections import defaultdict
            rels_by_type = defaultdict(list)
            
            for rel in relationships:
                rels_by_type[rel['rel_type']].append(rel)
            
            for rel_type, rels in sorted(rels_by_type.items()):
                f.write(f"\n[{rel_type}] - {len(rels)} relationships:\n")
                for rel in rels[:10]:  # First 10
                    f.write(f"  - ({rel['source']}) --[{rel_type}]--> ({rel['target']})\n")
                    f.write(f"    Types: {rel['source_type']} --> {rel['target_type']}\n")
                if len(rels) > 10:
                    f.write(f"  ... and {len(rels) - 10} more\n")
            
            # 3. Statistics
            f.write("\n" + "=" * 60 + "\n")
            f.write("STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            # Node count by type
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(*) as count
                ORDER BY count DESC
            """)
            
            f.write("Nodes by type:\n")
            for record in result:
                f.write(f"  - {record['type'] or 'unknown'}: {record['count']}\n")
            
            # Relationship count by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            
            f.write("\nRelationships by type:\n")
            for record in result:
                f.write(f"  - {record['type']}: {record['count']}\n")
    
    print(f"✅ Graph structure logged to: {output_file}")


def analyze_query_result(query, db_connection, output_file="query_analysis.txt"):
    """Analyze tại sao query không trả về kết quả"""
    
    with db_connection.get_session() as session:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"QUERY ANALYSIS: {query}\n")
            f.write("=" * 80 + "\n\n")
            
            # Extract query terms
            import re
            terms = re.findall(r'\w+', query.lower())
            terms = [t for t in terms if len(t) > 2 and t not in ['thầy', 'cô', 'dạy', 'học', 'phần', 'môn']]
            
            f.write(f"Query terms: {terms}\n\n")
            
            # Step 1: Find matching entities
            f.write("=" * 60 + "\n")
            f.write("STEP 1: Find matching entities\n")
            f.write("=" * 60 + "\n\n")
            
            for term in terms:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($term)
                    RETURN e.name as name, e.type as type
                    LIMIT 10
                """, term=term)
                
                matches = result.data()
                f.write(f"\nTerm '{term}' matches ({len(matches)} found):\n")
                for m in matches:
                    f.write(f"  - {m['name']} (type: {m.get('type', 'unknown')})\n")
            
            # Step 2: Check for GIẢNG_DẠY relationships
            f.write("\n" + "=" * 60 + "\n")
            f.write("STEP 2: GIẢNG_DẠY relationships\n")
            f.write("=" * 60 + "\n\n")
            
            result = session.run("""
                MATCH (gv:Entity)-[r:GIẢNG_DẠY]->(hp:Entity)
                RETURN gv.name as instructor, 
                       gv.type as gv_type,
                       hp.name as course,
                       hp.type as hp_type
            """)
            
            teaching = result.data()
            f.write(f"Total GIẢNG_DẠY relationships: {len(teaching)}\n\n")
            
            for rel in teaching:
                f.write(f"  - {rel['instructor']} ({rel.get('gv_type', '?')}) "
                       f"-> {rel['course']} ({rel.get('hp_type', '?')})\n")
            
            # Step 3: Check for instructors
            f.write("\n" + "=" * 60 + "\n")
            f.write("STEP 3: All instructors\n")
            f.write("=" * 60 + "\n\n")
            
            result = session.run("""
                MATCH (gv:Entity {type: "giảng_viên"})
                RETURN gv.name as name, properties(gv) as props
            """)
            
            instructors = result.data()
            f.write(f"Total instructors: {len(instructors)}\n\n")
            for ins in instructors:
                f.write(f"  - {ins['name']}\n")
                f.write(f"    Properties: {ins['props']}\n")
            
            # Step 4: Check for courses
            f.write("\n" + "=" * 60 + "\n")
            f.write("STEP 4: All courses\n")
            f.write("=" * 60 + "\n\n")
            
            result = session.run("""
                MATCH (hp:Entity {type: "học_phần"})
                RETURN hp.name as name, properties(hp) as props
                LIMIT 20
            """)
            
            courses = result.data()
            f.write(f"Total courses: {len(courses)}\n\n")
            for course in courses:
                f.write(f"  - {course['name']}\n")
                f.write(f"    Properties: {course['props']}\n")
    
    print(f"✅ Query analysis logged to: {output_file}")


# =========================================================
# ENHANCED PROCESSING PIPELINE
# =========================================================

def process_documents_with_logging(documents, cache_prefix="default"):
    """Process documents với detailed logging"""
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("DOCUMENT PROCESSING PIPELINE (WITH DETAILED LOGGING)")
    logger.info("=" * 80)
    
    # Initialize components
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    document_processor = DocumentProcessor(
        client=client,
        model=MODEL,
        max_workers=MAX_WORKERS
    )
    
    db_connection = GraphDatabaseConnection(
        uri=DB_URL,
        user=DB_USERNAME,
        password=DB_PASSWORD
    )
    
    graph_manager = GraphManager(
        db_connection=db_connection,
        auto_clear=True,
        openai_client=client
    )
    
    # Step 1: Chunking
    logger.info(f"[1/3] Chunking {len(documents)} documents...")
    chunks = document_processor.split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE
    )
    logger.info(f"  ✓ Created {len(chunks)} chunks")
    
    # Log first chunk as sample
    with open("sample_chunk.txt", 'w', encoding='utf-8') as f:
        f.write("SAMPLE CHUNK (first one):\n")
        f.write("=" * 80 + "\n")
        f.write(chunks[0] if chunks else "No chunks")
    print("✅ Sample chunk saved to: sample_chunk.txt")
    
    # Step 2: Entity & Relation Extraction
    logger.info(f"[2/3] Extracting entities & relations...")
    elements = document_processor.load_or_process(
        f"data/{cache_prefix}_elements.pkl",
        document_processor.extract_elements,
        chunks,
        use_parallel=True
    )
    logger.info(f"  ✓ Extracted {len(elements)} element sets")
    
    # LOG EXTRACTION DETAILS
    log_extraction_details(elements)
    
    # Step 3: Build Graph
    logger.info(f"[3/3] Building knowledge graph in Neo4j...")
    graph_stats = graph_manager.build_graph_from_elements(elements)
    logger.info(f"  ✓ Graph built: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    
    # LOG GRAPH STRUCTURE
    log_graph_structure(db_connection)
    
    # ANALYZE QUERY
    analyze_query_result("thầy cô dạy an ninh không gian mạng", db_connection)
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Processing complete in {elapsed:.1f}s")
    
    db_connection.close()
    
    return {
        'chunks': len(chunks),
        'elements': len(elements),
        'graph': graph_stats,
        'time': elapsed
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED BUILD WITH DETAILED LOGGING")
    print("=" * 80)
    print("\nThis will create detailed log files:")
    print("  1. extraction_details.txt - All extracted elements")
    print("  2. graph_structure.txt - Complete graph structure")
    print("  3. query_analysis.txt - Query analysis")
    print("  4. sample_chunk.txt - Sample input chunk")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted")
        exit(0)
    
    try:
        # Load documents
        logger.info(f"Loading documents from {DOCUMENT_DIR}...")
        documents = read_docx_from_directory(DOCUMENT_DIR)
        
        if not documents:
            logger.error(f"No documents found in {DOCUMENT_DIR}")
            exit(1)
        
        logger.info(f"✓ Loaded {len(documents)} documents")
        
        # Process with detailed logging
        results = process_documents_with_logging(documents, cache_prefix="neo4j_khop")
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {results['chunks']}")
        print(f"Elements: {results['elements']}")
        print(f"Nodes: {results['graph']['nodes']}")
        print(f"Edges: {results['graph']['edges']}")
        print(f"Time: {results['time']:.1f}s")
        print("=" * 80)
        
        print("\n📁 Output files created:")
        print("  - extraction_details.txt")
        print("  - graph_structure.txt")
        print("  - query_analysis.txt")
        print("  - sample_chunk.txt")
        print("\n👀 Review these files to understand what went wrong!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
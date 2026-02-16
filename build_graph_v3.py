# build_graph_v4.py
"""
ENHANCED VERSION V4: Build knowledge graph with Entity Linking
Uses MinioLoaderV3, GraphManagerV3, and NEW EntityLinker + DatabaseEntityLinker
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from minio_loader_v3 import MinioLoaderV3
from graph_database import GraphDatabaseConnection
from graph_manager_v3 import GraphManagerV3
from entity_linker_v3 import EntityLinker, RelationLinker, link_extracted_elements
from database_entity_linker import (
    DatabaseEntityLinker, 
    get_existing_entities_from_db,
    merge_duplicate_relations_in_db,
    update_entity_statistics
)
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()

logger = Logger("BuildGraphV4").get_logger()

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

# Entity Linking Configuration
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "85"))

# =========================================================
# PROCESSING PIPELINE WITH ENTITY LINKING
# =========================================================

def process_documents_from_minio(cache_prefix="minio_v4"):
    """
    Load documents from MinIO and build knowledge graph with Entity Linking.
    
    NEW in V4:
    - Entity Linking after extraction
    - Database entity merging
    - Duplicate relation handling
    - Entity statistics
    """
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("ENHANCED MINIO DOCUMENT PROCESSING PIPELINE V4 (WITH ENTITY LINKING)")
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
        print(f"   2. Merge new data with existing (recommended for incremental updates)")
        print(f"   3. Cancel")
        
        choice = input("\n   Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            logger.info("User chose to clear and rebuild database")
            auto_clear = True
            merge_mode = False
        elif choice == "2":
            logger.info("User chose to merge with existing data")
            auto_clear = False
            merge_mode = True
        else:
            logger.info("User cancelled operation")
            db_connection.close()
            return None
    else:
        auto_clear = False
        merge_mode = False
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
    logger.info(f"[1/7] Loading JSON documents from MinIO...")
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
    logger.info(f"[2/7] Creating intelligent chunks...")
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
    logger.info(f"[3/7] Extracting entities & relations with LLM...")
    elements = minio_loader.load_or_process(
        f"data/{cache_prefix}_elements.pkl",
        minio_loader.extract_elements,
        chunks,
        use_parallel=True
    )
    logger.info(f"  ✓ Extracted {len(elements)} element sets")
    
    # Log extraction stats
    total_entities_raw = sum(len(e.get('entities', [])) for e in elements)
    total_relations_raw = sum(len(e.get('relations', [])) for e in elements)
    logger.info(f"  Raw entities: {total_entities_raw}")
    logger.info(f"  Raw relations: {total_relations_raw}")
    
    # ★★★ NEW STEP 4: ENTITY LINKING ★★★
    logger.info(f"[4/7] Performing Entity Linking...")
    logger.info(f"  Fuzzy matching threshold: {FUZZY_THRESHOLD}")
    
    # Link entities and relations
    canonical_entities, canonical_relations, linking_stats = link_extracted_elements(
        elements,
        fuzzy_threshold=FUZZY_THRESHOLD
    )
    
    logger.info(f"  ✓ Entity Linking complete:")
    logger.info(f"    Entities: {linking_stats['original_entities']} → {linking_stats['canonical_entities']} (-{linking_stats['entity_reduction_pct']:.1f}%)")
    logger.info(f"    Relations: {linking_stats['original_relations']} → {linking_stats['canonical_relations']} (-{linking_stats['relation_reduction_pct']:.1f}%)")
    
    # ★★★ NEW STEP 5: LINK WITH DATABASE (if merge mode) ★★★
    if merge_mode:
        logger.info(f"[5/7] Linking with existing database entities...")
        
        # Get existing entities from DB
        existing_entities = get_existing_entities_from_db(db_connection)
        
        if existing_entities:
            logger.info(f"  Found {len(existing_entities)} existing entities")
            
            # Initialize entity linker for database linking
            entity_linker = EntityLinker(fuzzy_threshold=FUZZY_THRESHOLD)
            
            # Link with existing
            db_linker = DatabaseEntityLinker(entity_linker)
            final_entities, db_mapping = db_linker.link_with_existing(
                canonical_entities,
                existing_entities
            )
            
            # Update relations with database mapping
            # Combine mappings: extraction → canonical → database
            full_mapping = db_mapping
            
            relation_linker = RelationLinker(full_mapping)
            final_relations = relation_linker.update_relations(canonical_relations)
            
            logger.info(f"  ✓ Database linking complete:")
            logger.info(f"    Final entities: {len(final_entities)}")
            logger.info(f"    Final relations: {len(final_relations)}")
        else:
            logger.info(f"  No existing entities found, using canonical entities")
            final_entities = canonical_entities
            final_relations = canonical_relations
    else:
        logger.info(f"[5/7] Skipping database linking (fresh build mode)")
        final_entities = canonical_entities
        final_relations = canonical_relations
    
    # ★★★ STEP 6: BUILD GRAPH ★★★
    logger.info(f"[6/7] Building knowledge graph...")
    
    # Create consolidated element set
    final_elements = [{
        'entities': final_entities,
        'relations': final_relations,
        'chunk_metadata': {
            'source_file': 'consolidated',
            'chunk_type': 'linked',
            'entity_linking_applied': True
        }
    }]
    
    graph_stats = graph_manager.build_graph_from_elements(final_elements)
    logger.info(f"  ✓ Graph built: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    
    # ★★★ STEP 7: POST-PROCESSING ★★★
    logger.info(f"[7/7] Post-processing...")
    
    # Merge any remaining duplicate relations
    if merge_mode:
        merge_duplicate_relations_in_db(db_connection)
    
    # Update entity statistics
    update_entity_statistics(db_connection)
    
    logger.info(f"  ✓ Post-processing complete")
    
    # Verify entity types
    logger.info(f"Verifying entity types...")
    with db_connection.get_session() as session:
        type_stats = session.run("""
            MATCH (e:Entity)
            RETURN e.type AS type, count(*) AS count
            ORDER BY count DESC
        """).data()
        
        logger.info("Entity type distribution:")
        for stat in type_stats[:15]:
            logger.info(f"  - {stat['type']}: {stat['count']}")
    
    # Sample entities with high importance
    logger.info(f"Top entities by importance:")
    with db_connection.get_session() as session:
        samples = session.run("""
            MATCH (e:Entity)
            WHERE e.importance IS NOT NULL
            RETURN e.type as type, e.name as name, 
                   e.importance as importance,
                   e.source_count as mentions
            ORDER BY e.importance DESC
            LIMIT 20
        """).data()
        
        for sample in samples:
            logger.info(f"  - [{sample['type']}] {sample['name']} (importance: {sample['importance']:.2f}, mentions: {sample['mentions']})")
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Processing complete in {elapsed:.1f}s")
    
    db_connection.close()
    
    return {
        'documents': len(documents),
        'doc_types': doc_types,
        'chunks': len(chunks),
        'chunk_types': chunk_types,
        'elements': len(elements),
        'raw_entities': total_entities_raw,
        'raw_relations': total_relations_raw,
        'linking_stats': linking_stats,
        'final_entities': len(final_entities),
        'final_relations': len(final_relations),
        'graph': graph_stats,
        'entity_types': type_stats,
        'time': elapsed,
        'merge_mode': merge_mode
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BUILD ENHANCED KNOWLEDGE GRAPH FROM MINIO (V4 - WITH ENTITY LINKING)")
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
    print(f"\n★ V4 NEW FEATURES:")
    print(f"  ✓ Entity Linking (fuzzy threshold: {FUZZY_THRESHOLD})")
    print(f"  ✓ Abbreviation expansion")
    print(f"  ✓ Database entity merging")
    print(f"  ✓ Duplicate relation handling")
    print(f"  ✓ Entity importance scoring")
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
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Mode: {'MERGE' if results['merge_mode'] else 'FRESH BUILD'}")
        print(f"\nDocuments: {results['documents']}")
        print(f"  Document types: {results['doc_types']}")
        print(f"\nChunks: {results['chunks']}")
        print(f"  - Text chunks: {results['chunk_types'].get('text', 0)}")
        print(f"  - Table chunks: {results['chunk_types'].get('table', 0)}")
        print(f"\nExtraction:")
        print(f"  - Element sets: {results['elements']}")
        print(f"  - Raw entities: {results['raw_entities']}")
        print(f"  - Raw relations: {results['raw_relations']}")
        print(f"\n★ Entity Linking:")
        print(f"  - Canonical entities: {results['linking_stats']['canonical_entities']}")
        print(f"  - Entity reduction: {results['linking_stats']['entity_reduction_pct']:.1f}%")
        print(f"  - Canonical relations: {results['linking_stats']['canonical_relations']}")
        print(f"  - Relation reduction: {results['linking_stats']['relation_reduction_pct']:.1f}%")
        print(f"\nFinal Graph:")
        print(f"  - Nodes: {results['graph']['nodes']}")
        print(f"  - Edges: {results['graph']['edges']}")
        print(f"\nTop Entity Types:")
        for stat in results.get('entity_types', [])[:10]:
            print(f"  - {stat['type']}: {stat['count']}")
        print(f"\nTime: {results['time']:.1f}s")
        print("=" * 80)
        
        # Suggestions
        print("\n💡 Next Steps:")
        print("  1. Run 'python debug_graph.py' to inspect the graph")
        print("  2. Run 'python query_cli_v3.py' to query the graph")
        print("  3. Check entity variants and importance scores in Neo4j Browser")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

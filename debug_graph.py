# debug_graph.py
"""
Debug script to check what data exists in the knowledge graph
"""

from dotenv import load_dotenv
import os
from graph_database import GraphDatabaseConnection

load_dotenv()

DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def check_graph_content():
    """Check what entities and relationships exist in the graph."""
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH DEBUG REPORT")
    print("="*80)
    
    db = GraphDatabaseConnection(DB_URL, DB_USERNAME, DB_PASSWORD)
    
    try:
        # 1. Basic stats
        print("\n1. BASIC STATISTICS")
        print("-" * 80)
        stats = db.get_database_stats()
        print(f"Total Nodes: {stats['nodes']}")
        print(f"Total Relationships: {stats['relationships']}")
        print(f"Labels: {', '.join(stats['labels'])}")
        
        # 2. Entity types distribution
        print("\n2. ENTITY TYPES DISTRIBUTION")
        print("-" * 80)
        with db.get_session() as session:
            type_counts = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """).data()
            
            for item in type_counts:
                print(f"  {item['type']}: {item['count']}")
        
        # 3. Sample học_phần entities
        print("\n3. SAMPLE HỌC PHẦN ENTITIES")
        print("-" * 80)
        with db.get_session() as session:
            hoc_phan = session.run("""
                MATCH (e:Entity)
                WHERE e.type = 'học_phần'
                RETURN e.name as name, e.mã_học_phần as ma, e.số_tín_chỉ as tin_chi
                LIMIT 20
            """).data()
            
            if hoc_phan:
                for hp in hoc_phan:
                    print(f"  - {hp['name']} (Mã: {hp.get('ma', 'N/A')}, Tín chỉ: {hp.get('tin_chi', 'N/A')})")
            else:
                print("  ❌ No học_phần entities found!")
        
        # 4. Search for "vật lý"
        print("\n4. SEARCH FOR 'VẬT LÝ' ENTITIES")
        print("-" * 80)
        with db.get_session() as session:
            vat_ly = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS 'vật lý'
                RETURN e.name as name, e.type as type, properties(e) as props
            """).data()
            
            if vat_ly:
                print(f"  ✅ Found {len(vat_ly)} entities containing 'vật lý':")
                for vl in vat_ly:
                    print(f"\n  Entity: {vl['name']}")
                    print(f"  Type: {vl['type']}")
                    print(f"  Properties: {vl['props']}")
            else:
                print("  ❌ No entities found containing 'vật lý'")
                print("\n  Possible reasons:")
                print("  1. Document not loaded from MinIO")
                print("  2. Entity extraction failed")
                print("  3. Name normalization issue")
        
        # 5. Search for "java"
        print("\n5. SEARCH FOR 'JAVA' ENTITIES (FOR COMPARISON)")
        print("-" * 80)
        with db.get_session() as session:
            java = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS 'java'
                RETURN e.name as name, e.type as type
                LIMIT 5
            """).data()
            
            if java:
                print(f"  ✅ Found {len(java)} entities containing 'java':")
                for j in java:
                    print(f"  - {j['name']} ({j['type']})")
        
        # 6. Sample relationships
        print("\n6. SAMPLE RELATIONSHIPS")
        print("-" * 80)
        with db.get_session() as session:
            rels = session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.name as source, type(r) as rel_type, b.name as target
                LIMIT 10
            """).data()
            
            for rel in rels:
                print(f"  {rel['source']} --[{rel['rel_type']}]--> {rel['target']}")
        
        # 7. Check all unique entity names
        print("\n7. ALL UNIQUE ENTITY NAMES (First 50)")
        print("-" * 80)
        with db.get_session() as session:
            all_names = session.run("""
                MATCH (e:Entity)
                RETURN DISTINCT e.name as name
                ORDER BY name
                LIMIT 50
            """).data()
            
            for item in all_names:
                print(f"  - {item['name']}")
        
        # 8. Check if embeddings exist
        print("\n8. EMBEDDINGS STATUS")
        print("-" * 80)
        with db.get_session() as session:
            embed_count = session.run("""
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN count(*) as count
            """).single()['count']
            
            total = stats['nodes']
            percentage = (embed_count / total * 100) if total > 0 else 0
            
            print(f"  Entities with embeddings: {embed_count}/{total} ({percentage:.1f}%)")
            
            if percentage < 50:
                print("  ⚠️  Warning: Less than 50% entities have embeddings")
                print("  This may affect search quality")
        
        print("\n" + "="*80)
        print("DEBUG REPORT COMPLETE")
        print("="*80)
        
    finally:
        db.close()

def test_search(query_term):
    """Test search for a specific term."""
    
    print(f"\n" + "="*80)
    print(f"TESTING SEARCH FOR: '{query_term}'")
    print("="*80)
    
    db = GraphDatabaseConnection(DB_URL, DB_USERNAME, DB_PASSWORD)
    
    try:
        with db.get_session() as session:
            # Exact match
            print("\n1. Exact Match Search:")
            exact = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($term)
                RETURN e.name as name, e.type as type
            """, term=query_term).data()
            
            if exact:
                print(f"  ✅ Found exact match:")
                for item in exact:
                    print(f"  - {item['name']} ({item['type']})")
            else:
                print(f"  ❌ No exact match for '{query_term}'")
            
            # Contains match
            print("\n2. Contains Match Search:")
            contains = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                RETURN e.name as name, e.type as type
            """, term=query_term).data()
            
            if contains:
                print(f"  ✅ Found {len(contains)} entities containing '{query_term}':")
                for item in contains:
                    print(f"  - {item['name']} ({item['type']})")
            else:
                print(f"  ❌ No entities contain '{query_term}'")
            
            # Fuzzy match (split words)
            print("\n3. Word-based Search:")
            words = query_term.lower().split()
            for word in words:
                if len(word) < 3:
                    continue
                    
                fuzzy = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($word)
                    RETURN e.name as name, e.type as type
                    LIMIT 5
                """, word=word).data()
                
                if fuzzy:
                    print(f"  Word '{word}': Found {len(fuzzy)} matches")
                    for item in fuzzy:
                        print(f"    - {item['name']}")
    
    finally:
        db.close()

if __name__ == "__main__":
    import sys
    
    # Check graph content
    check_graph_content()
    
    # Test specific search if provided
    if len(sys.argv) > 1:
        search_term = " ".join(sys.argv[1:])
        test_search(search_term)
    else:
        # Default tests
        test_search("vật lý")
        test_search("vật lý đại cương")
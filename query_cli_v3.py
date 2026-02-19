# query_cli_v3.py
"""
Enhanced CLI Tool for GraphRAG Query V3
Optimized for Vietnamese educational content queries
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

from graph_database import GraphDatabaseConnection
from graph_manager_v3 import GraphManagerV3
from query_handler_v3 import QueryHandlerV3
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()
logger = Logger("QueryCLI_V3").get_logger()

# Configuration from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not OPENAI_API_KEY or not DB_URL or not DB_PASSWORD:
    print("❌ Error: Missing configuration in .env file")
    print("Required: OPENAI_API_KEY, DB_URL, DB_PASSWORD")
    sys.exit(1)

# =========================================================
# INITIALIZE QUERY SYSTEM
# =========================================================

def initialize_query_system():
    """Initialize connection to knowledge graph."""
    try:
        print("🔌 Connecting to Neo4j...")
        
        db_connection = GraphDatabaseConnection(
            uri=DB_URL,
            user=DB_USERNAME,
            password=DB_PASSWORD
        )
        
        # Check if graph exists
        stats = db_connection.get_database_stats()
        
        if stats['nodes'] == 0:
            print("❌ Database is empty!")
            print("Please run 'python build_graph_v3.py' first to build the knowledge graph")
            db_connection.close()
            return None, None, None
        
        print(f"✅ Connected to knowledge graph")
        print(f"   Nodes: {stats['nodes']}")
        print(f"   Relationships: {stats['relationships']}")
        
        # Get entity type distribution
        with db_connection.get_session() as session:
            type_stats = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
                LIMIT 5
            """).data()
            
            print(f"   Top entity types:")
            for stat in type_stats:
                print(f"     - {stat['type']}: {stat['count']}")
        
        print()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize graph manager
        graph_manager = GraphManagerV3(
            db_connection=db_connection,
            auto_clear=False,
            openai_client=client
        )
        
        # Initialize query handler
        query_handler = QueryHandlerV3(
            graph_manager=graph_manager,
            client=client,
            model=MODEL
        )
        
        return db_connection, graph_manager, query_handler
        
    except Exception as e:
        print(f"❌ Error connecting to graph: {e}")
        logger.error(f"Connection error: {e}", exc_info=True)
        return None, None, None

# =========================================================
# QUERY INTERFACE
# =========================================================

def ask_question(
    query_handler, 
    question, 
    k=2, 
    top_k_seeds=5, 
    max_nodes=80, 
    use_embeddings=True,
    verbose=False
):
    """Ask a question and get answer."""
    try:
        if verbose:
            print(f"\n🔍 Processing query...")
            print(f"   Settings: k={k}, seeds={top_k_seeds}, max_nodes={max_nodes}")
        
        answer = query_handler.ask_question(
            query=question,
            k=k,
            top_k_seeds=top_k_seeds,
            max_nodes=max_nodes,
            use_embeddings=use_embeddings
        )
        
        print(f"\n💡 Answer:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        return answer
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        logger.error(f"Query error: {e}", exc_info=True)
        return None

# =========================================================
# INTERACTIVE MODE
# =========================================================

def interactive_mode(query_handler):
    """Interactive query mode - ask multiple questions."""
    
    print("\n" + "=" * 80)
    print("🤖 GRAPHRAG INTERACTIVE QUERY MODE V3")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your question to search")
    print("  - 'examples' - Show example questions")
    print("  - 'settings' - Adjust query settings")
    print("  - 'help' - Show help")
    print("  - 'quit' or 'exit' - Exit program")
    print()
    
    # Default settings
    settings = {
        'k': 2,
        'top_k_seeds': 5,
        'max_nodes': 80,
        'use_embeddings': True,
        'verbose': False
    }
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif question.lower() == 'examples':
                show_examples()
                continue
            
            elif question.lower() == 'settings':
                settings = adjust_settings(settings)
                continue
            
            elif question.lower() == 'help':
                show_help()
                continue
            
            elif question.lower() == 'stats':
                show_quick_stats(query_handler.graph_manager.db)
                continue
            
            # Ask question
            ask_question(
                query_handler,
                question,
                k=settings['k'],
                top_k_seeds=settings['top_k_seeds'],
                max_nodes=settings['max_nodes'],
                use_embeddings=settings['use_embeddings'],
                verbose=settings['verbose']
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def show_examples():
    """Show example questions organized by category."""
    print("\n" + "=" * 80)
    print("📚 EXAMPLE QUESTIONS")
    print("=" * 80)
    
    examples = {
        "Về Học phần (Course Information)": [
            "Môn Phân tích và thiết kế hệ thống có bao nhiêu tín chỉ?",
            "Mã học phần của môn Phân tích và thiết kế hệ thống là gì?",
            "Mô tả về môn Phân tích và thiết kế hệ thống?",
            "Số giờ trên lớp của môn Phân tích và thiết kế hệ thống?"
        ],
        "Về Giảng viên (Instructors)": [
            "Giảng viên nào giảng dạy môn Phân tích và thiết kế hệ thống?",
            "Email của giảng viên Trần Thị Mỹ Diệp là gì?",
            "Danh sách giảng viên khoa Công nghệ thông tin"
        ],
        "Về Học phần tiên quyết (Prerequisites)": [
            "Các học phần tiên quyết của môn Phân tích và thiết kế hệ thống?",
            "Môn nào cần học trước khi học Phân tích và thiết kế hệ thống?",
            "Điều kiện tiên quyết để học môn này?"
        ],
        "Về Tài liệu (Materials)": [
            "Tài liệu tham khảo cho môn Phân tích và thiết kế hệ thống?",
            "Sách giáo trình nào được sử dụng?",
            "Phần mềm nào được dùng trong môn này?"
        ],
        "Về Mục tiêu & Chuẩn đầu ra (Objectives & Outcomes)": [
            "Mục tiêu của học phần Phân tích và thiết kế hệ thống?",
            "Chuẩn đầu ra của môn này?",
            "Sinh viên học môn này sẽ đạt được kỹ năng gì?"
        ],
        "Về Đánh giá (Assessment)": [
            "Cách thức đánh giá môn Phân tích và thiết kế hệ thống?",
            "Cơ cấu điểm của môn này?",
            "Tỷ lệ điểm chuyên cần là bao nhiêu?"
        ]
    }
    
    for category, questions in examples.items():
        print(f"\n{category}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print("\n💡 Tip: Copy and paste these questions to test the system!")
    print()

def adjust_settings(current_settings):
    """Adjust query settings interactively."""
    print("\n" + "=" * 80)
    print("⚙️  QUERY SETTINGS")
    print("=" * 80)
    print(f"\nCurrent settings:")
    print(f"  1. K-hop depth: {current_settings['k']}")
    print(f"  2. Top K seeds: {current_settings['top_k_seeds']}")
    print(f"  3. Max nodes: {current_settings['max_nodes']}")
    print(f"  4. Use embeddings: {current_settings['use_embeddings']}")
    print(f"  5. Verbose mode: {current_settings['verbose']}")
    print("\nEnter new values (or press Enter to keep current):")
    
    try:
        # K-hop
        k_input = input(f"K-hop depth (1-3) [{current_settings['k']}]: ").strip()
        if k_input:
            current_settings['k'] = max(1, min(3, int(k_input)))
        
        # Top K seeds
        seeds_input = input(f"Top K seeds (3-10) [{current_settings['top_k_seeds']}]: ").strip()
        if seeds_input:
            current_settings['top_k_seeds'] = max(3, min(10, int(seeds_input)))
        
        # Max nodes
        nodes_input = input(f"Max nodes (50-200) [{current_settings['max_nodes']}]: ").strip()
        if nodes_input:
            current_settings['max_nodes'] = max(50, min(200, int(nodes_input)))
        
        # Embeddings
        embed_input = input(f"Use embeddings (yes/no) [{'yes' if current_settings['use_embeddings'] else 'no'}]: ").strip()
        if embed_input:
            current_settings['use_embeddings'] = embed_input.lower() in ['yes', 'y', 'true', '1']
        
        # Verbose
        verbose_input = input(f"Verbose mode (yes/no) [{'yes' if current_settings['verbose'] else 'no'}]: ").strip()
        if verbose_input:
            current_settings['verbose'] = verbose_input.lower() in ['yes', 'y', 'true', '1']
        
        print("\n✅ Settings updated!")
        
    except ValueError:
        print("\n⚠️  Invalid input, keeping current settings")
    
    return current_settings

def show_help():
    """Show help information."""
    print("\n" + "=" * 80)
    print("📖 HELP")
    print("=" * 80)
    print("""
COMMANDS:
  - Type any question in Vietnamese to search the knowledge graph
  - 'examples' - Show example questions by category
  - 'settings' - Adjust query parameters
  - 'stats' - Show quick database statistics
  - 'help' - Show this help message
  - 'quit' or 'exit' - Exit the program

QUERY PARAMETERS:
  - K-hop depth (1-3): How deep to traverse the graph
    * k=1: Direct connections only
    * k=2: Friends of friends (recommended)
    * k=3: Wider network (may be slower)
  
  - Top K seeds (3-10): Number of starting entities
    * Lower: More focused results
    * Higher: More comprehensive but may include noise
  
  - Max nodes (50-200): Maximum nodes in subgraph
    * Lower: Faster, more focused
    * Higher: More complete but slower
  
  - Use embeddings (yes/no): Semantic search vs keyword search
    * yes: Better semantic matching (recommended)
    * no: Exact keyword matching only

TIPS:
  - Be specific in your questions
  - Use full names when possible (e.g., "Phân tích và thiết kế hệ thống")
  - For complex queries, increase k-hop depth and max_nodes
  - If results seem incomplete, try different seed counts
  - Use embeddings for better semantic understanding
    """)

def show_quick_stats(db):
    """Show quick database statistics."""
    print("\n" + "=" * 80)
    print("📊 QUICK STATISTICS")
    print("=" * 80)
    
    with db.get_session() as session:
        # Entity types
        types = session.run("""
            MATCH (e:Entity)
            RETURN e.type as type, count(*) as count
            ORDER BY count DESC
            LIMIT 5
        """).data()
        
        print("\nTop Entity Types:")
        for t in types:
            print(f"  • {t['type']}: {t['count']}")
        
        # Relationship types
        rels = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
            LIMIT 5
        """).data()
        
        print("\nTop Relationship Types:")
        for r in rels:
            print(f"  • {r['type']}: {r['count']}")
    
    print()

# =========================================================
# SINGLE QUERY MODE
# =========================================================

def single_query_mode(query_handler, question, settings=None):
    """Process a single question and exit."""
    
    if settings is None:
        settings = {
            'k': 2,
            'top_k_seeds': 5,
            'max_nodes': 80,
            'use_embeddings': True,
            'verbose': True
        }
    
    print("\n" + "=" * 80)
    print("🤖 GRAPHRAG SINGLE QUERY V3")
    print("=" * 80)
    print(f"\n❓ Question: {question}\n")
    
    ask_question(
        query_handler,
        question,
        **settings
    )

# =========================================================
# MAIN
# =========================================================

def main():
    """Main entry point."""
    
    # Initialize system
    db_connection, graph_manager, query_handler = initialize_query_system()
    
    if not query_handler:
        sys.exit(1)
    
    try:
        # Check if question provided as argument
        if len(sys.argv) > 1:
            # Single query mode
            question = " ".join(sys.argv[1:])
            single_query_mode(query_handler, question)
        else:
            # Interactive mode
            interactive_mode(query_handler)
    
    finally:
        # Cleanup
        if db_connection:
            db_connection.close()
            print("\n🔌 Connection closed")

if __name__ == "__main__":
    main()
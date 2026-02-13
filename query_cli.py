# query_cli.py
"""
CLI Tool for GraphRAG Query
Simple command-line interface to query the knowledge graph without Streamlit
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

from graph_database import GraphDatabaseConnection
from graph_manager import GraphManagerV2
from query_handler import QueryHandler
from logger import Logger

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()
logger = Logger("QueryCLI").get_logger()

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
            print("Please run 'python build_graph_v2.py' first to build the knowledge graph")
            db_connection.close()
            return None, None, None
        
        print(f"✅ Connected to knowledge graph")
        print(f"   Nodes: {stats['nodes']}")
        print(f"   Relationships: {stats['relationships']}")
        print()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize graph manager
        graph_manager = GraphManagerV2(
            db_connection=db_connection,
            auto_clear=False,
            openai_client=client
        )
        
        # Initialize query handler
        query_handler = QueryHandler(
            graph_manager=graph_manager,
            client=client,
            model=MODEL
        )
        
        return db_connection, graph_manager, query_handler
        
    except Exception as e:
        print(f"❌ Error connecting to graph: {e}")
        return None, None, None

# =========================================================
# QUERY INTERFACE
# =========================================================

def ask_question(query_handler, question, k=2, top_k_seeds=5, max_nodes=80, use_embeddings=True):
    """Ask a question and get answer."""
    try:
        print(f"\n🔍 Searching knowledge graph...")
        
        answer = query_handler.ask_question(
            query=question,
            method="khop",
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
    print("🤖 GRAPHRAG INTERACTIVE QUERY MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your question to search")
    print("  - 'examples' - Show example questions")
    print("  - 'settings' - Adjust query settings")
    print("  - 'quit' or 'exit' - Exit program")
    print()
    
    # Default settings
    settings = {
        'k': 2,
        'top_k_seeds': 5,
        'max_nodes': 80,
        'use_embeddings': True
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
            
            # Ask question
            ask_question(
                query_handler,
                question,
                k=settings['k'],
                top_k_seeds=settings['top_k_seeds'],
                max_nodes=settings['max_nodes'],
                use_embeddings=settings['use_embeddings']
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
    """Show example questions."""
    print("\n" + "=" * 80)
    print("📝 EXAMPLE QUESTIONS")
    print("=" * 80)
    
    examples = {
        "Về Giảng viên": [
            "Giảng viên nào giảng dạy môn Lập trình Java?",
            "Email của giảng viên Phạm Xuân Lâm là gì?",
            "Danh sách tất cả giảng viên có chức danh Tiến sĩ"
        ],
        "Về Học phần": [
            "Môn Lập trình Java có bao nhiêu tín chỉ?",
            "Mã học phần của Lập trình Java là gì?",
            "Các học phần tiên quyết của môn Lập trình Java?",
            "Số giờ trên lớp của môn Lập trình Java?"
        ],
        "Về Tài liệu": [
            "Tài liệu tham khảo cho môn Lập trình Java?",
            "Sách giáo trình nào được sử dụng?",
            "Phần mềm nào được dùng trong môn Lập trình Java?"
        ],
        "Về Mục tiêu": [
            "Mục tiêu của học phần Lập trình Java là gì?",
            "Chuẩn đầu ra của môn Lập trình Java?",
            "Sinh viên học môn này sẽ đạt được kỹ năng gì?"
        ]
    }
    
    for category, questions in examples.items():
        print(f"\n{category}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print()

def adjust_settings(current_settings):
    """Adjust query settings."""
    print("\n" + "=" * 80)
    print("⚙️  QUERY SETTINGS")
    print("=" * 80)
    print(f"\nCurrent settings:")
    print(f"  1. K-hop depth: {current_settings['k']}")
    print(f"  2. Top K seeds: {current_settings['top_k_seeds']}")
    print(f"  3. Max nodes: {current_settings['max_nodes']}")
    print(f"  4. Use embeddings: {current_settings['use_embeddings']}")
    print("\nEnter new values (or press Enter to keep current):")
    
    try:
        # K-hop
        k_input = input(f"K-hop depth (1-3) [{current_settings['k']}]: ").strip()
        if k_input:
            current_settings['k'] = int(k_input)
        
        # Top K seeds
        seeds_input = input(f"Top K seeds (3-10) [{current_settings['top_k_seeds']}]: ").strip()
        if seeds_input:
            current_settings['top_k_seeds'] = int(seeds_input)
        
        # Max nodes
        nodes_input = input(f"Max nodes (50-200) [{current_settings['max_nodes']}]: ").strip()
        if nodes_input:
            current_settings['max_nodes'] = int(nodes_input)
        
        # Embeddings
        embed_input = input(f"Use embeddings (yes/no) [{'yes' if current_settings['use_embeddings'] else 'no'}]: ").strip()
        if embed_input:
            current_settings['use_embeddings'] = embed_input.lower() in ['yes', 'y', 'true', '1']
        
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
Commands:
  - Type any question to search the knowledge graph
  - 'examples' - Show example questions
  - 'settings' - Adjust query parameters
  - 'help' - Show this help message
  - 'quit' or 'exit' - Exit the program

Query Parameters:
  - K-hop depth: How deep to traverse the graph (1-3)
  - Top K seeds: Number of starting entities (3-10)
  - Max nodes: Maximum nodes in subgraph (50-200)
  - Use embeddings: Vector similarity search (yes/no)

Tips:
  - Be specific in your questions
  - Use exact names when possible
  - Adjust settings for complex queries
    """)

# =========================================================
# SINGLE QUERY MODE
# =========================================================

def single_query_mode(query_handler, question):
    """Process a single question and exit."""
    print("\n" + "=" * 80)
    print("🤖 GRAPHRAG SINGLE QUERY")
    print("=" * 80)
    print(f"\n❓ Question: {question}")
    
    ask_question(query_handler, question)

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
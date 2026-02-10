# streamlit_app.py
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time

from document_processor import DocumentProcessor
from graph_database import GraphDatabaseConnection
from graph_manager import GraphManager
from query_handler import QueryHandler

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="GraphRAG Knowledge System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD ENVIRONMENT & INITIALIZE
# =========================================================
@st.cache_resource
def initialize_system():
    """Initialize all system components"""
    load_dotenv()
    
    # Get environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_URL = os.getenv("DB_URL")
    DB_USERNAME = os.getenv("DB_USERNAME", "neo4j")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    MODEL = os.getenv("MODEL", "gpt-4o-mini")
    
    # Validation
    if not OPENAI_API_KEY or not DB_URL or not DB_PASSWORD:
        st.error("⚠️ Missing environment variables. Please check .env file")
        st.stop()
    
    # Initialize components
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    db_connection = GraphDatabaseConnection(
        uri=DB_URL,
        user=DB_USERNAME,
        password=DB_PASSWORD
    )
    
    # Initialize GraphManager without clearing database
    graph_manager = GraphManager(
        db_connection=db_connection,
        auto_clear=False,
        openai_client=client  # Pass OpenAI client for embeddings
    )
    
    query_handler = QueryHandler(
        graph_manager=graph_manager,
        client=client,
        model="gpt-4o"
    )
    
    document_processor = DocumentProcessor(
        client=client,
        model=MODEL,
        max_workers=10
    )
    
    return {
        'client': client,
        'db_connection': db_connection,
        'graph_manager': graph_manager,
        'query_handler': query_handler,
        'document_processor': document_processor,
        'model': MODEL
    }

# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner("🔧 Initializing system..."):
        components = initialize_system()
        st.session_state.update(components)
        st.session_state.initialized = True
        st.session_state.chat_history = []
        st.session_state.k_hops = 2
        st.session_state.top_k_seeds = 5
        st.session_state.use_embeddings = True  # Enable embeddings by default

# =========================================================
# SIDEBAR - SETTINGS & STATS
# =========================================================
with st.sidebar:
    st.title("⚙️ GraphRAG Settings")
    
    # K-hop Configuration
    st.subheader("K-hop Retrieval Config")
    st.session_state.k_hops = st.slider(
        "Number of hops (k)",
        min_value=1,
        max_value=5,
        value=st.session_state.k_hops,
        help="How many relationship hops to traverse from seed entities"
    )
    
    st.session_state.top_k_seeds = st.slider(
        "Top-K seed entities",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k_seeds,
        help="Number of most relevant entities to use as starting points"
    )
    
    st.divider()
    
    # Embedding Configuration
    st.subheader("🔮 Semantic Search")
    st.session_state.use_embeddings = st.checkbox(
        "Use embeddings for entity search",
        value=st.session_state.use_embeddings,
        help="Use semantic similarity (embeddings) instead of keyword matching"
    )
    
    if st.session_state.use_embeddings:
        st.caption("✨ Semantic search enabled")
    else:
        st.caption("📝 Keyword search only")
    
    st.divider()
    
    # Graph Statistics
    st.subheader("📊 Graph Statistics")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        with st.spinner("Loading stats..."):
            stats = st.session_state.db_connection.get_database_stats()
            st.session_state.graph_stats = stats
    
    if 'graph_stats' in st.session_state:
        stats = st.session_state.graph_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", stats['nodes'])
        with col2:
            st.metric("Edges", stats['relationships'])
        
        if stats['labels']:
            st.caption(f"Labels: {', '.join(stats['labels'])}")
    
    st.divider()
    
    # Clear Chat History
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# =========================================================
# MAIN INTERFACE - TABS
# =========================================================
st.title("🧠 GraphRAG Knowledge System")
st.caption("AI-powered knowledge graph retrieval with K-hop traversal")

tab1, tab2, tab3 = st.tabs([
    "💬 Chat & Query",
    "🔍 Entity Search",
    "📈 Graph Explorer"
])

# =========================================================
# TAB 1: CHAT & QUERY
# =========================================================
with tab1:
    st.header("💬 Ask Questions")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            # Show context graph if available
            if 'context' in chat and chat['context']:
                with st.expander("🔗 View Knowledge Graph Context"):
                    st.code(chat['context'], language="text")
            
            # Show seed entities
            if 'seeds' in chat and chat['seeds']:
                with st.expander("🌱 Seed Entities Used"):
                    st.write(", ".join(chat['seeds']))
    
    # Query input
    query = st.chat_input("Type your question here...")
    
    if query:
        # Add user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Extract query terms for seed finding
                    query_handler = st.session_state.query_handler
                    graph_manager = st.session_state.graph_manager
                    
                    # Get seed entities
                    query_terms = query_handler._extract_query_terms(query)
                    seed_entities = graph_manager.find_relevant_entities(
                        query_terms=query_terms,
                        top_k=st.session_state.top_k_seeds,
                        use_embeddings=st.session_state.use_embeddings
                    )
                    
                    # Get K-hop subgraph
                    subgraph = graph_manager.get_k_hop_subgraph(
                        seed_entities=seed_entities,
                        k=st.session_state.k_hops,
                        max_nodes=80
                    )
                    
                    # Format context
                    context = graph_manager.format_subgraph_for_context(subgraph)
                    
                    # Get answer
                    answer = query_handler.ask_question(
                        query,
                        method='khop',
                        k=st.session_state.k_hops,
                        top_k_seeds=st.session_state.top_k_seeds
                    )
                    
                    # Display answer
                    st.write(answer)
                    
                    # Show context graph
                    with st.expander("🔗 View Knowledge Graph Context"):
                        st.code(context, language="text")
                        
                        # Display graph metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nodes", len(subgraph.get('nodes', [])))
                        with col2:
                            st.metric("Edges", len(subgraph.get('edges', [])))
                        with col3:
                            st.metric("Seeds", len(seed_entities))
                    
                    # Show seed entities
                    with st.expander("🌱 Seed Entities Used"):
                        st.write(", ".join(seed_entities))
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'context': context,
                        'seeds': seed_entities,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# =========================================================
# TAB 2: ENTITY SEARCH
# =========================================================
with tab2:
    st.header("🔍 Entity Search & Neighbors")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "Search for entities",
            placeholder="Enter search term...",
            help="Case-insensitive search in entity names"
        )
    
    with col2:
        search_limit = st.number_input(
            "Max results",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
    
    if search_term:
        with st.spinner("🔍 Searching..."):
            results = st.session_state.graph_manager.search_entities(
                search_term,
                limit=search_limit
            )
            
            if results:
                st.success(f"✅ Found {len(results)} entities")
                
                # Display results in expandable sections
                for entity in results:
                    with st.expander(f"📌 {entity}"):
                        # Get neighbors
                        neighbors = st.session_state.graph_manager.get_entity_neighbors(
                            entity,
                            max_depth=2
                        )
                        
                        if neighbors:
                            st.write(f"**Connected entities ({len(neighbors)}):**")
                            # Display as columns for better layout
                            num_cols = 3
                            cols = st.columns(num_cols)
                            for i, neighbor in enumerate(neighbors[:30]):
                                with cols[i % num_cols]:
                                    st.caption(f"• {neighbor}")
                            
                            if len(neighbors) > 30:
                                st.caption(f"... and {len(neighbors) - 30} more")
                        else:
                            st.info("No connected entities found")
            else:
                st.warning(f"No entities found matching '{search_term}'")
    
    st.divider()
    
    # Direct neighbor lookup
    st.subheader("🌐 Get Entity Neighbors")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        entity_name = st.text_input(
            "Entity name",
            placeholder="Enter exact entity name...",
            help="Case-insensitive entity name"
        )
    
    with col2:
        max_depth = st.selectbox(
            "Max depth",
            options=[1, 2, 3],
            index=1,
            help="Maximum relationship depth to traverse"
        )
    
    if st.button("🔗 Get Neighbors", use_container_width=True):
        if entity_name:
            with st.spinner("Fetching neighbors..."):
                neighbors = st.session_state.graph_manager.get_entity_neighbors(
                    entity_name,
                    max_depth=max_depth
                )
                
                if neighbors:
                    st.success(f"✅ Found {len(neighbors)} connected entities")
                    
                    # Display in grid
                    num_cols = 4
                    cols = st.columns(num_cols)
                    for i, neighbor in enumerate(neighbors):
                        with cols[i % num_cols]:
                            st.info(neighbor)
                else:
                    st.warning("No neighbors found")
        else:
            st.error("Please enter an entity name")

# =========================================================
# TAB 3: GRAPH EXPLORER
# =========================================================
with tab3:
    st.header("📈 Graph Analysis & Exploration")
    
    # Centrality Analysis
    st.subheader("🎯 Centrality Analysis")
    
    if st.button("📊 Calculate Centrality Measures", use_container_width=True):
        with st.spinner("Calculating centrality measures..."):
            try:
                centrality_data = st.session_state.graph_manager.calculate_centrality_measures()
                st.session_state.centrality_data = centrality_data
                st.success("✅ Centrality measures calculated")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if 'centrality_data' in st.session_state:
        data = st.session_state.centrality_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🔗 Degree Centrality**")
            st.caption("Most connected nodes")
            for record in data.get('degree', [])[:10]:
                st.metric(
                    record['entityName'],
                    f"{record['score']:.2f}"
                )
        
        with col2:
            st.write("**🌉 Betweenness Centrality**")
            st.caption("Important intermediaries")
            for record in data.get('betweenness', [])[:10]:
                st.metric(
                    record['entityName'],
                    f"{record['score']:.2f}"
                )
        
        with col3:
            st.write("**⭐ PageRank**")
            st.caption("Most influential nodes")
            for record in data.get('pagerank', [])[:10]:
                st.metric(
                    record['entityName'],
                    f"{record['score']:.4f}"
                )
    
    st.divider()
    
    # Graph Summary
    st.subheader("📋 Graph Summary")
    
    if st.button("📄 Generate Full Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                summary = st.session_state.graph_manager.get_graph_summary()
                st.session_state.graph_summary = summary
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if 'graph_summary' in st.session_state:
        summary = st.session_state.graph_summary
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Nodes", summary['nodes'])
        with col2:
            st.metric("Total Edges", summary['edges'])
        
        # Display centrality summary
        if 'centrality' in summary:
            centrality_text = st.session_state.graph_manager.summarize_centrality_measures(
                summary['centrality']
            )
            st.markdown(centrality_text)

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(f"🤖 Powered by GraphRAG • Model: {st.session_state.model} • K-hop Retrieval")
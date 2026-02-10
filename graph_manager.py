# graph_manager.py
from logger import Logger
import re
import unicodedata


class GraphManager:
    """
    Manages graph construction and analysis using Neo4j only.
    IMPROVED:
    - Vietnamese Unicode normalization
    - Better relationship name handling
    - Enhanced entity extraction from tables
    """
    
    logger = Logger('GraphManager').get_logger()

    def __init__(self, db_connection, auto_clear=True, openai_client=None):
        """
        Initialize graph manager with Neo4j connection.
        
        Args:
            db_connection: GraphDatabaseConnection instance (required)
            auto_clear: Whether to clear database on initialization (default: True)
            openai_client: OpenAI client for embeddings (optional)
        """
        if not db_connection:
            raise ValueError("Neo4j database connection is required")
        
        self.db_connection = db_connection
        self.openai_client = openai_client
        self.logger.info("GraphManager initialized with Neo4j backend")
        
        if auto_clear:
            self.logger.warning("Auto-clearing database...")
            self.db_connection.clear_database()
        else:
            self.logger.info("Database will NOT be cleared (auto_clear=False)")

    def build_graph_from_elements(self, elements):
        """
        Build graph from extracted elements directly in Neo4j.
        IMPROVED: Better handling of Vietnamese text and entity properties.
        
        Args:
            elements: List of extraction results with ENTITY and RELATION lines
            
        Returns:
            Dict with graph statistics
        """
        self.logger.info(f"Building graph from {len(elements)} element sets...")
        
        entity_count = 0
        relation_count = 0
        
        with self.db_connection.get_session() as session:
            for elem_idx, elem in enumerate(elements):
                lines = [l.strip() for l in elem.split("\n") if l.strip()]
                
                current_entity = None
                current_type = None
                current_props = {}
                
                for line in lines:
                    # Normalize Unicode
                    line = unicodedata.normalize('NFC', line)
                    
                    if line.startswith("ENTITY:"):
                        # Save previous entity if exists
                        if current_entity:
                            self._create_entity_with_properties(
                                session, current_entity, current_type, current_props
                            )
                            entity_count += 1
                        
                        # Start new entity
                        entity = line.replace("ENTITY:", "").strip()
                        if len(entity) > 2:
                            current_entity = self._normalize_entity_name(entity)
                            current_type = None
                            current_props = {}
                    
                    elif line.startswith("TYPE:") and current_entity:
                        entity_type = line.replace("TYPE:", "").strip()
                        current_type = entity_type
                    
                    elif line.startswith("PROPERTIES:") and current_entity:
                        props_text = line.replace("PROPERTIES:", "").strip()
                        # Simple parsing: "key1: value1, key2: value2"
                        for prop in props_text.split(','):
                            if ':' in prop:
                                key, val = prop.split(':', 1)
                                current_props[key.strip()] = val.strip()
                    
                    elif line.startswith("RELATION:"):
                        try:
                            _, rel = line.split(":", 1)
                            parts = [p.strip() for p in rel.split("->")]
                            if len(parts) == 3:
                                src, rel_type, tgt = parts
                                if len(src) > 2 and len(tgt) > 2:
                                    src_norm = self._normalize_entity_name(src)
                                    tgt_norm = self._normalize_entity_name(tgt)
                                    rel_clean = self._sanitize_relationship_name(rel_type)
                                    
                                    if rel_clean:
                                        session.run(
                                            f"MERGE (a:Entity {{name: $source}}) "
                                            f"MERGE (b:Entity {{name: $target}}) "
                                            f"MERGE (a)-[r:{rel_clean}]->(b)",
                                            source=src_norm,
                                            target=tgt_norm
                                        )
                                        relation_count += 1
                        except Exception as e:
                            self.logger.debug(f"Failed to parse relation: {line} - {e}")
                
                # Don't forget last entity
                if current_entity:
                    self._create_entity_with_properties(
                        session, current_entity, current_type, current_props
                    )
                    entity_count += 1
                
                if (elem_idx + 1) % 10 == 0:
                    self.logger.debug(f"Processed {elem_idx + 1}/{len(elements)} element sets")
        
        # Get actual counts from database
        stats = self.db_connection.get_database_stats()
        
        self.logger.info(f"Graph built: {stats['nodes']} nodes, {stats['relationships']} edges")
        
        # Add embeddings if OpenAI client is available
        if self.openai_client:
            self.logger.info("Adding embeddings to entities...")
            self.add_embeddings_to_entities()
        
        return {
            'nodes': stats['nodes'],
            'edges': stats['relationships']
        }

    def _create_entity_with_properties(self, session, entity_name, entity_type, properties):
        """
        Create or merge an entity with type and properties.
        
        Args:
            session: Neo4j session
            entity_name: Normalized entity name
            entity_type: Entity type (giảng_viên, học_phần, etc.)
            properties: Dict of additional properties
        """
        # Build Cypher query
        query = "MERGE (e:Entity {name: $name}) SET e.type = $type"
        params = {
            'name': entity_name,
            'type': entity_type if entity_type else 'unknown'
        }
        
        # Add properties
        for key, value in properties.items():
            # Sanitize property key
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()
            if safe_key and not safe_key[0].isdigit():
                query += f", e.{safe_key} = ${safe_key}"
                params[safe_key] = value
        
        session.run(query, params)

    def add_embeddings_to_entities(self):
        """
        Add vector embeddings to all entities in the graph.
        Uses OpenAI's text-embedding-3-small model.
        """
        if not self.openai_client:
            self.logger.warning("No OpenAI client available for embeddings")
            return
        
        with self.db_connection.get_session() as session:
            # Get all entities without embeddings
            entities = session.run("""
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.name AS name
            """).data()
            
            if not entities:
                self.logger.info("All entities already have embeddings")
                return
            
            self.logger.info(f"Creating embeddings for {len(entities)} entities...")
            
            # Batch process embeddings
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                names = [e['name'] for e in batch]
                
                try:
                    # Get embeddings from OpenAI
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=names
                    )
                    
                    # Update entities with embeddings
                    for j, embedding_obj in enumerate(response.data):
                        entity_name = names[j]
                        embedding = embedding_obj.embedding
                        
                        session.run("""
                            MATCH (e:Entity {name: $name})
                            SET e.embedding = $embedding
                        """, name=entity_name, embedding=embedding)
                    
                    self.logger.debug(f"Added embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}: {e}")
            
            self.logger.info("✓ Embeddings added to all entities")

    def _get_query_embedding(self, query_text):
        """Get embedding vector for query text."""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[query_text]
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to create query embedding: {e}")
            return None

    # =========================================================
    # K-HOP RETRIEVAL METHODS
    # =========================================================
    
    def find_relevant_entities(self, query_terms, top_k=10, use_embeddings=True):
        """
        Find entities relevant to query using hybrid search.
        IMPROVED: Better Vietnamese text matching.
        """
        self.logger.info(f"Finding entities matching: {query_terms}")
        
        entities = set()
        
        # Normalize query terms
        normalized_terms = [unicodedata.normalize('NFC', term.lower()) for term in query_terms]
        
        with self.db_connection.get_session() as session:
            # Semantic search with embeddings
            if use_embeddings and self.openai_client:
                query_text = " ".join(query_terms)
                query_embedding = self._get_query_embedding(query_text)
                
                if query_embedding:
                    try:
                        # Vector similarity search
                        result = session.run("""
                            MATCH (e:Entity)
                            WHERE e.embedding IS NOT NULL
                            WITH e, 
                                 reduce(dot = 0.0, i IN range(0, size(e.embedding)-1) | 
                                   dot + e.embedding[i] * $query_embedding[i]) as similarity
                            RETURN e.name as entity, similarity
                            ORDER BY similarity DESC
                            LIMIT $top_k
                        """, query_embedding=query_embedding, top_k=top_k).data()
                        
                        for record in result:
                            entities.add(record['entity'])
                        
                        self.logger.info(f"Found {len(entities)} entities via embeddings")
                    except Exception as e:
                        self.logger.warning(f"Embedding search failed: {e}")
            
            # Lexical search (fuzzy matching for Vietnamese)
            for term in normalized_terms:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $term
                       OR toLower(e.type) CONTAINS $term
                    RETURN DISTINCT e.name as entity
                    LIMIT $top_k
                """, term=term, top_k=top_k).data()
                
                for record in result:
                    entities.add(record['entity'])
        
        entities_list = list(entities)[:top_k]
        self.logger.info(f"Total relevant entities found: {len(entities_list)}")
        
        return entities_list

    def get_k_hop_subgraph(self, seed_entities, k=2, max_nodes=80):
        """
        Get k-hop subgraph starting from seed entities.
        """
        if not seed_entities:
            self.logger.warning("No seed entities provided")
            return {'nodes': [], 'edges': []}
        
        self.logger.info(f"Getting {k}-hop subgraph from {len(seed_entities)} seeds")
        
        with self.db_connection.get_session() as session:
            # Use APOC if available, otherwise standard Cypher
            try:
                result = session.run(f"""
                    MATCH (seed:Entity)
                    WHERE seed.name IN $seeds
                    CALL apoc.path.subgraphAll(seed, {{
                        maxLevel: $k,
                        limit: $max_nodes
                    }})
                    YIELD nodes, relationships
                    RETURN nodes, relationships
                """, seeds=seed_entities, k=k, max_nodes=max_nodes).single()
                
                if result:
                    nodes = [{'name': n['name'], 'type': n.get('type', 'unknown')} 
                            for n in result['nodes']]
                    edges = [{'source': r.start_node['name'], 
                             'target': r.end_node['name'],
                             'type': r.type} 
                            for r in result['relationships']]
                    
                    self.logger.info(f"Subgraph: {len(nodes)} nodes, {len(edges)} edges")
                    return {'nodes': nodes, 'edges': edges}
            
            except Exception as e:
                self.logger.warning(f"APOC not available, using standard Cypher: {e}")
            
            # Fallback: standard Cypher
            result = session.run(f"""
                MATCH (seed:Entity)
                WHERE seed.name IN $seeds
                MATCH path = (seed)-[*0..{k}]-(connected:Entity)
                WITH collect(DISTINCT connected) as nodes, collect(DISTINCT relationships(path)) as rels
                UNWIND rels as relList
                UNWIND relList as rel
                WITH nodes, collect(DISTINCT rel) as edges
                RETURN nodes[0..{max_nodes}] as nodes, edges
            """, seeds=seed_entities).single()
            
            if result:
                nodes = [{'name': n['name'], 'type': n.get('type', 'unknown')} 
                        for n in result['nodes']]
                edges = [{'source': r.start_node['name'],
                         'target': r.end_node['name'],
                         'type': r.type}
                        for r in result['edges']]
                
                self.logger.info(f"Subgraph: {len(nodes)} nodes, {len(edges)} edges")
                return {'nodes': nodes, 'edges': edges}
            
            return {'nodes': [], 'edges': []}

    def format_subgraph_for_context(self, subgraph):
        """Format subgraph as context for LLM."""
        if not subgraph or not subgraph.get('nodes'):
            return "No relevant data found."
        
        context_lines = []
        
        # Entities
        context_lines.append("=== ENTITIES ===")
        for node in subgraph['nodes']:
            entity_type = node.get('type', 'unknown')
            context_lines.append(f"- {node['name']} (type: {entity_type})")
        
        # Relationships
        if subgraph.get('edges'):
            context_lines.append("\n=== RELATIONSHIPS ===")
            for edge in subgraph['edges']:
                context_lines.append(
                    f"- {edge['source']} --[{edge['type']}]--> {edge['target']}"
                )
        
        return "\n".join(context_lines)

    # =========================================================
    # UTILITY METHODS
    # =========================================================
    
    def calculate_centrality_measures(self):
        """Calculate centrality measures using GDS if available."""
        try:
            return self._calculate_centrality_with_gds()
        except Exception as e:
            self.logger.warning(f"GDS not available: {e}")
            return self._calculate_centrality_fallback()
    
    def _calculate_centrality_with_gds(self):
        """Calculate centrality with Neo4j GDS plugin."""
        graph_name = 'temp_centrality_graph'
        
        with self.db_connection.get_session() as session:
            # Drop existing projection if exists
            self._drop_projection_if_exists(graph_name)
            
            # Create graph projection
            session.run(f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    'Entity',
                    {{
                        ALL: {{
                            type: '*',
                            orientation: 'UNDIRECTED'
                        }}
                    }}
                )
            """)
            
            # Degree centrality
            degree_result = session.run(f"""
                CALL gds.degree.stream('{graph_name}')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityName, score
                ORDER BY score DESC
                LIMIT 10
            """).data()
            
            # Betweenness centrality
            betweenness_result = session.run(f"""
                CALL gds.betweenness.stream('{graph_name}')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityName, score
                ORDER BY score DESC
                LIMIT 10
            """).data()
            
            # PageRank
            pagerank_result = session.run(f"""
                CALL gds.pageRank.stream('{graph_name}')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS entityName, score
                ORDER BY score DESC
                LIMIT 10
            """).data()
            
            # Clean up
            session.run(f"CALL gds.graph.drop('{graph_name}')")
            
            return {
                'degree': degree_result,
                'betweenness': betweenness_result,
                'pagerank': pagerank_result
            }
    
    def _calculate_centrality_fallback(self):
        """Calculate basic degree centrality without GDS."""
        with self.db_connection.get_session() as session:
            degree_result = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) as degree
                WHERE degree > 0
                RETURN e.name AS entityName, toFloat(degree) AS score
                ORDER BY degree DESC
                LIMIT 10
            """).data()
            
            return {
                'degree': degree_result,
                'betweenness': [],
                'pagerank': []
            }

    def summarize_centrality_measures(self, centrality_data):
        """Create a text summary of centrality measures."""
        summary = "### Centrality Measures Summary:\n\n"
        
        if centrality_data.get("degree"):
            summary += "#### Top Degree Centrality Nodes (most connected):\n"
            for record in centrality_data.get("degree", []):
                summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        else:
            summary += "#### Degree Centrality: No data available\n"
        
        if centrality_data.get("betweenness"):
            summary += "\n#### Top Betweenness Centrality Nodes (influential intermediaries):\n"
            for record in centrality_data.get("betweenness", []):
                summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        else:
            summary += "\n#### Betweenness Centrality: Requires Neo4j GDS plugin\n"
        
        if centrality_data.get("pagerank"):
            summary += "\n#### Top PageRank Nodes (most important):\n"
            for record in centrality_data.get("pagerank", []):
                summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        else:
            summary += "\n#### PageRank: Requires Neo4j GDS plugin\n"
        
        return summary

    def get_graph_summary(self):
        """Get comprehensive summary of the graph."""
        stats = self.db_connection.get_database_stats()
        centrality = self.calculate_centrality_measures()
        
        return {
            'nodes': stats['nodes'],
            'edges': stats['relationships'],
            'centrality': centrality
        }

    def get_entity_neighbors(self, entity_name, max_depth=2):
        """Get neighboring entities using Cypher."""
        normalized = self._normalize_entity_name(entity_name)
        
        with self.db_connection.get_session() as session:
            result = session.run(f"""
                MATCH (start:Entity {{name: $name}})
                MATCH path = (start)-[*1..{max_depth}]-(neighbor:Entity)
                RETURN DISTINCT neighbor.name AS neighborName
                LIMIT 50
            """, name=normalized).data()
            
            return [r['neighborName'] for r in result]

    def search_entities(self, search_term, limit=10):
        """Search for entities by name (case-insensitive)."""
        # Normalize search term
        search_term = unicodedata.normalize('NFC', search_term)
        
        with self.db_connection.get_session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                RETURN e.name AS entityName
                LIMIT $limit
            """, term=search_term, limit=limit).data()
            
            return [r['entityName'] for r in result]

    def _drop_projection_if_exists(self, graph_name):
        """Drop graph projection if it exists."""
        with self.db_connection.get_session() as session:
            try:
                exists = session.run(
                    "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                    name=graph_name
                ).single()["exists"]
                
                if exists:
                    session.run(
                        "CALL gds.graph.drop($name)",
                        name=graph_name
                    )
                    self.logger.debug(f"Dropped existing projection: {graph_name}")
            except Exception:
                pass

    def _normalize_entity_name(self, name):
        """
        Normalize entity name for consistency.
        IMPROVED: Preserve Vietnamese diacritics.
        """
        # Normalize Unicode to NFC form (important for Vietnamese)
        name = unicodedata.normalize('NFC', name)
        # Strip and lowercase
        name = name.strip().lower()
        return name

    def _sanitize_relationship_name(self, name):
        """
        Sanitize relationship name for Neo4j.
        IMPROVED: Better handling of Vietnamese text.
        """
        # Normalize Unicode first
        name = unicodedata.normalize('NFC', name)
        name = name.strip()
        
        # For Vietnamese relationships, keep the diacritics
        # Replace spaces and special chars with underscore
        sanitized = re.sub(r'[\s\-\.,;:!?(){}[\]]+', '_', name)
        
        # Remove any remaining invalid characters (keep Vietnamese)
        sanitized = re.sub(r'[^\w]', '', sanitized, flags=re.UNICODE)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "RELATED_TO"
        
        # Ensure it doesn't start with a number
        if sanitized[0].isdigit():
            sanitized = "REL_" + sanitized
        
        # Convert to uppercase
        return sanitized.upper()
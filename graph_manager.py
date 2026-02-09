from logger import Logger
import re


class GraphManager:
    """
    Manages graph construction and analysis using Neo4j only.
    No NetworkX dependencies - all operations use Cypher queries.
    """
    
    logger = Logger('GraphManager').get_logger()

    def __init__(self, db_connection):
        """
        Initialize graph manager with Neo4j connection.
        
        Args:
            db_connection: GraphDatabaseConnection instance (required)
        """
        if not db_connection:
            raise ValueError("Neo4j database connection is required")
        
        self.db_connection = db_connection
        self.logger.info("GraphManager initialized with Neo4j backend")
        
        # Clear database on initialization
        self.db_connection.clear_database()

    def build_graph_from_elements(self, elements):
        """
        Build graph from extracted elements directly in Neo4j.
        
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
                
                for line in lines:
                    if line.startswith("ENTITY:"):
                        entity = line.replace("ENTITY:", "").strip()
                        if len(entity) > 2:
                            normalized = self._normalize_entity_name(entity)
                            session.run(
                                "MERGE (e:Entity {name: $name})",
                                name=normalized
                            )
                            entity_count += 1
                    
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
                
                if (elem_idx + 1) % 10 == 0:
                    self.logger.debug(f"Processed {elem_idx + 1}/{len(elements)} element sets")
        
        # Get actual counts from database
        stats = self.db_connection.get_database_stats()
        
        self.logger.info(f"Graph built: {stats['nodes']} nodes, {stats['relationships']} edges")
        
        return {
            'nodes': stats['nodes'],
            'edges': stats['relationships']
        }

    def detect_communities(self, min_community_size=3):
        """
        Detect communities using Neo4j GDS Leiden algorithm.
        
        Args:
            min_community_size: Minimum size to consider a community
            
        Returns:
            List of communities (each is a list of entity names)
        """
        self.logger.info("Detecting communities using Neo4j GDS...")
        
        graph_name = "communityGraph"
        
        try:
            # Drop existing projection if exists
            self._drop_projection_if_exists(graph_name)
            
            # Create graph projection
            with self.db_connection.get_session() as session:
                # Project the graph
                session.run(f"""
                    CALL gds.graph.project(
                        '{graph_name}',
                        'Entity',
                        {{
                            REL: {{
                                type: '*',
                                orientation: 'UNDIRECTED'
                            }}
                        }}
                    )
                """)
                
                self.logger.debug("Graph projected for community detection")
                
                # Run Leiden algorithm
                result = session.run(f"""
                    CALL gds.leiden.stream('{graph_name}')
                    YIELD nodeId, communityId
                    RETURN gds.util.asNode(nodeId).name AS entityName, communityId
                    ORDER BY communityId
                """).data()
                
                # Group by community
                communities_dict = {}
                for record in result:
                    comm_id = record['communityId']
                    entity = record['entityName']
                    if comm_id not in communities_dict:
                        communities_dict[comm_id] = []
                    communities_dict[comm_id].append(entity)
                
                # Filter by minimum size
                communities = [
                    comm for comm in communities_dict.values()
                    if len(comm) >= min_community_size
                ]
                
                # If no large communities, keep all
                if not communities:
                    communities = list(communities_dict.values())
                
                # Clean up projection
                session.run(f"CALL gds.graph.drop('{graph_name}')")
                
                self.logger.info(f"Detected {len(communities)} communities")
                return communities
                
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            # Fallback: return connected components
            return self._get_connected_components(min_community_size)

    def _get_connected_components(self, min_size=3):
        """
        Fallback method: get connected components as communities.
        
        Args:
            min_size: Minimum component size
            
        Returns:
            List of components (each is a list of entity names)
        """
        self.logger.info("Using connected components as fallback...")
        
        with self.db_connection.get_session() as session:
            # Get all nodes and their relationships
            result = session.run("""
                MATCH (n:Entity)
                OPTIONAL MATCH path = (n)-[*]-(m:Entity)
                WITH n, collect(DISTINCT m.name) + n.name AS component
                RETURN DISTINCT component
            """).data()
            
            components = [r['component'] for r in result if len(r['component']) >= min_size]
            
            if not components:
                # Get all components regardless of size
                components = [r['component'] for r in result]
            
            self.logger.info(f"Found {len(components)} components")
            return components

    def calculate_centrality_measures(self):
        """
        Calculate centrality measures using Neo4j GDS algorithms.
        
        Returns:
            Dict with degree, betweenness, and closeness centrality
        """
        self.logger.info("Calculating centrality measures...")
        
        graph_name = "centralityGraph"
        
        try:
            # Drop existing projection
            self._drop_projection_if_exists(graph_name)
            
            with self.db_connection.get_session() as session:
                # Project graph
                session.run(f"""
                    CALL gds.graph.project(
                        '{graph_name}',
                        'Entity',
                        {{
                            REL: {{
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
                
                # PageRank (as alternative to closeness for disconnected graphs)
                pagerank_result = session.run(f"""
                    CALL gds.pageRank.stream('{graph_name}')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).name AS entityName, score
                    ORDER BY score DESC
                    LIMIT 10
                """).data()
                
                # Clean up
                session.run(f"CALL gds.graph.drop('{graph_name}')")
                
                centrality_data = {
                    'degree': degree_result,
                    'betweenness': betweenness_result,
                    'pagerank': pagerank_result
                }
                
                self.logger.info("Centrality measures calculated")
                return centrality_data
                
        except Exception as e:
            self.logger.error(f"Centrality calculation failed: {e}")
            return {'degree': [], 'betweenness': [], 'pagerank': []}

    def summarize_centrality_measures(self, centrality_data):
        """
        Create a text summary of centrality measures.
        
        Args:
            centrality_data: Dict from calculate_centrality_measures()
            
        Returns:
            Formatted summary string
        """
        summary = "### Centrality Measures Summary:\n\n"
        
        summary += "#### Top Degree Centrality Nodes (most connected):\n"
        for record in centrality_data.get("degree", []):
            summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        
        summary += "\n#### Top Betweenness Centrality Nodes (influential intermediaries):\n"
        for record in centrality_data.get("betweenness", []):
            summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        
        summary += "\n#### Top PageRank Nodes (most important):\n"
        for record in centrality_data.get("pagerank", []):
            summary += f" - {record['entityName']} with score {record['score']:.4f}\n"
        
        return summary

    def get_graph_summary(self):
        """
        Get comprehensive summary of the graph.
        
        Returns:
            Dict with graph statistics and top communities
        """
        stats = self.db_connection.get_database_stats()
        communities = self.detect_communities()
        centrality = self.calculate_centrality_measures()
        
        # Sort communities by size
        sorted_communities = sorted(communities, key=len, reverse=True)
        
        return {
            'nodes': stats['nodes'],
            'edges': stats['relationships'],
            'communities': len(communities),
            'top_communities': sorted_communities[:5],
            'centrality': centrality
        }

    def get_entity_neighbors(self, entity_name, max_depth=2):
        """
        Get neighboring entities using Cypher.
        
        Args:
            entity_name: Name of the entity
            max_depth: Maximum relationship depth
            
        Returns:
            List of neighboring entity names
        """
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
        """
        Search for entities by name (case-insensitive).
        
        Args:
            search_term: Search term
            limit: Maximum results
            
        Returns:
            List of matching entity names
        """
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

    def _normalize_entity_name(self, name):
        """Normalize entity name for consistency."""
        return name.strip().lower()

    def _sanitize_relationship_name(self, name):
        """Sanitize relationship name for Neo4j (must be valid Cypher identifier)."""
        # Replace non-alphanumeric with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "RELATED_TO"
        # Ensure it doesn't start with a number
        if sanitized[0].isdigit():
            sanitized = "REL_" + sanitized
        return sanitized.upper()
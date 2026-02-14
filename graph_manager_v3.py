# graph_manager_v3.py
"""
ENHANCED Graph Manager V3
- Better entity type handling
- Improved property extraction
- Enhanced search capabilities
- Support for Vietnamese text normalization
"""

from graph_database import GraphDatabaseConnection
from openai import OpenAI
from logger import Logger
from typing import List, Dict, Any, Optional
import unicodedata
import re


class GraphManagerV3:
    """
    Enhanced Graph Manager with:
    - Type-based entity labels
    - Rich property support
    - Advanced search with embeddings
    - Vietnamese text handling
    """
    
    logger = Logger("GraphManagerV3").get_logger()
    
    def __init__(
        self,
        db_connection: GraphDatabaseConnection,
        auto_clear: bool = False,
        openai_client: Optional[OpenAI] = None
    ):
        """Initialize graph manager."""
        self.db = db_connection
        self.client = openai_client
        
        if auto_clear:
            self.db.clear_database()
            self.logger.info("Database cleared")
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes and constraints."""
        self.logger.info("Creating indexes...")
        
        with self.db.get_session() as session:
            # Unique constraint on entity names (case-insensitive)
            session.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name_normalized IS UNIQUE
            """)
            
            # Index on entity type
            session.run("""
                CREATE INDEX entity_type_idx IF NOT EXISTS
                FOR (e:Entity) ON (e.type)
            """)
            
            # Fulltext index on entity names
            session.run("""
                CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name, e.name_normalized]
            """)
    
    # =========================================================
    # GRAPH BUILDING
    # =========================================================
    
    def build_graph_from_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Build knowledge graph from extracted elements.
        
        Args:
            elements: List of {entities: [...], relations: [...]}
            
        Returns:
            Statistics dict with counts
        """
        self.logger.info(f"Building graph from {len(elements)} element sets...")
        
        total_entities = 0
        total_relations = 0
        
        for i, element_set in enumerate(elements, 1):
            try:
                # Extract entities and relations
                entities = element_set.get('entities', [])
                relations = element_set.get('relations', [])
                chunk_meta = element_set.get('chunk_metadata', {})
                
                # Create entities
                for entity in entities:
                    self._create_entity(entity, chunk_meta)
                    total_entities += 1
                
                # Create relations
                for relation in relations:
                    self._create_relation(relation, chunk_meta)
                    total_relations += 1
                
                if i % 50 == 0:
                    self.logger.info(f"  Processed {i}/{len(elements)} element sets")
                    
            except Exception as e:
                self.logger.error(f"Error processing element set {i}: {e}")
        
        # Generate embeddings if client available
        if self.client:
            self.logger.info("Generating embeddings for entities...")
            self._generate_embeddings()
        
        stats = self.db.get_database_stats()
        self.logger.info(f"Graph built: {stats['nodes']} nodes, {stats['relationships']} edges")
        
        return {
            'nodes': stats['nodes'],
            'edges': stats['relationships'],
            'entities_created': total_entities,
            'relations_created': total_relations
        }
    
    def _create_entity(
        self,
        entity: Dict[str, Any],
        chunk_meta: Dict[str, Any]
    ):
        """Create or update an entity node with properties."""
        
        name = entity.get('name', '').strip()
        if not name:
            return
        
        # Normalize name for matching
        name_normalized = self._normalize_text(name)
        
        entity_type = entity.get('type', 'unknown').lower()
        properties = entity.get('properties', {})
        
        # Build property dictionary
        props = {
            'name': name,
            'name_normalized': name_normalized,
            'type': entity_type,
            'source_file': chunk_meta.get('source_file', 'unknown'),
            'chunk_type': chunk_meta.get('chunk_type', 'unknown')
        }
        
        # Add custom properties
        props.update(properties)
        
        # Clean properties (remove None/empty values)
        props = {k: v for k, v in props.items() if v is not None and str(v).strip()}
        
        with self.db.get_session() as session:
            # MERGE on normalized name to avoid duplicates
            session.run("""
                MERGE (e:Entity {name_normalized: $name_normalized})
                ON CREATE SET e = $props
                ON MATCH SET e += $props
            """, name_normalized=name_normalized, props=props)
    
    def _create_relation(
        self,
        relation: Dict[str, Any],
        chunk_meta: Dict[str, Any]
    ):
        """Create relationship between entities."""
        
        source = relation.get('source', '').strip()
        target = relation.get('target', '').strip()
        rel_type = relation.get('type', 'RELATED').upper()
        
        if not source or not target:
            return
        
        # Normalize names
        source_norm = self._normalize_text(source)
        target_norm = self._normalize_text(target)
        
        # Get properties
        properties = relation.get('properties', {})
        
        # Build property dict
        props = {
            'source_file': chunk_meta.get('source_file', 'unknown'),
            'weight': properties.get('weight', 1.0),
            'description': properties.get('description', '')
        }
        
        # Add custom properties
        props.update({k: v for k, v in properties.items() if k not in ['weight', 'description']})
        
        # Clean properties
        props = {k: v for k, v in props.items() if v is not None and str(v).strip()}
        
        # Sanitize relationship type (Neo4j constraint)
        rel_type = self._sanitize_rel_type(rel_type)
        
        with self.db.get_session() as session:
            # Use normalized names for matching
            query = f"""
                MATCH (a:Entity {{name_normalized: $source_norm}})
                MATCH (b:Entity {{name_normalized: $target_norm}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r = $props
                ON MATCH SET r += $props
            """
            
            session.run(
                query,
                source_norm=source_norm,
                target_norm=target_norm,
                props=props
            )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text for matching.
        - Unicode normalization (NFC)
        - Lowercase
        - Remove extra whitespace
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _sanitize_rel_type(self, rel_type: str) -> str:
        """
        Sanitize relationship type for Neo4j.
        - Replace spaces and special chars with underscore
        - Uppercase
        """
        # Remove special characters, keep Vietnamese letters
        rel_type = re.sub(r'[^\w\s]', '', rel_type)
        
        # Replace spaces with underscore
        rel_type = rel_type.replace(' ', '_')
        
        # Uppercase
        rel_type = rel_type.upper()
        
        return rel_type or 'RELATED'
    
    # =========================================================
    # EMBEDDINGS
    # =========================================================
    
    def _generate_embeddings(self):
        """Generate embeddings for all entities."""
        
        if not self.client:
            self.logger.warning("No OpenAI client, skipping embeddings")
            return
        
        with self.db.get_session() as session:
            # Get entities without embeddings
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.name as name, e.type as type, id(e) as id
            """)
            
            entities = result.data()
        
        if not entities:
            self.logger.info("All entities already have embeddings")
            return
        
        self.logger.info(f"Generating embeddings for {len(entities)} entities...")
        
        # Batch processing
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            
            try:
                # Create embedding texts
                texts = [
                    f"{e['type']}: {e['name']}"
                    for e in batch
                ]
                
                # Generate embeddings
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                
                # Update database
                with self.db.get_session() as session:
                    for entity, embedding_data in zip(batch, response.data):
                        embedding = embedding_data.embedding
                        
                        session.run("""
                            MATCH (e:Entity)
                            WHERE id(e) = $id
                            SET e.embedding = $embedding
                        """, id=entity['id'], embedding=embedding)
                
                self.logger.info(f"  Generated embeddings: {i + len(batch)}/{len(entities)}")
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i}: {e}")
        
        self.logger.info("✓ Embedding generation complete")
    
    # =========================================================
    # ENTITY SEARCH
    # =========================================================
    
    def find_relevant_entities(
        self,
        query_terms: List[str],
        top_k: int = 5,
        use_embeddings: bool = True
    ) -> List[str]:
        """
        Find relevant entities based on query terms.
        
        Args:
            query_terms: List of search terms
            top_k: Number of entities to return
            use_embeddings: Use semantic search with embeddings
            
        Returns:
            List of entity names
        """
        if use_embeddings and self.client:
            return self._find_entities_with_embeddings(query_terms, top_k)
        else:
            return self._find_entities_with_fulltext(query_terms, top_k)
    
    def _find_entities_with_embeddings(
        self,
        query_terms: List[str],
        top_k: int
    ) -> List[str]:
        """Find entities using embedding similarity."""
        
        # Create query text
        query_text = ' '.join(query_terms)
        
        try:
            # Generate query embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[query_text]
            )
            
            query_embedding = response.data[0].embedding
            
            # Find similar entities
            with self.db.get_session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    WITH e, 
                         gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN e.name as name, similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                """, query_embedding=query_embedding, top_k=top_k)
                
                entities = [record['name'] for record in result]
            
            if entities:
                self.logger.info(f"Found {len(entities)} entities via embeddings")
                return entities
            
        except Exception as e:
            self.logger.error(f"Embedding search error: {e}")
        
        # Fallback to fulltext
        self.logger.info("Falling back to fulltext search")
        return self._find_entities_with_fulltext(query_terms, top_k)
    
    def _find_entities_with_fulltext(
        self,
        query_terms: List[str],
        top_k: int
    ) -> List[str]:
        """Find entities using fulltext search."""
        
        # Build fulltext query
        query_text = ' OR '.join(query_terms)
        
        with self.db.get_session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes('entity_name_ft', $query)
                YIELD node, score
                RETURN node.name as name, score
                ORDER BY score DESC
                LIMIT $top_k
            """, query=query_text, top_k=top_k)
            
            entities = [record['name'] for record in result]
        
        self.logger.info(f"Found {len(entities)} entities via fulltext")
        return entities
    
    def search_entities(
        self,
        term: str,
        limit: int = 10
    ) -> List[str]:
        """Simple entity search by term."""
        
        term_normalized = self._normalize_text(term)
        
        with self.db.get_session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name_normalized CONTAINS $term
                RETURN e.name as name
                LIMIT $limit
            """, term=term_normalized, limit=limit)
            
            return [record['name'] for record in result]
    
    # =========================================================
    # K-HOP SUBGRAPH RETRIEVAL
    # =========================================================
    
    def get_k_hop_subgraph(
        self,
        seed_entities: List[str],
        k: int = 2,
        max_nodes: int = 80
    ) -> Dict[str, Any]:
        """
        Get k-hop subgraph around seed entities.
        
        Args:
            seed_entities: Starting entities
            k: Number of hops
            max_nodes: Maximum nodes to return
            
        Returns:
            Subgraph dict with nodes and edges
        """
        if not seed_entities:
            return {'nodes': [], 'edges': []}
        
        # Normalize seed entity names
        seeds_normalized = [self._normalize_text(name) for name in seed_entities]
        
        with self.db.get_session() as session:
            # Get k-hop subgraph
            result = session.run("""
                MATCH (start:Entity)
                WHERE start.name_normalized IN $seeds
                CALL apoc.path.subgraphAll(start, {
                    maxLevel: $k,
                    limit: $max_nodes
                })
                YIELD nodes, relationships
                RETURN nodes, relationships
            """, seeds=seeds_normalized, k=k, max_nodes=max_nodes)
            
            record = result.single()
            
            if not record:
                return {'nodes': [], 'edges': []}
            
            # Extract nodes
            nodes = []
            for node in record['nodes']:
                node_dict = dict(node)
                node_dict['id'] = node.id
                nodes.append(node_dict)
            
            # Extract relationships
            edges = []
            for rel in record['relationships']:
                edge_dict = dict(rel)
                edge_dict['id'] = rel.id
                edge_dict['source'] = rel.start_node.id
                edge_dict['target'] = rel.end_node.id
                edge_dict['type'] = rel.type
                edges.append(edge_dict)
            
            self.logger.info(f"Retrieved {len(nodes)} nodes, {len(edges)} edges")
            
            return {
                'nodes': nodes,
                'edges': edges
            }
    
    # =========================================================
    # CONTEXT FORMATTING
    # =========================================================
    
    def format_subgraph_for_context(
        self,
        subgraph: Dict[str, Any]
    ) -> str:
        """
        Format subgraph as text context for LLM.
        
        Args:
            subgraph: Subgraph dict from get_k_hop_subgraph
            
        Returns:
            Formatted text context
        """
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        if not nodes:
            return "No relevant information found."
        
        # Group nodes by type
        nodes_by_type = {}
        node_map = {}  # id -> node
        
        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
            node_map[node['id']] = node
        
        # Build context
        lines = ["=== KNOWLEDGE GRAPH CONTEXT ===\n"]
        
        # Section 1: Entities by type
        lines.append("ENTITIES:")
        for entity_type, entity_nodes in sorted(nodes_by_type.items()):
            lines.append(f"\n{entity_type.upper()}:")
            for node in entity_nodes:
                name = node.get('name', 'unknown')
                lines.append(f"  - {name}")
                
                # Add important properties
                for key in ['mã_học_phần', 'số_tín_chỉ', 'email', 'chức_danh']:
                    if key in node and node[key]:
                        lines.append(f"    + {key}: {node[key]}")
        
        # Section 2: Relationships
        if edges:
            lines.append("\n\nRELATIONSHIPS:")
            for edge in edges:
                source_node = node_map.get(edge['source'])
                target_node = node_map.get(edge['target'])
                
                if source_node and target_node:
                    source_name = source_node.get('name', 'unknown')
                    target_name = target_node.get('name', 'unknown')
                    rel_type = edge.get('type', 'RELATED')
                    
                    line = f"  - {source_name} --[{rel_type}]--> {target_name}"
                    
                    # Add description if available
                    if 'description' in edge and edge['description']:
                        line += f" ({edge['description']})"
                    
                    lines.append(line)
        
        return '\n'.join(lines)
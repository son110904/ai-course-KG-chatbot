# graph_manager_v2.py
"""
ENHANCED VERSION: Better entity property handling and metadata extraction
Key improvements:
1. Parse METADATA from LLM extraction
2. Better entity property management
3. Enhanced relationship properties
4. Improved entity deduplication with properties
"""

from logger import Logger
import re
import unicodedata


class GraphManagerV2:
    """
    Enhanced graph manager with better property handling.
    """
    
    logger = Logger('GraphManagerV2').get_logger()

    def __init__(self, db_connection, auto_clear=False, openai_client=None):
        """
        Initialize graph manager with Neo4j connection.
        
        Args:
            db_connection: GraphDatabaseConnection instance (required)
            auto_clear: Whether to clear database on initialization (default: False)
            openai_client: OpenAI client for embeddings (optional)
        """
        if not db_connection:
            raise ValueError("Neo4j database connection is required")
        
        self.db_connection = db_connection
        self.openai_client = openai_client
        self.logger.info("GraphManagerV2 initialized with Neo4j backend")
        
        # Check if database has data
        stats = self.db_connection.get_database_stats()
        has_data = stats['nodes'] > 0 or stats['relationships'] > 0
        
        if has_data:
            self.logger.warning(f"Database contains {stats['nodes']} nodes and {stats['relationships']} relationships")
            
            if auto_clear:
                self.logger.warning("Auto-clearing database...")
                self.db_connection.clear_database()
            else:
                self.logger.info("Database will NOT be cleared (auto_clear=False)")
        else:
            self.logger.info("Database is empty, ready for new data")

    # =========================================================
    # ENHANCED ENTITY EXTRACTION WITH METADATA
    # =========================================================

    def parse_entity_block(self, lines: list, start_idx: int) -> dict:
        """
        Parse entity block with TYPE and METADATA support.
        
        Returns:
        {
            'name': str,
            'type': str,
            'properties': dict,
            'next_idx': int
        }
        """
        entity_info = {
            'name': None,
            'type': 'unknown',
            'properties': {},
            'next_idx': start_idx + 1
        }
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("ENTITY:"):
                entity_info['name'] = line.replace("ENTITY:", "").strip()
                i += 1
                continue
            
            elif line.startswith("TYPE:"):
                entity_info['type'] = line.replace("TYPE:", "").strip()
                i += 1
                continue
            
            elif line.startswith("METADATA:"):
                # Parse metadata: "key1=value1, key2=value2"
                metadata_str = line.replace("METADATA:", "").strip()
                props = self._parse_metadata_string(metadata_str)
                entity_info['properties'].update(props)
                i += 1
                continue
            
            elif line.startswith("PROPERTIES:"):
                # Alternative format
                props_str = line.replace("PROPERTIES:", "").strip()
                props = self._parse_metadata_string(props_str)
                entity_info['properties'].update(props)
                i += 1
                continue
            
            elif line.startswith("ENTITY:") or line.startswith("RELATION:") or not line:
                # Next entity or relation or empty line
                entity_info['next_idx'] = i
                break
            
            else:
                # Unknown line, skip
                i += 1
                continue
        
        return entity_info

    def _parse_metadata_string(self, metadata_str: str) -> dict:
        """
        Parse metadata string into dict.
        
        Examples:
        "email=lampx@neu.edu.vn, chức_danh=TS" -> {'email': 'lampx@neu.edu.vn', 'chức_danh': 'TS'}
        "số_tín_chỉ=3, mã_học_phần=CNTT1153" -> {'số_tín_chỉ': '3', 'mã_học_phần': 'CNTT1153'}
        """
        properties = {}
        
        # Split by comma
        parts = metadata_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Sanitize key for Neo4j
                safe_key = self._sanitize_property_key(key)
                if safe_key:
                    properties[safe_key] = value
        
        return properties

    def _sanitize_property_key(self, key: str) -> str:
        """
        Sanitize property key for Neo4j.
        Neo4j property keys can't have spaces or special chars.
        """
        # Replace spaces and dashes with underscore
        key = re.sub(r'[\s\-]+', '_', key)
        
        # Remove invalid characters
        key = re.sub(r'[^a-zA-Z0-9_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', key, flags=re.UNICODE)
        
        # Ensure it doesn't start with a number
        if key and key[0].isdigit():
            key = '_' + key
        
        return key.lower()

    # =========================================================
    # BUILD GRAPH WITH ENHANCED PARSING
    # =========================================================

    def build_graph_from_elements(self, elements):
        """
        Build graph from extracted elements with enhanced metadata support.
        
        Args:
            elements: List of extraction results
            
        Returns:
            Dict with graph statistics
        """
        self.logger.info(f"Building graph from {len(elements)} element sets...")
        
        entity_count = 0
        relation_count = 0
        
        with self.db_connection.get_session() as session:
            for elem_idx, elem in enumerate(elements):
                lines = [l.strip() for l in elem.split("\n") if l.strip()]
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    
                    # Normalize Unicode
                    line = unicodedata.normalize('NFC', line)
                    
                    if line.startswith("ENTITY:"):
                        # Parse entity block
                        entity_info = self.parse_entity_block(lines, i)
                        
                        if entity_info['name'] and len(entity_info['name']) > 2:
                            normalized_name = self._normalize_entity_name(entity_info['name'])
                            
                            # Create entity with properties
                            self._create_entity_with_type_and_properties(
                                session,
                                normalized_name,
                                entity_info['type'],
                                entity_info['properties']
                            )
                            entity_count += 1
                        
                        i = entity_info['next_idx']
                    
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
                        
                        i += 1
                    
                    else:
                        i += 1
                
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

    def _create_entity_with_type_and_properties(
        self,
        session,
        entity_name: str,
        entity_type: str,
        properties: dict
    ):
        """
        Create or merge an entity with type and properties.
        
        Improvements:
        1. Set entity type as both property AND label
        2. Handle Vietnamese property names
        3. Better property type handling
        """
        # Build label from entity type
        type_label = self._type_to_label(entity_type)
        
        # Build Cypher query with multiple labels
        query = f"MERGE (e:Entity:{type_label} {{name: $name}}) SET e.type = $type"
        params = {
            'name': entity_name,
            'type': entity_type
        }
        
        # Add properties
        for key, value in properties.items():
            safe_key = self._sanitize_property_key(key)
            if safe_key and not safe_key[0].isdigit():
                query += f", e.{safe_key} = ${safe_key}"
                params[safe_key] = value
        
        try:
            session.run(query, params)
        except Exception as e:
            self.logger.warning(f"Error creating entity {entity_name}: {e}")
            # Fallback: create without custom label
            query_fallback = "MERGE (e:Entity {name: $name}) SET e.type = $type"
            session.run(query_fallback, params)

    def _type_to_label(self, entity_type: str) -> str:
        """
        Convert entity type to Neo4j label.
        
        Examples:
        "học_phần" -> "HocPhan"
        "giảng_viên" -> "GiangVien"
        "tài_liệu" -> "TaiLieu"
        """
        if not entity_type or entity_type == 'unknown':
            return 'Unknown'
        
        # Remove underscores and capitalize words
        words = entity_type.split('_')
        label = ''.join(word.capitalize() for word in words)
        
        # Remove any remaining invalid characters
        label = re.sub(r'[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', label, flags=re.UNICODE)
        
        return label if label else 'Unknown'

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    def add_embeddings_to_entities(self):
        """Add vector embeddings to all entities in the graph."""
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
            
            self.logger.info(f"Adding embeddings to {len(entities)} entities...")
            
            # Batch processing
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
                        embedding = embedding_obj.embedding
                        name = names[j]
                        
                        session.run("""
                            MATCH (e:Entity {name: $name})
                            SET e.embedding = $embedding
                        """, name=name, embedding=embedding)
                    
                    self.logger.info(f"Processed {min(i + batch_size, len(entities))}/{len(entities)} entities")
                    
                except Exception as e:
                    self.logger.error(f"Error adding embeddings: {e}")

    # =========================================================
    # SEARCH & RETRIEVAL
    # =========================================================

    def find_relevant_entities(self, query_terms: list, top_k: int = 5, use_embeddings: bool = True):
        """Find relevant entities based on query terms."""
        if use_embeddings and self.openai_client:
            return self._find_entities_by_embedding(query_terms, top_k)
        else:
            return self._find_entities_by_keyword(query_terms, top_k)

    def _find_entities_by_embedding(self, query_terms: list, top_k: int):
        """Find entities using vector similarity."""
        query_text = " ".join(query_terms)
        
        try:
            # Get query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[query_text]
            )
            query_embedding = response.data[0].embedding
            
            # Find similar entities
            with self.db_connection.get_session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN e.name AS name, similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                """, query_embedding=query_embedding, top_k=top_k).data()
                
                entities = [r['name'] for r in result]
                
                if entities:
                    self.logger.info(f"Found {len(entities)} entities by embedding")
                    return entities
                
        except Exception as e:
            self.logger.warning(f"Embedding search failed: {e}")
        
        # Fallback to keyword search
        return self._find_entities_by_keyword(query_terms, top_k)

    def _find_entities_by_keyword(self, query_terms: list, top_k: int):
        """Find entities using keyword matching."""
        entities = set()
        
        with self.db_connection.get_session() as session:
            for term in query_terms:
                term = unicodedata.normalize('NFC', term)
                
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($term)
                    RETURN e.name AS name
                    LIMIT $limit
                """, term=term, limit=top_k).data()
                
                entities.update([r['name'] for r in result])
        
        entities = list(entities)[:top_k]
        self.logger.info(f"Found {len(entities)} entities by keyword")
        return entities

    # =========================================================
    # K-HOP SUBGRAPH
    # =========================================================

    def get_k_hop_subgraph(self, seed_entities: list, k: int = 2, max_nodes: int = 100):
        """
        Get k-hop subgraph around seed entities.
        
        Returns:
        {
            'nodes': [{'name': str, 'type': str, 'properties': dict}],
            'edges': [{'source': str, 'target': str, 'type': str}]
        }
        """
        with self.db_connection.get_session() as session:
            # Build seed entity filter
            seed_filter = " OR ".join([f"start.name = '{e}'" for e in seed_entities])
            
            # Get subgraph
            result = session.run(f"""
                MATCH (start:Entity)
                WHERE {seed_filter}
                CALL {{
                    WITH start
                    MATCH path = (start)-[*1..{k}]-(neighbor:Entity)
                    RETURN path
                    LIMIT {max_nodes}
                }}
                WITH collect(path) AS paths
                UNWIND paths AS path
                UNWIND nodes(path) AS node
                WITH collect(DISTINCT node) AS nodes
                UNWIND nodes AS n
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE m IN nodes
                RETURN 
                    collect(DISTINCT {{
                        name: n.name,
                        type: n.type,
                        properties: properties(n)
                    }}) AS nodes,
                    collect(DISTINCT {{
                        source: startNode(r).name,
                        target: endNode(r).name,
                        type: type(r)
                    }}) AS edges
            """).single()
            
            if result:
                return {
                    'nodes': result['nodes'],
                    'edges': [e for e in result['edges'] if e['source'] and e['target']]
                }
            
            return {'nodes': [], 'edges': []}

    def format_subgraph_for_context(self, subgraph: dict) -> str:
        """Format subgraph as text context for LLM."""
        lines = []
        
        # Nodes
        lines.append("ENTITIES:")
        for node in subgraph.get('nodes', []):
            name = node.get('name', 'unknown')
            node_type = node.get('type', 'unknown')
            props = node.get('properties', {})
            
            line = f"- {name} (type: {node_type})"
            
            # Add important properties
            for key, value in props.items():
                if key not in ['name', 'type', 'embedding']:
                    line += f", {key}: {value}"
            
            lines.append(line)
        
        lines.append("\nRELATIONSHIPS:")
        for edge in subgraph.get('edges', []):
            source = edge.get('source', '?')
            target = edge.get('target', '?')
            rel_type = edge.get('type', 'RELATED_TO')
            
            lines.append(f"- {source} --[{rel_type}]--> {target}")
        
        return "\n".join(lines)

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    def search_entities(self, search_term: str, limit: int = 10):
        """Search for entities by name (case-insensitive)."""
        search_term = unicodedata.normalize('NFC', search_term)
        
        with self.db_connection.get_session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                RETURN e.name AS entityName
                LIMIT $limit
            """, term=search_term, limit=limit).data()
            
            return [r['entityName'] for r in result]

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistency."""
        name = unicodedata.normalize('NFC', name)
        name = name.strip().lower()
        return name

    def _sanitize_relationship_name(self, name: str) -> str:
        """Sanitize relationship name for Neo4j."""
        name = unicodedata.normalize('NFC', name)
        name = name.strip()
        
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
        
        return sanitized.upper()
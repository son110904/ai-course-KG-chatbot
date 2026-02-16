# database_entity_linker.py
"""
Database Entity Linking Module
Link new entities with existing entities already in Neo4j database
"""

from typing import List, Dict, Any, Tuple, Optional
from entity_linker_v3 import EntityLinker
from graph_database import GraphDatabaseConnection
from logger import Logger


class DatabaseEntityLinker:
    """
    Link new entities with existing entities in database
    
    Features:
    - Fuzzy match with existing entities
    - Merge properties from both sources
    - Track entity provenance
    - Avoid duplicates in database
    """
    
    def __init__(self, entity_linker: EntityLinker):
        """
        Initialize DatabaseEntityLinker
        
        Args:
            entity_linker: EntityLinker instance (reuses matching logic)
        """
        self.entity_linker = entity_linker
        self.logger = Logger("DatabaseEntityLinker").get_logger()
    
    def link_with_existing(
        self,
        new_entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Link new entities with existing ones in database
        
        Args:
            new_entities: Canonical entities from current extraction
            existing_entities: Entities already in database
            
        Returns:
            Tuple of:
            - merged_entities: Final list of entities to ingest
            - mapping: Dict mapping new_entity_name -> final_canonical_name
        """
        self.logger.info(f"Linking {len(new_entities)} new entities with {len(existing_entities)} existing...")
        
        # Index existing entities by type for faster lookup
        existing_by_type = {}
        for entity in existing_entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in existing_by_type:
                existing_by_type[entity_type] = []
            existing_by_type[entity_type].append(entity)
        
        self.logger.info(f"  Existing entity types: {list(existing_by_type.keys())}")
        
        # Process each new entity
        final_entities = []
        mapping = {}
        merge_count = 0
        new_count = 0
        
        for new_entity in new_entities:
            entity_type = new_entity['type']
            existing_of_type = existing_by_type.get(entity_type, [])
            
            # Try to find match in existing entities
            match = self._find_match(new_entity, existing_of_type)
            
            if match:
                # Merge with existing entity
                merged = self._merge_entities(new_entity, match)
                final_entities.append(merged)
                
                # Map new name to existing name
                mapping[new_entity['name']] = match['name']
                merge_count += 1
                
                self.logger.debug(f"  Merged: '{new_entity['name']}' → '{match['name']}'")
            else:
                # New entity, add as is
                final_entities.append(new_entity)
                mapping[new_entity['name']] = new_entity['name']
                new_count += 1
        
        self.logger.info(f"✓ Linked with database:")
        self.logger.info(f"  - Merged with existing: {merge_count}")
        self.logger.info(f"  - New entities: {new_count}")
        self.logger.info(f"  - Total final entities: {len(final_entities)}")
        
        return final_entities, mapping
    
    def _find_match(
        self,
        new_entity: Dict[str, Any],
        existing_entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find matching entity in existing entities
        
        Args:
            new_entity: Entity to match
            existing_entities: List of existing entities of same type
            
        Returns:
            Matching existing entity or None
        """
        for existing in existing_entities:
            # Reuse entity linking logic
            if self.entity_linker._should_link(new_entity, existing):
                return existing
        return None
    
    def _merge_entities(
        self,
        new_entity: Dict[str, Any],
        existing_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge new entity into existing entity
        
        Strategy:
        - Keep existing entity's name (canonical in DB)
        - Merge properties (prefer non-empty values)
        - Track all variants
        - Update source count
        
        Args:
            new_entity: New entity from extraction
            existing_entity: Existing entity from database
            
        Returns:
            Merged entity
        """
        # Start with existing entity
        merged = existing_entity.copy()
        
        # Merge properties
        new_props = new_entity.get('properties', {})
        existing_props = merged.get('properties', {})
        
        for key, value in new_props.items():
            # Skip empty values
            if not value or (isinstance(value, str) and not value.strip()):
                continue
            
            if key not in existing_props or not existing_props[key]:
                # New property or existing is empty
                existing_props[key] = value
            elif existing_props[key] != value:
                # Conflict: keep both values as list
                if not isinstance(existing_props[key], list):
                    existing_props[key] = [existing_props[key]]
                if value not in existing_props[key]:
                    existing_props[key].append(value)
        
        merged['properties'] = existing_props
        
        # Update variants
        variants = merged.get('variants', [merged['name']])
        if not isinstance(variants, list):
            variants = [variants]
        
        # Add new entity name and its variants
        if new_entity['name'] not in variants:
            variants.append(new_entity['name'])
        
        for variant in new_entity.get('variants', []):
            if variant not in variants:
                variants.append(variant)
        
        merged['variants'] = variants
        
        # Update source count
        existing_count = merged.get('source_count', 1)
        new_count = new_entity.get('source_count', 1)
        merged['source_count'] = existing_count + new_count
        
        return merged


def get_existing_entities_from_db(db: GraphDatabaseConnection) -> List[Dict[str, Any]]:
    """
    Retrieve all existing entities from database
    
    Args:
        db: Database connection
        
    Returns:
        List of entities currently in database
    """
    logger = Logger("get_existing_entities_from_db").get_logger()
    
    with db.get_session() as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN 
                e.name as name,
                e.type as type,
                e.name_normalized as name_normalized,
                properties(e) as properties
        """)
        
        entities = []
        for record in result:
            # Extract properties
            props = dict(record['properties'])
            
            # Remove system properties
            props.pop('name', None)
            props.pop('type', None)
            props.pop('name_normalized', None)
            
            entity = {
                'name': record['name'],
                'type': record['type'],
                'name_normalized': record['name_normalized'],
                'properties': props
            }
            entities.append(entity)
        
        logger.info(f"Retrieved {len(entities)} existing entities from database")
        
        # Log entity type distribution
        if entities:
            type_counts = {}
            for entity in entities:
                entity_type = entity['type']
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            logger.info(f"  Entity types in DB: {type_counts}")
        
        return entities


def merge_duplicate_relations_in_db(db: GraphDatabaseConnection):
    """
    Merge duplicate relations in database after entity linking
    
    This should be called after entities have been linked and merged.
    It will find and merge duplicate relationships.
    
    Args:
        db: Database connection
    """
    logger = Logger("merge_duplicate_relations").get_logger()
    
    logger.info("Merging duplicate relations in database...")
    
    with db.get_session() as session:
        # Find duplicate relations
        result = session.run("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WITH a, b, type(r) as relType, collect(r) as rels
            WHERE size(rels) > 1
            RETURN count(*) as duplicate_count
        """)
        
        duplicate_count = result.single()['duplicate_count']
        
        if duplicate_count == 0:
            logger.info("  No duplicate relations found")
            return
        
        logger.info(f"  Found {duplicate_count} sets of duplicate relations")
        
        # Merge duplicates
        session.run("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WITH a, b, type(r) as relType, collect(r) as rels
            WHERE size(rels) > 1
            
            // Keep first relation, collect info from others
            WITH a, b, relType, rels[0] as keep, rels[1..] as remove
            
            // Merge properties
            WITH a, b, relType, keep, remove,
                 [r in remove | r.weight] as weights,
                 [r in remove | r.description] as descriptions
            
            // Update kept relation
            SET keep.weight = CASE 
                WHEN keep.weight IS NULL AND size(weights) > 0 THEN weights[0]
                WHEN size(weights) > 0 THEN keep.weight + reduce(s = 0, w IN weights | s + w)
                ELSE keep.weight
            END,
            keep.mention_count = size(remove) + 1
            
            // Delete duplicates
            FOREACH (r IN remove | DELETE r)
        """)
        
        logger.info(f"✓ Merged duplicate relations")


def update_entity_statistics(db: GraphDatabaseConnection):
    """
    Update entity statistics after entity linking
    
    This calculates:
    - Degree (number of connections)
    - Importance score based on connections and mentions
    
    Args:
        db: Database connection
    """
    logger = Logger("update_entity_statistics").get_logger()
    
    logger.info("Updating entity statistics...")
    
    with db.get_session() as session:
        # Update degree (number of connections)
        session.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as degree
            SET e.degree = degree
        """)
        
        # Update importance score
        # Score = source_count * log(1 + degree)
        session.run("""
            MATCH (e:Entity)
            WITH e, 
                 COALESCE(e.source_count, 1) as source_count,
                 COALESCE(e.degree, 0) as degree
            SET e.importance = source_count * log(1 + degree)
        """)
        
        logger.info("✓ Updated entity statistics")


# Convenience function for full database linking workflow
def link_with_database(
    db: GraphDatabaseConnection,
    new_entities: List[Dict[str, Any]],
    entity_linker: EntityLinker
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Complete workflow for linking new entities with database
    
    Steps:
    1. Get existing entities from DB
    2. Link new entities with existing
    3. Return merged entities and mapping
    
    Args:
        db: Database connection
        new_entities: New canonical entities from extraction
        entity_linker: EntityLinker instance
        
    Returns:
        Tuple of (final_entities, entity_mapping)
    """
    logger = Logger("link_with_database").get_logger()
    
    # Get existing entities
    existing_entities = get_existing_entities_from_db(db)
    
    if not existing_entities:
        logger.info("No existing entities in database, using all new entities")
        mapping = {e['name']: e['name'] for e in new_entities}
        return new_entities, mapping
    
    # Link with existing
    db_linker = DatabaseEntityLinker(entity_linker)
    final_entities, mapping = db_linker.link_with_existing(
        new_entities,
        existing_entities
    )
    
    return final_entities, mapping
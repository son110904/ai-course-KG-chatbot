# entity_linker.py
"""
Entity Linking Module
Deduplicate và normalize entities từ LLM extraction
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import unicodedata
import re
from rapidfuzz import fuzz
from logger import Logger


class EntityLinker:
    """
    Entity Linking component để deduplicate và normalize entities
    
    Features:
    - Multi-level matching (exact, abbreviation, fuzzy, semantic)
    - Type-aware clustering
    - Property-based disambiguation
    - Graph-based clustering algorithm
    """
    
    def __init__(self, abbreviation_dict: Dict[str, str] = None, fuzzy_threshold: int = 85):
        """
        Initialize EntityLinker
        
        Args:
            abbreviation_dict: Custom abbreviation mapping
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.logger = Logger("EntityLinker").get_logger()
        self.abbreviations = abbreviation_dict or self._default_abbreviations()
        self.fuzzy_threshold = fuzzy_threshold
        
        # Mapping: variant_name -> canonical_name
        self.canonical_mapping = {}
        
        # Mapping: canonical_name -> merged_properties
        self.canonical_entities = {}
    
    def _default_abbreviations(self) -> Dict[str, str]:
        """
        Default abbreviation dictionary for Vietnamese educational terms
        """
        return {
            # Học phần
            "pttkht": "phân tích thiết kế hệ thống",
            "pttk": "phân tích thiết kế",
            "csdl": "cơ sở dữ liệu",
            "httt": "hệ thống thông tin",
            "cntt": "công nghệ thông tin",
            "khmt": "khoa học máy tính",
            "ktmt": "kiến trúc máy tính",
            "mmtdl": "mạng máy tính và truyền dữ liệu",
            "hdh": "hệ điều hành",
            "ctdl": "cấu trúc dữ liệu",
            "gt": "giải thuật",
            "oop": "hướng đối tượng",
            "lthdt": "lập trình hướng đối tượng",
            "tkđt": "thiết kế đồ họa",
            "xldl": "xử lý dữ liệu",
            "ptdl": "phân tích dữ liệu",
            "ai": "trí tuệ nhân tạo",
            "ml": "máy học",
            "dl": "học sâu",
            "nlp": "xử lý ngôn ngữ tự nhiên",
            
            # Các từ viết tắt thông dụng
            "ht": "hệ thống",
            "tt": "thông tin",
            "pt": "phân tích",
            "tk": "thiết kế",
            "lt": "lập trình",
            "qt": "quản trị",
            "ud": "ứng dụng",
            "cb": "cơ bản",
            "nc": "nâng cao",
            "tt": "thực tế",
            "th": "thực hành",
        }
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize entity name for matching
        
        Steps:
        1. Unicode normalization (NFC)
        2. Lowercase
        3. Remove extra whitespace
        4. Expand abbreviations
        5. Remove special characters (optional)
        
        Args:
            name: Original entity name
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Unicode normalization (important for Vietnamese)
        name = unicodedata.normalize('NFC', name)
        
        # Lowercase
        name = name.lower()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Expand abbreviations
        words = name.split()
        expanded_words = []
        for word in words:
            # Check if word is abbreviation
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                expanded_words.append(word)
        name = ' '.join(expanded_words)
        
        # Remove special characters (keep Vietnamese diacritics)
        # name = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', name)
        
        return name
    
    def link_entities(
        self, 
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Link và deduplicate entities
        
        Args:
            entities: List of entities from LLM extraction
            Each entity: {
                'name': str,
                'type': str,
                'properties': dict
            }
            
        Returns:
            Tuple of:
            - canonical_entities: List of deduplicated entities
            - entity_mapping: Dict mapping original_name -> canonical_name
        """
        self.logger.info(f"Starting entity linking for {len(entities)} entities...")
        
        # Group entities by type (important for type-aware matching)
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            entities_by_type[entity_type].append(entity)
        
        self.logger.info(f"  Entity types found: {list(entities_by_type.keys())}")
        
        # Process each type separately
        canonical_entities = []
        entity_mapping = {}
        
        for entity_type, type_entities in entities_by_type.items():
            self.logger.info(f"  Processing {len(type_entities)} entities of type '{entity_type}'")
            
            # Build similarity graph and cluster
            clusters = self._cluster_entities(type_entities)
            
            self.logger.info(f"    Found {len(clusters)} clusters")
            
            # Create canonical entity for each cluster
            for i, cluster in enumerate(clusters, 1):
                canonical = self._merge_cluster(cluster)
                canonical_entities.append(canonical)
                
                # Map all variants to canonical name
                canonical_name = canonical['name']
                for entity in cluster:
                    original_name = entity['name']
                    entity_mapping[original_name] = canonical_name
                    
                    if len(cluster) > 1:
                        self.logger.debug(f"    Cluster {i}: '{original_name}' → '{canonical_name}'")
        
        reduction_pct = 100 * len(canonical_entities) / len(entities) if entities else 0
        self.logger.info(f"✓ Linked to {len(canonical_entities)} canonical entities (reduction: {100-reduction_pct:.1f}%)")
        
        return canonical_entities, entity_mapping
    
    def _cluster_entities(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster similar entities using graph-based approach
        
        Algorithm:
        1. Build similarity graph (nodes = entities, edges = similar pairs)
        2. Find connected components (BFS)
        3. Each component = one cluster
        
        Args:
            entities: List of entities of the same type
            
        Returns:
            List of clusters (each cluster is a list of similar entities)
        """
        n = len(entities)
        
        # Build adjacency list for similarity graph
        graph = defaultdict(set)
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_link(entities[i], entities[j]):
                    graph[i].add(j)
                    graph[j].add(i)
        
        # Find connected components using BFS
        visited = set()
        clusters = []
        
        for i in range(n):
            if i not in visited:
                # BFS to find cluster
                cluster_indices = set()
                queue = [i]
                
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    
                    visited.add(node)
                    cluster_indices.add(node)
                    
                    # Add neighbors to queue
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                # Convert indices to entities
                cluster = [entities[idx] for idx in sorted(cluster_indices)]
                clusters.append(cluster)
        
        return clusters
    
    def _should_link(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """
        Determine if two entities should be linked
        
        Matching strategy:
        1. Exact match after normalization
        2. Fuzzy string matching
        3. Property-based disambiguation
        
        Args:
            entity1, entity2: Entities to compare
            
        Returns:
            True if entities should be linked
        """
        # Must be same type
        if entity1['type'] != entity2['type']:
            return False
        
        name1 = entity1['name']
        name2 = entity2['name']
        
        # Normalize
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        # Strategy 1: Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Strategy 2: Fuzzy match
        # Use multiple fuzzy algorithms and take max
        ratio = fuzz.ratio(norm1, norm2)
        token_sort = fuzz.token_sort_ratio(norm1, norm2)
        token_set = fuzz.token_set_ratio(norm1, norm2)
        
        max_similarity = max(ratio, token_sort, token_set)
        
        if max_similarity >= self.fuzzy_threshold:
            # Additional check: properties disambiguation
            conflict = self._check_properties_conflict(entity1, entity2)
            if conflict:
                # Properties conflict -> different entities despite similar names
                return False
            return True
        
        return False
    
    def _check_properties_conflict(
        self, 
        entity1: Dict[str, Any], 
        entity2: Dict[str, Any]) -> bool:
        """ Check if properties conflict (indicating different entities) 
        Args: entity1, entity2: Entities to check Returns: True if properties conflict (different entities) 
        False if no conflict (same entity or uncertain) 
        """
        props1 = entity1.get('properties') or {}
        props2 = entity2.get('properties') or {}

        entity_type = entity1.get('type')

        # Type-specific key properties that must match
        if entity_type == 'học_phần':
            # Mã học phần must match if both exist
            ma1_raw = props1.get('mã_học_phần', '')
            ma2_raw = props2.get('mã_học_phần', '')
            
            # Handle None values safely
            if ma1_raw is None:
                ma1_raw = ''
            if ma2_raw is None:
                ma2_raw = ''
            
            ma1 = str(ma1_raw).strip()
            ma2 = str(ma2_raw).strip()
            
            if ma1 and ma2 and ma1 != ma2:
                return True  # Conflict

        elif entity_type == 'giảng_viên':
            # Email must match if both exist
            email1_raw = props1.get('email', '')
            email2_raw = props2.get('email', '')
            
            # Handle None values safely
            if email1_raw is None:
                email1_raw = ''
            if email2_raw is None:
                email2_raw = ''
            
            email1 = str(email1_raw).strip().lower()
            email2 = str(email2_raw).strip().lower()
            
            if email1 and email2 and email1 != email2:
                return True  # Conflict

        elif entity_type == 'chương_trình':
            # Mã chương trình must match
            ma1_raw = props1.get('mã_chương_trình', '')
            ma2_raw = props2.get('mã_chương_trình', '')
            
            # Handle None values safely
            if ma1_raw is None:
                ma1_raw = ''
            if ma2_raw is None:
                ma2_raw = ''
            
            ma1 = str(ma1_raw).strip()
            ma2 = str(ma2_raw).strip()
            
            if ma1 and ma2 and ma1 != ma2:
                return True  # Conflict

        # No conflict detected
        return False

    
    def _merge_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge entities in cluster into canonical entity
        
        Strategy:
        - Choose canonical name (longest or most informative)
        - Merge all properties
        - Track all variants
        - Count source mentions
        
        Args:
            cluster: List of similar entities
            
        Returns:
            Canonical merged entity
        """
        if len(cluster) == 1:
            # Single entity, just return it
            entity = cluster[0].copy()
            entity['variants'] = [entity['name']]
            entity['source_count'] = 1
            return entity
        
        # Choose canonical name
        # Strategy: longest name (usually most complete)
        names = [e['name'] for e in cluster]
        canonical_name = max(names, key=len)
        
        # Merge properties from all entities
        merged_props = {}
        
        for entity in cluster:
            props = entity.get('properties', {})
            for key, value in props.items():
                if not value or (isinstance(value, str) and not value.strip()):
                    continue
                    
                if key not in merged_props:
                    # First time seeing this property
                    merged_props[key] = value
                elif merged_props[key] != value:
                    # Conflict: keep both as list
                    if not isinstance(merged_props[key], list):
                        merged_props[key] = [merged_props[key]]
                    if value not in merged_props[key]:
                        merged_props[key].append(value)
        
        # Create canonical entity
        canonical = {
            'name': canonical_name,
            'type': cluster[0]['type'],
            'properties': merged_props,
            'variants': names,  # Keep track of all name variants
            'source_count': len(cluster)  # How many extraction mentions
        }
        
        return canonical


class RelationLinker:
    """
    Update relations to use canonical entity names
    """
    
    def __init__(self, entity_mapping: Dict[str, str]):
        """
        Initialize RelationLinker
        
        Args:
            entity_mapping: Dict mapping original_name -> canonical_name
        """
        self.entity_mapping = entity_mapping
        self.logger = Logger("RelationLinker").get_logger()
    
    def update_relations(
        self, 
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update relation endpoints to canonical names and deduplicate
        
        Args:
            relations: List of relations with original entity names
            Each relation: {
                'source': str,
                'target': str,
                'type': str,
                'properties': dict
            }
            
        Returns:
            Updated and deduplicated relations
        """
        self.logger.info(f"Updating {len(relations)} relations...")
        
        updated_relations = []
        
        for relation in relations:
            source = relation['source']
            target = relation['target']
            
            # Map to canonical names
            canonical_source = self.entity_mapping.get(source, source)
            canonical_target = self.entity_mapping.get(target, target)
            
            # Create updated relation
            updated_relation = relation.copy()
            updated_relation['source'] = canonical_source
            updated_relation['target'] = canonical_target
            
            updated_relations.append(updated_relation)
        
        # Deduplicate relations
        deduplicated = self._deduplicate_relations(updated_relations)
        
        reduction_pct = 100 * len(deduplicated) / len(relations) if relations else 0
        self.logger.info(f"✓ Updated to {len(deduplicated)} unique relations (reduction: {100-reduction_pct:.1f}%)")
        
        return deduplicated
    
    def _deduplicate_relations(
        self, 
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        relation_groups = defaultdict(list)
        
        for relation in relations:
            key = (
                relation['source'],
                relation['type'],
                relation['target']
            )
            relation_groups[key].append(relation)
        
        merged_relations = []
        
        for key, group in relation_groups.items():
            if len(group) == 1:
                merged_relations.append(group[0])
            else:
                merged = group[0].copy()
                
                descriptions = []
                weights = []
                
                for rel in group:
                    props = rel.get('properties', {}) or {}
                    
                    desc = (props.get('description') or "").strip()
                    if desc:
                        descriptions.append(desc)
                    
                    weight = props.get('weight')
                    if weight is not None:
                        try:
                            weights.append(float(weight))
                        except:
                            pass
                
                merged_props = merged.get('properties', {}) or {}
                
                if descriptions:
                    unique_descriptions = list(dict.fromkeys(descriptions))
                    merged_props['description'] = '; '.join(unique_descriptions)
                
                if weights:
                    merged_props["weight"] = max(weights)
                    merged_props['mention_count'] = len(group)
                else:
                    merged_props["weight"] = 1.0
                    merged_props['mention_count'] = len(group)
                
                merged['properties'] = merged_props
                merged_relations.append(merged)
        
        return merged_relations



# Convenience function
def link_extracted_elements(
    elements: List[Dict[str, Any]],
    abbreviation_dict: Dict[str, str] = None,
    fuzzy_threshold: int = 85
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to link entities and relations from extracted elements
    
    Args:
        elements: List of extraction results from LLM
        abbreviation_dict: Optional custom abbreviations
        fuzzy_threshold: Fuzzy matching threshold
        
    Returns:
        Tuple of (canonical_entities, canonical_relations, stats)
    """
    logger = Logger("link_extracted_elements").get_logger()
    
    # Collect all entities and relations
    all_entities = []
    all_relations = []
    
    for element_set in elements:
        all_entities.extend(element_set.get('entities', []))
        all_relations.extend(element_set.get('relations', []))
    
    logger.info(f"Collected {len(all_entities)} entities, {len(all_relations)} relations")
    
    # Link entities
    entity_linker = EntityLinker(
        abbreviation_dict=abbreviation_dict,
        fuzzy_threshold=fuzzy_threshold
    )
    canonical_entities, entity_mapping = entity_linker.link_entities(all_entities)
    
    # Update relations
    relation_linker = RelationLinker(entity_mapping)
    canonical_relations = relation_linker.update_relations(all_relations)
    
    # Statistics
    stats = {
        'original_entities': len(all_entities),
        'canonical_entities': len(canonical_entities),
        'entity_reduction_pct': 100 * (1 - len(canonical_entities) / len(all_entities)) if all_entities else 0,
        'original_relations': len(all_relations),
        'canonical_relations': len(canonical_relations),
        'relation_reduction_pct': 100 * (1 - len(canonical_relations) / len(all_relations)) if all_relations else 0,
    }
    
    logger.info(f"Entity linking complete:")
    logger.info(f"  Entities: {stats['original_entities']} → {stats['canonical_entities']} (-{stats['entity_reduction_pct']:.1f}%)")
    logger.info(f"  Relations: {stats['original_relations']} → {stats['canonical_relations']} (-{stats['relation_reduction_pct']:.1f}%)")
    
    return canonical_entities, canonical_relations, stats
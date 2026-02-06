"""
Graph builder để merge nhiều graphs và summarize descriptions.
"""

import logging
from typing import List
import uuid

import networkx as nx
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builder để merge nhiều graphs thành một graph duy nhất.
    Xử lý duplicate entities và relationships.
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize graph builder
        
        Args:
            llm: Language model for summarization (optional)
        """
        self.llm = llm
        
        if llm:
            from .prompts import DESCRIPTION_SUMMARIZATION_PROMPT
            self.summarization_prompt = PromptTemplate.from_template(
                DESCRIPTION_SUMMARIZATION_PROMPT
            )
    
    def _normalize_node_name(self, name: str) -> str:
        """Normalize node name"""
        return name.upper().strip()
    
    def _merge_node_attributes(
        self,
        target_attrs: dict,
        source_attrs: dict,
    ) -> None:
        """
        Merge attributes from source to target node
        
        Args:
            target_attrs: Target node attributes (modified in place)
            source_attrs: Source node attributes
        """
        # Merge chunk_ids
        target_chunks = set(target_attrs.get('chunk_ids', []))
        source_chunks = set(source_attrs.get('chunk_ids', []))
        target_attrs['chunk_ids'] = sorted(target_chunks | source_chunks)
        
        # Merge descriptions
        target_desc = target_attrs.get('description', '')
        source_desc = source_attrs.get('description', '')
        
        if isinstance(target_desc, list):
            target_desc = ' '.join(target_desc)
        if isinstance(source_desc, list):
            source_desc = ' '.join(source_desc)
        
        # Combine unique descriptions
        if source_desc and source_desc not in target_desc:
            target_attrs['description'] = f"{target_desc} {source_desc}".strip()
        
        # Update type if target has empty type
        if not target_attrs.get('type') and source_attrs.get('type'):
            target_attrs['type'] = source_attrs['type']
    
    def _merge_edge_attributes(
        self,
        target_attrs: dict,
        source_attrs: dict,
    ) -> None:
        """
        Merge attributes from source to target edge
        
        Args:
            target_attrs: Target edge attributes (modified in place)
            source_attrs: Source edge attributes
        """
        # Merge chunk_ids
        target_chunks = set(target_attrs.get('chunk_ids', []))
        source_chunks = set(source_attrs.get('chunk_ids', []))
        target_attrs['chunk_ids'] = sorted(target_chunks | source_chunks)
        
        # Merge descriptions
        target_desc = target_attrs.get('description', '')
        source_desc = source_attrs.get('description', '')
        
        if isinstance(target_desc, list):
            target_desc = ' '.join(target_desc)
        if isinstance(source_desc, list):
            source_desc = ' '.join(source_desc)
        
        if source_desc and source_desc not in target_desc:
            target_attrs['description'] = f"{target_desc} {source_desc}".strip()
        
        # Add weights
        target_attrs['weight'] = (
            target_attrs.get('weight', 1.0) + source_attrs.get('weight', 1.0)
        )
    
    def merge_graphs(self, graphs: List[nx.Graph]) -> nx.Graph:
        """
        Merge multiple graphs into one
        
        Args:
            graphs: List of graphs to merge
            
        Returns:
            nx.Graph: Merged graph
        """
        if not graphs:
            return nx.Graph()
        
        merged = nx.Graph()
        
        for graph in graphs:
            # Merge nodes
            for node, attrs in graph.nodes(data=True):
                normalized_name = self._normalize_node_name(node)
                
                if normalized_name in merged.nodes():
                    # Merge with existing node
                    self._merge_node_attributes(
                        merged.nodes[normalized_name],
                        attrs,
                    )
                else:
                    # Add new node
                    merged.add_node(normalized_name, **attrs)
            
            # Merge edges
            for source, target, attrs in graph.edges(data=True):
                norm_source = self._normalize_node_name(source)
                norm_target = self._normalize_node_name(target)
                
                if merged.has_edge(norm_source, norm_target):
                    # Merge with existing edge
                    self._merge_edge_attributes(
                        merged.edges[norm_source, norm_target],
                        attrs,
                    )
                else:
                    # Add new edge
                    merged.add_edge(norm_source, norm_target, **attrs)
        
        logger.info(
            f"Merged {len(graphs)} graphs into graph with "
            f"{merged.number_of_nodes()} nodes and {merged.number_of_edges()} edges"
        )
        
        return merged
    
    def _summarize_description(self, entity_name: str, descriptions: str) -> str:
        """
        Summarize multiple descriptions using LLM
        
        Args:
            entity_name: Name of entity
            descriptions: Combined descriptions
            
        Returns:
            str: Summarized description
        """
        if not self.llm:
            return descriptions
        
        try:
            prompt = self.summarization_prompt.format(
                entity_name=entity_name,
                description_list=descriptions,
            )
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
            
        except Exception as e:
            logger.error(f"Error summarizing description for {entity_name}: {e}")
            return descriptions
    
    def summarize_graph_descriptions(self, graph: nx.Graph) -> nx.Graph:
        """
        Summarize all descriptions in graph using LLM
        
        Args:
            graph: Input graph
            
        Returns:
            nx.Graph: Graph with summarized descriptions
        """
        if not self.llm:
            logger.warning("No LLM provided, skipping summarization")
            return graph
        
        # Summarize node descriptions
        for node in graph.nodes():
            attrs = graph.nodes[node]
            description = attrs.get('description', '')
            
            # Only summarize if description is long
            if len(description) > 200:
                summarized = self._summarize_description(node, description)
                attrs['description'] = summarized
        
        # Summarize edge descriptions
        for source, target in graph.edges():
            attrs = graph.edges[source, target]
            description = attrs.get('description', '')
            
            if len(description) > 200:
                edge_name = f"{source} -> {target}"
                summarized = self._summarize_description(edge_name, description)
                attrs['description'] = summarized
        
        logger.info("Summarized all descriptions in graph")
        return graph
    
    def add_graph_metadata(self, graph: nx.Graph) -> nx.Graph:
        """
        Add metadata to graph (IDs, degrees, ranks, etc.)
        
        Args:
            graph: Input graph
            
        Returns:
            nx.Graph: Graph with metadata
        """
        # Add node IDs and degrees
        for idx, node in enumerate(graph.nodes()):
            attrs = graph.nodes[node]
            attrs['id'] = str(uuid.uuid4())
            attrs['human_readable_id'] = idx
            attrs['degree'] = graph.degree(node)
        
        # Add edge IDs and ranks
        for idx, (source, target) in enumerate(graph.edges()):
            attrs = graph.edges[source, target]
            attrs['id'] = str(uuid.uuid4())
            attrs['human_readable_id'] = idx
            
            # Rank = sum of source and target degrees
            source_degree = graph.nodes[source]['degree']
            target_degree = graph.nodes[target]['degree']
            attrs['rank'] = source_degree + target_degree
            attrs['source_degree'] = source_degree
            attrs['target_degree'] = target_degree
        
        logger.info("Added metadata to graph")
        return graph
    
    def build(self, graphs: List[nx.Graph], summarize: bool = True) -> nx.Graph:
        """
        Complete build pipeline: merge → summarize → add metadata
        
        Args:
            graphs: List of graphs to merge
            summarize: Whether to summarize descriptions
            
        Returns:
            nx.Graph: Final built graph
        """
        # Step 1: Merge graphs
        merged = self.merge_graphs(graphs)
        
        # Step 2: Summarize (optional)
        if summarize and self.llm:
            merged = self.summarize_graph_descriptions(merged)
        
        # Step 3: Add metadata
        merged = self.add_graph_metadata(merged)
        
        return merged


from typing import Optional
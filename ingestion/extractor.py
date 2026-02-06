
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from ..config import GraphExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity được trích xuất từ text"""
    name: str
    type: str
    description: str
    
    def __hash__(self):
        return hash(self.name.upper())
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.name.upper() == other.name.upper()
        return False


@dataclass  
class Relationship:
    """Relationship giữa các entities"""
    source: str
    target: str
    description: str
    weight: float = 1.0
    
    def __hash__(self):
        return hash((self.source.upper(), self.target.upper()))


class GraphExtractionOutputParser(BaseOutputParser[nx.Graph]):
    """
    Parser để parse output từ LLM thành NetworkX graph.
    
    Expected format:
    ("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>DESCRIPTION)
    ##
    ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>WEIGHT)
    """
    
    def __init__(
        self,
        tuple_delimiter: str = "<|>",
        record_delimiter: str = "##",
    ):
        super().__init__()
        self.tuple_delimiter = tuple_delimiter
        self.record_delimiter = record_delimiter
    
    def _clean_str(self, s: str) -> str:
        """Clean và normalize string"""
        return s.strip().strip('"').strip("'")
    
    def _parse_entity(self, parts: List[str]) -> Optional[Entity]:
        """Parse entity từ parts"""
        if len(parts) < 4:
            return None
        
        if parts[0] != '"entity"':
            return None
        
        name = self._clean_str(parts[1]).upper()
        entity_type = self._clean_str(parts[2]).upper()
        description = self._clean_str(parts[3])
        
        if not name or not entity_type:
            return None
        
        return Entity(
            name=name,
            type=entity_type,
            description=description,
        )
    
    def _parse_relationship(self, parts: List[str]) -> Optional[Relationship]:
        """Parse relationship từ parts"""
        if len(parts) < 4:
            return None
        
        if parts[0] != '"relationship"':
            return None
        
        source = self._clean_str(parts[1]).upper()
        target = self._clean_str(parts[2]).upper()
        description = self._clean_str(parts[3])
        
        # Weight là optional
        weight = 1.0
        if len(parts) >= 5:
            try:
                weight = float(self._clean_str(parts[4]))
            except (ValueError, IndexError):
                weight = 1.0
        
        if not source or not target:
            return None
        
        return Relationship(
            source=source,
            target=target,
            description=description,
            weight=weight,
        )
    
    def parse(self, text: str) -> nx.Graph:
        """
        Parse LLM output thành graph
        
        Args:
            text: Output từ LLM
            
        Returns:
            nx.Graph: Graph chứa entities và relationships
        """
        graph = nx.Graph()
        
        # Split records
        records = text.split(self.record_delimiter)
        
        entities: Dict[str, Entity] = {}
        relationships: List[Relationship] = []
        
        for record in records:
            record = record.strip()
            if not record:
                continue
            
            # Remove parentheses
            record = re.sub(r'^\(|\)$', '', record)
            
            # Split parts
            parts = record.split(self.tuple_delimiter)
            
            # Try parse as entity
            entity = self._parse_entity(parts)
            if entity:
                # Merge descriptions if entity exists
                if entity.name in entities:
                    existing = entities[entity.name]
                    # Combine descriptions
                    if entity.description not in existing.description:
                        existing.description += f" {entity.description}"
                    # Update type if empty
                    if not existing.type and entity.type:
                        existing.type = entity.type
                else:
                    entities[entity.name] = entity
                continue
            
            # Try parse as relationship
            rel = self._parse_relationship(parts)
            if rel:
                relationships.append(rel)
        
        # Add entities to graph
        for entity in entities.values():
            graph.add_node(
                entity.name,
                type=entity.type,
                description=entity.description,
            )
        
        # Add relationships to graph
        for rel in relationships:
            # Ensure both entities exist
            if rel.source not in graph.nodes():
                graph.add_node(rel.source, type="", description="")
            if rel.target not in graph.nodes():
                graph.add_node(rel.target, type="", description="")
            
            # Add or update edge
            if graph.has_edge(rel.source, rel.target):
                # Merge with existing edge
                edge_data = graph[rel.source][rel.target]
                edge_data['weight'] = edge_data.get('weight', 1.0) + rel.weight
                
                # Merge descriptions
                existing_desc = edge_data.get('description', '')
                if rel.description not in existing_desc:
                    edge_data['description'] = f"{existing_desc} {rel.description}".strip()
            else:
                graph.add_edge(
                    rel.source,
                    rel.target,
                    description=rel.description,
                    weight=rel.weight,
                )
        
        logger.info(
            f"Parsed graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        
        return graph
    
    @property
    def _type(self) -> str:
        return "graph_extraction_output_parser"


class EntityRelationshipExtractor:
    """
    Extractor để trích xuất entities và relationships từ text chunks.
    Sử dụng LLM với few-shot prompting.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        config: GraphExtractionConfig,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize extractor
        
        Args:
            llm: Language model
            config: Graph extraction config
            prompt_template: Custom prompt template (optional)
        """
        self.llm = llm
        self.config = config
        
        # Use default prompt if not provided
        if prompt_template is None:
            from ..retrieval.prompts import DEFAULT_ENTITY_EXTRACTION_PROMPT
            prompt_template = DEFAULT_ENTITY_EXTRACTION_PROMPT
        
        # Create prompt
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.prompt = self.prompt.partial(
            entity_types=",".join(config.entity_types),
            tuple_delimiter=config.tuple_delimiter,
            record_delimiter=config.record_delimiter,
            completion_delimiter=config.completion_delimiter,
        )
        
        # Output parser
        self.output_parser = GraphExtractionOutputParser(
            tuple_delimiter=config.tuple_delimiter,
            record_delimiter=config.record_delimiter,
        )
    
    def extract_from_chunk(
        self,
        text: str,
        chunk_id: str,
    ) -> nx.Graph:
        """
        Trích xuất graph từ một text chunk
        
        Args:
            text: Text chunk
            chunk_id: ID của chunk
            
        Returns:
            nx.Graph: Graph chứa entities và relationships
        """
        try:
            # Format prompt
            formatted_prompt = self.prompt.format(input_text=text)
            
            # Call LLM
            logger.debug(f"Extracting graph from chunk {chunk_id}")
            response = self.llm.invoke(formatted_prompt)
            
            # Parse response
            if hasattr(response, 'content'):
                output_text = response.content
            else:
                output_text = str(response)
            
            # Remove completion delimiter if present
            output_text = output_text.split(self.config.completion_delimiter)[0]
            
            graph = self.output_parser.parse(output_text)
            
            # Add chunk_id to all nodes and edges
            for node in graph.nodes():
                graph.nodes[node]['chunk_ids'] = [chunk_id]
            
            for edge in graph.edges():
                graph.edges[edge]['chunk_ids'] = [chunk_id]
            
            return graph
            
        except Exception as e:
            logger.error(f"Error extracting graph from chunk {chunk_id}: {e}")
            # Return empty graph on error
            return nx.Graph()
    
    def extract_from_chunks(
        self,
        chunks: List[Tuple[str, str]],  # [(chunk_id, text), ...]
    ) -> List[nx.Graph]:
        """
        Trích xuất graphs từ nhiều chunks
        
        Args:
            chunks: List of (chunk_id, text) tuples
            
        Returns:
            List[nx.Graph]: List of graphs
        """
        graphs = []
        
        for chunk_id, text in chunks:
            graph = self.extract_from_chunk(text, chunk_id)
            if graph.number_of_nodes() > 0:
                graphs.append(graph)
        
        logger.info(f"Extracted {len(graphs)} graphs from {len(chunks)} chunks")
        return graphs
"""
STEP 3: Graph Building
Input: List of extraction results (entities and relations)
Output: NetworkX graph with nodes and edges
"""

from typing import List
import networkx as nx
from config import MIN_ENTITY_LENGTH


class GraphOutput:
    """Output của bước graph building"""
    def __init__(self, graph: nx.Graph, stats: dict):
        self.graph = graph
        self.stats = stats
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("STEP 3: GRAPH BUILDING - OUTPUT")
        print("=" * 80)
        print(f"📥 Số extraction results đầu vào: {self.stats['num_extractions']}")
        print(f"🔵 Số nodes (entities): {self.stats['num_nodes']}")
        print(f"🔗 Số edges (relations): {self.stats['num_edges']}")
        print(f"📊 Mật độ graph: {self.stats['density']:.4f}")
        print(f"🌐 Số connected components: {self.stats['num_components']}")
        
        if self.stats['sample_nodes']:
            print(f"\n📋 Sample nodes (10 đầu tiên):")
            for node in self.stats['sample_nodes'][:10]:
                print(f"   - {node}")
        
        if self.stats['sample_edges']:
            print(f"\n🔗 Sample edges (5 đầu tiên):")
            for edge in self.stats['sample_edges'][:5]:
                print(f"   - {edge[0]} ↔ {edge[1]}")
        
        print("=" * 80)
    
    def save_to_file(self, output_dir: str = "pipeline_outputs"):
        """Lưu output ra file txt"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "step3_graph_output.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 3: GRAPH BUILDING - DETAILED OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            
            # Stats
            f.write("📊 THỐNG KÊ:\n")
            f.write(f"   - Số extraction results đầu vào: {self.stats['num_extractions']}\n")
            f.write(f"   - Số nodes (entities): {self.stats['num_nodes']}\n")
            f.write(f"   - Số edges (relations): {self.stats['num_edges']}\n")
            f.write(f"   - Mật độ graph: {self.stats['density']:.4f}\n")
            f.write(f"   - Số connected components: {self.stats['num_components']}\n\n")
            
            # All nodes
            f.write("=" * 80 + "\n")
            f.write(f"📝 TẤT CẢ NODES ({len(self.graph.nodes())} nodes):\n")
            f.write("=" * 80 + "\n")
            for i, node in enumerate(sorted(self.graph.nodes()), 1):
                f.write(f"{i}. {node}\n")
            
            # All edges
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"🔗 TẤT CẢ EDGES ({len(self.graph.edges())} edges):\n")
            f.write("=" * 80 + "\n")
            for i, (src, tgt) in enumerate(sorted(self.graph.edges()), 1):
                f.write(f"{i}. {src} ↔ {tgt}\n")
        
        print(f"💾 Đã lưu output vào: {filepath}")
        return filepath


class GraphBuilder:
    """Class để xây dựng knowledge graph từ extraction results"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def _parse_extraction_line(self, line: str):
        """
        Parse một dòng extraction để lấy entity hoặc relation
        
        Returns:
            tuple: ('entity', node_name) hoặc ('relation', src, tgt)
        """
        line = line.strip()
        
        if line.startswith("RELATION:"):
            try:
                # RELATION: entity_1 -> relation -> entity_2
                _, rel_text = line.split(":", 1)
                parts = [p.strip() for p in rel_text.split("->")]
                
                if len(parts) == 3:
                    src, relation_type, tgt = parts
                    if len(src) > MIN_ENTITY_LENGTH and len(tgt) > MIN_ENTITY_LENGTH:
                        return ('relation', src, tgt)
            except Exception:
                pass
        
        elif line.startswith("ENTITY:"):
            node = line.replace("ENTITY:", "").strip()
            if len(node) > MIN_ENTITY_LENGTH:
                return ('entity', node)
        
        return None
    
    def build(self, extractions: List[str]) -> GraphOutput:
        """
        Xây dựng graph từ extraction results
        
        Args:
            extractions: List of extraction text results
            
        Returns:
            GraphOutput: Object chứa graph và stats
        """
        self.graph = nx.Graph()
        
        # Parse tất cả extraction results
        for extraction in extractions:
            lines = [l.strip() for l in extraction.split("\n") if l.strip()]
            
            for line in lines:
                parsed = self._parse_extraction_line(line)
                
                if parsed:
                    if parsed[0] == 'relation':
                        _, src, tgt = parsed
                        self.graph.add_edge(src, tgt)
                    
                    elif parsed[0] == 'entity':
                        _, node = parsed
                        self.graph.add_node(node)
        
        # Tính toán stats
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        stats = {
            'num_extractions': len(extractions),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': nx.density(self.graph) if num_nodes > 0 else 0,
            'num_components': nx.number_connected_components(self.graph),
            'sample_nodes': list(self.graph.nodes())[:10],
            'sample_edges': list(self.graph.edges())[:5]
        }
        
        return GraphOutput(self.graph, stats)


if __name__ == "__main__":
    # Test graph building
    test_extractions = [
        """ENTITY: Hệ điều hành
ENTITY: Tiến trình
ENTITY: CPU
RELATION: Hệ điều hành -> quản lý -> Tiến trình
RELATION: CPU -> thực thi -> Tiến trình""",
        """ENTITY: Python
ENTITY: Django
RELATION: Django -> là framework của -> Python"""
    ]
    
    builder = GraphBuilder()
    output = builder.build(test_extractions)
    output.print_summary()
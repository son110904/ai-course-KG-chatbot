"""
STEP 4: Community Detection
Input: NetworkX graph
Output: List of communities (clusters of related entities)
"""

from typing import List
import networkx as nx
from cdlib import algorithms
from config import MIN_COMMUNITY_SIZE


class CommunityOutput:
    """Output của bước community detection"""
    def __init__(self, communities: List[List[str]], stats: dict):
        self.communities = communities
        self.stats = stats
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("STEP 4: COMMUNITY DETECTION - OUTPUT")
        print("=" * 80)
        print(f"📥 Số nodes trong graph: {self.stats['num_nodes']}")
        print(f"📥 Số edges trong graph: {self.stats['num_edges']}")
        print(f"🌐 Số connected components: {self.stats['num_components']}")
        print(f"👥 Tổng số communities phát hiện: {self.stats['total_communities']}")
        print(f"⭐ Số communities lớn (>={MIN_COMMUNITY_SIZE} nodes): {self.stats['large_communities']}")
        
        print(f"\n📊 Phân bố kích thước communities:")
        for size, count in sorted(self.stats['size_distribution'].items()):
            print(f"   - Size {size}: {count} communities")
        
        print(f"\n📋 Top 5 communities lớn nhất:")
        for i, comm in enumerate(self.communities[:5]):
            print(f"   {i+1}. [{len(comm)} nodes] {', '.join(comm[:5])}{'...' if len(comm) > 5 else ''}")
        
        print("=" * 80)
    
    def save_to_file(self, output_dir: str = "pipeline_outputs"):
        """Lưu output ra file txt"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "step4_communities_output.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 4: COMMUNITY DETECTION - DETAILED OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            
            # Stats
            f.write("📊 THỐNG KÊ:\n")
            f.write(f"   - Số nodes trong graph: {self.stats['num_nodes']}\n")
            f.write(f"   - Số edges trong graph: {self.stats['num_edges']}\n")
            f.write(f"   - Số connected components: {self.stats['num_components']}\n")
            f.write(f"   - Tổng số communities: {self.stats['total_communities']}\n")
            f.write(f"   - Communities lớn (>={MIN_COMMUNITY_SIZE}): {self.stats['large_communities']}\n\n")
            
            f.write("📊 PHÂN BỐ KÍCH THƯỚC:\n")
            for size, count in sorted(self.stats['size_distribution'].items()):
                f.write(f"   - Size {size}: {count} communities\n")
            
            # All communities
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"📝 TẤT CẢ COMMUNITIES ({len(self.communities)} communities):\n")
            f.write("=" * 80 + "\n\n")
            
            for i, comm in enumerate(self.communities, 1):
                f.write(f"--- COMMUNITY {i} (Size: {len(comm)}) ---\n")
                for j, entity in enumerate(comm, 1):
                    f.write(f"   {j}. {entity}\n")
                f.write("\n")
        
        print(f"💾 Đã lưu output vào: {filepath}")
        return filepath


class CommunityDetector:
    """Class để phát hiện communities trong graph"""
    
    def detect(self, graph: nx.Graph) -> CommunityOutput:
        """
        Phát hiện communities sử dụng Leiden algorithm
        
        Args:
            graph: NetworkX graph
            
        Returns:
            CommunityOutput: Object chứa communities và stats
        """
        communities = []
        
        # Xử lý từng connected component
        for component in nx.connected_components(graph):
            if len(component) > 2:
                # Component đủ lớn -> dùng Leiden
                subgraph = graph.subgraph(component)
                try:
                    comms = algorithms.leiden(subgraph)
                    communities.extend([list(c) for c in comms.communities])
                except Exception as e:
                    # Nếu Leiden thất bại, coi cả component là 1 community
                    print(f"[WARN] Leiden failed for component size {len(component)}: {e}")
                    communities.append(list(component))
            else:
                # Component nhỏ -> coi là 1 community
                communities.append(list(component))
        
        # Lọc communities lớn
        large_communities = [c for c in communities if len(c) >= MIN_COMMUNITY_SIZE]
        
        # Nếu không có community lớn, giữ tất cả
        if not large_communities:
            large_communities = communities
        
        # Sắp xếp theo kích thước giảm dần
        large_communities.sort(key=len, reverse=True)
        
        # Tính toán stats
        size_distribution = {}
        for comm in communities:
            size = len(comm)
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_components': nx.number_connected_components(graph),
            'total_communities': len(communities),
            'large_communities': len(large_communities),
            'size_distribution': size_distribution,
            'avg_community_size': sum(len(c) for c in large_communities) / len(large_communities) if large_communities else 0
        }
        
        return CommunityOutput(large_communities, stats)


if __name__ == "__main__":
    # Test community detection
    G = nx.Graph()
    
    # Tạo graph test với 2 communities rõ ràng
    # Community 1: OS-related
    G.add_edge("Hệ điều hành", "Tiến trình")
    G.add_edge("Hệ điều hành", "Bộ nhớ")
    G.add_edge("Tiến trình", "CPU")
    G.add_edge("Bộ nhớ", "CPU")
    
    # Community 2: Programming-related
    G.add_edge("Python", "Django")
    G.add_edge("Python", "Flask")
    G.add_edge("Django", "Web Framework")
    
    detector = CommunityDetector()
    output = detector.detect(G)
    output.print_summary()
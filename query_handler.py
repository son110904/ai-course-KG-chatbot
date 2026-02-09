from graph_manager import GraphManager
from logger import Logger
from typing import List


class QueryHandler:
    """
    Handle user queries using graph-based retrieval strategies
    such as community detection and centrality analysis.
    """

    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client, model: str):
        self.graph_manager = graph_manager
        self.client = client
        self.model = model
        self.logger.info(f"QueryHandler initialized with model={model}")

    # ------------------------------------------------------------------
    # COMMUNITY-BASED QUERYING (MAIN GRAPH RAG METHOD)
    # ------------------------------------------------------------------

    def _filter_relevant_communities(
        self,
        query: str,
        communities: List[List[str]],
        min_overlap: int = 1
    ) -> List[List[str]]:
        """
        Filter communities based on lexical overlap with query.
        No embedding is used.

        Args:
            query: user question
            communities: list of communities (list of entity names)
            min_overlap: minimum matched entities to keep community

        Returns:
            Sorted list of relevant communities
        """
        query_terms = set(query.lower().split())
        scored = []

        for comm in communities:
            overlap = 0
            for entity in comm:
                entity_l = entity.lower()
                if any(term in entity_l for term in query_terms):
                    overlap += 1

            if overlap >= min_overlap:
                scored.append((overlap, comm))

        # Sort by relevance score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [comm for _, comm in scored]

    def ask_question_with_communities(
        self,
        query: str,
        max_communities: int = 3,
        max_entities_per_comm: int = 20
    ) -> str:
        """
        Answer query using ONLY relevant communities.

        Flow:
        Query -> detect communities -> filter by relevance -> LLM synthesis
        """
        self.logger.info(f"Processing query with communities: {query}")

        communities = self.graph_manager.detect_communities()

        relevant_communities = self._filter_relevant_communities(
            query=query,
            communities=communities
        )[:max_communities]

        if not relevant_communities:
            self.logger.warning("No relevant communities found.")
            return "Không tìm thấy cụm kiến thức liên quan trong đồ thị."

        # Format context for LLM
        comm_desc = []
        for i, comm in enumerate(relevant_communities, start=1):
            entities = ", ".join(comm[:max_entities_per_comm])
            comm_desc.append(
                f"Community {i} (relevant):\n"
                f"- Entities: {entities}"
            )

        community_text = "\n\n".join(comm_desc)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là hệ thống trả lời câu hỏi dựa trên đồ thị tri thức.\n"
                        "Chỉ sử dụng các community được cung cấp.\n"
                        "Các community này đã được lọc vì có liên quan trực tiếp đến câu hỏi.\n"
                        "Không suy đoán hoặc bổ sung kiến thức bên ngoài đồ thị."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Câu hỏi:\n{query}\n\n"
                        f"Các community liên quan:\n{community_text}\n\n"
                        "Hãy tổng hợp câu trả lời mạch lạc và có cấu trúc."
                    )
                }
            ],
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        self.logger.debug(f"Generated answer: {answer[:120]}...")
        return answer

    # ------------------------------------------------------------------
    # CENTRALITY-BASED QUERYING (SECONDARY METHOD)
    # ------------------------------------------------------------------

    def ask_question_with_centrality(self, query: str) -> str:
        """
        Answer query using centrality measures to highlight key entities.
        """
        self.logger.info(f"Processing query with centrality: {query}")

        centrality_data = self.graph_manager.calculate_centrality_measures()
        centrality_summary = self.graph_manager.summarize_centrality_measures(
            centrality_data
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sử dụng thông tin centrality để trả lời câu hỏi.\n"
                        "Các entity có centrality cao đại diện cho khái niệm cốt lõi."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Câu hỏi:\n{query}\n\n"
                        f"Tóm tắt centrality:\n{centrality_summary}"
                    )
                }
            ],
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        self.logger.debug(f"Generated answer: {answer[:120]}...")
        return answer

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    def ask_question(self, query: str, method: str = "communities") -> str:
        """
        Unified entry point.

        method:
            - 'communities' (default, GraphRAG-style)
            - 'centrality'
        """
        if method == "centrality":
            return self.ask_question_with_centrality(query)
        return self.ask_question_with_communities(query)

    # ------------------------------------------------------------------
    # GRAPH METADATA
    # ------------------------------------------------------------------

    def get_graph_stats(self):
        """
        Get graph statistics for debugging / monitoring.
        """
        summary = self.graph_manager.get_graph_summary()
        self.logger.info(
            f"Graph stats: {summary['nodes']} nodes, "
            f"{summary['edges']} edges, "
            f"{summary['communities']} communities"
        )
        return summary

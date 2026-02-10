# graph_database.py
from neo4j import GraphDatabase
from logger import Logger


class GraphDatabaseConnection:
    """
    Manages Neo4j database connection with session handling.
    Supports both local and cloud (Aura) connections.
    """
    
    logger = Logger('GraphDatabaseConnection').get_logger()

    def __init__(self, uri, user, password):
        """
        Initialize database connection.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j://localhost:7687 or neo4j+s://xxx.databases.neo4j.io)
            user: Database username
            password: Database password
        """
        if not uri or not user or not password:
            raise ValueError(
                "URI, user, and password must be provided to initialize the DatabaseConnection."
            )
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self.driver.verify_connectivity()
            self.logger.info(f"Successfully connected to Neo4j at {uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Database connection closed")

    def get_session(self):
        """
        Get a new database session.
        
        Returns:
            Neo4j session object
        """
        return self.driver.session()

    def clear_database(self):
        """
        Delete all nodes and relationships in the database.
        WARNING: This is destructive and cannot be undone.
        """
        self.logger.warning("Clearing entire database...")
        with self.get_session() as session:
            result = session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Database cleared successfully")

    def execute_query(self, query, parameters=None):
        """
        Execute a single query and return results.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            Query results as list of records
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return result.data()

    def get_database_stats(self):
        """
        Get basic database statistics.
        
        Returns:
            Dict with node count, relationship count, and label info
        """
        with self.get_session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            labels = session.run("CALL db.labels()").data()
            
            stats = {
                "nodes": node_count,
                "relationships": rel_count,
                "labels": [record["label"] for record in labels]
            }
            
            self.logger.info(f"Database stats: {stats}")
            return stats
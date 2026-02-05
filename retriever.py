from neo4j_store import driver

def graph_retrieve(question: str):
    query = """
    MATCH (c:Course)-[:HAS_SUBJECT]->(s)
    WHERE toLower(s.id) CONTAINS toLower($q)
    RETURN c.id AS course, s.id AS subject
    LIMIT 5
    """

    with driver.session() as session:
        return session.run(query, q=question).data()

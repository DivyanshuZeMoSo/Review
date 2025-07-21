from src.embeddings import get_embedding

def search_movies(collection, user_query, genre=None, director=None):
    """
    Searches the ChromaDB collection for the top 5 movies most semantically
    similar to the user query, with optional filters for genre and director.

    Args:
        collection: The ChromaDB collection object.
        user_query (str): The user's natural language search input.
        genre (str): Optional filter by genre.
        director (str): Optional filter by director.

    Returns:
        A dictionary with keys: 'ids', 'documents', 'metadatas', 'distances'
    """
    query_embedding = get_embedding(user_query)

    filters = {}
    if genre:
        filters["genre"] = genre
    if director:
        filters["director"] = director

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=filters
    )

    return results
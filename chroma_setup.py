import chromadb
from chromadb.config import Settings

def setup_chroma_collection(collection_name="movies"):
    """
    Initializes and returns a ChromaDB collection.
    """
    client = chromadb.Client(Settings(
        persist_directory="./chroma_storage",  # Optional: for persistence
        anonymized_telemetry=False
    ))

    collection = client.get_or_create_collection(name=collection_name)
    return collection
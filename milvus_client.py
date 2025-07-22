from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import list_collections
from sentence_transformers import SentenceTransformer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Embed model
model = SentenceTransformer('BAAI/bge-small-en')

def embed_and_store(chunks):
    embeddings = model.encode(chunks)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]

    schema = CollectionSchema(fields, "RAG storage")

    collection_name = "company_docs"
    if collection_name in list_collections():
        collection = Collection(collection_name)
    else:
        collection = Collection(collection_name, schema)

    # Insert
    collection.insert([embeddings, chunks])

    # Create index on vector field
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
    )

    collection.load()
    print("✅ Documents inserted and indexed successfully.")

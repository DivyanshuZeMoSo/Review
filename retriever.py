from pymilvus import Collection
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")

def query_milvus(query, top_k=3):
    collection = Collection("company_docs")
    collection.load()  # <-- this line is critical

    query_embedding = model.encode([query])[0]

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )

    return [hit.entity.get("text") for hit in results[0]]

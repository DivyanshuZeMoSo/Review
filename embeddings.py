from sentence_transformers import SentenceTransformer

# Local model used for generating sentence embeddings (small & fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()
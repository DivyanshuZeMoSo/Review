from embeddings.milvus_client import embed_and_store
from retrieval.retriever import query_milvus
from llm.llama_client import ask_llama

def main():
    # Load and chunk
    with open("docs/storage.txt", "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")  # naive chunking; replace with langchain if needed

    # Embed and store
    embed_and_store(docs)

    while True:
        query = input("Ask a business question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        contexts = query_milvus(query)
        combined_context = "\n".join(contexts)
        response = ask_llama(combined_context, query)

        print("\nAnswer:")
        print(response)
        print("\nSource Snippets:")
        for i, ctx in enumerate(contexts, 1):
            print(f"[{i}] {ctx[:200]}...\n")

if __name__ == "__main__":
    main()

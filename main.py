from src.load_data import load_movie_data
from src.embeddings import get_embedding
from src.chroma_setup import setup_chroma_collection
from src.search import search_movies
from src.query_llama import refine_query_with_llama  # Optional

def main():
    data_path = "data/Movies Dataset.xlsx"
    df = load_movie_data(data_path)
    collection = setup_chroma_collection()

    print("Inserting data into ChromaDB...")
    for idx, row in df.iterrows():
        embedding = get_embedding(row['Description'])
        collection.add(
            ids=[str(idx)],
            documents=[row['Description']],
            embeddings=[embedding],
            metadatas=[{
                "title": row["Title"],
                "genre": row["Genre"],
                "director": row["Director"]
            }]
        )

    print("\n Enter your movie preferences to get recommendations.")
    user_query = input("Your movie query (e.g., 'thrilling time travel adventure'): ")

    refined_query = refine_query_with_llama(user_query)
    print(f" Refined Query: {refined_query}")

    genre = input("Optional genre filter (press Enter to skip): ").strip()
    director = input("Optional director filter (press Enter to skip): ").strip()

    genre = genre if genre else None
    director = director if director else None

    results = search_movies(collection, refined_query, genre=genre, director=director)

    print("\n Top Recommendations:")
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        print(f" {meta['title']} ({meta['genre']}) — {doc[:100]}...")

if __name__ == "__main__":
    main()
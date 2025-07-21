# Movie Recommendation System using Chroma Vector DB

## 6. Vector Storage

**Building a Movie Recommendation System with Chroma Vector DB**

**Scenario:** Develop a Python script that utilizes Chroma Vector DB to create a movie recommendation system based on user preferences via query and movie descriptions.

**Tasks:**

1. For each movie, extract the description text.
2. Use any embedding_model to generate an embedding for the description.
3. Insert the movie description, its embedding, and the metadata (genre and director) into the ChromaDB collection
4. Based on the user query form a proper filter expression.
5. Retrieve the top 5 results based on the user query.

**Dataset:**

[**Movies Dataset**](https://docs.google.com/spreadsheets/d/17zn3h5pDWTz1CEiouAc0c5KWpobk7OFg8D0wmO6QJoE/edit?usp=sharing)

This system recommends movies based on semantic similarity between user queries and movie descriptions. Uses OpenAI for embedding generation and ChromaDB as the vector store.

##  Structure
- `main.py`: Main script to run the entire pipeline
- `src/`: Contains all the modular code
- `data/`: Contains the dataset (Excel format)

##  Setup Instructions
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-openai-key
python main.py


User Query: "A mind-bending sci-fi thriller"
Genre: Science Fiction





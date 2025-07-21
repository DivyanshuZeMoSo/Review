# Movie Recommendation System using Chroma Vector DB

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





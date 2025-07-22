# RAG Evaluation using RAGAS (LLaMA 3.1v 8B)

This project demonstrates retrieval-augmented generation (RAG) evaluation using the RAGAS framework, leveraging a locally hosted LLaMA 3.1v 8B model.

## Features

- LangChain RAG pipeline
- Chroma vector store
- HuggingFace embeddings
- Local LLaMA endpoint at `http://127.0.0.1:1234`
- RAGAS metrics: `context_precision`, `context_recall`, `answer_relevancy`

## How to Run

1. Create virtual environment:

python -m venv .venv
.venv\Scripts\activate


2. Install dependencies:
pip install -r requirements.txt


3. Add your context file in `docs/storage.txt`.

4. Run:
python main.py


## Notes

- Ensure your LLaMA model is running at `http://127.0.0.1:1234` (e.g., via LM Studio).
- Compatible with `ragas==0.1.8` and `langchain==0.2.5`

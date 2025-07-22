import sys
import asyncio
import requests
import numpy as np
from typing import List
from langchain.llms.base import LLM
from langchain_core.outputs import LLMResult, Generation
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# --- Generation LLM: LLaMA via LM Studio ---
class LMStudioGenerationLLM(LLM):
    def _call(self, prompt: str, stop=None, run_manager=None):
        try:
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-lexi-uncensored-v2",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                timeout=60
            )
            return {"text": response.json()["choices"][0]["message"]["content"]}
        except Exception as e:
            print(f"Error calling LLaMA model: {e}")
            return {"text": "Error: LLaMA failed."}

    @property
    def _llm_type(self) -> str:
        return "llama_generation"

# --- Evaluation LLM: Phi-3-mini via LM Studio ---
class Phi3EvaluationLLM(BaseRagasLLM):
    def __init__(self):
        self.model_url = "http://127.0.0.1:1234/v1/chat/completions"
        self.model_name = "phi-3-mini-128k-instruct-imatrix-smashed"

    def generate_text(self, prompt: str, **kwargs) -> LLMResult:
        full_prompt = f"""You are a strict evaluation assistant.
Return your answer ONLY in valid JSON in the following format:

{{
  "classifications": [
    {{
      "statement": "...",
      "reason": "...",
      "attributed": true
    }}
  ]
}}

Evaluate based on the given context, question, answer, and reference.

{prompt}
"""
        try:
            response = requests.post(
                self.model_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 512
                },
                timeout=60
            )
            content = response.json()["choices"][0]["message"]["content"]
            return LLMResult(generations=[[Generation(text=content)]])
        except Exception as e:
            print(f"Error in Phi-3 Evaluation LLM: {e}")
            return LLMResult(generations=[[Generation(text="Error: Phi-3 failed.")]])

    async def agenerate_text(self, prompt: str, n: int = 1, **kwargs) -> LLMResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_text, prompt)

    @property
    def model(self) -> str:
        return self.model_name

# --- RAG Pipeline ---
class RAG:
    def __init__(self):
        self.llm = LMStudioGenerationLLM()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.docs = None
        self.doc_embeddings = None

    def load_documents(self, documents: List[str]):
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query: str) -> List[str]:
        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_index = np.argmax(similarities)
        return [self.docs[most_relevant_index]]

    def generate_answer(self, query: str, relevant_doc: List[str]) -> str:
        prompt = f"""You are a helpful assistant. Use only the following context to answer the question.

Context: {relevant_doc[0]}

Question: {query}

Answer:"""
        return self.llm._call(prompt)["text"]

# --- Prepare Input ---
sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]
query = "Who introduced the theory of relativity?"
reference = "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity."

# --- Run RAG ---
rag = RAG()
rag.load_documents(sample_docs)
relevant_docs = rag.get_most_relevant_docs(query)
response = rag.generate_answer(query, relevant_docs)

# --- Evaluation Dataset ---
dataset = [{
    "user_input": query,
    "retrieved_contexts": relevant_docs,
    "response": response,
    "reference": reference
}]
evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = Phi3EvaluationLLM()

# --- Metric Selection ---
metric_choice = sys.argv[1] if len(sys.argv) > 1 else "context_recall"
if metric_choice == "context_recall":
    selected_metrics = [LLMContextRecall()]
elif metric_choice == "faithfulness":
    selected_metrics = [Faithfulness()]
elif metric_choice == "factual_correctness":
    selected_metrics = [FactualCorrectness()]
else:
    raise ValueError(f"Invalid metric selected: {metric_choice}")

# --- Run Evaluation ---
print(f"\nEvaluating with metric: {metric_choice}")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=selected_metrics,
    llm=evaluator_llm
)

# --- Output ---
print("\nEvaluation Results:")
print(result)
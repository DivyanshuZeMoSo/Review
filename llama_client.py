import requests

def ask_llama(context, question):
    prompt = f"""Use the following context to answer the question:
    
Context:
{context}

Question:
{question}

Answer:"""

    payload = {
        "prompt": prompt,
        "temperature": 0.3,
        "max_tokens": 300,
        "stop": ["\n\n"]
    }
    response = requests.post("http://127.0.0.1:1234/v1/completions", json=payload)
    return response.json()["choices"][0]["text"].strip()

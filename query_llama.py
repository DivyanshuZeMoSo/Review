import requests

def refine_query_with_llama(user_input):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-lexi-uncensored-v2:2",  # As shown in LM Studio
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that rewrites vague movie queries into clear, searchable phrases."},
            {"role": "user", "content": f"Refine this movie search query: {user_input}"}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()

    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"].strip()
    else:
        print("Unexpected response format:", response_json)
        return user_input  # fallback
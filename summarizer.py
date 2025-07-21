import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime
import re
import urllib.request

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "lmstudio"
JSON_FILE = "summaries.json"


def extract_article(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html = response.read()

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = soup.title.string.strip() if soup.title else "News Article"

        # Extract paragraph text
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()

        return title, text
    except Exception as e:
        print("Error while extracting article:", e)
        return None, None


def query_llm(prompt, system_msg, temperature=0.5, max_tokens=300):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(API_URL, json=payload)
    return response.json()["choices"][0]["message"]["content"].strip()


def summarize_article(title, text):
    prompt = f"Summarize the following news article:\n\nTitle: {title}\n\n{text}"
    return query_llm(prompt, "You are a professional news summarizer.", temperature=0.7, max_tokens=400)


def extract_article_type(summary):
    prompt = f"Categorize the following news summary into one word like financial, sports, politics, tech, health, etc.:\n\n{summary}"
    return query_llm(prompt, "You are an expert in tagging articles with appropriate types.", temperature=0.2)


def store_summary(date, summary, article_type):
    data = {"date": date, "summary": summary, "articleType": article_type}

    try:
        with open(JSON_FILE, "r") as f:
            summaries = json.load(f)
    except FileNotFoundError:
        summaries = []

    summaries.append(data)

    with open(JSON_FILE, "w") as f:
        json.dump(summaries, f, indent=4)

    print("\nSummary stored successfully.")


def ask_question():
    try:
        with open(JSON_FILE, "r") as f:
            summaries = json.load(f)
    except FileNotFoundError:
        print("No summaries found.")
        return

    print("\nYou can now ask a question based on stored summaries.")
    question = input("Your question: ")

    context = "\n\n".join([f"Date: {s['date']}\nType: {s['articleType']}\nSummary: {s['summary']}" for s in summaries])

    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    answer = query_llm(prompt, "Answer questions based on summaries provided.")
    print("\n💡 Answer:")
    print(answer)


def main():
    print("\n=== News Summarizer Tool ===")
    url = input("Enter a news article URL: ").strip()

    title, text = extract_article(url)
    if not title or not text:
        return

    print("\n🔍 Summarizing article...")
    summary = summarize_article(title, text)
    print("\nSummary:\n", summary)

    print("\n🔎 Extracting article type...")
    article_type = extract_article_type(summary)
    print(f"\nArticle Type: {article_type}")

    date = datetime.today().strftime("%d %B %Y")
    store_summary(date, summary, article_type)

    ask_now = input("\nDo you want to ask a question from the summaries? (y/n): ").lower()
    if ask_now == "y":
        ask_question()


if __name__ == "__main__":
    main()

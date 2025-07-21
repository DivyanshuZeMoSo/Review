# News Summarization Tool (LLM + BeautifulSoup)

A command-line Python tool that allows you to:

- Summarize any news article using a locally running **Large Language Model (LLM)** via **LM Studio**
- Store those summaries with auto-generated **article types** (e.g., financial, tech, sports)
- Ask natural language questions from previously stored summaries
- No frontend needed — fast, minimal CLI setup

---

## Features

-  Input: News article URL
-  Scrapes content using **BeautifulSoup**
-  Summarizes article using **LLM (LM Studio API)**
-  Auto-tags article type (e.g., financial, tech, etc.) using LLM
-  Saves summary, date, and type in `summaries.json`
-  Allows Q&A over stored summaries via LLM

---

##  How It Works

1. You paste a news article URL
2. The tool scrapes the text using BeautifulSoup
3. It sends the article to your **local LLM (e.g., LLaMA, Mistral)** via LM Studio's API
4. Summary is generated and categorized automatically
5. Optionally ask questions across stored summaries

---

##  Project Structure

news_summarizer/
├── summarizer.py # Main script
├── summaries.json # Output file storing all summaries
├── README.md # Documentation
|── requirements.txt # all required libraries and modules


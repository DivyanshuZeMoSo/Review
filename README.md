# RAG-Powered Business Analysis System

## Overview
This system enables analysts to query company-related documents (financial reports, blog posts, press releases) using Retrieval-Augmented Generation.

## Features
- Document chunking + embedding using BGE-small-en
- Vector storage with Milvus
- Question answering using a locally hosted LLaMA model via LM Studio
- Cited source snippets

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Start Milvus locally.
3. Start LM Studio at `http://127.0.0.1:1234`.
4. Add documents to `docs/storage.txt`.
5. Run the main pipeline: `python main.py`
# Semantic Search & QA System with FAISS, Hybrid, and Cross-Reranked Retrieval

This project implements a **robust semantic search and question-answering system** using:

- **FAISS embeddings** for vector search
- **Hybrid retrieval** (vector + SQLite FTS keyword scoring)
- **Cross-encoder reranking** for precise, context-aware results

---

## Features

### 1. Retrieval Modes

- **Vector-only**  
  Uses FAISS embeddings for semantic similarity search. Fast and efficient, but may return related results that aren’t directly on-topic.

- **Hybrid (Vector + Keyword)**  
  Combines semantic similarity (FAISS) with keyword relevance (SQLite FTS OR-tokenized queries).  
  - Supports fallback scoring when BM25 is unavailable.  
  - Scores are normalized and blended using a configurable `ALPHA` weight.  

- **Cross-Reranked (CrossEncoder)**  
  Uses a **sentence-transformers cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank a larger candidate pool from FAISS for highly accurate results.  
  - Lazy-loaded only when cross-reranking is requested.  
  - Default candidate pool size is 20 (configurable via `CROSS_RERANK_TOP_K`).

---

## Installation

```bash
git clone <repo_url>
cd <repo_folder>
pip install -r requirements.txt

import os
import re
import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import random
import fitz
from pathlib import Path

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FILE = "docs.db"
INDEX_FILE = "faiss.index"
THRESHOLD = 0.15
ALPHA = 0.6
CROSS_RERANK_TOP_K = 20
CHUNK_MIN_SIZE = 50
CHUNK_MAX_SIZE = 400

model_name = EMBED_MODEL
model = SentenceTransformer(model_name)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
query = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    id UNINDEXED,
    text,
    link
);
"""
c.execute(query)
conn.commit()

embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(embedding_dim)
docid_mapping: Dict[int, str] = {}
next_idx = 0

file_exists = os.path.exists(INDEX_FILE)
if file_exists:
    index = faiss.read_index(INDEX_FILE)
    with sqlite3.connect(DB_FILE) as conn2:
        c2 = conn2.cursor()
        c2.execute("SELECT id FROM docs ORDER BY rowid")
        rows = c2.fetchall()
        for idx, row in enumerate(rows):
            docid_mapping[idx] = row[0]
        next_idx = len(docid_mapping)

_cross_reranker: Optional[CrossEncoder] = None


def pdf_to_text(pdf_path: str) -> str:
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text()
        if page_text:
            combined = text + page_text + "\n\n"
            text = combined
    return text


def chunk_text(text: str, min_size: int = CHUNK_MIN_SIZE, max_size: int = CHUNK_MAX_SIZE) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    for p in paragraphs:
        stripped = p.strip()
        if not stripped:
            continue
        if len(stripped) <= max_size:
            if len(stripped) >= min_size:
                chunks.append(stripped)
            else:
                condition = chunks and len(chunks[-1]) + len(stripped) + 1 <= max_size
                if condition:
                    chunks[-1] = chunks[-1] + " " + stripped
                else:
                    chunks.append(stripped)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', stripped)
            cur = ""
            for s in sentences:
                condition2 = len(cur) + len(s) + 1 <= max_size
                if condition2:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        if len(cur) >= min_size:
                            chunks.append(cur)
                        else:
                            chunks.append(cur)
                    cur = s
            if cur:
                chunks.append(cur)
    return chunks


def store_chunks(chunks: List[str], source: str):
    global next_idx, docid_mapping
    for chunk in chunks:
        doc_id = f"{source}_{next_idx}"
        values = (doc_id, chunk, source)
        c.execute("INSERT INTO docs (id, text, link) VALUES (?, ?, ?)", values)
        vec = model.encode([chunk])[0].astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        index.add_with_ids(np.array([vec]), np.array([next_idx], dtype=np.int64))
        docid_mapping[next_idx] = doc_id
        next_idx = next_idx + 1
    conn.commit()


def ingest_folder(folder_path: str):
    folder_path_clean = folder_path.rstrip("/\\")
    folder = Path(folder_path_clean)
    exists = folder.exists()
    if not exists:
        raise FileNotFoundError(f"{folder_path_clean} not found")
    for filepath in folder.iterdir():
        condition = filepath.is_file() and filepath.suffix.lower() in {".pdf", ".txt"}
        if condition:
            print(f"Ingesting {filepath} ...")
            if filepath.suffix.lower() == ".pdf":
                text = pdf_to_text(str(filepath))
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            chunks = chunk_text(text)
            store_chunks(chunks, source=filepath.name)
    faiss.write_index(index, INDEX_FILE)
    print("Ingestion complete. FAISS index saved.")


def similarity_search(q: str, k: int = 5) -> List[Dict]:
    vec = model.encode([q])[0].astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    search_vec = np.array([vec])
    scores, ids = index.search(search_vec, k)
    results: List[Dict] = []
    with sqlite3.connect(DB_FILE) as conn2:
        c2 = conn2.cursor()
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            idx = int(idx)
            doc_id = docid_mapping.get(idx)
            if not doc_id:
                continue
            values = (doc_id,)
            c2.execute("SELECT id, text, link FROM docs WHERE id=?", values)
            row = c2.fetchone()
            if row:
                result = {
                    "id": row[0],
                    "text": row[1],
                    "link": row[2],
                    "score_vec": float(score)
                }
                results.append(result)
    return results


def hybrid_rerank(q: str, base_results: List[Dict], k: int = 5) -> List[Dict]:
    safe = re.sub(r"[^\w\s]", " ", q).lower()
    tokens = [t for t in re.split(r"\s+", safe) if len(t) > 1]
    if not tokens:
        return base_results
    fts_query = " OR ".join(tokens)
    keyword_results = []
    with sqlite3.connect(DB_FILE) as conn2:
        c2 = conn2.cursor()
        try:
            values = (fts_query, k)
            c2.execute("SELECT id, text, link, bm25(docs) as score FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?", values)
            rows = c2.fetchall()
            for r in rows:
                keyword_results.append({"id": r[0], "text": r[1], "link": r[2], "score_kw_raw": -float(r[3])})
        except sqlite3.OperationalError:
            c2.execute("SELECT id, text, link FROM docs")
            for r in c2.fetchall():
                text = r[1].lower()
                matches = sum(1 for t in tokens if t in text)
                if matches:
                    keyword_results.append({"id": r[0], "text": r[1], "link": r[2], "score_kw_raw": float(matches)})
    vec_scores = [r.get("score_vec", 0.0) for r in base_results]
    kw_scores = [r["score_kw_raw"] for r in keyword_results] if keyword_results else [0.0]
    vec_min, vec_max = (min(vec_scores), max(vec_scores)) if vec_scores else (0.0, 0.0)
    kw_min, kw_max = (min(kw_scores), max(kw_scores)) if kw_scores else (0.0, 0.0)

    def norm(x, a, b):
        diff = b - a
        if diff == 0:
            return 0.0
        return (x - a) / diff

    merged = {r["id"]: dict(r) for r in base_results}
    for kr in keyword_results:
        if kr["id"] in merged:
            merged[kr["id"]]["score_kw_raw"] = kr["score_kw_raw"]
        else:
            merged[kr["id"]] = {"id": kr["id"], "text": kr["text"], "link": kr["link"], "score_vec": 0.0, "score_kw_raw": kr["score_kw_raw"]}
    final_list = []
    for r in merged.values():
        sv = r.get("score_vec", 0.0)
        sk_raw = r.get("score_kw_raw", 0.0)
        sv_n = norm(sv, vec_min, vec_max) if vec_max != vec_min else sv
        sk_n = norm(sk_raw, kw_min, kw_max) if kw_max != kw_min else sk_raw
        blended = ALPHA * sv_n + (1 - ALPHA) * sk_n
        r["score_vec_norm"] = sv_n
        r["score_kw_norm"] = sk_n
        r["score"] = blended
        final_list.append(r)
    final_sorted = sorted(final_list, key=lambda x: x["score"], reverse=True)[:k]
    return final_sorted


def cross_rerank(q: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    global _cross_reranker
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder is not available in this environment. Install sentence-transformers with the reranker models.")
    if _cross_reranker is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _cross_reranker = CrossEncoder(model_name)
    pairs = [(q, c["text"]) for c in candidates]
    scores = _cross_reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    return sorted_candidates


def search(q: str, k: int = 5, mode: str = "hybrid") -> List[Dict]:
    if mode == "vector":
        results = similarity_search(q, k)
        return results
    elif mode == "hybrid":
        base = similarity_search(q, k)
        reranked = hybrid_rerank(q, base, k)
        return reranked
    elif mode == "cross":
        base = similarity_search(q, k=CROSS_RERANK_TOP_K)
        reranked = cross_rerank(q, base, top_k=k)
        return reranked
    else:
        raise ValueError("mode must be one of: vector, hybrid, cross")


def extract_answer(text: str, query: str, max_sentences: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = []
    qtokens = [t.lower() for t in re.split(r"\s+", re.sub(r"[^\w\s]", " ", query)) if t]
    for s in sentences:
        low = s.lower()
        condition = any(q in low for q in qtokens)
        if condition:
            selected.append(s.strip())
        if len(selected) >= max_sentences:
            break
    if not selected:
        selected = sentences[:max_sentences]
    return " ".join(selected).strip()


def answer_question(q: str, k: int = 5, mode: str = "hybrid") -> Dict:
    results = search(q, k, mode)
    if not results:
        return {"answer": None, "contexts": [], "reranker_used": mode}
    top = results[0]
    score = top.get("score") or top.get("rerank_score") or top.get("score_vec", 0.0)
    if score < THRESHOLD:
        return {"answer": None, "contexts": results, "reranker_used": mode}
    answer_text = extract_answer(top["text"], q)
    return {"answer": f"{answer_text}\n\n(Source: {top['id']})", "contexts": results, "reranker_used": mode}


def compare_search(q: str, k: int = 5) -> Dict:
    vec_results = search(q, k=k, mode="vector")
    hybrid_results = search(q, k=k, mode="hybrid")
    cross_results = None
    try:
        cross_results = search(q, k=k, mode="cross")
    except Exception as e:
        cross_results = {"error": str(e)}

    def tidy(results):
        out = []
        if isinstance(results, list):
            for r in results:
                value = {
                    "id": r.get("id"),
                    "score_vec": r.get("score_vec"),
                    "score": r.get("score"),
                    "score_kw": r.get("score_kw_norm") or r.get("score_kw_raw") if r.get("score_kw_raw") is not None else r.get("score_kw_norm"),
                    "rerank_score": r.get("rerank_score"),
                    "text": r.get("text")[:300] + ("..." if len(r.get("text","")) > 300 else ""),
                    "link": r.get("link")
                }
                out.append(value)
        else:
            out = results
        return out

    return {
        "query": q,
        "vector_only": tidy(vec_results),
        "hybrid": tidy(hybrid_results),
        "cross_reranked": tidy(cross_results)
    }


app = FastAPI()


class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "hybrid"


class CompareRequest(BaseModel):
    q: str
    k: int = 5


@app.post("/ask")
def ask(req: AskRequest):
    cleaned = req.q.replace("\n", " ").replace("\r", " ").strip()
    req.q = cleaned
    response = answer_question(req.q, req.k, req.mode)
    return response


@app.post("/compare")
def compare(req: CompareRequest):
    cleaned = req.q.replace("\n", " ").replace("\r", " ").strip()
    req.q = cleaned
    response = compare_search(req.q, req.k)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

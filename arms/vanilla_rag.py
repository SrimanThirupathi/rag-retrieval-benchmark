"""Arm 1 — vanilla vector RAG. FAISS flat index, float32, no compression.

This is the reference point. Per HotpotQA-distractor question we index the
10 candidate paragraphs, embed the question, retrieve top-k by cosine
similarity, and stuff them into the prompt.
"""
import numpy as np
import faiss
import config
import llm

_model = None


def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def retrieve(item, k=None):
    """Returns top-k paragraphs by cosine similarity. Shared by subclass arms."""
    k = k or config.TOP_K
    texts = [p["title"] + ": " + p["text"] for p in item["paragraphs"]]
    emb = _embedder().encode(texts, normalize_embeddings=True)
    q = _embedder().encode([item["question"]], normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    _, idx = index.search(q.astype(np.float32), k)
    return [item["paragraphs"][i] for i in idx[0]]


def build_context(paragraphs):
    """Assemble context under the shared token budget (rough 4 chars/token)."""
    budget_chars = config.CONTEXT_TOKEN_BUDGET * 4
    parts, used = [], 0
    for p in paragraphs:
        block = f"[{p['title']}] {p['text']}"
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts), used // 4


def answer(item):
    retrieved = retrieve(item)
    context, ctx_tokens = build_context(retrieved)
    prompt = llm.PROMPT_TEMPLATE.format(context=context, question=item["question"])
    pred = llm.generate(prompt)
    return {
        "prediction": pred,
        "retrieved_titles": [p["title"] for p in retrieved],
        "context_tokens": ctx_tokens,
    }

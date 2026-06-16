"""Arm — vanilla RAG, pooled setting. FAISS float32 over the shared corpus."""
import numpy as np
import faiss
import config
import llm
from arms import vanilla_rag, pooled
from data.dataset import load_sample

_index = None
_corpus = None


def _get_index():
    global _index, _corpus
    if _index is None:
        items = load_sample()
        _corpus, emb = pooled.get_corpus_and_embeddings(
            items, vanilla_rag._embedder())
        _index = faiss.IndexFlatIP(emb.shape[1])
        _index.add(emb)
    return _index, _corpus


def retrieve(item, k=None):
    k = k or config.TOP_K
    index, corpus = _get_index()
    q = vanilla_rag._embedder().encode(
        [item["question"]], normalize_embeddings=True).astype(np.float32)
    _, idx = index.search(q, k)
    return [corpus[i] for i in idx[0]]


def answer(item):
    retrieved = retrieve(item)
    context, ctx_tokens = vanilla_rag.build_context(retrieved)
    prompt = llm.PROMPT_TEMPLATE.format(context=context, question=item["question"])
    pred = llm.generate(prompt)
    return {
        "prediction": pred,
        "retrieved_titles": [p["title"] for p in retrieved],
        "context_tokens": ctx_tokens,
    }

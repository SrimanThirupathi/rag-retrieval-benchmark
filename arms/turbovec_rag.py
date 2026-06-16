"""Arm 2 — TurboVec. Identical to vanilla_rag except the index.

The ONLY difference vs arm 1 is the vector store: TurboQuant-compressed
at config.TURBOVEC_BIT_WIDTH bits instead of FAISS float32. Any answer-quality
gap between arm 1 and arm 2 is therefore attributable to compression alone.

Requires: pip install turbovec
"""
import numpy as np
import config
import llm
from arms import vanilla_rag


def retrieve(item, k=None):
    from turbovec import TurboQuantIndex
    k = k or config.TOP_K
    texts = [p["title"] + ": " + p["text"] for p in item["paragraphs"]]
    emb = vanilla_rag._embedder().encode(texts, normalize_embeddings=True)
    q = vanilla_rag._embedder().encode([item["question"]], normalize_embeddings=True)
    index = TurboQuantIndex(dim=emb.shape[1], bit_width=config.TURBOVEC_BIT_WIDTH)
    index.add(emb.astype(np.float32))
    _, idx = index.search(q.astype(np.float32), k=k)
    return [item["paragraphs"][i] for i in np.asarray(idx).reshape(-1)[:k]]


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

"""Pooled corpus: ONE shared index over all paragraphs from all questions.

In the per-question setting each retrieval picks 5 from 10 candidates — too
easy for compression distortion to matter. Here we merge every paragraph
from the whole sample (~500, deduplicated by title) into a single index,
so each query must find its 2 gold paragraphs among ~500. This is both a
harder test and far closer to real RAG deployments.

Embeddings are computed once per run and cached at module level.
"""
import numpy as np

_corpus = None      # list of {"title", "text"}
_embeddings = None  # float32 [N, dim], L2-normalized


def get_corpus_and_embeddings(items, embedder):
    global _corpus, _embeddings
    if _corpus is None:
        seen = {}
        for item in items:
            for p in item["paragraphs"]:
                seen.setdefault(p["title"], p["text"])  # dedupe by title
        _corpus = [{"title": t, "text": x} for t, x in seen.items()]
        texts = [p["title"] + ": " + p["text"] for p in _corpus]
        print(f"  [pooled] embedding shared corpus: {len(_corpus)} paragraphs...")
        _embeddings = embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
    return _corpus, _embeddings

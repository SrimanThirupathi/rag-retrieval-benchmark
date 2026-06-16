"""Full-scale retrieval-only benchmark — no LLM involved.

Question under test: when searching the ENTIRE HotpotQA paragraph corpus
(~66k unique paragraphs) instead of a per-question pool of 10, do the
compressed (TurboVec, 4-bit) and uncompressed (FAISS, float32) indexes still
retrieve the same gold paragraphs?

We measure recall@k: for each question, what fraction of its 2 gold paragraphs
appear in the top-k retrieved. Averaged over all questions. We report both
indexes side by side at several k values, plus how often they AGREE on the
exact retrieved set (the most sensitive signal for compression distortion).

This needs no generator model, so it runs in minutes: one embedding pass over
the corpus, then two batched searches.

Usage:
  python retrieval_only.py                # all questions, k in {1,2,5,10,20}
  python retrieval_only.py --n 1000       # subset of questions (corpus still full)
  python retrieval_only.py --bits 3       # try 3-bit compression
"""
import argparse
import time
import numpy as np

import config
from arms import vanilla_rag  # reuse the shared embedder


def build_corpus():
    """All unique paragraphs across the whole HotpotQA validation split,
    plus each question with the titles of its gold paragraphs."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    title_to_id, corpus_titles, corpus_texts = {}, [], []
    questions = []
    for ex in ds:
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        for tt, ss in zip(titles, sents):
            if tt not in title_to_id:
                title_to_id[tt] = len(corpus_titles)
                corpus_titles.append(tt)
                corpus_texts.append(tt + ": " + " ".join(ss))
        gold = sorted(set(ex["supporting_facts"]["title"]))
        questions.append({"question": ex["question"],
                          "gold_ids": [title_to_id[g] for g in gold if g in title_to_id]})
    return corpus_titles, corpus_texts, questions


def recall_at_k(retrieved_ids, gold_ids_list, k):
    """Mean fraction of gold paragraphs found within top-k, over all questions."""
    tot = 0.0
    for row, gold in zip(retrieved_ids, gold_ids_list):
        if not gold:
            continue
        topk = set(row[:k])
        tot += sum(1 for g in gold if g in topk) / len(gold)
    return tot / len(gold_ids_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit number of questions")
    ap.add_argument("--bits", type=int, default=config.TURBOVEC_BIT_WIDTH)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    args = ap.parse_args()
    maxk = max(args.ks)

    print("Building corpus from full HotpotQA validation split...")
    titles, texts, questions = build_corpus()
    if args.n:
        questions = questions[:args.n]
    print(f"  corpus: {len(texts):,} unique paragraphs")
    print(f"  questions: {len(questions):,}")

    print("Embedding corpus (one-time, slow part)...")
    t0 = time.time()
    embedder = vanilla_rag._embedder()
    emb = embedder.encode(texts, normalize_embeddings=True,
                          show_progress_bar=True, batch_size=256).astype(np.float32)
    print(f"  embedded in {time.time()-t0:.0f}s, shape {emb.shape}")

    q_emb = embedder.encode([q["question"] for q in questions],
                            normalize_embeddings=True,
                            show_progress_bar=True, batch_size=256).astype(np.float32)
    gold_ids_list = [q["gold_ids"] for q in questions]

    # ---- FAISS float32 ----
    import faiss
    print("\nFAISS (float32) search...")
    t0 = time.time()
    findex = faiss.IndexFlatIP(emb.shape[1])
    findex.add(emb)
    _, faiss_ids = findex.search(q_emb, maxk)
    faiss_t = time.time() - t0
    faiss_mem = emb.nbytes

    # ---- TurboVec 4-bit ----
    from turbovec import TurboQuantIndex
    print(f"TurboVec ({args.bits}-bit) search...")
    t0 = time.time()
    tindex = TurboQuantIndex(dim=emb.shape[1], bit_width=args.bits)
    tindex.add(emb)
    _, tv_ids = tindex.search(q_emb, k=maxk)
    tv_ids = np.asarray(tv_ids)
    tv_t = time.time() - t0
    tv_mem = emb.nbytes * args.bits / 32.0  # approximate stored size

    # ---- results ----
    print("\n" + "=" * 64)
    print(f"{'k':>4} | {'FAISS recall':>13} | {'TurboVec recall':>15} | {'set agree@k':>11}")
    print("-" * 64)
    for k in args.ks:
        fr = recall_at_k(faiss_ids, gold_ids_list, k)
        tr = recall_at_k(tv_ids, gold_ids_list, k)
        agree = np.mean([len(set(a[:k]) & set(b[:k])) / k
                         for a, b in zip(faiss_ids, tv_ids)])
        print(f"{k:>4} | {fr:>13.4f} | {tr:>15.4f} | {agree:>11.4f}")
    print("=" * 64)
    print(f"corpus memory:  FAISS {faiss_mem/1e6:.0f} MB   "
          f"TurboVec ~{tv_mem/1e6:.0f} MB   ({32/args.bits:.0f}x smaller)")
    print(f"build+search:   FAISS {faiss_t:.1f}s   TurboVec {tv_t:.1f}s")


if __name__ == "__main__":
    main()

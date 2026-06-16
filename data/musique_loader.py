"""Load MuSiQue (answerable multi-hop QA) and draw a fixed random sample.

Why MuSiQue for the structure study: its questions are built by composing
2-4 single-hop questions into a chain, so the supporting paragraphs are
genuinely connected (the answer to hop 1 is the subject of hop 2). That
interconnection is exactly what the wiki method needs to demonstrate its
claimed advantage — unlike HotpotQA-distractor's mostly-independent puzzles.

Returns the SAME item shape as data/loader.py so every arm works unchanged:
  {id, question, answer, paragraphs:[{title,text}], gold_titles:[...]}

MuSiQue schema (HF, e.g. "dgslibisey/MuSiQue"):
  id, question, answer
  paragraphs: list of {idx, title, paragraph_text, is_supporting(bool)}
  question_decomposition: the hop chain (we don't need it for retrieval)

Downloads from HF on first run, cached after. If the dataset id differs on
your machine, edit MUSIQUE_HF_ID below.
"""
import random
import config

MUSIQUE_HF_ID = "dgslibisey/MuSiQue"   # fallback ids tried if this fails
_FALLBACK_IDS = ["bdsaglam/musique", "musique"]


def load_sample(n=None, seed=None):
    n = n or config.N_QUESTIONS
    seed = seed if seed is not None else config.RANDOM_SEED
    data = _from_huggingface()
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))


def _load_any():
    from datasets import load_dataset
    last = None
    for ds_id in [MUSIQUE_HF_ID, *_FALLBACK_IDS]:
        try:
            return load_dataset(ds_id, split="validation")
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not load MuSiQue from any known id: {last}")


def _from_huggingface():
    print("Loading MuSiQue from Hugging Face (first run only)...")
    ds = _load_any()
    items = []
    for ex in ds:
        paragraphs, gold_titles = [], []
        for p in ex["paragraphs"]:
            title = p.get("title", "")
            text = p.get("paragraph_text", p.get("text", ""))
            paragraphs.append({"title": title, "text": text})
            if p.get("is_supporting"):
                gold_titles.append(title)
        items.append({
            "id": str(ex["id"]),
            "question": ex["question"],
            "answer": ex["answer"],
            "paragraphs": paragraphs,
            "gold_titles": sorted(set(gold_titles)),
            "n_hops": len(ex.get("question_decomposition", [])) or None,
        })
    return items


if __name__ == "__main__":
    items = load_sample()
    print(f"Loaded {len(items)} questions. First one:")
    print(" Q:", items[0]["question"])
    print(" A:", items[0]["answer"])
    print(" paragraphs:", len(items[0]["paragraphs"]),
          "| gold (supporting):", len(items[0]["gold_titles"]),
          "| hops:", items[0]["n_hops"])
    print(" gold titles:", items[0]["gold_titles"])

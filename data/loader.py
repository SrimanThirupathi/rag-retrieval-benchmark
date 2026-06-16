"""Load HotpotQA (distractor setting) and draw a fixed random sample.

Two sources, tried in order:
  1. Local file at config.DATASET_PATH (the original CMU JSON), if present
  2. Hugging Face hub ("hotpotqa/hotpot_qa", distractor config) — automatic
     download, cached locally after first run. Requires: pip install datasets

Each item gives us:
  - question, answer (gold)
  - paragraphs: 10 candidates (2 gold "supporting" + 8 distractors)
  - gold_titles: which paragraphs are the true evidence
"""
import json
import os
import random
import config


def load_sample():
    if os.path.exists(config.DATASET_PATH):
        data = _from_local_json()
    else:
        data = _from_huggingface()
    rng = random.Random(config.RANDOM_SEED)
    return rng.sample(data, config.N_QUESTIONS)


def _from_local_json():
    with open(config.DATASET_PATH) as f:
        raw = json.load(f)
    items = []
    for ex in raw:
        paragraphs = [{"title": t, "text": " ".join(s)} for t, s in ex["context"]]
        gold_titles = sorted({t for t, _ in ex["supporting_facts"]})
        items.append({"id": ex["_id"], "question": ex["question"],
                      "answer": ex["answer"], "paragraphs": paragraphs,
                      "gold_titles": gold_titles})
    return items


def _from_huggingface():
    from datasets import load_dataset
    print("Downloading HotpotQA from Hugging Face (first run only, ~50 MB)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        paragraphs = [{"title": t, "text": " ".join(s)}
                      for t, s in zip(titles, sents)]
        gold_titles = sorted(set(ex["supporting_facts"]["title"]))
        items.append({"id": ex["id"], "question": ex["question"],
                      "answer": ex["answer"], "paragraphs": paragraphs,
                      "gold_titles": gold_titles})
    return items


if __name__ == "__main__":
    items = load_sample()
    print(f"Loaded {len(items)} questions. First one:")
    print(" Q:", items[0]["question"])
    print(" A:", items[0]["answer"])
    print(" gold paragraphs:", items[0]["gold_titles"])

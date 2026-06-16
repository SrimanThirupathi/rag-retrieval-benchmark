"""Run the benchmark.

  python run_benchmark.py closed_book vanilla turbovec

Per arm, per question: prediction, EM, F1, retrieval recall, context tokens,
wall-clock latency. Saves JSONL per arm + prints a summary table.
"""
import importlib
import json
import os
import statistics
import sys
import time

import config
from data.dataset import load_sample
from eval import scoring

ARMS = {
    "closed_book": "arms.closed_book",
    "vanilla": "arms.vanilla_rag",
    "turbovec": "arms.turbovec_rag",
    "vanilla_pooled": "arms.vanilla_pooled",
    "turbovec_pooled": "arms.turbovec_pooled",
    "wiki": "arms.wiki_rag",
}


def run_arm(name, items):
    arm = importlib.import_module(ARMS[name])
    rows = []
    for i, item in enumerate(items):
        t0 = time.time()
        out = arm.answer(item)
        latency = time.time() - t0
        row = {
            "id": item["id"],
            "question": item["question"],
            "gold": item["answer"],
            "prediction": out["prediction"],
            "em": scoring.exact_match(out["prediction"], item["answer"]),
            "f1": scoring.f1(out["prediction"], item["answer"]),
            "retrieval_recall": scoring.retrieval_recall(
                out["retrieved_titles"], item["gold_titles"]),
            "multihop_recall": scoring.multihop_recall(
                out["retrieved_titles"], item["gold_titles"]),
            "context_tokens": out["context_tokens"],
            "compile_tokens": out.get("compile_tokens", 0),
            "latency_s": round(latency, 2),
        }
        rows.append(row)
        print(f"  [{name}] {i+1}/{len(items)} EM={row['em']:.0f} F1={row['f1']:.2f} "
              f"({latency:.1f}s)  {item['question'][:60]}")
    return rows


def summarize(name, rows):
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else float("nan")
    return {
        "arm": name,
        "n": len(rows),
        "EM": round(mean("em"), 3),
        "F1": round(mean("f1"), 3),
        "retr_recall": round(mean("retrieval_recall"), 3),
        "mhop_recall": round(mean("multihop_recall"), 3),
        "ctx_tokens": round(mean("context_tokens")),
        "compile_tok": round(mean("compile_tokens")),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
    }


def main():
    arm_names = sys.argv[1:] or ["closed_book", "vanilla"]
    items = load_sample()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summaries = []
    for name in arm_names:
        print(f"\n=== Running arm: {name} ===")
        rows = run_arm(name, items)
        path = os.path.join(config.RESULTS_DIR, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summaries.append(summarize(name, rows))

    print("\n" + "=" * 70)
    keys = ["arm", "n", "EM", "F1", "retr_recall", "mhop_recall", "ctx_tokens", "compile_tok", "p50_latency_s"]
    print(" | ".join(f"{k:>13}" for k in keys))
    for s in summaries:
        print(" | ".join(f"{str(s[k]):>13}" for k in keys))


if __name__ == "__main__":
    main()

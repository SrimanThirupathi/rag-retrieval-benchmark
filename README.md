# rag-bench — Vector RAG vs TurboVec vs Wiki-style Retrieval

A small, fair, end-to-end benchmark harness comparing retrieval philosophies on HotpotQA (multi-hop QA).

## Arms

| Arm | What it tests |
|---|---|
| `closed_book` | Pretraining contamination floor — no retrieval at all |
| `vanilla` | FAISS float32 flat index (the reference) |
| `turbovec` | Identical pipeline, TurboQuant 4-bit compressed index — isolates the cost of compression |
| `wiki` (todo) | Karpathy-style compiled wiki / Graphify navigation |

Fairness controls baked in: same generator LLM, same prompt template, same fixed question sample (seeded), same context token budget for every arm.

## Setup

```bash
pip install sentence-transformers faiss-cpu turbovec datasets
```

Generator LLM — pick one in `config.py`:
- **Ollama (free, local, default):** install Ollama, `ollama pull llama3.1:8b`
- **Anthropic API:** set `LLM_BACKEND = "anthropic"` and export `ANTHROPIC_API_KEY`

Dataset — downloads automatically from Hugging Face on first run (no manual step). Just make sure `datasets` is installed:

```bash
pip install datasets
```

Sanity check: `python data/loader.py`

## Run

```bash
python run_benchmark.py closed_book vanilla        # step 1: prove harness works
python run_benchmark.py turbovec                   # step 2: drop in compression
```

Per-question results land in `results/<arm>.jsonl`; a summary table prints at the end.

## Reading the results

- **closed_book EM high (>0.5)?** Questions are memorized — increase N_QUESTIONS or note contamination in the writeup.
- **vanilla vs turbovec gap on EM/F1?** That gap is the *real* cost of 4-bit compression — the number TurboVec's own benchmarks (recall@k only) don't show.
- **retr_recall** of 0.5 means one of the two multi-hop paragraphs was missed — the question was unanswerable from context. Watch how often this differs between arms.

## Why HotpotQA distractor setting

Each question ships with 10 candidate paragraphs (2 gold + 8 distractors), so retrieval is testable per-question without indexing all of Wikipedia. Cheap, fair, multi-hop. Once this works, scale to the fullwiki setting or add Natural Questions / QASPER.

## Full-scale retrieval-only test

No LLM needed. Pools the entire HotpotQA validation corpus (~66k paragraphs) and compares FAISS vs TurboVec on recall@k:

```bash
python retrieval_only.py            # full corpus, all 7405 questions
python retrieval_only.py --n 1000   # fewer questions (corpus stays full)
```

The `set agree@k` column is the sensitive one: fraction of the top-k that the two indexes retrieve identically. If compression ever distorts ranking, it shows here first.

## Structure study (MuSiQue): wiki vs vanilla

Switch dataset in config.py: `DATASET = "musique"`. Then re-baseline the existing arms on MuSiQue and add the wiki arm:

```bash
python run_benchmark.py closed_book vanilla turbovec wiki
```

New metrics in the table:
- `mhop_recall` — all-or-nothing: did retrieval find ALL gold (bridge) paragraphs? The decisive wiki-vs-vanilla number.
- `compile_tok` — upfront LLM tokens the wiki spends organizing (the cost vanilla doesn't pay).

The wiki arm calls the LLM once per paragraph to compile articles, so it is much slower per question. Start with a small N_QUESTIONS (e.g. 20) in config.py to iterate.

## Next steps (in order)

1. Run closed_book + vanilla on 50 Qs → verify table looks sane
2. Add turbovec arm (one `pip install`)
3. Build the wiki arm: compile sampled paragraphs into a linked Markdown wiki via an LLM pass, retrieval = load relevant wiki page(s) under the same token budget. Count compilation tokens as amortized indexing cost.
4. Scale N_QUESTIONS to 200–500, add a second dataset, plot quality-vs-tokens Pareto curve

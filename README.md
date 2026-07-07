# RAG Retrieval Benchmark

A controlled, end-to-end benchmark comparing three families of retrieval for question answering: compressed, vanilla, and structured (wiki-style). Every method is evaluated under identical, fair conditions (same generator model, same embedder, same context token budget, same fixed question sample), so that any difference in results is attributable to the retrieval strategy alone.

The project runs two studies:

1. **Compression study** (HotpotQA): does compressing the stored vectors hurt retrieval?
2. **Structure study** (MuSiQue): does organizing documents into a linked wiki beat plain similarity search on multi-hop questions?

---

## Key Results

### Compression study (HotpotQA, 66K paragraphs, 7,405 questions)

TurboQuant 4-bit vector compression matches full-precision FAISS recall to within 0.001 at every retrieval depth, giving an 8x memory reduction with no measurable loss. Pushing to 2-bit (16x reduction) introduces a small, consistent recall loss of about 1 percentage point, marking the practical limit of free compression.

| Retrieval depth | FAISS (float32) | TurboQuant 4-bit | TurboQuant 2-bit |
|-----------------|-----------------|------------------|------------------|
| Top 1  | 0.379 | 0.379 | 0.374 |
| Top 2  | 0.516 | 0.516 | 0.511 |
| Top 5  | 0.637 | 0.638 | 0.629 |
| Top 10 | 0.703 | 0.702 | 0.693 |
| Top 20 | 0.752 | 0.751 | 0.743 |

A question-by-question comparison confirmed the compressed and uncompressed indexes retrieve independently (agreeing on ~94% of the top-k at 4-bit), rather than one silently mirroring the other.

### Structure study (MuSiQue, multi-hop, n=20)

Wiki-style retrieval (an LLM compiles each paragraph into a linked article, then retrieval follows those links) doubles strict multi-hop recall over similarity search. This confirms the core claim that structured retrieval catches the connected bridge paragraph that plain similarity search tends to miss. The gain comes at a large upfront compilation cost and roughly 27x higher per-query latency.

| Method | Exact Match | F1 | Retrieval recall | Multi-hop recall |
|--------|-------------|-----|------------------|------------------|
| No documents (baseline) | 0.05 | 0.062 | n/a | n/a |
| Vanilla RAG | 0.15 | 0.236 | 0.629 | 0.25 |
| TurboVec (compressed) | 0.10 | 0.196 | 0.629 | 0.25 |
| Wiki-style | 0.25 | 0.397 | 0.758 | 0.50 |

Multi-hop recall is all-or-nothing: it scores 1 only when both required bridge paragraphs were retrieved, since finding one of two leaves a multi-hop question unanswerable.

> Note: the structure study uses a small sample (n=20). The doubling is a clear signal, not a precision measurement. The robust takeaway is the mechanism (link-following recovers the bridge paragraph), with a larger-sample run as the natural next step.

---

## Background

Retrieval-Augmented Generation (RAG) gives a language model access to outside knowledge by embedding document chunks into vectors, storing them in a searchable index, and retrieving the closest chunks to a question at query time. Two recent ideas challenge this standard design:

* **Compression (TurboQuant / TurboVec):** shrink each stored vector to a few bits to cut memory, claiming negligible loss. Their published numbers report retrieval recall only, not end-to-end answer quality.
* **Structured retrieval (Karpathy's "LLM Wiki" / Graphify):** replace similarity search with an LLM-compiled wiki of linked summary articles, claiming better answers on questions that combine multiple facts. This was demonstrated informally, never measured against a baseline on a standard benchmark.

This project tests both claims under controlled conditions.

---

## The Arms

Each retrieval method is a swappable module ("arm"). All arms share the same generator, embedder, prompt template, and context token budget.

| Arm | Retrieval method |
|-----|------------------|
| `closed_book` | No retrieval; the model answers from its own knowledge (contamination floor) |
| `vanilla` | Standard RAG: embed the question, retrieve the top-k most similar paragraphs via FAISS |
| `turbovec` | Identical to vanilla, but the index is TurboQuant-compressed (4-bit by default) |
| `vanilla_pooled` | Vanilla over one shared corpus pooling all questions' paragraphs |
| `turbovec_pooled` | TurboVec over the same shared pooled corpus |
| `wiki` | LLM compiles paragraphs into linked articles; retrieval follows links to gather connected articles |

---

## Setup

```bash
pip install sentence-transformers faiss-cpu turbovec datasets
```

The generator model is configurable in `config.py`. Two backends are supported:

* **Ollama (default, free, local):** install Ollama, then `ollama pull llama3.1:8b`
* **Anthropic API:** set `LLM_BACKEND = "anthropic"` and export `ANTHROPIC_API_KEY`

Datasets download automatically from Hugging Face on first run (HotpotQA and MuSiQue). No manual download step is required.

Sanity check the data loaders:

```bash
python -m data.loader          # HotpotQA
python -m data.musique_loader  # MuSiQue
```

---

## Running

Select the dataset in `config.py` with `DATASET = "hotpot"` or `DATASET = "musique"`, then run any set of arms:

```bash
# Compression study (HotpotQA)
python run_benchmark.py closed_book vanilla turbovec

# Larger pooled-corpus variant
python run_benchmark.py vanilla_pooled turbovec_pooled

# Structure study (set DATASET = "musique" first)
python run_benchmark.py closed_book vanilla turbovec wiki
```

Per-question results are written to `results/<arm>.jsonl`, and a summary table is printed at the end.

### Full-scale retrieval test (no generator model)

Compares FAISS vs TurboVec on recall across the entire HotpotQA corpus (~66K paragraphs). This test involves only embedding and searching, so it runs in minutes:

```bash
python retrieval_only.py            # full corpus, all questions
python retrieval_only.py --n 1000   # fewer questions, full corpus
python retrieval_only.py --bits 2   # try 2-bit compression
```

---

## Metrics

* **Exact Match (EM):** strict; 1 only if the answer matches the gold answer exactly after normalization.
* **F1:** token-overlap partial credit between prediction and gold answer.
* **Retrieval recall:** fraction of gold supporting paragraphs retrieved (partial credit).
* **Multi-hop recall:** all-or-nothing; 1 only if every gold paragraph was retrieved. The decisive metric for the structure study.
* **Context tokens:** amount of retrieved text fed to the model per question.
* **Compile tokens:** upfront tokens the wiki arm spends organizing paragraphs (the cost vanilla does not pay).
* **Latency:** median time per question.

---

## Fairness Controls

For the comparisons to be valid, everything except the retrieval method is held constant across arms:

* Same generator model and settings.
* Same embedding model for all vector arms.
* Same context token budget everywhere.
* Same fixed, seeded sample of questions.

Comparisons are only ever made within a single dataset. When the structure study switched to MuSiQue, the vanilla and turbovec baselines were re-run on MuSiQue so that all four arms are directly comparable on the same data.

---

## Project Structure

```
config.py             Central config: dataset, model, k, token budget, seed
llm.py                Single generate() used by every arm (generator parity)
run_benchmark.py      Runner: executes selected arms, scores, prints summary
retrieval_only.py     Full-scale retrieval-only recall test (no generator)
data/
  loader.py           HotpotQA loader
  musique_loader.py   MuSiQue loader
  dataset.py          Dispatcher selecting the loader from config
arms/
  closed_book.py      No-retrieval baseline
  vanilla_rag.py      Standard FAISS RAG
  turbovec_rag.py     Compressed-index RAG
  vanilla_pooled.py   Vanilla over a shared pooled corpus
  turbovec_pooled.py  TurboVec over a shared pooled corpus
  wiki_rag.py         Wiki-style link-following retrieval
eval/
  scoring.py          EM, F1, retrieval recall, multi-hop recall
```

---

## Limitations and Next Steps

* The structure study uses n=20; a run of 100 or more questions is needed to state the size of the advantage with confidence.
* The wiki arm's quality depends on the compiler model being good enough to identify the right links; a stronger model may widen the advantage.
* The wiki arm is tested per-question, which exercises its core mechanism but not the claimed benefit on a single large interconnected corpus. Testing that would require a dataset built as one connected corpus.
* Natural extensions: scale up the structure study, inspect the compiled articles to confirm which links drove the wins, and add an LLM-as-judge metric to complement exact match.

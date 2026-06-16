"""Arm — Wiki-style retrieval (Karpathy's method, per-question scope).

Pipeline per question:
  BUILD (once per question):
    paragraphs -> LLM compiles each into an article {title, summary, links}
    where `links` names other articles this one connects to.
  RETRIEVE:
    embed question against article SUMMARIES -> pick best article
    -> follow its links -> pull connected articles (up to TOP_K total)
  ANSWER:
    same generator, same prompt, same token budget as vanilla.

This tests Karpathy's headline claim: that pre-built links between articles
help find the connected (bridge) paragraph that plain similarity search misses.

Costs are tracked: `compile_tokens` (the upfront LLM cost vanilla doesn't pay).
The compiler is instructed to SUMMARIZE/ORGANIZE, never to answer the
question — checked to avoid answer leakage.
"""
import json
import re
import numpy as np
import config
import llm
from arms import vanilla_rag

COMPILE_PROMPT = (
    "You are organizing source paragraphs into a small wiki. "
    "For the given paragraph, write a JSON object with:\n"
    '  "title": a short article title (the main entity/topic),\n'
    '  "summary": 1-2 sentences capturing the key facts,\n'
    '  "links": a list of other entities/topics this paragraph references '
    "that would each deserve their own article.\n"
    "Do NOT answer any question. Only summarize and identify references.\n"
    "Reply with ONLY the JSON object.\n\n"
    "Paragraph:\n{para}\n\nJSON:"
)


def _safe_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_wiki(item):
    """Compile each paragraph into an article. Returns (articles, compile_tokens)."""
    articles = []
    compile_tokens = 0
    for p in item["paragraphs"]:
        prompt = COMPILE_PROMPT.format(para=f"{p['title']}: {p['text']}")
        raw = llm.generate(prompt)
        compile_tokens += len(prompt) // 4 + len(raw) // 4  # rough token estimate
        obj = _safe_json(raw) or {}
        articles.append({
            "title": obj.get("title") or p["title"],
            "summary": obj.get("summary") or p["text"][:200],
            "links": [str(x) for x in (obj.get("links") or [])],
            "source_title": p["title"],
            "text": p["text"],
        })
    return articles, compile_tokens


def _match_link(link, articles):
    """Resolve a link string to an article index by fuzzy title overlap."""
    link_l = link.lower()
    best, best_score = None, 0
    for i, a in enumerate(articles):
        title_l = a["title"].lower()
        if link_l in title_l or title_l in link_l:
            return i
        overlap = len(set(link_l.split()) & set(title_l.split()))
        if overlap > best_score:
            best, best_score = i, overlap
    return best if best_score > 0 else None


def retrieve(item, articles, k=None):
    k = k or config.TOP_K
    emb = vanilla_rag._embedder()
    summaries = [a["title"] + ": " + a["summary"] for a in articles]
    svec = emb.encode(summaries, normalize_embeddings=True).astype(np.float32)
    qvec = emb.encode([item["question"]], normalize_embeddings=True).astype(np.float32)
    sims = (svec @ qvec[0])
    order = list(np.argsort(-sims))

    chosen = []
    # seed with best-matching article, then follow its links, then fill by similarity
    seed = order[0]
    chosen.append(seed)
    for link in articles[seed]["links"]:
        idx = _match_link(link, articles)
        if idx is not None and idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= k:
            break
    for idx in order:
        if len(chosen) >= k:
            break
        if idx not in chosen:
            chosen.append(idx)
    return [articles[i] for i in chosen[:k]]


def answer(item):
    articles, compile_tokens = build_wiki(item)
    retrieved = retrieve(item, articles)
    # build context from the retrieved articles' source text, same budget as vanilla
    paras = [{"title": a["source_title"], "text": a["text"]} for a in retrieved]
    context, ctx_tokens = vanilla_rag.build_context(paras)
    prompt = llm.PROMPT_TEMPLATE.format(context=context, question=item["question"])
    pred = llm.generate(prompt)
    return {
        "prediction": pred,
        "retrieved_titles": [a["source_title"] for a in retrieved],
        "context_tokens": ctx_tokens,
        "compile_tokens": compile_tokens,
    }

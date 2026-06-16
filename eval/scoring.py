"""Scoring: HotpotQA-style Exact Match and token-level F1, plus
retrieval recall (did the gold supporting paragraphs make it into context?).
Normalization mirrors the official HotpotQA eval script.
"""
import re
import string
from collections import Counter


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(retrieved_titles, gold_titles):
    """Fraction of gold supporting paragraphs that were retrieved.
    HotpotQA needs BOTH gold paragraphs for multi-hop — so 1.0 means the
    answer was reachable, 0.5 means one hop was missing."""
    if not gold_titles or not retrieved_titles:
        return None  # arm did no retrieval (e.g. closed book) -> not applicable
    hit = sum(1 for t in gold_titles if t in retrieved_titles)
    return hit / len(gold_titles)


def multihop_recall(retrieved_titles, gold_titles):
    """All-or-nothing recall: 1.0 only if EVERY gold paragraph was retrieved.
    For multi-hop questions, finding only some of the bridge paragraphs leaves
    the question unanswerable, so partial credit is misleading. This is the
    decisive metric for the wiki-vs-vanilla comparison."""
    if not gold_titles or not retrieved_titles:
        return None
    return 1.0 if all(t in retrieved_titles for t in gold_titles) else 0.0

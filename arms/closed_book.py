"""Arm 0 — closed book. No retrieval at all.

Purpose: quantify pretraining contamination. If the model already answers
70% correctly with zero context, the questions are too easy / memorized and
retrieval differences will be invisible.
"""
import llm


def answer(item):
    prompt = llm.CLOSED_BOOK_TEMPLATE.format(question=item["question"])
    pred = llm.generate(prompt)
    return {"prediction": pred, "retrieved_titles": [], "context_tokens": 0}

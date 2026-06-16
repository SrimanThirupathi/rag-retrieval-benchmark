"""One generate() function used by every arm — guarantees generator parity."""
import json
import urllib.request
import config

PROMPT_TEMPLATE = (
    "Answer the question using the context below. "
    "Reply with ONLY the answer phrase, no explanation.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

CLOSED_BOOK_TEMPLATE = (
    "Answer the question from your own knowledge. "
    "Reply with ONLY the answer phrase, no explanation.\n\n"
    "Question: {question}\nAnswer:"
)


def generate(prompt: str) -> str:
    if config.LLM_BACKEND == "ollama":
        return _ollama(prompt)
    elif config.LLM_BACKEND == "anthropic":
        return _anthropic(prompt)
    raise ValueError(f"unknown backend {config.LLM_BACKEND}")


def _ollama(prompt: str) -> str:
    body = json.dumps({
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "num_predict": config.MAX_TOKENS},
    }).encode()
    req = urllib.request.Request(
        config.OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


def _anthropic(prompt: str) -> str:
    import os
    body = json.dumps({
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        })
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data["content"]).strip()

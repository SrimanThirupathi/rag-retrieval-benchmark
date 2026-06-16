"""Central config — every experimental knob lives here so runs are reproducible."""

# ---- Data ----
DATASET = "musique"        # "hotpot" or "musique" — selects which loader is used
DATASET_PATH = "data/hotpot_dev_distractor_v1.json"  # see data/README.md
N_QUESTIONS = 20          # start small; scale to 200-500 later
RANDOM_SEED = 42          # fixed sample so all arms see identical questions

# ---- Generator LLM (pluggable: "ollama" or "anthropic") ----
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 256

# ---- Retrieval (vector arms) ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free
CHUNK_SIZE = 0            # 0 = use HotpotQA's natural paragraph units (recommended)
TOP_K = 5                 # chunks injected into the prompt
CONTEXT_TOKEN_BUDGET = 2000  # hard cap, enforced identically across all arms

# ---- TurboVec arm ----
TURBOVEC_BIT_WIDTH = 4

# ---- Output ----
RESULTS_DIR = "results"

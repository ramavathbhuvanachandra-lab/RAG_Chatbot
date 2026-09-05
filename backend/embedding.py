import os

from langchain_ollama import OllamaEmbeddings


# =========================================================
# Ollama Configuration
# =========================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# =========================================================
# Embedding Model
# =========================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url=OLLAMA_URL,
)
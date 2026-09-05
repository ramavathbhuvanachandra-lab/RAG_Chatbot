import os

from langchain_ollama import ChatOllama


# =========================================================
# Ollama Configuration
# =========================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# =========================================================
# Query / Conversation Processing LLM
# =========================================================

query_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url=OLLAMA_URL,
)


# =========================================================
# Final Answer Generation LLM
# =========================================================

answer_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url=OLLAMA_URL,
)


# =========================================================
# Backward Compatibility
# =========================================================

# Existing code may still import `llm`.
# Keep this alias pointed at the query-processing model so
# existing query/conversation code remains compatible.
llm = query_llm
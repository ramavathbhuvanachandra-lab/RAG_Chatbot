"""Reusable Ollama LLM runtime for the current English-only RAG core.

Current production milestone:
- qwen2.5:3b is used for query understanding.
- qwen2.5:3b is used for general query processing/rewrite helpers.
- qwen2.5:3b is used for final answer generation.

The model remains environment-overridable so the multilingual model can be
introduced later without changing the core architecture.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama


# ============================================================
# Environment helpers
# ============================================================


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().casefold()
    return value in {"1", "true", "yes", "on"}


# ============================================================
# Ollama
# ============================================================

OLLAMA_URL = _env(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_BASE_URL = OLLAMA_URL


# ============================================================
# Models
# ============================================================

# English-only baseline for the current architecture milestone.
# Keep these environment-overridable for the later multilingual model swap.
DEFAULT_MODEL = "qwen2.5:3b"

QUERY_LLM_MODEL = _env(
    "QUERY_LLM_MODEL",
    DEFAULT_MODEL,
)

QUERY_UNDERSTANDING_LLM_MODEL = _env(
    "QUERY_UNDERSTANDING_LLM_MODEL",
    QUERY_LLM_MODEL,
)

ANSWER_LLM_MODEL = _env(
    "ANSWER_LLM_MODEL",
    DEFAULT_MODEL,
)


# ============================================================
# Common generation settings
# ============================================================

LLM_TEMPERATURE = _env_float(
    "LLM_TEMPERATURE",
    0.0,
)


# ============================================================
# General query model
# ============================================================

QUERY_LLM_CONTEXT = _env_int(
    "QUERY_LLM_CONTEXT",
    4096,
)

QUERY_LLM_MAX_OUTPUT = _env_int(
    "QUERY_LLM_MAX_OUTPUT",
    160,
)

QUERY_LLM_TIMEOUT = _env_int(
    "QUERY_LLM_TIMEOUT",
    60,
)

QUERY_LLM_KEEP_ALIVE = _env(
    "QUERY_LLM_KEEP_ALIVE",
    "5m",
)

QUERY_LLM_THINKING = _env_bool(
    "QUERY_LLM_THINKING",
    False,
)


# ============================================================
# Semantic query-understanding model
# ============================================================

QUERY_UNDERSTANDING_CONTEXT = _env_int(
    "QUERY_UNDERSTANDING_CONTEXT",
    2048,
)

QUERY_UNDERSTANDING_MAX_OUTPUT = _env_int(
    "QUERY_UNDERSTANDING_MAX_OUTPUT",
    256,
)

QUERY_UNDERSTANDING_TIMEOUT = _env_int(
    "QUERY_UNDERSTANDING_TIMEOUT",
    45,
)

QUERY_UNDERSTANDING_KEEP_ALIVE = _env(
    "QUERY_UNDERSTANDING_KEEP_ALIVE",
    "5m",
)

QUERY_UNDERSTANDING_THINKING = _env_bool(
    "QUERY_UNDERSTANDING_THINKING",
    False,
)


# ============================================================
# Answer model
# ============================================================

ANSWER_LLM_CONTEXT = _env_int(
    "ANSWER_LLM_CONTEXT",
    4096,
)

ANSWER_LLM_TIMEOUT = _env_int(
    "ANSWER_LLM_TIMEOUT",
    120,
)

ANSWER_LLM_KEEP_ALIVE = _env(
    "ANSWER_LLM_KEEP_ALIVE",
    "5m",
)

# Answers should remain concise. This also prevents unnecessary generation
# time on the laptop during the architecture-quality milestone.
ANSWER_LLM_MAX_OUTPUT = _env_int(
    "ANSWER_LLM_MAX_OUTPUT",
    256,
)

ANSWER_LLM_THINKING = _env_bool(
    "ANSWER_LLM_THINKING",
    False,
)


# ============================================================
# Ollama client timeout kwargs
# ============================================================


def _client_kwargs(timeout: int) -> dict:
    return {
        "timeout": timeout,
    }


# ============================================================
# Semantic query-understanding model
# ============================================================

query_understanding_llm = ChatOllama(
    model=QUERY_UNDERSTANDING_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_URL,
    reasoning=QUERY_UNDERSTANDING_THINKING,
    num_ctx=QUERY_UNDERSTANDING_CONTEXT,
    num_predict=QUERY_UNDERSTANDING_MAX_OUTPUT,
    format="json",
    keep_alive=QUERY_UNDERSTANDING_KEEP_ALIVE,
    client_kwargs=_client_kwargs(
        QUERY_UNDERSTANDING_TIMEOUT,
    ),
)


# ============================================================
# General query-processing model
# ============================================================

query_llm = ChatOllama(
    model=QUERY_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_URL,
    reasoning=QUERY_LLM_THINKING,
    num_ctx=QUERY_LLM_CONTEXT,
    num_predict=QUERY_LLM_MAX_OUTPUT,
    keep_alive=QUERY_LLM_KEEP_ALIVE,
    client_kwargs=_client_kwargs(
        QUERY_LLM_TIMEOUT,
    ),
)


# ============================================================
# Final answer model
# ============================================================

answer_llm = ChatOllama(
    model=ANSWER_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_URL,
    reasoning=ANSWER_LLM_THINKING,
    num_ctx=ANSWER_LLM_CONTEXT,
    num_predict=ANSWER_LLM_MAX_OUTPUT,
    keep_alive=ANSWER_LLM_KEEP_ALIVE,
    client_kwargs=_client_kwargs(
        ANSWER_LLM_TIMEOUT,
    ),
)


# Existing query/rewrite code uses `llm`.
llm = query_llm


__all__ = [
    "OLLAMA_URL",
    "OLLAMA_BASE_URL",
    "DEFAULT_MODEL",
    "QUERY_LLM_MODEL",
    "QUERY_UNDERSTANDING_LLM_MODEL",
    "ANSWER_LLM_MODEL",
    "LLM_TEMPERATURE",
    "QUERY_LLM_CONTEXT",
    "QUERY_LLM_MAX_OUTPUT",
    "QUERY_LLM_TIMEOUT",
    "QUERY_LLM_KEEP_ALIVE",
    "QUERY_LLM_THINKING",
    "QUERY_UNDERSTANDING_CONTEXT",
    "QUERY_UNDERSTANDING_MAX_OUTPUT",
    "QUERY_UNDERSTANDING_TIMEOUT",
    "QUERY_UNDERSTANDING_KEEP_ALIVE",
    "QUERY_UNDERSTANDING_THINKING",
    "ANSWER_LLM_CONTEXT",
    "ANSWER_LLM_TIMEOUT",
    "ANSWER_LLM_KEEP_ALIVE",
    "ANSWER_LLM_MAX_OUTPUT",
    "ANSWER_LLM_THINKING",
    "query_understanding_llm",
    "query_llm",
    "answer_llm",
    "llm",
]
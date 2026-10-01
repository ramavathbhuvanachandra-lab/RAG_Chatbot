"""Ollama model configuration for the reusable RAG engine.

Model selection and inference budgets are deployment configuration,
not application architecture.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama


# ============================================================
# Helpers
# ============================================================


def _env(
    name: str,
    default: str,
) -> str:
    return os.getenv(
        name,
        default,
    ).strip()


def _env_int(
    name: str,
    default: int,
) -> int:
    try:
        return int(
            os.getenv(
                name,
                str(default),
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _env_float(
    name: str,
    default: float,
) -> float:
    try:
        return float(
            os.getenv(
                name,
                str(default),
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(
        name,
        str(default),
    ).strip().casefold()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# Ollama
# ============================================================


OLLAMA_URL = _env(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ============================================================
# Models
# ============================================================


QUERY_LLM_MODEL = _env(
    "QUERY_LLM_MODEL",
    "qwen3:4b",
)

QUERY_UNDERSTANDING_LLM_MODEL = _env(
    "QUERY_UNDERSTANDING_LLM_MODEL",
    QUERY_LLM_MODEL,
)

ANSWER_LLM_MODEL = _env(
    "ANSWER_LLM_MODEL",
    "qwen3:4b",
)


# ============================================================
# Common generation settings
# ============================================================


LLM_TEMPERATURE = _env_float(
    "LLM_TEMPERATURE",
    0.0,
)


# ============================================================
# Query model settings
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
# Semantic query-understanding settings
# ============================================================


QUERY_UNDERSTANDING_CONTEXT = _env_int(
    "QUERY_UNDERSTANDING_CONTEXT",
    2048,
)

# 256 is intentionally large enough to prevent structured JSON
# truncation while still keeping the semantic parser cheap.
QUERY_UNDERSTANDING_MAX_OUTPUT = _env_int(
    "QUERY_UNDERSTANDING_MAX_OUTPUT",
    256,
)

QUERY_UNDERSTANDING_TIMEOUT = _env_int(
    "QUERY_UNDERSTANDING_TIMEOUT",
    30,
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
# Answer model settings
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

ANSWER_LLM_THINKING = _env_bool(
    "ANSWER_LLM_THINKING",
    False,
)


# ============================================================
# Shared Ollama client kwargs
# ============================================================


def _client_kwargs(
    timeout: int,
) -> dict:
    return {
        "timeout": timeout,
    }


# ============================================================
# Query understanding model
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
        QUERY_UNDERSTANDING_TIMEOUT
    ),
)


# ============================================================
# General query model
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
        QUERY_LLM_TIMEOUT
    ),
)


# ============================================================
# Answer model
# ============================================================


answer_llm = ChatOllama(
    model=ANSWER_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_URL,
    reasoning=ANSWER_LLM_THINKING,
    num_ctx=ANSWER_LLM_CONTEXT,
    keep_alive=ANSWER_LLM_KEEP_ALIVE,
    client_kwargs=_client_kwargs(
        ANSWER_LLM_TIMEOUT
    ),
)


# ============================================================
# Backward compatibility
# ============================================================


llm = query_llm

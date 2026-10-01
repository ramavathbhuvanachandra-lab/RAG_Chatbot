"""
Application-level assistant router.

Responsibilities
----------------
- Detect structured application services before RAG.
- Delegate institution-specific structured services to the active
  institution package.
- Decide whether the RAG pipeline should also run.
- Keep the routing contract expected by UI/chat.py.

This module must not contain institution-specific factual knowledge.

Architecture
------------
User question
    ↓
Assistant Router
    ├── Active institution navigation service
    ├── Active institution emergency service
    └── RAG decision
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Any, Callable

from backend.config import INSTITUTION


logger = logging.getLogger(__name__)


# =========================================================
# Generic routing vocabulary
# =========================================================

NAVIGATION_KEYWORDS = (
    "where is",
    "location",
    "locate",
    "map",
    "maps",
    "navigate",
    "navigation",
    "direction",
    "directions",
    "how do i reach",
    "how to reach",
    "take me to",
    "route to",
    "way to",
)


INFORMATION_KEYWORDS = (
    "timing",
    "timings",
    "time",
    "hours",
    "working hours",
    "open",
    "close",
    "fee",
    "fees",
    "admission",
    "hostel",
    "facility",
    "facilities",
    "department",
    "course",
    "syllabus",
    "process",
    "procedure",
    "rules",
    "eligibility",
    "contact",
    "email",
    "what",
    "when",
    "why",
    "who",
    "which",
    "explain",
    "tell me",
    "provide",
    "information",
    "details",
)


# =========================================================
# Active institution
# =========================================================

def _active_institution_id() -> str:
    """
    Return the normalized active institution identifier.
    """

    institution_id = str(
        getattr(
            INSTITUTION,
            "institution_id",
            "",
        )
        or ""
    ).strip().lower()

    if not institution_id:
        raise RuntimeError(
            "Active institution profile does not define institution_id."
        )

    return institution_id


# =========================================================
# Optional institution service loading
# =========================================================

@lru_cache(maxsize=None)
def _load_service(
    module_name: str,
    function_name: str,
) -> Callable[[str], Any] | None:
    """
    Load an optional institution-specific service.

    Missing optional services are valid for institutions that do not
    provide that capability.

    A real import failure is logged and treated as unavailable so the
    chatbot can continue through the normal RAG path.
    """

    institution_id = _active_institution_id()

    module_path = (
        f"backend.institutions."
        f"{institution_id}."
        f"{module_name}"
    )

    try:
        module = importlib.import_module(
            module_path
        )

    except ModuleNotFoundError as exc:

        # Only treat the requested institution module itself as optional.
        # Unexpected missing dependencies should remain visible.
        if exc.name == module_path:
            return None

        logger.exception(
            "Failed to load institution service %s.%s",
            module_path,
            function_name,
        )

        return None

    except Exception:

        logger.exception(
            "Failed to load institution service %s.%s",
            module_path,
            function_name,
        )

        return None

    service = getattr(
        module,
        function_name,
        None,
    )

    if not callable(service):

        logger.warning(
            "Institution service %s.%s is unavailable.",
            module_path,
            function_name,
        )

        return None

    return service


def _get_navigation_service() -> (
    Callable[[str], Any] | None
):
    """Return the active institution navigation lookup."""

    return _load_service(
        "campus_navigation",
        "find_location",
    )


def _get_emergency_service() -> (
    Callable[[str], Any] | None
):
    """Return the active institution emergency lookup."""

    return _load_service(
        "emergency",
        "find_emergency",
    )


# =========================================================
# Routing helpers
# =========================================================

def _is_navigation_question(
    question: str,
) -> bool:
    """
    Detect generic navigation language.
    """

    question_lower = question.casefold()

    return any(
        keyword in question_lower
        for keyword in NAVIGATION_KEYWORDS
    )


def _needs_rag(
    question: str,
    structured: list[dict[str, Any]],
) -> bool:
    """
    Decide whether the normal RAG pipeline should also run.

    Preserve the existing application behavior:
    - recognized information questions use RAG;
    - if no structured service matched, RAG is always used.
    """

    question_lower = question.casefold()

    needs_information = any(
        keyword in question_lower
        for keyword in INFORMATION_KEYWORDS
    )

    if not structured:
        return True

    return needs_information


# =========================================================
# Public router
# =========================================================

def assistant_router(
    question: str,
) -> dict[str, Any]:
    """
    Route one user question to structured institution services and/or RAG.

    Returns:
        {
            "structured": [...],
            "need_rag": bool,
        }
    """

    normalized_question = str(
        question or ""
    ).strip()

    structured: list[
        dict[str, Any]
    ] = []

    if not normalized_question:
        return {
            "structured": structured,
            "need_rag": True,
        }

    # -----------------------------------------------------
    # Campus navigation
    # -----------------------------------------------------

    if _is_navigation_question(
        normalized_question
    ):

        navigation_service = (
            _get_navigation_service()
        )

        if navigation_service is not None:

            try:

                location = navigation_service(
                    normalized_question
                )

            except Exception:

                logger.exception(
                    "Institution navigation service failed."
                )

                location = None

            if location:

                structured.append(
                    {
                        "type": "navigation",
                        "data": location,
                    }
                )

    # -----------------------------------------------------
    # Emergency contacts
    # -----------------------------------------------------

    emergency_service = (
        _get_emergency_service()
    )

    if emergency_service is not None:

        try:

            emergency = emergency_service(
                normalized_question
            )

        except Exception:

            logger.exception(
                "Institution emergency service failed."
            )

            emergency = None

        if emergency:

            structured.append(
                {
                    "type": "emergency",
                    "data": emergency,
                }
            )

    # -----------------------------------------------------
    # RAG decision
    # -----------------------------------------------------

    need_rag = _needs_rag(
        normalized_question,
        structured,
    )

    return {
        "structured": structured,
        "need_rag": need_rag,
    }


__all__ = [
    "assistant_router",
]
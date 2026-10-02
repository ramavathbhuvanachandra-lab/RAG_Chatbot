"""Generic institution deployment contract for the reusable RAG application.

This module is intentionally institution-agnostic. It describes how an
institutional deployment is configured, not what the institution knows.
Institutional facts remain in the institution knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstitutionProfile:
    """Deployment configuration for one institution."""

    institution_id: str
    display_name: str

    data_path: Path
    vectorstore_path: Path
    vectorstore_collection: str | None = None
    structured_data_path: Path | None = None

    default_language: str = "English"
    supported_languages: tuple[str, ...] = ("English",)

    answer_tone: str = "student_friendly"
    answer_verbosity: str = "moderate"

    fallback_message: str = (
        "I'm sorry, I don't know based on the available information."
    )

    enable_semantic_query_understanding: bool = True
    enable_multi_intent: bool = True
    enable_hybrid_retrieval: bool = True
    enable_local_context: bool = True
    enable_evidence_guard: bool = True

    environment: str = "development"

    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    prompt_additions: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """Validate the complete deployment contract."""

        if not self.institution_id.strip():
            raise ValueError("institution_id cannot be empty")

        if not self.display_name.strip():
            raise ValueError("display_name cannot be empty")

        if not self.data_path:
            raise ValueError("data_path cannot be empty")

        if not self.vectorstore_path:
            raise ValueError("vectorstore_path cannot be empty")

        if self.vectorstore_collection is not None:
            if not self.vectorstore_collection.strip():
                raise ValueError(
                    "vectorstore_collection cannot be empty when provided"
                )

        if self.structured_data_path is not None and not self.structured_data_path:
            raise ValueError("structured_data_path cannot be empty")

        if not self.supported_languages:
            raise ValueError("supported_languages cannot be empty")

        for language in self.supported_languages:
            if not language.strip():
                raise ValueError(
                    "supported_languages cannot contain empty values"
                )

        if not self.default_language.strip():
            raise ValueError("default_language cannot be empty")

        if not self.answer_tone.strip():
            raise ValueError("answer_tone cannot be empty")

        if not self.answer_verbosity.strip():
            raise ValueError("answer_verbosity cannot be empty")

        if not self.fallback_message.strip():
            raise ValueError("fallback_message cannot be empty")

        if not self.environment.strip():
            raise ValueError("environment cannot be empty")

        for key, value in self.metadata:
            if not key.strip():
                raise ValueError("metadata keys cannot be empty")
            if not value.strip():
                raise ValueError("metadata values cannot be empty")

    @property
    def institution_data_root(self) -> Path:
        """Alias for the main institution data root."""

        return self.data_path

    @property
    def vector_db_root(self) -> Path:
        """Alias for the institution vector-store root."""

        return self.vectorstore_path

    @property
    def has_structured_data(self) -> bool:
        """Whether a separate structured-data root is configured."""

        return self.structured_data_path is not None


__all__ = ["InstitutionProfile"]
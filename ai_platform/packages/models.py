"""Data models for knowledge package discovery, validation, and release management.

These models define the normalized contract between external knowledge packages
(produced by the Knowledge Ingestion Engine) and the AI platform retrieval runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class PackageMetadata:
    """Core package-level manifest metadata."""

    package_name: str
    version: str
    institution_id: str
    institution_name: str
    release_timestamp: str
    total_canonical_documents: int = 0
    total_structured_datasets: int = 0
    schema_version: str = "2.0"
    parity_guarantee: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """One verified canonical document paired with its metadata sidecar."""

    document_id: str
    title: str
    category: str
    subcategory: str
    document_type: str
    docx_path: Path
    markdown_path: Path
    metadata_path: Path
    source_urls: tuple[str, ...] = ()
    unit_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    docx_hash: str = ""
    markdown_hash: str = ""
    is_fallback: bool = False
    validation_status: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    """Formally logged factual conflict and resolution policy."""

    conflict_id: str
    entity: str
    source_1: dict[str, Any]
    source_2: dict[str, Any]
    disposition: str
    explanation: str


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """Claim-level evidence entry from the provenance ledger."""

    claim_id: str
    domain: str
    entity: str
    claim_value: str
    conditions: str
    source_url: str
    verified_in_documents: tuple[str, ...]
    verification_status: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Detailed validation results for a package release."""

    valid: bool
    package_root: Path
    institution_id: str
    release_id: str
    accepted_documents: int
    skipped_documents: int
    duplicated_documents: int
    invalid_documents: int
    missing_files: tuple[str, ...] = ()
    hash_mismatches: tuple[str, ...] = ()
    schema_warnings: tuple[str, ...] = ()
    datasets_discovered: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> str:
        status_str = "VALID (PASS)" if self.valid else "INVALID (FAIL)"
        lines = [
            f"Package Validation Report: {status_str}",
            f"  Institution ID:      {self.institution_id}",
            f"  Release ID:          {self.release_id}",
            f"  Package Root:        {self.package_root}",
            f"  Accepted Documents:  {self.accepted_documents}",
            f"  Skipped Documents:   {self.skipped_documents} (quarantined fallback/companion)",
            f"  Duplicated Docs:     {self.duplicated_documents}",
            f"  Invalid Documents:   {self.invalid_documents}",
            f"  Datasets Discovered: {len(self.datasets_discovered)}",
            f"  Hash Mismatches:     {len(self.hash_mismatches)}",
            f"  Schema Warnings:     {len(self.schema_warnings)}",
        ]
        if self.missing_files:
            lines.append(f"  Missing Files ({len(self.missing_files)}): {list(self.missing_files)[:5]}")
        if self.hash_mismatches:
            lines.append(f"  Hash Mismatches ({len(self.hash_mismatches)}): {list(self.hash_mismatches)[:3]}")
        if self.schema_warnings:
            lines.append(f"  Warnings: {list(self.schema_warnings)[:3]}")
        return "\n".join(lines)


@dataclass
class InstitutionalRelease:
    """Complete, normalized, validated institutional release ready for indexing and retrieval."""

    institution_id: str
    release_id: str
    display_name: str
    package_root: Path
    metadata: PackageMetadata
    documents: list[CanonicalDocument]
    structured_datasets: dict[str, Any]
    structured_records: list[Document]
    conflicts: list[ConflictRecord]
    claims: list[ClaimRecord]
    lexical_resources: dict[str, Any]
    validation_report: ValidationReport

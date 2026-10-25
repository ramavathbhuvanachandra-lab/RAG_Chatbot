"""Release registry and configuration manager for institutional knowledge packages.

Provides zero-disruption staging, release isolation, atomic cutover, and instant rollback.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _active_config_path(institution_id: str) -> Path:
    return PROJECT_ROOT / "data" / f"active_release_{institution_id.strip().lower()}.json"


def _staging_config_path(institution_id: str) -> Path:
    return PROJECT_ROOT / "data" / f"staging_release_{institution_id.strip().lower()}.json"


def get_active_release_config(institution_id: str | None = None) -> dict[str, Any]:
    """Retrieve the currently active release configuration for an institution."""
    inst_id = (institution_id or os.getenv("INSTITUTION_ID", "iitj")).strip().lower()

    # 0. Staging preview mode if explicitly enabled via environment
    if os.getenv("RAG_USE_STAGING", "").strip().casefold() in {"1", "true", "yes"}:
        staging_cfg = get_staging_release_config(inst_id)
        if staging_cfg:
            return staging_cfg

    # 1. Environment variable overrides (highest priority for container/CI workflows)
    env_collection = os.getenv("VECTORSTORE_COLLECTION") or os.getenv("RAG_ACTIVE_COLLECTION")
    env_root = os.getenv("RAG_PACKAGE_ROOT")
    env_release = os.getenv("RAG_ACTIVE_RELEASE")

    # 2. Persisted active release file
    cfg_file = _active_config_path(inst_id)
    if cfg_file.exists():
        try:
            with cfg_file.open("r", encoding="utf-8") as f:
                saved = json.load(f)
                if env_collection:
                    saved["vectorstore_collection"] = env_collection
                if env_root:
                    saved["package_root"] = env_root
                if env_release:
                    saved["active_release"] = env_release
                return saved
        except Exception as exc:
            logger.warning("Error reading active release file %s: %s", cfg_file, exc)

    # 3. Default fallback to V1 historical baseline
    default_root = str(PROJECT_ROOT / "data" / f"data_{inst_id}")
    default_v2_root = str(PROJECT_ROOT / "data" / f"data_{inst_id}" / f"{inst_id}_jodhpur_rag_package_v2")

    return {
        "institution_id": inst_id,
        "active_release": env_release or "iit_jodhpur_rag_package_v1",
        "package_root": env_root or default_root,
        "vectorstore_collection": env_collection or f"{inst_id}_v1",
        "vectorstore_path": str(PROJECT_ROOT / "chroma_db"),
        "lexical_cache_filename": ".canonical_chunks_cache.json",
        "status": "baseline_active",
        "activated_at": None,
        "previous_release": None,
    }


def get_staging_release_config(institution_id: str | None = None) -> dict[str, Any] | None:
    """Retrieve the staging release configuration if staged."""
    inst_id = (institution_id or os.getenv("INSTITUTION_ID", "iitj")).strip().lower()
    staging_file = _staging_config_path(inst_id)
    if not staging_file.exists():
        return None
    try:
        with staging_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Error reading staging release file %s: %s", staging_file, exc)
        return None


def register_staging_release(
    institution_id: str,
    release_id: str,
    package_root: Path | str,
    vectorstore_collection: str,
    persist_dir: Path | str,
    lexical_cache_path: Path | str,
    evaluation_passed: bool = False,
) -> dict[str, Any]:
    """Persist staging metadata for a validated release."""
    inst_id = institution_id.strip().lower()
    staging_payload = {
        "institution_id": inst_id,
        "release_id": release_id.strip(),
        "package_root": str(Path(package_root).resolve()),
        "vectorstore_collection": vectorstore_collection.strip(),
        "vectorstore_path": str(Path(persist_dir).resolve()),
        "lexical_cache_path": str(Path(lexical_cache_path).resolve()),
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_passed": evaluation_passed,
        "status": "staged",
    }
    staging_file = _staging_config_path(inst_id)
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    with staging_file.open("w", encoding="utf-8") as f:
        json.dump(staging_payload, f, indent=2)
    return staging_payload


def mark_staging_evaluated(institution_id: str, passed: bool, eval_report: dict[str, Any] | None = None) -> None:
    """Update staging status after running evaluation."""
    inst_id = institution_id.strip().lower()
    staged = get_staging_release_config(inst_id)
    if not staged:
        return
    staged["evaluation_passed"] = passed
    staged["evaluation_report"] = eval_report or {}
    staged["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    staging_file = _staging_config_path(inst_id)
    with staging_file.open("w", encoding="utf-8") as f:
        json.dump(staged, f, indent=2)


def activate_staged_release(
    institution_id: str,
    active_collection_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically activate the staged release while preserving the previous release for rollback."""
    inst_id = institution_id.strip().lower()
    staged = get_staging_release_config(inst_id)
    if not staged:
        raise RuntimeError(f"No staged release found for institution '{inst_id}'. Stage a package first.")

    if not force and not staged.get("evaluation_passed"):
        raise RuntimeError(
            f"Staged release '{staged.get('release_id')}' has not passed required evaluation checks. "
            "Run evaluate before activating or specify force=True."
        )

    current_active = get_active_release_config(inst_id)

    # Use specified collection name or keep the staged collection
    target_collection = active_collection_name or staged["vectorstore_collection"]

    new_active_payload = {
        "institution_id": inst_id,
        "active_release": staged["release_id"],
        "package_root": staged["package_root"],
        "vectorstore_collection": target_collection,
        "vectorstore_path": staged["vectorstore_path"],
        "lexical_cache_path": staged["lexical_cache_path"],
        "status": "active",
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "previous_release": {
            "active_release": current_active.get("active_release"),
            "package_root": current_active.get("package_root"),
            "vectorstore_collection": current_active.get("vectorstore_collection"),
            "vectorstore_path": current_active.get("vectorstore_path"),
            "lexical_cache_path": current_active.get("lexical_cache_path") or current_active.get("lexical_cache_filename"),
            "status": current_active.get("status"),
        },
    }

    active_file = _active_config_path(inst_id)
    active_file.parent.mkdir(parents=True, exist_ok=True)
    with active_file.open("w", encoding="utf-8") as f:
        json.dump(new_active_payload, f, indent=2)

    # Invalidate runtime caches
    _clear_runtime_caches()

    return new_active_payload


def rollback_active_release(institution_id: str) -> dict[str, Any]:
    """Restore the previous active release configuration."""
    inst_id = institution_id.strip().lower()
    current_active = get_active_release_config(inst_id)
    prev = current_active.get("previous_release")
    if not prev or not isinstance(prev, dict):
        raise RuntimeError(f"No previous release recorded for institution '{inst_id}' to roll back to.")

    rolled_back_payload = {
        "institution_id": inst_id,
        "active_release": prev.get("active_release", "iit_jodhpur_rag_package_v1"),
        "package_root": prev.get("package_root"),
        "vectorstore_collection": prev.get("vectorstore_collection", f"{inst_id}_v1"),
        "vectorstore_path": prev.get("vectorstore_path", str(PROJECT_ROOT / "chroma_db")),
        "lexical_cache_path": prev.get("lexical_cache_path", ".canonical_chunks_cache.json"),
        "status": "rolled_back",
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "previous_release": current_active,
    }

    active_file = _active_config_path(inst_id)
    with active_file.open("w", encoding="utf-8") as f:
        json.dump(rolled_back_payload, f, indent=2)

    _clear_runtime_caches()
    return rolled_back_payload


def _clear_runtime_caches() -> None:
    """Clear all in-memory retrieval, vectorstore, and embedding caches."""
    try:
        from ai_platform.runtime.retrieval import (
            _cached_canonical_chunks,
            _cached_bm25_retriever,
            _cached_dense_retriever,
            clear_query_embedding_cache,
        )
        _cached_canonical_chunks.cache_clear()
        _cached_bm25_retriever.cache_clear()
        _cached_dense_retriever.cache_clear()
        clear_query_embedding_cache()
    except Exception as exc:
        logger.debug("Could not clear retrieval caches: %s", exc)

    try:
        from ai_platform.runtime.vectorstore import _cached_vectorstore
        _cached_vectorstore.cache_clear()
    except Exception as exc:
        logger.debug("Could not clear vectorstore cache: %s", exc)

    try:
        from ai_platform.institutions.loader import load_institution_profile
        if hasattr(load_institution_profile, "cache_clear"):
            load_institution_profile.cache_clear()
    except Exception as exc:
        logger.debug("Could not clear institution profile cache: %s", exc)

    try:
        from ai_platform.runtime.engine import get_app_graph, get_inspect_nodes
        get_app_graph.cache_clear()
        get_inspect_nodes.cache_clear()
    except Exception as exc:
        logger.debug("Could not clear engine graph caches: %s", exc)

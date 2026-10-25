"""Knowledge Package integration for the College AI Platform.

Provides package validation, loading, metadata-aware chunking, staging, atomic cutover,
and release rollback for versioned institutional knowledge releases.
"""

from ai_platform.packages.models import (
    CanonicalDocument,
    ClaimRecord,
    ConflictRecord,
    InstitutionalRelease,
    PackageMetadata,
    ValidationReport,
)
from ai_platform.packages.package_adapter import (
    chunk_package_documents,
    load_institutional_release,
    validate_package,
)
from ai_platform.packages.registry import (
    activate_staged_release,
    get_active_release_config,
    get_staging_release_config,
    register_staging_release,
    rollback_active_release,
)

__all__ = [
    "PackageMetadata",
    "CanonicalDocument",
    "ConflictRecord",
    "ClaimRecord",
    "ValidationReport",
    "InstitutionalRelease",
    "validate_package",
    "load_institutional_release",
    "chunk_package_documents",
    "get_active_release_config",
    "get_staging_release_config",
    "register_staging_release",
    "activate_staged_release",
    "rollback_active_release",
]

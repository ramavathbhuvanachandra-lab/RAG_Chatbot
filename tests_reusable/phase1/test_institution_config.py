from pathlib import Path

from backend.config import (
    CHROMA_DB_PATH,
    DATA_PATH,
    INSTITUTION_ID,
    INSTITUTION_NAME,
    SUPPORTED_LANGUAGES,
)
from backend.institutions.iitj.profile import PROFILE


def test_iitj_profile_is_small_and_deployment_focused():
    assert PROFILE.institution_id == "iitj"
    assert PROFILE.display_name == "IIT Jodhpur"

    assert PROFILE.data_path.name == "data_iitj"
    assert PROFILE.vectorstore_path.name == "chroma_db"

    assert PROFILE.supported_languages == (
        "English",
        "Hindi",
    )


def test_legacy_config_exports_remain_compatible():
    assert INSTITUTION_ID == PROFILE.institution_id
    assert INSTITUTION_NAME == PROFILE.display_name

    assert DATA_PATH == PROFILE.data_path
    assert CHROMA_DB_PATH == PROFILE.vectorstore_path

    assert all(
        isinstance(path, Path)
        for path in (
            DATA_PATH,
            CHROMA_DB_PATH,
        )
    )


def test_core_profile_does_not_contain_institutional_facts():
    from backend.core.institution import InstitutionProfile

    fields = set(
        InstitutionProfile.__dataclass_fields__
    )

    assert "programs" not in fields
    assert "hostel_rules" not in fields
    assert "fee_rules" not in fields
    assert "admission_rules" not in fields

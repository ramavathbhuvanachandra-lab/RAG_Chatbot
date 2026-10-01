from __future__ import annotations

from pathlib import Path

from backend.config import (
    CHROMA_DB_PATH,
    INSTITUTION_ID,
)
from backend.vectorstore import (
    VECTORSTORE_COLLECTION,
    VECTORSTORE_PATH,
    vectorstore,
)


def test_vectorstore_collection_is_institution_derived() -> None:
    assert VECTORSTORE_COLLECTION == f"{INSTITUTION_ID}_v1"


def test_iitj_collection_name_is_unchanged() -> None:
    assert INSTITUTION_ID == "iitj"
    assert VECTORSTORE_COLLECTION == "iitj_v1"


def test_vectorstore_path_comes_from_institution_config() -> None:
    assert Path(VECTORSTORE_PATH) == Path(CHROMA_DB_PATH)


def test_vectorstore_object_is_initialized() -> None:
    assert vectorstore is not None


if __name__ == "__main__":
    print("Vectorstore institution configuration tests passed.")

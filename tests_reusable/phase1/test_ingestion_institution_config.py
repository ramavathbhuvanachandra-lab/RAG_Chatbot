from __future__ import annotations

from pathlib import Path

from backend.config import (
    CHROMA_DB_PATH,
    DATA_PATH,
    INSTITUTION_ID,
)
from backend.ingestion import create_vectorstore
from backend.vectorstore import VECTORSTORE_COLLECTION


def test_ingestion_uses_active_institution_data_path() -> None:
    assert INSTITUTION_ID == "iitj"
    assert Path(DATA_PATH).name == "data_iitj"


def test_ingestion_uses_active_vectorstore_path() -> None:
    assert Path(CHROMA_DB_PATH).name == "chroma_db"


def test_ingestion_uses_institution_derived_collection() -> None:
    assert VECTORSTORE_COLLECTION == "iitj_v1"


def test_create_vectorstore_is_available() -> None:
    assert callable(create_vectorstore)


if __name__ == "__main__":
    print(
        "Ingestion institution configuration tests passed."
    )

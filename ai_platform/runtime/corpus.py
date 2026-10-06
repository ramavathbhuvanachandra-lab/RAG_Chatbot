"""Corpus loading and chunking runtime.

The runtime owns document I/O and chunk construction. The reusable core owns
retrieval policy; the institution profile supplies the corpus location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)


def load_documents(data_path: str | Path) -> list[Document]:
    """Load all DOCX documents recursively from one institution corpus."""
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    loader = DirectoryLoader(
        str(root),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=False,
        use_multithreading=False,
    )
    return list(loader.load())


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Create deterministic chunks using the production chunking contract."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=list(CHUNK_SEPARATORS),
    )
    return list(splitter.split_documents(list(documents)))


def load_and_split(data_path: str | Path) -> list[Document]:
    """Load and chunk one institution corpus."""
    return split_documents(load_documents(data_path))


__all__ = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_SEPARATORS",
    "load_documents",
    "split_documents",
    "load_and_split",
]

"""
Reusable document ingestion layer.

This module is institution-agnostic.

Responsibilities
----------------
1. Load source documents from the active institution's data directory.
2. Remove ingestion-generated artifacts from retrieval content.
3. Remove URLs from retrieval content while preserving them as metadata.
4. Normalize extracted document text.
5. Preserve stable source/document/chunk metadata.
6. Split documents into retrieval-friendly chunks.
7. Build the configured vector store.

Architecture boundary
---------------------
This module does NOT know:

- institution names
- programs
- departments
- courses
- hostel rules
- admission rules
- fee rules
- campus information
- institution-specific terminology

The active institution supplies deployment configuration only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from backend.config import (
    CHROMA_DB_PATH,
    DATA_PATH,
)

from backend.vectorstore import (
    VECTORSTORE_COLLECTION,
)


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_CHUNK_SIZE = 800

DEFAULT_CHUNK_OVERLAP = 150

BATCH_SIZE = 50

DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)


# =====================================================================
# URL handling
# =====================================================================

_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+",
    re.IGNORECASE,
)


# =====================================================================
# Generated ingestion-artifact handling
# =====================================================================

_RETRIEVAL_REPRESENTATION_PATTERN = re.compile(
    r"""
    (?is)
    \b
    command
    \s*
    \d+
    \s*
    [•|:]
    \s*
    retrieval
    \s+
    representation
    \s*
    [•|:]?
    """,
    re.VERBOSE,
)


_SOURCE_WRAPPER_PATTERN = re.compile(
    r"""
    (?is)
    \b
    source
    \s*
    \n?
    \s*
    original
    \s+
    source
    \s+
    urls?
    \s+
    preserved
    .*?
    (?=
        \n\s*
        command
        \s*
        \d+
        \s*
        [•|:]
        \s*
        retrieval
        \s+
        representation
        |
        \Z
    )
    """,
    re.VERBOSE,
)


_SOURCE_URL_BLOCK_PATTERN = re.compile(
    r"""
    (?is)
    \b
    original
    \s+
    source
    \s+
    urls?
    \s+
    preserved
    .*?
    (?=
        \n\s*
        command
        \s*
        \d+
        \s*
        [•|:]
        \s*
        retrieval
        \s+
        representation
        |
        \Z
    )
    """,
    re.VERBOSE,
)


_GENERIC_RETRIEVAL_MARKER_PATTERN = re.compile(
    r"""
    (?im)
    ^\s*
    command
    \s*
    \d+
    \s*
    [•|:]
    \s*
    retrieval
    \s+
    representation
    \s*
    [•|:]?
    \s*$
    """,
    re.VERBOSE,
)


# =====================================================================
# Text normalization
# =====================================================================

def _normalize_whitespace(
    text: str,
) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """

    value = str(
        text or ""
    )

    value = value.replace(
        "\r\n",
        "\n",
    )

    value = value.replace(
        "\r",
        "\n",
    )

    value = re.sub(
        r"[ \t]+\n",
        "\n",
        value,
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    return value.strip()


# =====================================================================
# URL extraction
# =====================================================================

def _clean_url(
    url: str,
) -> str:
    """
    Remove punctuation accidentally attached to a URL.
    """

    return str(
        url or ""
    ).rstrip(
        ".,;:)]}>\"'"
    )


def _extract_urls(
    text: str,
) -> list[str]:
    """
    Extract unique URLs from source text.
    """

    found = _URL_PATTERN.findall(
        str(text or "")
    )

    urls: list[str] = []

    for raw_url in found:

        url = _clean_url(
            raw_url
        )

        if (
            url
            and
            url not in urls
        ):
            urls.append(
                url
            )

    return urls


# =====================================================================
# Markdown cleanup
# =====================================================================

def _find_markdown_label_start(
    text: str,
    closing_bracket_index: int,
) -> int | None:
    """
    Find the matching opening '[' for a Markdown label.

    This supports nested brackets.

    Example:

        [[Back to index]]

    The matching opening bracket is the outermost '['.
    """

    depth = 0

    index = closing_bracket_index

    while index >= 0:

        character = text[index]

        if character == "]":
            depth += 1

        elif character == "[":
            depth -= 1

            if depth == 0:
                return index

        index -= 1

    return None


def _replace_markdown_urls(
    text: str,
) -> str:
    """
    Remove Markdown URLs while preserving visible labels.

    This function deliberately uses a small parser rather than a single
    regular expression because the corpus contains:

    - nested labels
    - labels with spaces
    - labels containing brackets
    - malformed links produced by document extraction

    Examples
    --------
    [Placement Statistics](https://example.edu/file)

    becomes:

        Placement Statistics

    [[Back to index]](https://example.edu/page)

    becomes:

        [Back to index]
    """

    value = str(
        text or ""
    )

    # --------------------------------------------------------------
    # Process URLs from left to right.
    # --------------------------------------------------------------

    search_position = 0

    while True:

        match = _URL_PATTERN.search(
            value,
            search_position,
        )

        if match is None:
            break

        url_start = match.start()
        url_end = match.end()

        prefix = value[
            :url_start
        ]

        # ----------------------------------------------------------
        # Look immediately before the URL for Markdown link syntax:
        #
        #     ...](https://...)
        # ----------------------------------------------------------

        before_url_match = re.search(
            r"\]\s*\(\s*$",
            prefix,
            re.DOTALL,
        )

        if before_url_match is None:

            # This is a bare URL.
            value = (
                value[:url_start]
                +
                value[url_end:]
            )

            search_position = (
                url_start
            )

            continue

        closing_bracket_index = (
            before_url_match.start()
        )

        # ----------------------------------------------------------
        # Find the matching opening '['.
        # ----------------------------------------------------------

        opening_bracket_index = (
            _find_markdown_label_start(
                value,
                closing_bracket_index,
            )
        )

        if (
            opening_bracket_index
            is None
        ):

            # Malformed case where no opening bracket can be found.
            #
            # Remove only the Markdown link syntax around the URL.
            #
            syntax_start = (
                before_url_match.start()
            )

            value = (
                value[:syntax_start]
                +
                value[url_end:]
            )

            search_position = (
                syntax_start
            )

            continue

        # ----------------------------------------------------------
        # Extract visible label.
        # ----------------------------------------------------------

        label_start = (
            opening_bracket_index + 1
        )

        label_end = (
            closing_bracket_index
        )

        label = value[
            label_start:
            label_end
        ]

        # ----------------------------------------------------------
        # Replace complete Markdown link with label.
        # ----------------------------------------------------------

        value = (
            value[:opening_bracket_index]
            +
            label
            +
            value[url_end:]
        )

        search_position = (
            opening_bracket_index
            +
            len(label)
        )

    return value


def _remove_reference_style_links(
    text: str,
) -> str:
    """
    Remove Markdown reference-link definitions while preserving their
    visible labels when possible.
    """

    value = str(
        text or ""
    )

    # Example:
    #
    # [Placement Statistics]:
    # https://example.edu/file
    #

    pattern = re.compile(
        r"""
        (?im)
        ^\s*
        \[
            (?P<label>[^\]\n]+)
        \]
        \s*:
        \s*
        https?://\S+
        \s*$
        """,
        re.VERBOSE,
    )

    value = pattern.sub(
        lambda match: match.group(
            "label"
        ),
        value,
    )

    return value


def _repair_orphaned_markdown(
    text: str,
) -> str:
    """
    Remove malformed Markdown syntax remaining after URL removal.

    Handles corpus patterns such as:

        Label](

        Label] (

        [Label](

        [[Label]](

        [](

    Only Markdown-link artifacts are removed.
    """

    value = str(
        text or ""
    )

    # --------------------------------------------------------------
    # Case 1:
    #
    #     [Label](
    #
    # or:
    #
    #     [[Label]](
    #
    # Keep the visible label.
    # --------------------------------------------------------------

    previous_value = None

    while previous_value != value:

        previous_value = value

        value = re.sub(
            r"\[([^\]\n]*)\]\s*\(\s*(?=\n|$)",
            r"\1",
            value,
        )

        value = re.sub(
            r"\[\[([^\]\n]*)\]\]\s*\(\s*(?=\n|$)",
            r"[\1]",
            value,
        )

    # --------------------------------------------------------------
    # Case 2:
    #
    #     Label](
    #
    # This is the exact artifact seen in the corpus.
    # --------------------------------------------------------------

    value = re.sub(
        r"(?m)^([^\]\n]+)\]\s*\(\s*$",
        r"\1",
        value,
    )

    # --------------------------------------------------------------
    # Case 3:
    #
    #     Label](
    #
    # embedded inside text / adjacent links.
    # --------------------------------------------------------------

    value = re.sub(
        r"\]\s*\(\s*(?=\n|$)",
        "",
        value,
    )

    # --------------------------------------------------------------
    # Case 4:
    #
    # Remove orphan closing-link syntax where no URL remains.
    # --------------------------------------------------------------

    value = re.sub(
        r"\]\s*\(\s*",
        "",
        value,
    )

    # --------------------------------------------------------------
    # Remove empty link shells.
    # --------------------------------------------------------------

    value = re.sub(
        r"\[\s*\]",
        "",
        value,
    )

    value = re.sub(
        r"\(\s*\)",
        "",
        value,
    )

    return value


def _remove_remaining_markdown_artifacts(
    text: str,
) -> str:
    """
    Final defensive cleanup of obvious Markdown artifacts.

    This intentionally does NOT remove ordinary factual brackets
    indiscriminately.
    """

    value = str(
        text or ""
    )

    # Remove empty bracket/parenthesis fragments.
    value = re.sub(
        r"(?m)^\s*\[\s*$",
        "",
        value,
    )

    value = re.sub(
        r"(?m)^\s*\]\s*$",
        "",
        value,
    )

    value = re.sub(
        r"(?m)^\s*\(\s*$",
        "",
        value,
    )

    value = re.sub(
        r"(?m)^\s*\)\s*$",
        "",
        value,
    )

    return value


def _remove_urls_from_content(
    text: str,
) -> str:
    """
    Remove URLs from retrieval content.

    Priority:

        1. Complete Markdown links
        2. Reference-style links
        3. Bare URLs
        4. Broken Markdown syntax
    """

    value = str(
        text or ""
    )

    # Complete / nested Markdown links.
    value = _replace_markdown_urls(
        value
    )

    # Reference-style Markdown links.
    value = _remove_reference_style_links(
        value
    )

    # Bare URLs.
    value = _URL_PATTERN.sub(
        "",
        value,
    )

    # Broken/orphaned Markdown artifacts.
    value = _repair_orphaned_markdown(
        value
    )

    # Final artifact cleanup.
    value = _remove_remaining_markdown_artifacts(
        value
    )

    return value


# =====================================================================
# Stable identities
# =====================================================================

def _content_hash(
    text: str,
) -> str:
    """
    Return a stable SHA-256 hash of normalized content.
    """

    normalized = _normalize_whitespace(
        text
    )

    return sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def _source_hash(
    source: str,
) -> str:
    """
    Return a stable short source identifier.
    """

    return sha256(
        str(source).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


# =====================================================================
# Content cleaning
# =====================================================================

def clean_document_content(
    text: str,
) -> tuple[str, list[str]]:
    """
    Clean one source document.

    Returns
    -------
    cleaned_content:
        Factual retrieval content.

    source_urls:
        URLs extracted from the source document.

    Important
    ---------
    This function performs structural cleaning only.

    It does NOT:

    - summarize content
    - rewrite facts
    - infer entities
    - classify programs
    - classify departments
    - apply institution-specific rules
    """

    original = str(
        text or ""
    )

    # --------------------------------------------------------------
    # Extract URLs before modifying the source text.
    # --------------------------------------------------------------

    source_urls = _extract_urls(
        original
    )

    cleaned = original

    # --------------------------------------------------------------
    # Remove generated source wrappers.
    # --------------------------------------------------------------

    cleaned = _SOURCE_WRAPPER_PATTERN.sub(
        "",
        cleaned,
    )

    cleaned = _SOURCE_URL_BLOCK_PATTERN.sub(
        "",
        cleaned,
    )

    # --------------------------------------------------------------
    # Remove retrieval representation markers.
    # --------------------------------------------------------------

    cleaned = _RETRIEVAL_REPRESENTATION_PATTERN.sub(
        "",
        cleaned,
    )

    cleaned = _GENERIC_RETRIEVAL_MARKER_PATTERN.sub(
        "",
        cleaned,
    )

    # --------------------------------------------------------------
    # Remove standalone Source labels left after wrapper removal.
    # --------------------------------------------------------------

    cleaned = re.sub(
        r"(?im)^\s*source\s*$",
        "",
        cleaned,
    )

    # --------------------------------------------------------------
    # Remove URLs while preserving human-readable labels.
    # --------------------------------------------------------------

    cleaned = _remove_urls_from_content(
        cleaned
    )

    # --------------------------------------------------------------
    # Final whitespace normalization.
    # --------------------------------------------------------------

    cleaned = _normalize_whitespace(
        cleaned
    )

    return (
        cleaned,
        source_urls,
    )


# =====================================================================
# Metadata
# =====================================================================

def _safe_source_path(
    source: str | Path,
) -> Path:
    """
    Convert loader source to a stable absolute path.
    """

    return Path(
        source
    ).expanduser().resolve()


def _build_document_metadata(
    document: Document,
    cleaned_text: str,
    source_urls: Iterable[str],
) -> dict:
    """
    Build generic document metadata.

    Metadata remains separate from factual retrieval content.
    """

    raw_source = str(
        document.metadata.get(
            "source",
            "",
        )
    )

    source_path = _safe_source_path(
        raw_source
    )

    source_id = _source_hash(
        str(source_path)
    )

    urls = list(
        source_urls
    )

    metadata = dict(
        document.metadata or {}
    )

    metadata.update(
        {
            "source": str(
                source_path
            ),
            "source_path": str(
                source_path
            ),
            "source_id": source_id,
            "document_type": (
                source_path.suffix
                .lower()
                .lstrip(".")
            ),

            # JSON string keeps metadata compatible with scalar
            # vector-store metadata requirements.
            "source_urls": json.dumps(
                urls,
                ensure_ascii=False,
            ),

            "source_url_count": len(
                urls
            ),

            "document_content_hash": (
                _content_hash(
                    cleaned_text
                )
            ),
        }
    )

    return metadata


# =====================================================================
# Document loading
# =====================================================================

def load_documents(
    data_path: str | Path | None = None,
) -> list[Document]:
    """
    Load and clean all DOCX documents recursively.

    When data_path is omitted, the active institution's configured
    DATA_PATH is used.
    """

    resolved_data_path = (
        Path(
            data_path
            if data_path is not None
            else DATA_PATH
        )
        .expanduser()
        .resolve()
    )

    if not resolved_data_path.exists():
        raise FileNotFoundError(
            "Knowledge data directory does not exist: "
            f"{resolved_data_path}"
        )

    if not resolved_data_path.is_dir():
        raise NotADirectoryError(
            "Knowledge data path is not a directory: "
            f"{resolved_data_path}"
        )

    loader = DirectoryLoader(
        str(
            resolved_data_path
        ),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=False,
        use_multithreading=False,
    )

    loaded_documents = loader.load()

    cleaned_documents: list[Document] = []

    for document in loaded_documents:

        cleaned_text, source_urls = (
            clean_document_content(
                document.page_content
            )
        )

        if not cleaned_text.strip():
            continue

        metadata = _build_document_metadata(
            document=document,
            cleaned_text=cleaned_text,
            source_urls=source_urls,
        )

        cleaned_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=metadata,
            )
        )

    return cleaned_documents


# =====================================================================
# Document splitting
# =====================================================================

def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split cleaned documents into retrieval chunks.

    The splitter remains intentionally conservative at this stage.

    More advanced section/table-aware chunking will be introduced in
    the retrieval-core refactor rather than being mixed into the
    cleaning layer.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(
                DEFAULT_SEPARATORS
            ),
            keep_separator=True,
        )
    )

    chunks: list[Document] = []

    for document in documents:

        source = str(
            document.metadata.get(
                "source",
                "",
            )
        )

        source_id = str(
            document.metadata.get(
                "source_id",
                _source_hash(
                    source
                ),
            )
        )

        document_hash = str(
            document.metadata.get(
                "document_content_hash",
                _content_hash(
                    document.page_content
                ),
            )
        )

        document_chunks = (
            text_splitter.split_documents(
                [
                    document
                ]
            )
        )

        chunk_count = len(
            document_chunks
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            chunk_text = (
                _normalize_whitespace(
                    chunk.page_content
                )
            )

            if not chunk_text:
                continue

            chunk_metadata = dict(
                document.metadata or {}
            )

            chunk_id_material = (
                f"{source_id}\n"
                f"{document_hash}\n"
                f"{index}\n"
                f"{chunk_text}"
            )

            chunk_id = sha256(
                chunk_id_material.encode(
                    "utf-8"
                )
            ).hexdigest()

            chunk_metadata.update(
                {
                    "source": source,
                    "source_id": source_id,
                    "document_content_hash": (
                        document_hash
                    ),
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "chunk_id": chunk_id,
                    "chunk_character_count": (
                        len(
                            chunk_text
                        )
                    ),
                }
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    return chunks


# =====================================================================
# Vectorstore creation
# =====================================================================

def create_vectorstore(
    chunks: list[Document],
) -> Chroma:
    """
    Destructively rebuild the vectorstore for the active institution.

    IMPORTANT
    ---------
    This deletes the configured vectorstore directory.

    Do not call this until the ingestion and retrieval representations
    have been fully validated.
    """

    if not chunks:
        raise ValueError(
            "Cannot create vectorstore from an empty chunk list."
        )

    persist_directory = str(
        Path(
            CHROMA_DB_PATH
        )
        .expanduser()
        .resolve()
    )

    if os.path.exists(
        persist_directory
    ):

        print(
            "Deleting existing Chroma database..."
        )

        shutil.rmtree(
            persist_directory
        )

    print(
        "Creating new Chroma vector store..."
    )

    vectorstore: Chroma | None = None

    total_chunks = len(
        chunks
    )

    total_batches = (
        (
            total_chunks
            +
            BATCH_SIZE
            -
            1
        )
        //
        BATCH_SIZE
    )

    for start_index in range(
        0,
        total_chunks,
        BATCH_SIZE,
    ):

        batch = chunks[
            start_index:
            start_index + BATCH_SIZE
        ]

        batch_number = (
            start_index
            //
            BATCH_SIZE
        ) + 1

        batch_start = (
            start_index + 1
        )

        batch_end = min(
            start_index + BATCH_SIZE,
            total_chunks,
        )

        print(
            f"Embedding batch "
            f"{batch_number}/{total_batches} "
            f"({batch_start}-{batch_end} "
            f"of {total_chunks})..."
        )

        if vectorstore is None:

            vectorstore = (
                Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=(
                        VECTORSTORE_COLLECTION
                    ),
                    persist_directory=(
                        persist_directory
                    ),
                )
            )

        else:

            vectorstore.add_documents(
                batch
            )

    if vectorstore is None:
        raise RuntimeError(
            "Vectorstore creation failed."
        )

    return vectorstore


# =====================================================================
# Public API
# =====================================================================

__all__ = [
    "load_documents",
    "split_documents",
    "create_vectorstore",
    "clean_document_content",
]
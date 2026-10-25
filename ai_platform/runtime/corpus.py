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


CHUNK_CACHE_FILENAME = ".canonical_chunks_cache.json"


def is_quarantined_fallback(path: Path | str) -> bool:
    """Return True if path represents a quarantined heterogeneous fallback buffer."""
    norm = str(path).replace("\\", "/").casefold()
    return "/fallback/" in norm or "knowledge_buffer" in norm


def get_corpus_signature(data_path: str | Path) -> str:
    """Calculate a fast filesystem signature based on file count and latest mtime."""
    root = Path(data_path)
    if not root.exists() or not root.is_dir():
        return ""
    files = [f for f in sorted(root.glob("**/*.docx")) if not is_quarantined_fallback(f)]
    if not files:
        files = [
            f for f in sorted(root.glob("**/*.md")) + sorted(root.glob("**/*.txt"))
            if not is_quarantined_fallback(f)
        ]
    if not files:
        return "empty_0"
    latest_mtime = max(f.stat().st_mtime for f in files)
    return f"{len(files)}_{latest_mtime}"


def load_documents(data_path: str | Path, include_fallback: bool = False) -> list[Document]:
    """Load all valid institutional documents recursively from one institution corpus.

    Heterogeneous catch-all fallback documents are quarantined from normal answer evidence.
    """
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    docx_files = list(root.glob("**/*.docx"))
    if not include_fallback:
        docx_files = [f for f in docx_files if not is_quarantined_fallback(f)]

    if docx_files:
        docs: list[Document] = []
        for file_path in sorted(docx_files):
            try:
                loader = Docx2txtLoader(str(file_path))
                loaded = loader.load()
                for d in loaded:
                    d.metadata["source"] = str(file_path)
                docs.extend(loaded)
            except Exception:
                pass
        return docs

    from langchain_community.document_loaders import TextLoader
    docs: list[Document] = []
    for ext in ("*.txt", "*.md"):
        for path in sorted(root.glob(f"**/{ext}")):
            if not include_fallback and is_quarantined_fallback(path):
                continue
            try:
                loaded = TextLoader(str(path), encoding="utf-8").load()
                docs.extend(loaded)
            except Exception:
                pass
    return docs


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Create deterministic chunks using the production chunking contract."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=list(CHUNK_SEPARATORS),
    )
    return list(splitter.split_documents(list(documents)))


def load_and_split(data_path: str | Path, use_cache: bool = True) -> list[Document]:
    """Load and chunk one institution corpus, using persistent disk cache when available."""
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    # Check if root is or contains a V2 knowledge package
    pkg_manifest = root / "manifests" / "primary_corpus_manifest.json"
    pkg_root = root if pkg_manifest.exists() else None
    if not pkg_root and root.is_dir():
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and (sub / "manifests" / "primary_corpus_manifest.json").exists():
                pkg_root = sub
                break

    if pkg_root is not None:
        try:
            from ai_platform.packages.package_adapter import (
                load_institutional_release,
                chunk_package_documents,
            )
            cache_file = pkg_root / CHUNK_CACHE_FILENAME
            if not cache_file.exists():
                cache_file = root / CHUNK_CACHE_FILENAME

            current_sig = get_corpus_signature(pkg_root / "documents") or get_corpus_signature(root)

            if use_cache and cache_file.exists():
                try:
                    import json
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "chunks" in data and data["chunks"] and len(data["chunks"][0].get("metadata", {})) > 2:
                        return [
                            Document(
                                page_content=item["page_content"],
                                metadata=item.get("metadata", {}),
                            )
                            for item in data["chunks"]
                        ]
                except Exception:
                    pass

            release = load_institutional_release(pkg_root)
            chunks = chunk_package_documents(release)

            if use_cache and chunks:
                try:
                    import json
                    cache_payload = {
                        "signature": current_sig,
                        "release_id": release.release_id,
                        "count": len(chunks),
                        "chunks": [
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                            for doc in chunks
                        ],
                    }
                    tmp_cache = cache_file.with_suffix(".tmp")
                    with open(tmp_cache, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f)
                    tmp_cache.replace(cache_file)
                except Exception:
                    pass

            return chunks
        except Exception:
            pass

    cache_file = root / CHUNK_CACHE_FILENAME
    current_sig = get_corpus_signature(root)

    if use_cache and cache_file.exists():
        try:
            import json
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_sig = data.get("signature")
            if cached_sig == current_sig and "chunks" in data:
                return [
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {}),
                    )
                    for item in data["chunks"]
                ]
        except Exception:
            pass

    chunks = split_documents(load_documents(root))

    if use_cache and chunks:
        try:
            import json
            cache_payload = {
                "signature": current_sig,
                "count": len(chunks),
                "chunks": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in chunks
                ],
            }
            tmp_cache = cache_file.with_suffix(".tmp")
            with open(tmp_cache, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f)
            tmp_cache.replace(cache_file)
        except Exception:
            pass

    return chunks


def invalidate_chunk_cache(data_path: str | Path) -> bool:
    """Remove persistent chunk cache to force re-indexing."""
    cache_file = Path(data_path) / CHUNK_CACHE_FILENAME
    if cache_file.exists():
        cache_file.unlink()
        return True
    return False


__all__ = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_SEPARATORS",
    "CHUNK_CACHE_FILENAME",
    "get_corpus_signature",
    "load_documents",
    "split_documents",
    "load_and_split",
    "invalidate_chunk_cache",
]

"""
Build the vectorstore for the active institution.

Institution selection and data paths come from backend.config.
"""

from backend.config import DATA_PATH
from backend.ingestion import (
    load_documents,
    split_documents,
    create_vectorstore,
)


def main() -> None:
    print(f"Loading documents from: {DATA_PATH}")

    documents = load_documents(DATA_PATH)

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating Chroma vector store...")

    create_vectorstore(chunks)

    print("✅ Ingestion completed successfully!")


if __name__ == "__main__":
    main()

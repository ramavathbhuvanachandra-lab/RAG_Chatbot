from backend.ingestion import (
    load_documents,
    split_documents,
    create_vectorstore,
)


DATA_PATH = "./data/data_iitj"


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents(DATA_PATH)
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating Chroma vector store...")
    create_vectorstore(chunks)

    print("✅ Ingestion completed successfully!")
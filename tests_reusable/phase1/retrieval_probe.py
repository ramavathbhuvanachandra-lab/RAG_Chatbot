from __future__ import annotations

from pathlib import Path

from backend.config import DATA_PATH
from backend.ingestion import load_documents, split_documents
from backend.embedding import embeddings
from backend.vectorstore import vectorstore


QUERIES = [
    "What is the fee structure?",
    "Can you tell me about tuition fees?",
    "For which students is tuition fee waived?",
    "What are the different admission categories?",
    "What programs are offered?",
    "What is the hostel charge for double occupancy without bedding?",
    "What is the hostel charge for a visitor staying in a single room with bedding?",
    "How does summer course registration work for M.Tech students?",
    "What documents are required for registration?",
    "What happens if registration is late?",
    "Which research areas are available?",
    "I am interested in artificial intelligence research. What is available?",
    "Abey yaar, kya free sachar hai?",
]


def show_document(index: int, document) -> None:
    print("\n" + "=" * 100)
    print(f"RESULT #{index}")
    print("=" * 100)

    print("SOURCE:")
    print(document.metadata.get("source"))

    print("\nMETADATA:")
    for key, value in document.metadata.items():
        print(f"{key}: {value}")

    content = str(document.page_content or "").strip()

    print("\nCONTENT:")
    print(content[:1800])


def main() -> None:
    print("\n" + "#" * 100)
    print("SEMANTIC RETRIEVAL PROBE")
    print("#" * 100)

    print("\nDATA PATH:")
    print(DATA_PATH)

    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)

    print("\nDOCUMENTS:", len(documents))
    print("CHUNKS:", len(chunks))

    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 20}
        )
    except Exception as exc:
        print("\nFAILED TO CREATE VECTOR RETRIEVER:")
        print(type(exc).__name__, exc)
        return

    for query_number, query in enumerate(QUERIES, start=1):
        print("\n\n")
        print("#" * 100)
        print(f"QUERY {query_number}/{len(QUERIES)}")
        print("#" * 100)
        print(query)

        try:
            results = retriever.invoke(query)
        except Exception as exc:
            print("\nRETRIEVAL FAILED:")
            print(type(exc).__name__, exc)
            continue

        print("\nRETRIEVED RESULTS:", len(results))

        for index, document in enumerate(results, start=1):
            show_document(index, document)


if __name__ == "__main__":
    main()

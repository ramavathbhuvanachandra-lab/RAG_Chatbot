from backend.retriever import dense_retrieve

query = "How many dining halls are there?"

docs = dense_retrieve(query)

print("=" * 80)

for i, doc in enumerate(docs, 1):
    print(f"\nRank {i}")
    print(doc.metadata["source"])

    if "two dining halls" in doc.page_content.lower():
        print("✅ FOUND THE ANSWER CHUNK")

    print(doc.page_content[:300])
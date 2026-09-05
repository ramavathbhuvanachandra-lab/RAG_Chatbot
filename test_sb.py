from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
)

query = "How many dining halls are there?"

print("=" * 80)
print("QUERY:", query)

dense_docs = dense_retrieve(query)
keyword_docs = keyword_retrieve(query)

print("\n========== DENSE ==========\n")

for i, doc in enumerate(dense_docs, 1):
    print(f"\n--- Dense {i} ---")
    print(doc.metadata)
    print(doc.page_content[:500])

print("\n========== BM25 ==========\n")

for i, doc in enumerate(keyword_docs, 1):
    print(f"\n--- BM25 {i} ---")
    print(doc.metadata)
    print(doc.page_content[:500])

fused = reciprocal_rank_fusion([dense_docs, keyword_docs])

print("\n========== RRF ==========\n")

for i, doc in enumerate(fused[:10], 1):
    print(f"\n--- RRF {i} ---")
    print(doc.metadata)
    print(doc.page_content[:500])
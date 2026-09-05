from backend.graph import create_graph

graph = create_graph()

question ="Provide complete information about the IIT Jodhpur admission process"

state = {
    "question": question,
    "chat_history": []
}

result = graph.invoke(state)

print("=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\nREWRITTEN QUERY")
print(result["rewritten_question"])

print("\nGENERATED QUERIES")
for query in result["generated_queries"]:
    print("-", query)

print("\nRETRIEVAL LISTS :", len(result["retrieval_results"]))
print("FUSED DOCS      :", len(result["fused_docs"]))
print("COMPRESSED DOCS :", len(result["compressed_docs"]))

print("\nFINAL ANSWER")
print(result["answer"])
from backend.graph import create_graph

graph = create_graph()


def chatbot(question: str):
    result = graph.invoke(
        {
            "question": question,
            "chat_history": [],
        }
    )

    return {
        "answer": result["answer"],
        "context": result["context"],
    }
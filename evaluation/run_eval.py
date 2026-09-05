from pathlib import Path

import pandas as pd
from langsmith import traceable

from backend.chatbot import chatbot


@traceable(name="IITJ Chatbot")
def run_chatbot(question: str):
    return chatbot(question)


def main():
    evaluation_dir = Path(__file__).parent

    dataset_path = evaluation_dir / "dataset_v1.csv"
    results_path = evaluation_dir / "results.csv"

    df = pd.read_csv(dataset_path)

    results = []

    print("\nStarting Evaluation...\n")

    for i, row in df.iterrows():

        question = row["question"]
        category = row["category"]

        response = run_chatbot(question)

        answer = response["answer"]

        results.append({
            "category": category,
            "question": question,
            "answer": answer
        })

        print("=" * 80)
        print(f"{i+1}. {question}")
        print()
        print(answer)
        print()

    pd.DataFrame(results).to_csv(results_path, index=False)

    print("=" * 80)
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
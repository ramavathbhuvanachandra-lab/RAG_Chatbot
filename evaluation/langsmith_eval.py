from langsmith import Client
from langsmith.evaluation import evaluate

from backend.chatbot import chatbot
from evaluation.evaluators import (
    correctness,
    relevance,
    helpfulness,
    faithfulness,
)

# Connect to LangSmith
client = Client()

# Function that LangSmith will call for every question
def predict(inputs):
    question = inputs["question"]

    result = chatbot(question)

    return {
        "answer": result["answer"],
        "context": result["context"],
    }


def correctness_evaluator(run, example):

    result = correctness(
        example.inputs["question"],
        example.outputs["answer"],
        run.outputs["answer"],
    )

    return {
        "key": "correctness",
        "score": result.get("score", 0) / 10.0,
       "comment": result.get("reason", "")
    }

def relevance_evaluator(run, example):

    result = relevance(
        example.inputs["question"],
        run.outputs["answer"],
    )

    return {
        "key": "relevance",
       "score": result.get("score", 0) / 10.0,
        "comment": result.get("reason", "")
    }


def helpfulness_evaluator(run, example):

    result = helpfulness(
        example.inputs["question"],
        run.outputs["answer"],
    )

    return {
        "key": "helpfulness",
        "score": result.get("score", 0) / 10.0,
        "comment": result.get("reason", ""),
    }

def faithfulness_evaluator(run, example):

    result = faithfulness(
        run.outputs["context"],
        run.outputs["answer"],
    )

    return {
        "key": "faithfulness",
       "score": result.get("score", 0) / 10.0,
       "comment": result.get("reason", "")
    }
# Run evaluation
experiment = evaluate(
    predict,
    data="dataset_v2",
    evaluators=[
        correctness_evaluator,
        relevance_evaluator,
        helpfulness_evaluator,
        faithfulness_evaluator,
    ],
    experiment_prefix="IITJ Chatbot V2",
)

print(experiment)
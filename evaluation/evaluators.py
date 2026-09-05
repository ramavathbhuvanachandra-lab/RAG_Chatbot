import json

from langchain_ollama import ChatOllama

from evaluation.prompts import (
    CORRECTNESS_PROMPT,
    RELEVANCE_PROMPT,
    HELPFULNESS_PROMPT,
    FAITHFULNESS_PROMPT,
)

judge = ChatOllama(
    model="qwen2.5:3b ",
    temperature=0,
)



def run_judge(prompt):

    try:
        response = judge.invoke(prompt)
        text = response.content

        start = text.find("{")
        end = text.rfind("}") + 1

        result = json.loads(text[start:end])

        # Accept either "score" or "rating"
        score = result.get("score", result.get("rating"))

        if score is None:
            return {
                "score": 0,
                "reason": "Judge did not return a score."
            }

        score = float(score)

        # Clamp to [0, 10]
        score = max(0, min(score, 10))

        result["score"] = score

        if "reason" not in result:
            result["reason"] = ""

        return result

    except Exception as e:
        return {
            "score": 0,
            "reason": f"Judge parsing failed: {str(e)}"
        }


def correctness(question, reference_answer, answer):
    return run_judge(
        CORRECTNESS_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            answer=answer,
        )
    )


def relevance(question, answer):
    return run_judge(
        RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
    )


def helpfulness(question, answer):
    return run_judge(
        HELPFULNESS_PROMPT.format(
            question=question,
            answer=answer,
        )
    )


def faithfulness(context, answer):
    return run_judge(
        FAITHFULNESS_PROMPT.format(
            context=context,
            answer=answer,
        )
    )
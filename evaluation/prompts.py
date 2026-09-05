CORRECTNESS_PROMPT = """
You are an expert evaluator for an IIT Jodhpur RAG chatbot.

Question:
{question}

Reference Answer:
{reference_answer}

Student Answer:
{answer}

Evaluate ONLY the factual correctness of the student's answer.

Scoring Rubric:
10 = Completely correct.
8-9 = Mostly correct with only minor omissions.
6-7 = Partially correct with important missing information.
3-5 = Mostly incorrect.
1-2 = Completely incorrect.

Return ONLY valid JSON.

Do NOT include markdown.
Do NOT include explanations outside JSON.
Do NOT include ```json.

The output MUST exactly follow this schema:

{{
    "score": 9,
    "reason": "One short sentence."
}}
"""

RELEVANCE_PROMPT = """
You are evaluating answer relevance.

Question:
{question}

Answer:
{answer}

Evaluate whether the answer directly addresses the user's question.

Scoring Rubric:
10 = Completely relevant.
8-9 = Mostly relevant with minor extra information.
6-7 = Partially relevant.
3-5 = Mostly irrelevant.
1-2 = Completely irrelevant.

Return ONLY valid JSON.

Do NOT include markdown.
Do NOT include explanations outside JSON.

The output MUST exactly follow this schema:

{{
    "score": 8,
    "reason": "One short sentence."
}}
"""


HELPFULNESS_PROMPT = """
You are evaluating answer helpfulness.

Question:
{question}

Answer:
{answer}

Assume the user is a new IIT Jodhpur student.

Would this answer genuinely help them?

Scoring Rubric:
10 = Extremely helpful.
8-9 = Helpful with small missing details.
6-7 = Somewhat helpful.
3-5 = Not very helpful.
1-2 = Not helpful.

Return ONLY valid JSON.

Do NOT include markdown.
Do NOT include explanations outside JSON.

The output MUST exactly follow this schema:

{{
    "score": 9,
    "reason": "One short sentence."
}}
"""


FAITHFULNESS_PROMPT = """
You are evaluating factual grounding.

Context:
{context}

Answer:
{answer}

Check whether every factual claim in the answer is supported by the provided context.

Do NOT use outside knowledge.

Scoring Rubric:
10 = Fully supported.
8-9 = Almost fully supported.
6-7 = Some unsupported claims.
3-5 = Mostly unsupported.
1-2 = Hallucinated.

Return ONLY valid JSON.

Do NOT include markdown.
Do NOT include explanations outside JSON.

The output MUST exactly follow this schema:

{{
    "score": 10,
    "reason": "One short sentence."
}}
"""
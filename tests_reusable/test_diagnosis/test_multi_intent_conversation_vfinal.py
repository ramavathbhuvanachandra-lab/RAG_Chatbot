from ai_platform.core.conversation.multi_intent import decompose_multi_intent


def test_three_explicit_intents():
    units = decompose_multi_intent(
        "What is M.Tech eligibility? What documents are needed? What is the application fee?"
    )
    assert [u.question for u in units] == [
        "What is M.Tech eligibility?",
        "What documents are needed?",
        "What is the application fee?",
    ]


def test_coordinated_questions_split():
    units = decompose_multi_intent(
        "What is the MBA eligibility and what is the application deadline?"
    )
    assert len(units) == 2


def test_ordinary_coordinated_phrase_stays_single():
    units = decompose_multi_intent("What are the admission fees and charges?")
    assert len(units) == 1


def test_single_question_stays_single():
    units = decompose_multi_intent("How do I apply for M.Tech admission?")
    assert len(units) == 1

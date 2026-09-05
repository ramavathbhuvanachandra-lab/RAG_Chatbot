from backend.answer_guard import (
    UNKNOWN_RESPONSE,
    contains_internal_leak,
    contains_contact_fallback,
    guard_answer,
)


def test_unknown_response_constant():
    assert UNKNOWN_RESPONSE == (
        "I'm sorry, I don't know based on the available information."
    )


def test_internal_document_leak_detected():
    assert contains_internal_leak(
        "This information is supported by Document 1."
    )


def test_rrf_leak_detected():
    assert contains_internal_leak(
        "The RRF score was highest for this result."
    )


def test_contact_fallback_detected():
    assert contains_contact_fallback(
        "Please contact the Office of Admissions."
    )


def test_internal_reference_is_removed_but_answer_preserved():
    result = guard_answer(
        "Non-degree students may take courses for credit. "
        "This information is based on Document 1."
    )

    assert result["answer"] == (
        "Non-degree students may take courses for credit."
    )

    assert result["status"] == (
        "sanitized_internal_reference"
    )


def test_contact_fallback_is_removed_but_answer_preserved():
    result = guard_answer(
        "The admission process varies by program. "
        "Please contact the relevant office for more details."
    )

    assert result["answer"] == (
        "The admission process varies by program."
    )

    assert result["status"] == (
        "sanitized_contact_fallback"
    )


def test_empty_answer_uses_unknown():
    result = guard_answer("")

    assert result["answer"] == UNKNOWN_RESPONSE
    assert result["status"] == "empty_answer"


def test_internal_only_answer_becomes_unknown():
    result = guard_answer(
        "According to Document 1."
    )

    assert result["answer"] == UNKNOWN_RESPONSE
    assert result["status"] == "empty_answer"


def test_valid_answer_is_preserved():
    answer = (
        "IIT Jodhpur offers multiple academic programs."
    )

    result = guard_answer(answer)

    assert result["answer"] == answer
    assert result["status"] == "clean"
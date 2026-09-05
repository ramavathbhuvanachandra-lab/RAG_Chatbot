import sys
from pathlib import Path


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Imports
# =========================================================

from backend.nodes import answer_chain


# =========================================================
# Test Cases
# =========================================================

TEST_CASES = [
    {
        "id": "strong_hostel",
        "question": "What are the hostel facilities for students?",
        "context": """
Document 1
IIT Jodhpur hostel facilities include electricity and water supply,
Wi-Fi and LAN connectivity, TV rooms, table tennis rooms,
study rooms, music rooms, indoor gym facilities, laundry facilities,
RO/filtered drinking water, housekeeping, 24x7 security, and access
to medical support and emergency services.
""",
        "expected_behavior": (
            "Answer directly using the provided hostel facilities."
        ),
    },

    {
        "id": "ambiguous_btech_fees",
        "question": "What are the fees for B.Tech students?",
        "context": """
Document 1
Program Fee Structure (2026-2028)

The tuition fee for the program is INR 2,25,000 per semester.
In addition to the tuition fee, students are required to pay
semester fee, admission fee, convocation fee, refundable deposits,
dining fee, and other applicable charges.

The document does not identify this program as B.Tech.
""",
        "expected_behavior": (
            "Do not claim that INR 2,25,000 is definitely the B.Tech fee."
        ),
    },

    {
        "id": "unknown_btech_hostel_discount",
        "question": "Do B.Tech students get a hostel fee discount?",
        "context": """
Document 1
IIT Jodhpur hostels provide Wi-Fi, LAN connectivity,
common rooms, indoor gyms, laundry facilities, housekeeping,
security, and medical support.

No information about hostel fee discounts is provided.
""",
        "expected_behavior": (
            "State that the provided information does not contain "
            "the hostel fee discount."
        ),
    },
]


# =========================================================
# Run One Test
# =========================================================

def run_test(test_case):

    print()
    print("=" * 100)
    print(
        f"PHASE 6 — {test_case['id']}"
    )
    print("=" * 100)

    print(
        f"\nQuestion:\n{test_case['question']}"
    )

    print(
        f"\nExpected behavior:\n"
        f"{test_case['expected_behavior']}"
    )

    print(
        "\nGenerating answer..."
    )

    response = answer_chain.invoke(
        {
            "context":
                test_case["context"],

            "question":
                test_case["question"],

            "chat_history":
                [],
        }
    )

    answer = response.content.strip()

    print()
    print(
        "ANSWER:"
    )
    print(answer)


# =========================================================
# Main
# =========================================================

def main():

    for test_case in TEST_CASES:
        run_test(
            test_case
        )

    print()
    print("=" * 100)
    print(
        "PHASE 6 DECISION"
    )
    print("=" * 100)

    print(
        "Check:"
    )

    print(
        "1. Strong evidence → direct answer"
    )

    print(
        "2. Ambiguous evidence → no unsupported assumption"
    )

    print(
        "3. Missing evidence → explicit uncertainty / I don't know"
    )

    print()
    print(
        "Do not modify answer_prompt yet."
    )


if __name__ == "__main__":
    main()
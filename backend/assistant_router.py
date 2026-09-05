from backend.campus_navigation import find_location
from backend.emergency import find_emergency


def assistant_router(question: str):
    """
    Route the question to structured modules first.
    """

    structured = []

    question_lower = question.lower()

    # Navigation intent keywords
    NAVIGATION_KEYWORDS = [
    "where is",
    "location",
    "locate",
    "map",
    "maps",
    "navigate",
    "navigation",
    "direction",
    "directions",
    "how do i reach",
    "how to reach",
    "take me to",
    "route to",
    "way to"
]
    # Check if the user is asking for navigation
    is_navigation = any(
        keyword in question_lower
        for keyword in NAVIGATION_KEYWORDS
    )

    # -------------------------------
    # Campus Navigation
    # -------------------------------
    if is_navigation:
        location = find_location(question)

        if location:
            structured.append({
                "type": "navigation",
                "data": location
            })

    # -------------------------------
    # Emergency Contacts
    # -------------------------------
    emergency = find_emergency(question)

    if emergency:
        structured.append({
            "type": "emergency",
            "data": emergency
        })

    # -------------------------------
    # RAG Decision
    # -------------------------------
    information_keywords = [
        "timing",
        "timings",
        "time",
        "hours",
        "working hours",
        "open",
        "close",
        "fee",
        "fees",
        "admission",
        "hostel",
        "facility",
        "facilities",
        "department",
        "course",
        "syllabus",
        "process",
        "procedure",
        "rules",
        "eligibility",
        "contact",
        "email",
        "what",
        "when",
        "why",
        "who",
        "which",
        "explain",
        "tell me",
        "provide",
        "information",
        "details"
    ]

    need_rag = any(
        keyword in question_lower
        for keyword in information_keywords
    )

    # If nothing matched in structured routing,
    # always use RAG.
    if not structured:
        need_rag = True

    return {
        "structured": structured,
        "need_rag": need_rag,
    }
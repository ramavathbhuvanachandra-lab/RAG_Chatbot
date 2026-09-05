from backend.database import supabase
from datetime import datetime, timezone
from collections import Counter

TOPICS = {
    "Admission": ["admission", "join", "registration", "document"],
    "Hostel": ["hostel", "room", "accommodation"],
    "Mess": ["mess", "food", "canteen"],
    "Fees": ["fee", "fees", "payment"],
    "WiFi": ["wifi", "internet"],
    "Laundry": ["laundry", "washing"],
    "Medical": ["medical", "hospital", "health"],
    "Scholarship": ["scholarship"],
    "Campus Map": ["map", "location", "where"],
    "Departments": ["department", "branch", "electrical", "cse", "ai", "mechanical"],
    "Research": ["research", "lab", "professor", "phd"],
    "Clubs": ["club", "society", "prometeo", "varchas"],
    "Library": ["library"],
    "Placement": ["placement", "internship"],
    "Drinking Water": ["water", "drinking water"]
}
def get_total_users():
    response = (
        supabase.table("users")
        .select("*", count="exact")
        .execute()
    )
    return response.count or 0


def get_total_sessions():
    response = (
        supabase.table("sessions")
        .select("*", count="exact")
        .execute()
    )
    return response.count or 0


def get_total_questions():
    response = (
        supabase.table("messages")
        .select("*", count="exact")
        .eq("role", "user")
        .execute()
    )
    return response.count or 0


def get_average_response_time():
    response = (
        supabase.table("messages")
        .select("response_time")
        .eq("role", "assistant")
        .execute()
    )

    times = [
        row["response_time"]
        for row in response.data
        if row.get("response_time") is not None
    ]

    if not times:
        return 0

    return round(sum(times) / len(times), 2)


def get_latest_questions(limit=10):
    response = (
        supabase.table("messages")
        .select("message, timestamp")
        .eq("role", "user")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data




def get_today_users():
    today = datetime.now(timezone.utc).date()

    response = (
        supabase.table("users")
        .select("created_at")
        .execute()
    )

    return sum(
        1
        for row in response.data
        if datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).date() == today
    )


def get_today_questions():
    today = datetime.now(timezone.utc).date()

    response = (
        supabase.table("messages")
        .select("timestamp")
        .eq("role", "user")
        .execute()
    )

    return sum(
        1
        for row in response.data
        if datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).date() == today
    )


def get_questions_per_user():
    users = get_total_users()
    questions = get_total_questions()

    if users == 0:
        return 0

    return round(questions / users, 2)


def get_average_questions_per_session():
    sessions = get_total_sessions()
    questions = get_total_questions()

    if sessions == 0:
        return 0

    return round(questions / sessions, 2)


def get_recent_users(limit=10):
    response = (
        supabase.table("users")
        .select("name, phone, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data




def get_top_topics(limit=10):
    response = (
        supabase.table("messages")
        .select("message")
        .eq("role", "user")
        .execute()
    )

    counter = Counter()

    for row in response.data:
        question = row["message"].lower()

        for topic, keywords in TOPICS.items():
            if any(keyword in question for keyword in keywords):
                counter[topic] += 1

    return counter.most_common(limit)
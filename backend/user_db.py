import uuid
from backend.database import supabase


def get_user_by_phone(phone: str):
    response = (
        supabase.table("users")
        .select("*")
        .eq("phone", phone)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def create_user(name: str, phone: str):
    user_id = f"USR_{uuid.uuid4().hex[:8].upper()}"

    response = (
        supabase.table("users")
        .insert({
            "user_id": user_id,
            "name": name,
            "phone": phone
        })
        .execute()
    )

    return response.data[0]


def get_or_create_user(name: str, phone: str):
    """
    Returns an existing user if the phone number is already registered.
    Otherwise creates a new user.
    """

    user = get_user_by_phone(phone)

    if user:
        return user

    return create_user(name, phone)
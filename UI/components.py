import streamlit as st


# ---------------------------------------------------------
# Section Divider
# ---------------------------------------------------------

def divider():

    st.divider()


# ---------------------------------------------------------
# Section Title
# ---------------------------------------------------------

def section_title(title: str):

    st.subheader(title)


# ---------------------------------------------------------
# Information Box
# ---------------------------------------------------------

def info_box(message: str):

    st.info(message)


# ---------------------------------------------------------
# Success Box
# ---------------------------------------------------------

def success_box(message: str):

    st.success(message)


# ---------------------------------------------------------
# Warning Box
# ---------------------------------------------------------

def warning_box(message: str):

    st.warning(message)


# ---------------------------------------------------------
# Error Box
# ---------------------------------------------------------

def error_box(message: str):

    st.error(message)

# ---------------------------------------------------------
# Quick Action Cards
# ---------------------------------------------------------

QUICK_ACTIONS = {
    "🎓 Admissions":
        "Provide complete information about IIT Jodhpur admissions, reporting process, document verification, orientation, and joining formalities.",

    "🏠 Hostel & Mess":
        "Provide complete information about IIT Jodhpur hostels, hostel allocation, facilities, mess, Wi-Fi, laundry, and hostel rules.",

    "🚨 Emergency":
        "Provide all IIT Jodhpur emergency contacts, Medical Centre information, ambulance, security, and student support services.",

    "📚 Library & IT":
        "Provide complete information about the library, Wi-Fi, ERP, institute email, computer centre, and IT services."
}
def quick_action_cards():
    """
    Display quick action buttons.

    Returns:
        str | None
            The predefined question corresponding to the clicked button.
    """

    st.markdown("### Quick Actions")

    # Full-width Campus Map button
    if st.button("🗺️ Campus Map", use_container_width=True):
        return "CAMPUS_MAP"

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎓 Admissions", use_container_width=True):
            return QUICK_ACTIONS["🎓 Admissions"]

        if st.button("🚨 Emergency", use_container_width=True):
            return QUICK_ACTIONS["🚨 Emergency"]

    with col2:
        if st.button("🏠 Hostel & Mess", use_container_width=True):
            return QUICK_ACTIONS["🏠 Hostel & Mess"]

        if st.button("📚 Library & IT", use_container_width=True):
            return QUICK_ACTIONS["📚 Library & IT"]

    return None
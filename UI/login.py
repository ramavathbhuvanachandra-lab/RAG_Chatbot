import streamlit as st

from backend.authentication import authenticate_user


def render_login():
    """
    Render the demo phone-number login.

    Authentication and persistence decisions are handled by the
    backend authentication layer so this UI remains reusable when
    the frontend is replaced later.
    """

    st.title("🎓 IIT Jodhpur AI Assistant")

    st.subheader("Welcome!")

    st.write("Please enter your details to continue.")

    name = st.text_input("Full Name")

    phone = st.text_input(
        "Phone Number",
        max_chars=10,
        placeholder="Enter 10-digit mobile number",
    )

    if st.button("Continue", use_container_width=True):

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------

        name = name.strip()
        phone = phone.strip()

        if len(name) < 3:
            st.error("Please enter your full name.")
            return

        if len(phone) != 10 or not phone.isdigit():
            st.error("Please enter a valid 10-digit phone number.")
            return

        # ---------------------------------------------------------
        # Authenticate through application layer
        # ---------------------------------------------------------

        auth_result = authenticate_user(
            name=name,
            phone=phone,
        )

        if not auth_result["authenticated"]:
            st.error("Unable to start your session. Please try again.")
            return

        # ---------------------------------------------------------
        # Store session state
        # ---------------------------------------------------------

        st.session_state.logged_in = True
        st.session_state.user_id = auth_result["user_id"]
        st.session_state.user_name = auth_result["user_name"]
        st.session_state.session_id = auth_result["session_id"]

        st.session_state.persistence_available = (
            auth_result["persistence_available"]
        )

        st.session_state.persistence_mode = (
            auth_result["persistence_mode"]
        )

        st.rerun()
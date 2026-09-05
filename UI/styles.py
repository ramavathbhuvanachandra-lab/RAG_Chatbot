import streamlit as st


def load_css():
    st.markdown(
        """
        <style>

        /* Remove Streamlit Header */
        header {
            visibility: hidden;
        }

        /* Reduce top padding */
        .block-container {
            padding-top: 2rem;
        }

        /* Sidebar width */
        section[data-testid="stSidebar"] {
            width: 320px !important;
        }

        /* Chat input */
        .stChatInput {
            padding-top: 12px;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
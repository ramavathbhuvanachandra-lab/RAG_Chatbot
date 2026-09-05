import streamlit as st

from UI.chat import create_new_chat


def render_sidebar():

    with st.sidebar:

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        st.title("🎓 IITJ Assistant")

        st.markdown("---")

        # -------------------------------------------------
        # New Chat
        # -------------------------------------------------

        if st.button(
            "➕ New Chat",
            use_container_width=True,
        ):
            create_new_chat()
            st.rerun()

        st.markdown("---")

        # -------------------------------------------------
        # Previous Chats
        # -------------------------------------------------

        st.subheader("💬 Previous Chats")

        for chat_id, chat in st.session_state.conversations.items():
            is_active = chat_id == st.session_state.active_chat
            label = f"🟢 {chat['title']}" if is_active else f"⚪ {chat['title']}"

            if st.button(
                label,
                
        
                key=chat_id,
                use_container_width=True,
            ):

                st.session_state.active_chat = chat_id
                st.rerun()

        st.markdown("---")

        # -------------------------------------------------
        # About
        # -------------------------------------------------

        st.subheader("About")

        st.write(
            """
This chatbot answers questions related to IIT Jodhpur.

### Supported Topics

- Admissions
- Academics
- Departments
- Research
- Hostel
- Campus Life
- Facilities
"""
        )

        st.markdown("---")

        # -------------------------------------------------
        # Footer
        # -------------------------------------------------

        st.caption("Built with ❤️ using")
        st.caption("• LangGraph")
        st.caption("• ChromaDB")
        st.caption("• Streamlit")
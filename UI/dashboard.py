import streamlit as st

from backend.dashboard_db import (
    get_total_users,
    get_total_sessions,
    get_total_questions,
    get_average_response_time,
    get_latest_questions,
    get_today_users,
    get_today_questions,
    get_questions_per_user,
    get_average_questions_per_session,
    get_top_topics,
)
from datetime import datetime, timezone
from collections import Counter

def show_dashboard():

    st.title("📊 IIT Jodhpur AI Assistant Dashboard")

    try:

        users = get_total_users()
        sessions = get_total_sessions()
        questions = get_total_questions()
        avg_time = get_average_response_time()

        today_users = get_today_users()
        today_questions = get_today_questions()
        questions_per_user = get_questions_per_user()
        questions_per_session = get_average_questions_per_session()

        top_topics = get_top_topics()
        latest = get_latest_questions()

        # -----------------------------
        # Overview
        # -----------------------------

        st.subheader("Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Users", users)
        c2.metric("Sessions", sessions)
        c3.metric("Questions", questions)
        c4.metric("Avg Response (s)", avg_time)

        c5, c6, c7, c8 = st.columns(4)

        c5.metric("Today's Users", today_users)
        c6.metric("Today's Questions", today_questions)
        c7.metric("Questions / User", questions_per_user)
        c8.metric("Questions / Session", questions_per_session)

        st.divider()

        # -----------------------------
        # Top Topics
        # -----------------------------

        st.subheader("🔥 Top Asked Topics")

        if top_topics:

            topic_data = [
                {
                    "Topic": topic,
                    "Count": count,
                }
                for topic, count in top_topics
            ]

            st.table(topic_data)

        else:

            st.info("No topic data available.")

        st.divider()

        # -----------------------------
        # Latest Questions
        # -----------------------------

        with st.expander("🕒 Latest Questions"):

            if latest:
                st.table(latest)
            else:
                st.info("No recent questions.")

    except Exception as e:

        st.error(f"Dashboard Error: {e}")


import json
from pathlib import Path

import streamlit as st

def load_locations():

    json_path = Path(__file__).parent.parent / "data" / "campus_locations.json"

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_campus_map():

    st.title("🗺️ Campus Map")
    st.caption("Select a location to get directions.")

    if st.button("⬅ Back", use_container_width=True):
        st.session_state.show_campus_map = False
        st.rerun()

    st.write("")

    locations = load_locations()

    cols = st.columns(2)

    button_index = 0

    for location in locations:

        if not location.get("show_in_map", False):
            continue

        with cols[button_index % 2]:

            if st.button(
                f"📍 {location['name']}",
                use_container_width=True,
                key=f"location_{location['id']}",
            ):
                return location["name"]

        button_index += 1

    return None
"""Structured Cards and Campus Map Presentation.

Renders navigation cards, emergency contact cards, and the interactive campus
location selector grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import streamlit as st

if TYPE_CHECKING:
    from frontend.core.models import CampusLocationItem


def render_structured_card(block: dict[str, Any]) -> None:
    """Render a structured navigation or emergency card inside a chat message."""
    if not isinstance(block, dict):
        return

    block_type = block.get("type")
    data = block.get("data")
    if not isinstance(data, dict):
        return

    if block_type == "navigation":
        name = data.get("name") or "Campus Location"
        desc = data.get("description") or ""
        timings = data.get("timings") or ""
        maps_url = data.get("google_maps") or ""
        directions = data.get("directions") or ""

        with st.container():
            st.markdown(
                f"""
                <div class="structured-card">
                    <div class="structured-card-title">📍 {name}</div>
                    {f"<p>{desc}</p>" if desc else ""}
                    {f"<p><strong>🕒 Timings:</strong> {timings}</p>" if timings else ""}
                    {f"<p><strong>🧭 Directions:</strong> {directions}</p>" if directions else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if maps_url and maps_url.startswith("http"):
                st.link_button("🗺️ Open in Google Maps", maps_url, use_container_width=False)

    elif block_type == "emergency":
        name = data.get("name") or "Emergency Service"
        phone = data.get("phone") or ""
        desc = data.get("description") or ""
        help_text = data.get("help_text") or ""

        st.markdown(
            f"""
            <div class="structured-card" style="border-left-color: #EF4444;">
                <div class="structured-card-title" style="color: #EF4444;">🚨 {name}</div>
                {f"<p style='font-size: 1.1rem; font-weight: 600;'>📞 Phone: {phone}</p>" if phone else ""}
                {f"<p>{desc}</p>" if desc else ""}
                {f"<p><em>{help_text}</em></p>" if help_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_campus_map_grid(locations: tuple[CampusLocationItem, ...]) -> str | None:
    """Render the 2-column interactive campus location selector grid.

    Returns:
        The location query string if a button is clicked, else None.
    """
    st.markdown("### 🗺️ Campus Location Directory")
    st.caption("Select a campus facility to view directions, operating hours, and location details.")

    if st.button("⬅ Back to Chat", use_container_width=False):
        st.session_state.show_campus_map = False
        st.rerun()

    st.write("")

    valid_locations = [loc for loc in locations if loc.show_in_map]
    if not valid_locations:
        st.info("No campus map locations configured for this institution.")
        return None

    cols = st.columns(2)
    clicked_query: str | None = None

    for idx, loc in enumerate(valid_locations):
        with cols[idx % 2]:
            category_icon = "🏢" if loc.category == "Academic" else ("🍽️" if loc.category == "Food" else "📍")
            label = f"{category_icon} {loc.name}"
            if st.button(label, key=f"loc_btn_{loc.id}", use_container_width=True):
                clicked_query = f"Where is the {loc.name}?"

    return clicked_query


__all__ = [
    "render_structured_card",
    "render_campus_map_grid",
]

"""Engine health status shown the way a cockpit display shows alarms.

Normal is quiet and warnings are loud: Healthy is plain green text, Watch is an
amber outline, and Replace soon is a solid red pill. The styles live in the
shared stylesheet in app.py, so every page uses one alarm language.
"""

import streamlit as st

_CLASS = {"Healthy": "healthy", "Watch": "watch", "Replace soon": "replace"}


def status_pill(label: str) -> None:
    """Show a health status label styled by its urgency."""
    st.html(
        f'<span class="status-pill status-{_CLASS[label]}">{label}</span>',
        width="content",
    )

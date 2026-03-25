"""
Streamlit dashboard placeholder.

This module is intentionally minimal so the repo still works even if
Streamlit is not installed.
"""

from __future__ import annotations


def create_dashboard():
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Streamlit is not installed.") from exc

    st.title("Real-Time Industrial Anomaly Detection (Orion ML)")
    st.write("Add charts + inference controls here.")


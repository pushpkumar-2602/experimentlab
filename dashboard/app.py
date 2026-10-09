import httpx
import streamlit as st

# The dashboard never touches the database. It only asks the API for data.
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ExperimentLab", layout="wide")
st.title("ExperimentLab")
st.caption("Experiment results, guardrails, and decisions")


def fetch(path: str):
    """Calls the API and returns the JSON response."""
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


try:
    experiments = fetch("/experiments")
except httpx.HTTPError:
    st.error("Can't reach the API. Start it with: python -m uvicorn src.api.main:app --reload")
    st.stop()  # nothing below this line runs if the API is down

st.subheader("Experiments")
st.dataframe(experiments, use_container_width=True)

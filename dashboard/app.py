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


def format_p_value(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


try:
    experiments = fetch("/experiments")
except httpx.HTTPError:
    st.error("Can't reach the API. Start it with: python -m uvicorn src.api.main:app --reload")
    st.stop()  # nothing below this line runs if the API is down

# --- Sidebar: choose an experiment -------------------------------------
options = {f"{e['name']} (#{e['experiment_id']})": e["experiment_id"] for e in experiments}
choice = st.sidebar.selectbox("Experiment", list(options.keys()))
experiment_id = options[choice]

report = fetch(f"/experiments/{experiment_id}/analysis")
analysis = report["analysis"]
guardrails = report["guardrails"]
decision = report["final_decision"]

# --- Verdict banner -----------------------------------------------------
st.header(choice)

if decision == "SHIP":
    st.success("Decision: SHIP. The experiment is valid and the lift is statistically significant.")
elif decision == "INVESTIGATE":
    st.warning("Decision: INVESTIGATE. A guardrail failed, so these results should not be trusted yet.")
else:
    st.error("Decision: DO NOT SHIP. The experiment is valid, but there is no significant improvement.")

# --- Key numbers --------------------------------------------------------
if report["experiment_status"] != "VALID":
    st.caption("The statistics below are shown for transparency only. Do not act on them until the guardrails pass.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control conversion", f"{analysis['rate_control']:.2%}")
col2.metric("Treatment conversion", f"{analysis['rate_treatment']:.2%}")
col3.metric("Absolute lift", f"{analysis['absolute_lift']:.2%}")
col4.metric("Relative lift", f"{analysis['relative_lift']:.1%}")

st.write(
    f"**95% confidence interval on the lift:** "
    f"[{analysis['ci_95_low']:.2%}, {analysis['ci_95_high']:.2%}]  |  "
    f"**p-value:** {format_p_value(analysis['p_value'])}"
)

# --- Guardrails ---------------------------------------------------------
st.subheader("Guardrails")
srm = guardrails["srm_check"]
size = guardrails["sample_size_check"]

left, right = st.columns(2)

with left:
    st.markdown("**Sample ratio check**")
    st.write(f"Treatment: {srm['n_treatment']:,} users  |  Control: {srm['n_control']:,} users")
    st.write(f"Observed split: {srm['observed_ratio']:.1%} treatment (expected {srm['expected_ratio']:.0%})")
    if srm["srm_detected"]:
        st.error(srm['status'])
    else:
        st.success("PASS")

with right:
    st.markdown("**Sample size check**")
    st.write(f"Required per group: {size['required_n_per_group']:,}")
    st.write(f"Actual per group: {size['actual_n_per_group']:,}")
    if size["has_enough_data"]:
        st.success("PASS")
    else:
        st.error(size['status'])

import altair as alt
import httpx
import pandas as pd
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

# --- For whom did it work? ----------------------------------------------
st.subheader("For whom did it work?")

slice_labels = {
    "is_new_user": "New vs returning users",
    "user_value_segment": "Customer value segment",
}
slice_by = st.radio(
    "Slice results by",
    list(slice_labels.keys()),
    format_func=lambda key: slice_labels[key],
    horizontal=True,
)

seg_report = fetch(f"/experiments/{experiment_id}/segments?by={slice_by}")

rows = []
for seg in seg_report["segments"]:
    if "skipped" in seg:
        continue  # too few users to test
    name = seg["segment"]
    if slice_by == "is_new_user":
        name = "New users" if name == "True" else "Returning users"
    rows.append({
        "segment": name,
        "users": seg["n_treatment"] + seg["n_control"],
        "lift": seg["absolute_lift"],
        "ci_low": seg["ci_95_low"],
        "ci_high": seg["ci_95_high"],
        "p_value": seg["p_value"],
        "significant_after_correction": seg["significant_after_correction"],
    })

if not rows:
    st.info("Not enough users in any segment to analyze.")
else:
    if report["experiment_status"] != "VALID":
        st.caption("These segment effects come from an experiment that failed its guardrails. Treat them as unreliable.")

    seg_df = pd.DataFrame(rows)

    base = alt.Chart(seg_df).encode(y=alt.Y("segment:N", title=None, sort=None))
    interval = base.mark_rule(strokeWidth=3).encode(
        x=alt.X("ci_low:Q", title="Absolute lift in conversion rate (with 95% interval)",
                axis=alt.Axis(format=".1%")),
        x2="ci_high:Q",
    )
    dots = base.mark_point(size=140, filled=True).encode(x="lift:Q")
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")

    st.altair_chart(interval + dots + zero_line, width="stretch")

    table = pd.DataFrame({
        "Segment": seg_df["segment"],
        "Users": seg_df["users"],
        "Lift": seg_df["lift"].map("{:.2%}".format),
        "95% interval": [f"[{lo:.2%}, {hi:.2%}]" for lo, hi in zip(seg_df["ci_low"], seg_df["ci_high"])],
        "p-value": seg_df["p_value"].map(format_p_value),
        "Significant after correction": seg_df["significant_after_correction"].map({True: "Yes", False: "No"}),
    })
    st.dataframe(table, hide_index=True, width="stretch")

    st.caption(
        f"Each segment is tested at a stricter threshold (p < {seg_report['alpha_per_segment_after_correction']:.4f}) "
        "because testing several segments raises the chance of a false positive. "
        "Overlapping intervals do not prove two segments are the same; comparing segments properly needs a test of the difference between them."
    )

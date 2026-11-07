# ExperimentLab

A causal experimentation platform. It takes an A/B test (or an observational study), checks whether the experiment can be trusted, estimates the effect with a method suited to how the data was collected, and reports a decision.

**Live demo:** [dashboard](https://experimentlab-rhmkucc4jc3crolmrzexrq.streamlit.app) | [API docs](https://experimentlab-api.onrender.com/docs)

> The demo runs on free hosting. If nobody has visited recently, the first load can take a minute or two while the servers wake up.

All data is simulated, with effects planted on purpose so each method can be checked against a known true answer. The results below show how the methods behave. They are not claims about real products.

## What it does

- **Guardrails first.** A sample ratio mismatch check and a minimum sample size check run before any result is trusted. If a guardrail fails, the decision is INVESTIGATE, however significant the lift looks.
- **A/B analysis.** Two-proportion z-test for conversion and Welch's t-test for continuous metrics, with lift and 95% confidence intervals.
- **For whom did it work?** Per-segment effects with a Bonferroni correction for multiple testing, plus one joint test of whether the effect truly differs between segments.
- **Causal methods.** CUPED, propensity scores with inverse probability weighting (IPW), and difference-in-differences, each validated against a planted true effect.

## Results (simulated data, known true effects)

| Method | Scenario | Result |
|---|---|---|
| A/B test + guardrails | Randomized recommendation test, 10,000 users | Conversion 7.74% to 11.05% (+3.32 points), 95% CI [2.17%, 4.46%], guardrails pass, SHIP |
| Segmented analysis | Same experiment, new vs returning users | New users +5.18 points, returning users +2.53 points |
| Segment-difference test | Same experiment | New vs returning p = 0.043 (borderline); value segments p = 0.39 (no evidence of a difference) |
| CUPED | Session duration (a dense metric) | Confidence interval 57% narrower (0.269 to 0.115 minutes) |
| CUPED | Revenue (over 90% of users have $0) | Under 1% narrower |
| IPW | Email campaign targeted at engaged users (confounded) | Naive lift 4.93 points vs true 4.00; IPW estimate 4.15, which removes 83.5% of the bias |
| Difference-in-differences | Regional rollout with a shared background trend | Naive 4.88 points vs true 3.00; DiD estimate 2.12, CI [1.21, 3.04], which removes 53.5% of the bias |

Each row comes from a single simulated dataset with a fixed random seed. The size of each correction would change with a different seed, so read the numbers as demonstrations of how each method behaves, not as expected performance.

## What the results taught me

- **CUPED only helps when the covariate explains the metric.** It cut the interval by more than half on a dense metric, and almost nothing on zero-inflated revenue, where most of the variance is the coin flip of whether someone converts at all.
- **Difference-in-differences undershot.** With 60 days of data and four noisy averages, the estimate landed below the true effect, and the true value sat only just inside the confidence interval. Both the correction and its uncertainty were real, but it was a weaker correction than IPW.
- **Segment rankings can be noise.** In the value-segment chart, the order of the three dots looks meaningful but is not: the planted effect is the same in every value segment. Comparing segments properly needs a joint test of the difference between them, which the platform now runs: it finds no evidence of a difference across value segments (p = 0.39), and only borderline evidence between new and returning users (p = 0.043), even though a real 3-point gap was planted, because detecting a difference between effects takes more data than detecting the effects themselves.
- **The guardrails override the statistics.** The email-campaign experiment shows a large, highly significant lift and is still marked INVESTIGATE, because it was not randomized and its group sizes fail the sample ratio check.

## Architecture

```
Browser -> Streamlit dashboard -> FastAPI -> PostgreSQL
```

Hosted on free tiers: Streamlit Community Cloud, Render (Docker), and Neon. Locally, Docker Compose runs the database, API and dashboard together. The dashboard has no statistics code. It only displays what the API returns.

## Run it locally

Requires Docker.

```
git clone https://github.com/pushpkumar-2602/experimentlab.git
cd experimentlab
docker compose up --build -d
docker compose exec api python -m src.database.load_data
```

Run the load command once only; it adds experiments each time. Then open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API).

## Tests

60 automated tests with pytest. The 46 unit tests need no database:

```
pip install -r requirements.txt
pytest tests/ -q -m "not integration"
```

The 14 integration tests read from a local PostgreSQL database with the data loaded.

## Project structure

```
src/data_generation/   simulated users
src/experiments/       assignment, outcome simulation, guardrails, simulated rollout data
src/analysis/          A/B tests, CUPED, IPW, difference-in-differences
src/database/          schema, loader, queries
src/api/               FastAPI app
dashboard/             Streamlit app
tests/                 pytest suite
run_pipeline.py        regenerates the CSV data
```

## Limitations and next steps

- CUPED, IPW and difference-in-differences run as scripts in `src/analysis`. The API and dashboard expose the A/B test, guardrails and segment analysis.
- The database does not store whether an experiment was randomized, so an observational experiment is caught by the sample ratio check and not routed to IPW.
- The API has no authentication. That is acceptable for fake data and would not be for real data.
- Not implemented: novelty-effect detection, peeking corrections, network-effect handling, and guardrails for missing data, outliers and covariate imbalance.
- Local and Docker databases run PostgreSQL 16, while the hosted Neon database runs a newer version.

## Tech stack

Python 3.11, pandas, NumPy, SciPy, statsmodels, scikit-learn, FastAPI, SQLAlchemy, PostgreSQL, Streamlit, Docker and Compose, pytest.

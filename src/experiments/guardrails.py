import pandas as pd
import numpy as np
from scipy import stats

def check_sample_ratio_mismatch(results_df: pd.DataFrame, expected_ratio: float = 0.5,
                                  group_col: str = "group", alpha: float = 0.01) -> dict:
    """
    Checks whether the observed treatment/control split significantly differs
    from the expected split. Uses a chi-square goodness-of-fit test.

    Note: alpha is stricter here (0.01, not 0.05) because SRM checks are meant
    to only fire on REAL problems, not everyday random noise.
    """
    counts = results_df[group_col].value_counts()
    n_treatment = counts.get("treatment", 0)
    n_control = counts.get("control", 0)
    n_total = n_treatment + n_control

    observed = [n_treatment, n_control]
    expected = [n_total * expected_ratio, n_total * (1 - expected_ratio)]

    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    srm_detected = p_value < alpha

    return {
        "n_treatment": n_treatment,
        "n_control": n_control,
        "expected_ratio": expected_ratio,
        "observed_ratio": n_treatment / n_total,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "srm_detected": srm_detected,
        "status": "FAIL - SRM DETECTED" if srm_detected else "PASS",
    }


def check_minimum_sample_size(baseline_rate: float, minimum_detectable_effect: float,
                                actual_n_per_group: int, alpha: float = 0.05,
                                power: float = 0.8) -> dict:
    """
    Estimates the minimum sample size needed PER GROUP to reliably detect
    a given effect size, and compares it to what we actually have.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_avg = (p1 + p2) / 2

    numerator = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                 z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    required_n = int(np.ceil(numerator / denominator))

    has_enough_data = actual_n_per_group >= required_n

    return {
        "required_n_per_group": required_n,
        "actual_n_per_group": actual_n_per_group,
        "has_enough_data": has_enough_data,
        "status": "PASS" if has_enough_data else "FAIL - UNDERPOWERED",
    }


def run_all_guardrails(results_df: pd.DataFrame, baseline_rate: float,
                        minimum_detectable_effect: float) -> dict:
    """Runs every guardrail check and returns a combined report."""
    srm_result = check_sample_ratio_mismatch(results_df)

    n_per_group = min(
        (results_df["group"] == "treatment").sum(),
        (results_df["group"] == "control").sum(),
    )
    sample_size_result = check_minimum_sample_size(
        baseline_rate, minimum_detectable_effect, n_per_group
    )

    overall_valid = (not srm_result["srm_detected"]) and sample_size_result["has_enough_data"]

    return {
        "srm_check": srm_result,
        "sample_size_check": sample_size_result,
        "experiment_status": "VALID" if overall_valid else "INVALID - CHECK GUARDRAILS",
    }


def print_guardrail_report(report: dict):
    print("=" * 45)
    print("EXPERIMENT GUARDRAILS")
    print("=" * 45)

    srm = report["srm_check"]
    print(f"\n[Sample Ratio Mismatch Check]")
    print(f"  Treatment: {srm['n_treatment']:,}  Control: {srm['n_control']:,}")
    print(f"  Observed ratio: {srm['observed_ratio']:.2%}  (expected {srm['expected_ratio']:.2%})")
    print(f"  Chi-square p-value: {srm['p_value']:.4f}")
    print(f"  Result: {srm['status']}")

    size = report["sample_size_check"]
    print(f"\n[Minimum Sample Size Check]")
    print(f"  Required per group: {size['required_n_per_group']:,}")
    print(f"  Actual per group:   {size['actual_n_per_group']:,}")
    print(f"  Result: {size['status']}")

    print(f"\nOVERALL EXPERIMENT STATUS: {report['experiment_status']}")
    print("=" * 45)


if __name__ == "__main__":
    df = pd.read_csv("data/experiment_results.csv")
    report = run_all_guardrails(
        df,
        baseline_rate=0.08,
        minimum_detectable_effect=0.02,
    )
    print_guardrail_report(report)
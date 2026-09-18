import pandas as pd
from src.analysis.ab_test_stats import analyze_ab_test, print_report

def run_segmented_analysis(results_df: pd.DataFrame, segment_col: str):
    """
    Run the A/B test analysis separately for each value of segment_col.
    e.g. segment_col='is_new_user' gives results for new users and returning users separately.
    """
    segments = results_df[segment_col].unique()

    print(f"\n{'#'*50}")
    print(f"OVERALL RESULT")
    print(f"{'#'*50}")
    overall = analyze_ab_test(results_df)
    print_report(overall)

    for segment_value in segments:
        subset = results_df[results_df[segment_col] == segment_value]
        print(f"\n{'#'*50}")
        print(f"SEGMENT: {segment_col} = {segment_value}  (n={len(subset):,})")
        print(f"{'#'*50}")
        result = analyze_ab_test(subset)
        print_report(result)


if __name__ == "__main__":
    df = pd.read_csv("data/experiment_results.csv")
    run_segmented_analysis(df, segment_col="is_new_user")
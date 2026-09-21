import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def estimate_propensity_scores(df: pd.DataFrame, treatment_col: str = "received_email") -> pd.DataFrame:
    """
    Estimates each user's propensity score: P(received treatment | their features),
    using ONLY pre-treatment features (never the outcome).
    """
    df = df.copy()

    # user_value_segment is categorical text ("low"/"medium"/"high") — logistic
    # regression needs numbers, so we convert it to separate 0/1 columns
    # (one-hot encoding) rather than arbitrarily numbering the categories,
    # which would falsely imply "high" > "medium" > "low" in a mathematical sense.
    segment_dummies = pd.get_dummies(df["user_value_segment"], prefix="segment")

    features = pd.concat([
        segment_dummies,
        df[["pre_experiment_purchases", "pre_experiment_avg_session_minutes", "is_new_user"]],
    ], axis=1)

    X = features.values
    y = df[treatment_col].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # predict_proba returns [P(class=0), P(class=1)] per row — we want P(class=1),
    # i.e. the probability of actually being treated.
    df["estimated_propensity"] = model.predict_proba(X)[:, 1]

    return df, model


if __name__ == "__main__":
    df = pd.read_csv("data/confounded_results.csv")
    df_scored, model = estimate_propensity_scores(df)

    print("Estimated propensity score summary:")
    print(df_scored["estimated_propensity"].describe())

    # Since this is SIMULATED data, we happen to know the true propensity
    # we used to generate treatment — let's check how close our model got.
    correlation = df_scored["estimated_propensity"].corr(df_scored["true_propensity"])
    print(f"\nCorrelation between estimated and TRUE propensity: {correlation:.4f}")

    df_scored.to_csv("data/confounded_results_with_propensity.csv", index=False)
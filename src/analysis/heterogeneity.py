import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def check_effect_heterogeneity(df: pd.DataFrame, by: str, outcome_col: str = "converted",
                                 group_col: str = "group") -> dict:
    """
    Tests whether the treatment effect differs across the segments in column `by`.

    Fits  outcome ~ treated * segment  as a linear model on the 0/1 outcome (so every
    coefficient is a difference in conversion rates), with HC3 robust standard errors.
    Then runs ONE joint Wald test that all treated-by-segment interaction terms are
    zero. A small p-value means the effect really does differ between segments.
    """
    data = df[[group_col, outcome_col, by]].copy()
    data["treated"] = (data[group_col] == "treatment").astype(int)
    data["segment"] = data[by].astype(str)

    if data["segment"].nunique() < 2:
        raise ValueError("Need at least two segments to compare effects.")

    model = smf.ols(f"{outcome_col} ~ treated * C(segment)", data=data).fit(cov_type="HC3")

    # Interaction coefficients are the ones named like "treated:C(segment)[T.x]".
    names = list(model.params.index)
    interaction_positions = [i for i, name in enumerate(names) if name.startswith("treated:")]

    # One row per interaction term: "this coefficient equals zero".
    restrictions = np.zeros((len(interaction_positions), len(names)))
    for row, column in enumerate(interaction_positions):
        restrictions[row, column] = 1.0

    wald = model.wald_test(restrictions, scalar=True)

    return {
        "by": by,
        "n_segments": int(data["segment"].nunique()),
        "df": len(interaction_positions),
        "statistic": float(wald.statistic),
        "p_value": float(wald.pvalue),
    }

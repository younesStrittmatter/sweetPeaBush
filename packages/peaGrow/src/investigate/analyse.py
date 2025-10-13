import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import chi2


def analyze_category_effect(
        df: pd.DataFrame,
        category_col: str,  # <- your "interested_column"
        covariates: list[str] | None = None,  # <- columns to "factor out"
        subject_col: str = "subj_id",
        rt_col: str = "rt",
        acc_col: str = "correct",
        transform_rt: str | None = "log",  # 'log' or None
        drop_na: bool = True
):
    """
    Runs:
      (1) Accuracy ~ Category + covariates, GEE(Binomial), clustered by subject.
      (2) RT (correct-only) ~ Category + covariates, MixedLM with random intercept by subject.
    Returns a dict with fitted models and convenient summaries.
    Assumes df has columns: rt, correct, <category_col>, subj_id, and any covariates.
    """

    if covariates is None:
        covariates = []

    # Work on a copy; drop NAs for used cols
    use_cols = [category_col, subject_col, rt_col, acc_col] + covariates
    d = df[use_cols].copy()
    if drop_na:
        d = d.dropna(subset=use_cols)

    # Build RHS with auto C() wrapping for categorical-looking columns
    def term_for(col: str) -> str:
        return f"C({col})" if (d[col].dtype == "O" or str(d[col].dtype).startswith("category")) else col

    rhs_terms = [term_for(category_col)] + [term_for(c) for c in covariates]
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"

    results = {}

    # ---------- (1) Accuracy model: GEE Binomial with subject clustering ----------
    # Formula: correct ~ C(category) + covariates
    formula_acc = f"{acc_col} ~ {rhs}"
    gee = smf.gee(
        formula=formula_acc,
        groups=subject_col,
        data=d,
        family=Binomial(),
        cov_struct=Exchangeable()
    ).fit()
    results["accuracy_model"] = gee

    # Quick descriptive: observed accuracy per category level
    acc_by_cat = (
        d.groupby(category_col, observed=True)[acc_col]
        .mean()
        .rename("observed_accuracy")
        .reset_index()
    )
    results["accuracy_observed_by_category"] = acc_by_cat

    # ---------- (2) RT model on correct trials: MixedLM ----------
    d_rt = d.loc[d[acc_col] == 1].copy()
    if transform_rt == "log":
        d_rt = d_rt.loc[d_rt[rt_col] > 0].copy()
        d_rt["_rt"] = np.log(d_rt[rt_col])
        rt_response = "_rt"
        results["rt_transform"] = "log"
    else:
        rt_response = rt_col
        results["rt_transform"] = None

    # Mixed model formula
    formula_rt = f"{rt_response} ~ {rhs}"
    mdf_full = smf.mixedlm(
        formula=formula_rt,
        data=d_rt,
        groups=d_rt[subject_col],
        re_formula="1"  # random intercept per subject
    ).fit(reml=True)
    results["rt_model_full"] = mdf_full

    # Likelihood-ratio test for the *overall* effect of the category (plus any covariates you want kept)
    # Compare full model vs reduced model that drops the category term only
    rhs_reduced_terms = [t for t in rhs_terms if t != term_for(category_col)]
    rhs_reduced = " + ".join(rhs_reduced_terms) if rhs_reduced_terms else "1"
    formula_rt_reduced = f"{rt_response} ~ {rhs_reduced}"
    mdf_reduced = smf.mixedlm(
        formula=formula_rt_reduced,
        data=d_rt,
        groups=d_rt[subject_col],
        re_formula="1"
    ).fit(reml=True)

    # LRT
    lr_stat = 2 * (mdf_full.llf - mdf_reduced.llf)
    # df = number of fixed params removed (handle categorical expansions)
    df_diff = mdf_full.df_modelwc - mdf_reduced.df_modelwc
    p_lrt = chi2.sf(lr_stat, df_diff)
    results["rt_lrt"] = {"lr_stat": float(lr_stat), "df": int(df_diff), "p_value": float(p_lrt)}

    # Predicted marginal means (simple): average fitted RT per category (holding data distribution of covariates)
    d_rt["_fitted_full"] = mdf_full.fittedvalues
    emmeans_rt = (
        d_rt.groupby(category_col, observed=True)["_fitted_full"]
        .mean()
        .rename("predicted_rt")
        .reset_index()
    )
    results["rt_predicted_by_category"] = emmeans_rt

    return results

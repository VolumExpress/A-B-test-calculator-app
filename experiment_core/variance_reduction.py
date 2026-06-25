from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _effect_ols(y: pd.Series, treatment: pd.Series, covariates: pd.DataFrame | None = None) -> dict[str, float]:
    X = pd.DataFrame({"treatment": treatment.astype(float)})
    if covariates is not None and not covariates.empty:
        X = pd.concat([X.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y.astype(float).reset_index(drop=True), X).fit(cov_type="HC3")
    return {
        "effect": float(model.params["treatment"]),
        "se": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
        "ci_low": float(model.conf_int().loc["treatment", 0]),
        "ci_high": float(model.conf_int().loc["treatment", 1]),
        "variance": float(model.bse["treatment"] ** 2),
    }


def cuped_analysis(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    preperiod_cols: Iterable[str],
) -> dict[str, Any]:
    """Фактический CUPED/ANCOVA по загруженным данным с HC3 standard errors."""
    cols = [outcome_col, treatment_col, *list(preperiod_cols)]
    work = df[cols].dropna(subset=[outcome_col, treatment_col]).copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
    work = work.dropna(subset=[outcome_col, treatment_col])
    pre = work[list(preperiod_cols)].copy()
    pre = pd.get_dummies(pre, drop_first=True, dtype=float)
    pre = pre.fillna(pre.median(numeric_only=True)).fillna(0)
    pre = pre - pre.mean()

    raw = _effect_ols(work[outcome_col], work[treatment_col])
    adjusted = _effect_ols(work[outcome_col], work[treatment_col], pre)
    vr = 1 - adjusted["variance"] / raw["variance"] if raw["variance"] > 0 else np.nan
    return {
        "summary": pd.DataFrame([
            {"method": "Unadjusted", **raw},
            {"method": "CUPED/ANCOVA", **adjusted},
        ]),
        "variance_reduction": float(vr),
        "sample_size_multiplier": float(1 - vr) if np.isfinite(vr) else np.nan,
        "n": len(work),
        "warnings": [] if vr >= 0 else ["Корректировка увеличила дисперсию. Не используйте её без повторной проверки на A/A."],
    }


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def cupac_analysis(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    feature_cols: Iterable[str],
    metric_type: str = "continuous",
    folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    CUPAC: строит out-of-fold прогноз outcome только по pre-treatment признакам,
    затем использует прогноз как ковариату в анализе treatment effect.
    """
    cols = [outcome_col, treatment_col, *list(feature_cols)]
    work = df[cols].dropna(subset=[outcome_col, treatment_col]).copy().reset_index(drop=True)
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    t = pd.to_numeric(work[treatment_col], errors="coerce")
    ok = y.notna() & t.notna()
    work, y, t = work.loc[ok].reset_index(drop=True), y.loc[ok].reset_index(drop=True), t.loc[ok].reset_index(drop=True)
    X = work[list(feature_cols)]
    prep = _make_preprocessor(X)
    if metric_type == "binary":
        model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, C=0.5))])
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        pred = cross_val_predict(model, X, y.astype(int), cv=cv, method="predict_proba")[:, 1]
    else:
        model = Pipeline([("prep", prep), ("model", Ridge(alpha=10.0))])
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    raw = _effect_ols(y, t)
    adjusted = _effect_ols(y, t, pd.DataFrame({"cupac_prediction": pred - np.mean(pred)}))
    vr = 1 - adjusted["variance"] / raw["variance"] if raw["variance"] > 0 else np.nan
    return {
        "summary": pd.DataFrame([
            {"method": "Unadjusted", **raw},
            {"method": "CUPAC cross-fit", **adjusted},
        ]),
        "variance_reduction": float(vr),
        "sample_size_multiplier": float(1 - vr) if np.isfinite(vr) else np.nan,
        "prediction_correlation": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan,
        "predictions": pd.DataFrame({"outcome": y, "treatment": t, "cupac_prediction": pred}),
        "n": len(work),
        "warnings": [] if vr >= 0 else ["CUPAC не снизил дисперсию. Проверьте признаки, cross-fitting и стабильность на A/A."],
    }

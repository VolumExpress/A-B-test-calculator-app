from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def cluster_randomized_design(
    *,
    individual_sample_size: int,
    mean_cluster_size: float,
    icc: float,
    coefficient_of_variation: float = 0.0,
    attrition: float = 0.0,
) -> dict[str, float]:
    """Приближённое увеличение выборки для кластерной рандомизации."""
    if mean_cluster_size <= 1 or not 0 <= icc < 1:
        raise ValueError("Проверьте размер кластера и ICC.")
    # Eldridge correction for unequal cluster sizes.
    deff = 1 + ((1 + coefficient_of_variation**2) * mean_cluster_size - 1) * icc
    total = int(np.ceil(individual_sample_size * deff / max(1 - attrition, 1e-9)))
    clusters = int(np.ceil(total / mean_cluster_size))
    return {
        "design_effect": float(deff),
        "inflated_sample_size": total,
        "approximate_clusters": clusters,
        "clusters_per_arm_for_two_arm": int(np.ceil(clusters / 2)),
    }


def stepped_wedge_schedule(clusters: int, periods: int, seed: int = 42) -> pd.DataFrame:
    """Создаёт сбалансированный stepped-wedge график перехода кластеров в treatment."""
    if clusters < 2 or periods < 3:
        raise ValueError("Нужно минимум 2 кластера и 3 периода.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(clusters)
    steps = max(1, periods - 1)
    switch_period = {int(c): int(1 + (rank % steps)) for rank, c in enumerate(order)}
    rows = []
    for c in range(clusters):
        for p in range(periods):
            rows.append({"cluster": c, "period": p, "treatment": int(p >= switch_period[c]), "switch_period": switch_period[c]})
    return pd.DataFrame(rows)


def analyze_cluster_period_design(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    cluster_col: str,
    period_col: str,
    metric_type: str = "continuous",
    carryover_col: str | None = None,
) -> dict[str, Any]:
    """GEE-анализ stepped-wedge/switchback с period fixed effects и cluster correlation."""
    cols = [outcome_col, treatment_col, cluster_col, period_col] + ([carryover_col] if carryover_col else [])
    work = df[cols].dropna().copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
    work = work.dropna()
    period_dummies = pd.get_dummies(work[period_col].astype(str), prefix="period", drop_first=True, dtype=float)
    X = pd.concat([work[[treatment_col]].astype(float).reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1)
    if carryover_col:
        X[carryover_col] = pd.to_numeric(work[carryover_col], errors="coerce").fillna(0).to_numpy()
    X = sm.add_constant(X, has_constant="add")
    family = Binomial() if metric_type == "binary" else Gaussian()
    model = GEE(work[outcome_col].astype(float), X, groups=work[cluster_col], family=family, cov_struct=Exchangeable()).fit()
    ci = model.conf_int().loc[treatment_col]
    effect = float(np.exp(model.params[treatment_col])) if metric_type == "binary" else float(model.params[treatment_col])
    return {
        "summary": model.summary2().tables[1].reset_index().rename(columns={"index": "term"}),
        "effect_summary": pd.DataFrame([{
            "effect_scale": "odds_ratio" if metric_type == "binary" else "mean_difference",
            "effect": effect,
            "ci_low": float(np.exp(ci[0])) if metric_type == "binary" else float(ci[0]),
            "ci_high": float(np.exp(ci[1])) if metric_type == "binary" else float(ci[1]),
            "p_value": float(model.pvalues[treatment_col]),
            "clusters": int(work[cluster_col].nunique()),
            "periods": int(work[period_col].nunique()),
        }]),
        "warnings": ["Проверьте отсутствие anticipatory effects и достаточную длину washout для switchback."],
    }


def generate_switchback_schedule(
    clusters: int,
    periods: int,
    *,
    block_length: int = 1,
    washout_periods: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Создаёт чередующийся switchback schedule с рандомным стартом по кластерам."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(clusters):
        start = int(rng.integers(0, 2))
        previous = None
        washout_remaining = 0
        for p in range(periods):
            treatment = (start + (p // block_length)) % 2
            changed = previous is not None and treatment != previous
            if changed:
                washout_remaining = washout_periods
            washout = int(washout_remaining > 0)
            rows.append({"cluster": c, "period": p, "treatment": treatment, "is_washout": washout})
            washout_remaining = max(0, washout_remaining - 1)
            previous = treatment
    return pd.DataFrame(rows)


def synthetic_control(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
    intervention_time: Any,
    placebo: bool = True,
) -> dict[str, Any]:
    """Классический synthetic control с неотрицательными весами, сумма весов=1."""
    panel = df.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()
    if treated_unit not in panel.columns:
        raise ValueError("treated_unit отсутствует в данных.")
    donors = [c for c in panel.columns if c != treated_unit]
    pre = panel.index < intervention_time
    post = ~pre
    panel = panel.dropna(axis=0, how="any")
    pre = panel.index < intervention_time
    post = ~pre
    y = panel.loc[pre, treated_unit].to_numpy(float)
    X = panel.loc[pre, donors].to_numpy(float)

    def obj(w: np.ndarray) -> float:
        return float(np.mean((y - X @ w) ** 2))
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    res = optimize.minimize(obj, np.repeat(1/len(donors), len(donors)), bounds=[(0, 1)]*len(donors), constraints=cons)
    if not res.success:
        raise RuntimeError(f"Не удалось подобрать synthetic control: {res.message}")
    w = res.x
    synth = panel[donors].to_numpy() @ w
    curve = pd.DataFrame({
        "time": panel.index,
        "treated": panel[treated_unit].to_numpy(float),
        "synthetic": synth,
    })
    curve["gap"] = curve["treated"] - curve["synthetic"]
    pre_rmse = float(np.sqrt(np.mean(curve.loc[pre, "gap"]**2)))
    post_effect = float(curve.loc[post, "gap"].mean())

    placebo_df = pd.DataFrame()
    if placebo and len(donors) >= 3:
        placebo_rows = []
        for pseudo in donors:
            pseudo_donors = [d for d in panel.columns if d != pseudo]
            yp = panel.loc[pre, pseudo].to_numpy(float)
            Xp = panel.loc[pre, pseudo_donors].to_numpy(float)
            rp = optimize.minimize(
                lambda ww: float(np.mean((yp - Xp @ ww)**2)),
                np.repeat(1/len(pseudo_donors), len(pseudo_donors)),
                bounds=[(0, 1)]*len(pseudo_donors),
                constraints={"type": "eq", "fun": lambda ww: np.sum(ww)-1},
            )
            if rp.success:
                gp = panel[pseudo].to_numpy(float) - panel[pseudo_donors].to_numpy(float) @ rp.x
                pre_p = float(np.sqrt(np.mean(gp[pre]**2)))
                post_p = float(np.mean(gp[post]))
                placebo_rows.append({"unit": pseudo, "pre_rmse": pre_p, "post_effect": post_p, "effect_to_pre_rmse": post_p/max(pre_p,1e-12)})
        placebo_df = pd.DataFrame(placebo_rows)

    return {
        "weights": pd.DataFrame({"donor": donors, "weight": w}).sort_values("weight", ascending=False),
        "curve": curve,
        "summary": pd.DataFrame([{
            "pre_rmse": pre_rmse,
            "average_post_effect": post_effect,
            "post_periods": int(post.sum()),
        }]),
        "placebos": placebo_df,
        "warnings": ["Synthetic control требует стабильной прединтервенционной связи и отсутствия других одновременных шоков."],
    }


def regression_discontinuity(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    running_col: str,
    cutoff: float,
    bandwidth: float,
    treatment_col: str | None = None,
    polynomial_order: int = 1,
) -> dict[str, Any]:
    """Sharp/fuzzy RDD через локальную полиномиальную регрессию с triangular kernel."""
    cols = [outcome_col, running_col] + ([treatment_col] if treatment_col else [])
    work = df[cols].dropna().copy()
    work["running_centered"] = pd.to_numeric(work[running_col], errors="coerce") - cutoff
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna()
    work = work[work["running_centered"].abs() <= bandwidth]
    work["above"] = (work["running_centered"] >= 0).astype(int)
    weights = 1 - work["running_centered"].abs() / bandwidth

    X = pd.DataFrame({"above": work["above"], "running": work["running_centered"]})
    X["above_x_running"] = X["above"] * X["running"]
    if polynomial_order >= 2:
        X["running2"] = X["running"]**2
        X["above_x_running2"] = X["above"] * X["running2"]
    X = sm.add_constant(X)
    outcome_model = sm.WLS(work[outcome_col], X, weights=weights).fit(cov_type="HC1")
    reduced_form = float(outcome_model.params["above"])
    effect = reduced_form
    first_stage = np.nan
    if treatment_col:
        work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
        stage = sm.WLS(work[treatment_col], X, weights=weights).fit(cov_type="HC1")
        first_stage = float(stage.params["above"])
        effect = reduced_form / first_stage if abs(first_stage) > 1e-8 else np.nan

    return {
        "summary": pd.DataFrame([{
            "design": "fuzzy" if treatment_col else "sharp",
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "n_in_bandwidth": len(work),
            "reduced_form_jump": reduced_form,
            "first_stage_jump": first_stage,
            "local_treatment_effect": effect,
            "outcome_jump_p_value": float(outcome_model.pvalues["above"]),
        }]),
        "model_summary": outcome_model.summary2().tables[1].reset_index().rename(columns={"index":"term"}),
        "plot_data": work[[running_col, outcome_col, "above", "running_centered"]],
        "warnings": ["RDD валиден только при отсутствии точного манипулирования running variable около cutoff и при непрерывности остальных факторов."],
    }


def dose_response(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    dose_col: str,
    covariate_cols: Iterable[str] = (),
    grid_points: int = 40,
    bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Гибкая dose-response кривая через spline outcome regression и bootstrap.

    Это observational adjustment, а не автоматическое доказательство причинности.
    """
    cols = [outcome_col, dose_col, *list(covariate_cols)]
    work = df[cols].dropna().copy()
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    dose = pd.to_numeric(work[dose_col], errors="coerce")
    ok = y.notna() & dose.notna()
    work, y, dose = work.loc[ok].reset_index(drop=True), y.loc[ok].to_numpy(), dose.loc[ok].to_numpy()
    covars = pd.get_dummies(work[list(covariate_cols)], drop_first=True, dtype=float).fillna(0) if covariate_cols else pd.DataFrame(index=work.index)
    grid = np.linspace(np.quantile(dose, 0.02), np.quantile(dose, 0.98), grid_points)

    def fit_predict(idx: np.ndarray) -> np.ndarray:
        d = dose[idx].reshape(-1,1)
        X_cov = covars.iloc[idx].to_numpy() if not covars.empty else np.empty((len(idx),0))
        spline = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        D = spline.fit_transform(d)
        X = np.column_stack([D, X_cov])
        model = LinearRegression().fit(X, y[idx])
        preds = []
        for g in grid:
            Dg = spline.transform(np.full((len(work),1), g))
            Xg = np.column_stack([Dg, covars.to_numpy() if not covars.empty else np.empty((len(work),0))])
            preds.append(model.predict(Xg).mean())
        return np.asarray(preds)

    base = fit_predict(np.arange(len(work)))
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap, grid_points))
    for b in range(bootstrap):
        boot[b] = fit_predict(rng.integers(0, len(work), len(work)))
    curve = pd.DataFrame({
        "dose": grid,
        "expected_outcome": base,
        "ci_low": np.quantile(boot, 0.025, axis=0),
        "ci_high": np.quantile(boot, 0.975, axis=0),
    })
    return {
        "curve": curve,
        "summary": pd.DataFrame([{
            "n": len(work),
            "dose_min_modeled": grid.min(),
            "dose_max_modeled": grid.max(),
            "best_observed_grid_dose": float(grid[np.argmax(base)]),
        }]),
        "warnings": ["Не экстраполируйте за область фактического overlap; скрытые confounders могут искажать dose-response."],
    }

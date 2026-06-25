from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Poisson, NegativeBinomial
from statsmodels.genmod.generalized_estimating_equations import GEE

from lifelines import AalenJohansenFitter, CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time


def _clean_survival(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> pd.DataFrame:
    work = df[[duration_col, event_col, group_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    work = work.dropna()
    work = work[(work[duration_col] >= 0) & work[event_col].isin([0, 1])]
    work[group_col] = work[group_col].astype(str)
    return work


def _rmst_bootstrap(
    work: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    control_label: str,
    treatment_label: str,
    tau: float,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    groups = {g: work[work[group_col] == g] for g in [control_label, treatment_label]}
    for _ in range(bootstrap):
        vals = {}
        for g, frame in groups.items():
            idx = rng.integers(0, len(frame), len(frame))
            sample = frame.iloc[idx]
            km = KaplanMeierFitter().fit(sample[duration_col], sample[event_col])
            vals[g] = restricted_mean_survival_time(km, t=tau)
        diffs.append(vals[treatment_label] - vals[control_label])
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def analyze_survival(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    group_col: str,
    control_label: str,
    treatment_label: str,
    tau: float | None = None,
    milestone: float | None = None,
    bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Survival-анализ с KM, RMST, milestone, Cox PH и тестами на non-PH.

    RMST - среднее время без события до горизонта tau. Оно остаётся понятным,
    даже если hazard ratio меняется во времени или кривые пересекаются.
    """
    work = _clean_survival(df, duration_col, event_col, group_col)
    work = work[work[group_col].isin([str(control_label), str(treatment_label)])]
    if work[group_col].nunique() != 2:
        raise ValueError("Нужны обе выбранные группы.")
    if tau is None:
        tau = float(np.quantile(work[duration_col], 0.90))
    milestone = milestone or tau

    curve_rows = []
    summary_rows = []
    km_models: dict[str, KaplanMeierFitter] = {}
    for g in [str(control_label), str(treatment_label)]:
        part = work[work[group_col] == g]
        km = KaplanMeierFitter(label=g).fit(part[duration_col], part[event_col])
        km_models[g] = km
        sf = km.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        sf["group"] = g
        curve_rows.append(sf)
        summary_rows.append({
            "group": g,
            "n": len(part),
            "events": int(part[event_col].sum()),
            "median_survival": float(km.median_survival_time_),
            "rmst_tau": float(restricted_mean_survival_time(km, t=tau)),
            "survival_at_milestone": float(km.predict(milestone)),
        })

    c = work[work[group_col] == str(control_label)]
    t = work[work[group_col] == str(treatment_label)]
    lr = logrank_test(t[duration_col], c[duration_col], event_observed_A=t[event_col], event_observed_B=c[event_col])
    weighted_rows = [{"test": "log-rank", "statistic": float(lr.test_statistic), "p_value": float(lr.p_value)}]
    for name, kwargs in [
        ("Wilcoxon / early-weighted", {"weightings": "wilcoxon"}),
        ("Fleming-Harrington late", {"weightings": "fleming-harrington", "p": 0, "q": 1}),
        ("Fleming-Harrington early", {"weightings": "fleming-harrington", "p": 1, "q": 0}),
    ]:
        test = logrank_test(
            t[duration_col], c[duration_col],
            event_observed_A=t[event_col], event_observed_B=c[event_col], **kwargs
        )
        weighted_rows.append({"test": name, "statistic": float(test.test_statistic), "p_value": float(test.p_value)})

    cox_data = work[[duration_col, event_col, group_col]].copy()
    cox_data["treatment"] = (cox_data[group_col] == str(treatment_label)).astype(int)
    cph = CoxPHFitter().fit(cox_data[[duration_col, event_col, "treatment"]], duration_col=duration_col, event_col=event_col)
    cox_summary = cph.summary.reset_index().rename(columns={"covariate": "term"})
    ph = proportional_hazard_test(cph, cox_data[[duration_col, event_col, "treatment"]], time_transform="rank")
    ph_p = float(ph.summary.loc["treatment", "p"])

    rmst_diff, rmst_low, rmst_high = _rmst_bootstrap(
        work, duration_col, event_col, group_col, str(control_label), str(treatment_label), tau, bootstrap, seed
    )
    milestone_diff = float(km_models[str(treatment_label)].predict(milestone) - km_models[str(control_label)].predict(milestone))
    warnings = []
    if ph_p < 0.05:
        warnings.append(
            "Тест proportional hazards указывает на меняющийся во времени эффект. "
            "Не делайте единый HR главным бизнес-выводом; используйте RMST, milestone и временные графики."
        )

    return {
        "curves": pd.concat(curve_rows, ignore_index=True),
        "group_summary": pd.DataFrame(summary_rows),
        "weighted_tests": pd.DataFrame(weighted_rows),
        "cox_summary": cox_summary,
        "ph_test": pd.DataFrame([{"term": "treatment", "p_value": ph_p, "ph_assumption_warning": ph_p < 0.05}]),
        "effect_summary": pd.DataFrame([{
            "tau": tau,
            "rmst_difference_treatment_minus_control": rmst_diff,
            "rmst_ci_low": rmst_low,
            "rmst_ci_high": rmst_high,
            "milestone": milestone,
            "survival_difference_at_milestone": milestone_diff,
            "cox_hazard_ratio": float(np.exp(cph.params_["treatment"])),
        }]),
        "warnings": warnings,
    }


def analyze_competing_risks(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_type_col: str,
    group_col: str,
    control_label: str,
    treatment_label: str,
    event_of_interest: int = 1,
    tau: float | None = None,
    bootstrap: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Competing-risks анализ через Aalen-Johansen CIF и cause-specific Cox.

    event_type=0 означает цензурирование; 1,2,... - взаимоисключающие события.
    """
    work = df[[duration_col, event_type_col, group_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_type_col] = pd.to_numeric(work[event_type_col], errors="coerce")
    work = work.dropna()
    work = work[(work[duration_col] >= 0) & (work[event_type_col] >= 0)]
    work[group_col] = work[group_col].astype(str)
    work = work[work[group_col].isin([str(control_label), str(treatment_label)])]
    tau = tau or float(np.quantile(work[duration_col], 0.90))

    curves, summaries = [], []
    for g in [str(control_label), str(treatment_label)]:
        part = work[work[group_col] == g]
        aj = AalenJohansenFitter().fit(part[duration_col], part[event_type_col], event_of_interest=event_of_interest)
        cif = aj.cumulative_density_.reset_index()
        cif.columns = ["time", "cif"]
        cif["group"] = g
        curves.append(cif)
        summaries.append({
            "group": g,
            "n": len(part),
            "event_of_interest": int((part[event_type_col] == event_of_interest).sum()),
            "competing_events": int(((part[event_type_col] != 0) & (part[event_type_col] != event_of_interest)).sum()),
            "cif_at_tau": float(aj.predict(tau)),
        })

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(bootstrap):
        vals = {}
        for g in [str(control_label), str(treatment_label)]:
            part = work[work[group_col] == g]
            sample = part.iloc[rng.integers(0, len(part), len(part))]
            aj = AalenJohansenFitter().fit(sample[duration_col], sample[event_type_col], event_of_interest=event_of_interest)
            vals[g] = float(aj.predict(tau))
        diffs.append(vals[str(treatment_label)] - vals[str(control_label)])

    cs = work.copy()
    cs["event_interest"] = (cs[event_type_col] == event_of_interest).astype(int)
    cs["treatment"] = (cs[group_col] == str(treatment_label)).astype(int)
    cph = CoxPHFitter().fit(cs[[duration_col, "event_interest", "treatment"]], duration_col=duration_col, event_col="event_interest")
    return {
        "cif_curves": pd.concat(curves, ignore_index=True),
        "group_summary": pd.DataFrame(summaries),
        "effect_summary": pd.DataFrame([{
            "tau": tau,
            "cif_difference_treatment_minus_control": float(np.mean(diffs)),
            "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)),
            "cause_specific_hazard_ratio": float(np.exp(cph.params_["treatment"])),
        }]),
        "cause_specific_cox": cph.summary.reset_index(),
        "warnings": [
            "Cause-specific HR и CIF отвечают на разные вопросы. Для бизнес-решения обычно показывайте CIF на фиксированном горизонте."
        ],
    }


def analyze_recurrent_events(
    df: pd.DataFrame,
    *,
    id_col: str,
    start_col: str,
    stop_col: str,
    event_col: str,
    treatment_col: str,
    model: str = "andersen-gill",
) -> dict[str, Any]:
    """Анализ повторных событий: Andersen-Gill или GEE rate ratio."""
    cols = [id_col, start_col, stop_col, event_col, treatment_col]
    work = df[cols].dropna().copy()
    for col in [start_col, stop_col, event_col, treatment_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    work = work[work[stop_col] > work[start_col]]
    work["exposure"] = work[stop_col] - work[start_col]

    if model == "andersen-gill":
        # CoxPHFitter с entry_col реализует counting-process представление.
        # cluster_col=id и robust=True дают sandwich SE с учётом повторных строк субъекта.
        ag = CoxPHFitter().fit(
            work[[id_col, start_col, stop_col, event_col, treatment_col]],
            duration_col=stop_col,
            event_col=event_col,
            entry_col=start_col,
            cluster_col=id_col,
            robust=True,
        )
        ci = ag.confidence_intervals_.loc[treatment_col]
        return {
            "summary": ag.summary.reset_index(),
            "effect_summary": pd.DataFrame([{
                "method": "Andersen-Gill",
                "hazard_ratio": float(np.exp(ag.params_[treatment_col])),
                "ci_low": float(np.exp(ci.iloc[0])),
                "ci_high": float(np.exp(ci.iloc[1])),
                "p_value": float(ag.summary.loc[treatment_col, "p"]),
            }]),
            "warnings": ["Andersen-Gill предполагает общий multiplicative effect и требует корректного start-stop формата."],
        }

    # GEE Poisson / Negative Binomial rate model with subject clustering.
    family = NegativeBinomial() if model == "negative-binomial" else Poisson()
    exog = sm.add_constant(work[[treatment_col]].astype(float))
    gee = GEE(
        endog=work[event_col].astype(float),
        exog=exog,
        groups=work[id_col],
        offset=np.log(np.maximum(work["exposure"], 1e-12)),
        family=family,
        cov_struct=Exchangeable(),
    ).fit()
    return {
        "summary": gee.summary2().tables[1].reset_index().rename(columns={"index": "term"}),
        "effect_summary": pd.DataFrame([{
            "method": f"GEE {model}",
            "rate_ratio": float(np.exp(gee.params[treatment_col])),
            "ci_low": float(np.exp(gee.conf_int().loc[treatment_col, 0])),
            "ci_high": float(np.exp(gee.conf_int().loc[treatment_col, 1])),
            "p_value": float(gee.pvalues[treatment_col]),
        }]),
        "warnings": [],
    }

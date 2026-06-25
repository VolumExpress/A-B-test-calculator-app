from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


def validate_dataset(
    df: pd.DataFrame,
    *,
    group_col: str,
    outcome_col: str,
    id_col: str | None = None,
    metric_type: str = "binary",
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    for col in [group_col, outcome_col]:
        if col not in df.columns:
            errors.append(f"Не найдена колонка '{col}'.")
    if errors:
        return ValidationResult(False, errors, warnings, {"rows": len(df)})

    if df[group_col].isna().any():
        errors.append("Есть строки без экспериментальной группы.")
    if df[outcome_col].isna().any():
        warnings.append("Есть пропуски в целевой метрике. Они будут исключены из анализа.")

    groups = df[group_col].dropna().astype(str).unique()
    if len(groups) < 2:
        errors.append("Для анализа нужны как минимум две группы.")

    if id_col and id_col in df.columns:
        duplicate_share = float(df[id_col].duplicated().mean())
        if duplicate_share > 0:
            warnings.append(
                f"Дубликаты идентификатора: {duplicate_share:.1%}. "
                "Проверьте, является ли строка клиентом или отдельным событием."
            )
    else:
        duplicate_share = float("nan")

    if metric_type == "binary":
        values = set(pd.to_numeric(df[outcome_col], errors="coerce").dropna().unique())
        if not values.issubset({0, 1}):
            errors.append("Для бинарной метрики значения должны быть 0 и 1.")
    elif metric_type == "continuous":
        numeric_share = pd.to_numeric(df[outcome_col], errors="coerce").notna().mean()
        if numeric_share < 0.99:
            errors.append("Непрерывная метрика должна быть числовой.")

    counts = df[group_col].value_counts(dropna=False)
    summary = {
        "rows": len(df),
        "groups": len(groups),
        "group_counts": counts.to_dict(),
        "missing_outcome": int(df[outcome_col].isna().sum()),
        "duplicate_share": duplicate_share,
    }
    return ValidationResult(not errors, errors, warnings, summary)


def _srm_test(counts: pd.Series, expected_shares: dict[str, float] | None = None) -> dict[str, float]:
    counts = counts.astype(float)
    labels = counts.index.astype(str)
    total = counts.sum()
    if expected_shares:
        shares = np.array([expected_shares.get(label, np.nan) for label in labels], dtype=float)
        if np.isnan(shares).any() or not np.isclose(shares.sum(), 1.0):
            shares = np.repeat(1 / len(counts), len(counts))
    else:
        shares = np.repeat(1 / len(counts), len(counts))
    expected = total * shares
    chi2, p = stats.chisquare(counts.to_numpy(), expected)
    return {"srm_chi2": float(chi2), "srm_p_value": float(p)}


def _binary_two_group(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str,
    treatment_label: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    work = df[[group_col, outcome_col]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    work[outcome_col] = pd.to_numeric(work[outcome_col])

    c = work[work[group_col] == str(control_label)][outcome_col]
    t = work[work[group_col] == str(treatment_label)][outcome_col]
    if len(c) == 0 or len(t) == 0:
        raise ValueError("Не найдены выбранные control/treatment группы.")

    x_c, n_c = int(c.sum()), len(c)
    x_t, n_t = int(t.sum()), len(t)
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    pooled = (x_c + x_t) / (n_c + n_t)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n_c + 1 / n_t), 1e-15))
    z = diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = math.sqrt(max(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t, 1e-15))
    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - zcrit * se_unpooled, diff + zcrit * se_unpooled

    table = np.array([[x_t, n_t - x_t], [x_c, n_c - x_c]])
    fisher_p = float(stats.fisher_exact(table, alternative="two-sided").pvalue)

    warnings: list[str] = []
    min_expected_events = min(n_c * p_c, n_t * p_t)
    if min_expected_events < 5:
        warnings.append(
            "Ожидаемое число событий в одной из групп меньше 5. "
            "Основным ориентиром используйте Fisher exact, а не нормальную аппроксимацию."
        )

    rr = p_t / p_c if p_c > 0 else float("nan")
    summary = pd.DataFrame(
        [
            {"group": control_label, "n": n_c, "events": x_c, "metric": p_c},
            {"group": treatment_label, "n": n_t, "events": x_t, "metric": p_t},
        ]
    )
    results = pd.DataFrame(
        [
            {
                "comparison": f"{treatment_label} vs {control_label}",
                "effect_absolute": diff,
                "effect_relative": rr - 1 if np.isfinite(rr) else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "z_stat": z,
                "p_value": p_value,
                "fisher_p_value": fisher_p,
                "significant": p_value < alpha,
            }
        ]
    )
    return summary, results, warnings


def _continuous_two_group(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str,
    treatment_label: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    work = df[[group_col, outcome_col]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    work[outcome_col] = pd.to_numeric(work[outcome_col])

    c = work[work[group_col] == str(control_label)][outcome_col].to_numpy(float)
    t = work[work[group_col] == str(treatment_label)][outcome_col].to_numpy(float)
    if len(c) < 2 or len(t) < 2:
        raise ValueError("В каждой группе требуется минимум два наблюдения.")

    test = stats.ttest_ind(t, c, equal_var=False)
    diff = float(np.mean(t) - np.mean(c))
    se = math.sqrt(np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c))
    df_num = (np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c)) ** 2
    df_den = (
        (np.var(t, ddof=1) / len(t)) ** 2 / (len(t) - 1)
        + (np.var(c, ddof=1) / len(c)) ** 2 / (len(c) - 1)
    )
    dof = df_num / df_den
    crit = stats.t.ppf(1 - alpha / 2, dof)

    pooled_std = math.sqrt(
        ((len(t) - 1) * np.var(t, ddof=1) + (len(c) - 1) * np.var(c, ddof=1))
        / (len(t) + len(c) - 2)
    )
    d = diff / pooled_std if pooled_std > 0 else np.nan

    summary = pd.DataFrame(
        [
            {"group": control_label, "n": len(c), "metric": np.mean(c), "std": np.std(c, ddof=1)},
            {"group": treatment_label, "n": len(t), "metric": np.mean(t), "std": np.std(t, ddof=1)},
        ]
    )
    results = pd.DataFrame(
        [
            {
                "comparison": f"{treatment_label} vs {control_label}",
                "effect_absolute": diff,
                "effect_relative": diff / np.mean(c) if np.mean(c) != 0 else np.nan,
                "ci_low": diff - crit * se,
                "ci_high": diff + crit * se,
                "t_stat": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohens_d": d,
                "significant": float(test.pvalue) < alpha,
            }
        ]
    )
    return summary, results, []


def _multiarm(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str,
    metric_type: str,
    alpha: float,
    correction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    work = df[[group_col, outcome_col]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    groups = [g for g in work[group_col].unique() if g != str(control_label)]
    rows: list[dict[str, Any]] = []
    summaries: list[pd.DataFrame] = []
    warnings: list[str] = []

    for arm in groups:
        if metric_type == "binary":
            s, r, w = _binary_two_group(work, group_col, outcome_col, str(control_label), arm, alpha)
        else:
            s, r, w = _continuous_two_group(work, group_col, outcome_col, str(control_label), arm, alpha)
        r = r.iloc[0].to_dict()
        r["arm"] = arm
        rows.append(r)
        summaries.append(s[s["group"] == arm])
        warnings.extend(w)

    if not rows:
        raise ValueError("Кроме контроля нужна хотя бы одна treatment-ветка.")
    result = pd.DataFrame(rows)
    method = correction if correction in {"holm", "bonferroni", "fdr_bh"} else "holm"
    reject, p_adj, _, _ = multipletests(result["p_value"], alpha=alpha, method=method)
    result["p_value_adjusted"] = p_adj
    result["significant_adjusted"] = reject

    control = work[work[group_col] == str(control_label)]
    if metric_type == "binary":
        control_row = pd.DataFrame([{
            "group": str(control_label), "n": len(control),
            "events": int(pd.to_numeric(control[outcome_col]).sum()),
            "metric": float(pd.to_numeric(control[outcome_col]).mean()),
        }])
    else:
        control_row = pd.DataFrame([{
            "group": str(control_label), "n": len(control),
            "metric": float(pd.to_numeric(control[outcome_col]).mean()),
            "std": float(pd.to_numeric(control[outcome_col]).std(ddof=1)),
        }])
    summary = pd.concat([control_row, *summaries], ignore_index=True)
    return summary, result, sorted(set(warnings))


def analyze_experiment(
    df: pd.DataFrame,
    *,
    group_col: str,
    outcome_col: str,
    control_label: str,
    metric_type: str = "binary",
    alpha: float = 0.05,
    correction: str = "holm",
    expected_shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    work = df[[group_col, outcome_col]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    groups = work[group_col].unique()
    counts = work[group_col].value_counts().sort_index()
    srm = _srm_test(counts, expected_shares)

    if len(groups) == 2:
        treatment_label = next(g for g in groups if g != str(control_label))
        if metric_type == "binary":
            summary, results, warnings = _binary_two_group(
                work, group_col, outcome_col, str(control_label), treatment_label, alpha
            )
        else:
            summary, results, warnings = _continuous_two_group(
                work, group_col, outcome_col, str(control_label), treatment_label, alpha
            )
    else:
        summary, results, warnings = _multiarm(
            work, group_col, outcome_col, str(control_label), metric_type, alpha, correction
        )

    if srm["srm_p_value"] < 0.01:
        warnings.append(
            "Обнаружен Sample Ratio Mismatch: размеры групп существенно отличаются от ожидаемых. "
            "До интерпретации эффекта проверьте рандомизацию и загрузку данных."
        )

    return {
        "group_summary": summary,
        "results": results,
        "quality": pd.DataFrame([
            {"check": "rows_analyzed", "value": len(work)},
            {"check": "groups", "value": len(groups)},
            {"check": "srm_chi2", "value": srm["srm_chi2"]},
            {"check": "srm_p_value", "value": srm["srm_p_value"]},
        ]),
        "warnings": warnings,
    }


def analyze_uplift(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    score_col: str,
    bins: int = 10,
) -> dict[str, pd.DataFrame | float | list[str]]:
    work = df[[treatment_col, outcome_col, score_col]].dropna().copy()
    work[treatment_col] = pd.to_numeric(work[treatment_col])
    work[outcome_col] = pd.to_numeric(work[outcome_col])
    work[score_col] = pd.to_numeric(work[score_col])
    if not set(work[treatment_col].unique()).issubset({0, 1}):
        raise ValueError("Для uplift MVP treatment должен быть бинарным: 0/1.")

    work = work.sort_values(score_col, ascending=False).reset_index(drop=True)
    effective_bins = min(bins, max(2, len(work) // 100))
    work["uplift_bin"] = pd.qcut(
        work.index, q=effective_bins, labels=False, duplicates="drop"
    ) + 1

    rows = []
    warnings: list[str] = []
    for bin_id, part in work.groupby("uplift_bin", sort=True):
        t = part[part[treatment_col] == 1][outcome_col]
        c = part[part[treatment_col] == 0][outcome_col]
        if len(t) == 0 or len(c) == 0:
            warnings.append(f"В uplift-группе {bin_id} нет treatment или control.")
            continue
        rows.append({
            "uplift_bin": int(bin_id),
            "n": len(part),
            "n_treatment": len(t),
            "n_control": len(c),
            "predicted_uplift_mean": float(part[score_col].mean()),
            "treatment_rate": float(t.mean()),
            "control_rate": float(c.mean()),
            "observed_uplift": float(t.mean() - c.mean()),
        })

    calibration = pd.DataFrame(rows)
    if calibration.empty:
        raise ValueError("Не удалось построить uplift-калибровку.")
    calibration["cumulative_n"] = calibration["n"].cumsum()
    calibration["incremental_events"] = calibration["observed_uplift"] * calibration["n"]
    calibration["cumulative_incremental_events"] = calibration["incremental_events"].cumsum()

    overall_t = work[work[treatment_col] == 1][outcome_col]
    overall_c = work[work[treatment_col] == 0][outcome_col]
    overall_uplift = float(overall_t.mean() - overall_c.mean())
    return {
        "calibration": calibration,
        "overall_uplift": overall_uplift,
        "warnings": warnings,
    }

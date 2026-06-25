from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def _corr_from_info(info: np.ndarray) -> np.ndarray:
    k = len(info)
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            out[i, j] = math.sqrt(min(info[i], info[j]) / max(info[i], info[j]))
    return out


def calibrate_gaussian_group_sequential(
    information_fractions: Iterable[float],
    *, alpha: float = 0.05,
    family: str = "obf",
    sided: str = "one-sided",
    simulations: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Симуляционно калибрует O'Brien-Fleming или Pocock границы."""
    info = np.asarray(list(information_fractions), dtype=float)
    if np.any(info <= 0) or np.any(info > 1) or np.any(np.diff(info) <= 0):
        raise ValueError("Доли информации должны возрастать и находиться в (0, 1].")
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(info)), _corr_from_info(info), simulations)

    def boundaries(c: float) -> np.ndarray:
        if family.lower() == "obf":
            return c / np.sqrt(info)
        if family.lower() == "pocock":
            return np.full(len(info), c)
        raise ValueError("family: obf или pocock")

    def cross_prob(c: float) -> float:
        b = boundaries(c)
        return float(np.mean(np.any(z >= b, axis=1))) if sided == "one-sided" else float(np.mean(np.any(np.abs(z) >= b, axis=1)))

    lo, hi = 0.2, 8.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if cross_prob(mid) > alpha:
            lo = mid
        else:
            hi = mid
    b = boundaries((lo + hi) / 2)
    return pd.DataFrame({
        "look": np.arange(1, len(info) + 1),
        "information_fraction": info,
        "z_boundary": b,
        "nominal_one_sided_p": 1 - stats.norm.cdf(b),
    })


def z_stat_two_proportions(x_c: int, n_c: int, x_t: int, n_t: int, benefit_direction: str = "increase") -> float:
    pc, pt = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n_c + 1 / n_t), 1e-15))
    return (pt - pc) / se if benefit_direction == "increase" else (pc - pt) / se


def sequential_monitoring_table(
    cumulative: pd.DataFrame,
    *,
    x_control_col: str,
    n_control_col: str,
    x_treatment_col: str,
    n_treatment_col: str,
    boundaries: pd.DataFrame,
    benefit_direction: str = "increase",
    assumed_effect_z_final: float | None = None,
    futility_threshold: float = 0.10,
) -> pd.DataFrame:
    """Применяет заранее откалиброванные границы к накопленным итогам пилота."""
    if len(cumulative) != len(boundaries):
        raise ValueError("Число строк cumulative должно совпадать с числом interim-анализов.")
    rows = []
    final_boundary = float(boundaries.iloc[-1]["z_boundary"])
    for i, (_, r) in enumerate(cumulative.reset_index(drop=True).iterrows()):
        z = z_stat_two_proportions(
            int(r[x_control_col]), int(r[n_control_col]), int(r[x_treatment_col]), int(r[n_treatment_col]), benefit_direction
        )
        info = float(boundaries.iloc[i]["information_fraction"])
        b = float(boundaries.iloc[i]["z_boundary"])
        efficacy = z >= b
        cp = np.nan
        futility = False
        if assumed_effect_z_final is not None and info < 1:
            mean_final = math.sqrt(info) * z + assumed_effect_z_final * (1 - info)
            sd_final = math.sqrt(1 - info)
            cp = float(1 - stats.norm.cdf((final_boundary - mean_final) / sd_final))
            futility = cp < futility_threshold
        rows.append({
            "look": i + 1,
            "information_fraction": info,
            "z_stat": z,
            "efficacy_boundary": b,
            "stop_for_efficacy": efficacy,
            "conditional_power": cp,
            "stop_for_futility_nonbinding": futility,
        })
    return pd.DataFrame(rows)


def _exact_pvalue(table: np.ndarray, method: str, alternative: str) -> float:
    method = method.lower()
    if method == "fisher":
        return float(stats.fisher_exact(table, alternative=alternative).pvalue)
    if method == "boschloo":
        return float(stats.boschloo_exact(table, alternative=alternative).pvalue)
    if method == "barnard":
        return float(stats.barnard_exact(table, alternative=alternative).pvalue)
    raise ValueError("method: fisher, boschloo или barnard")


def calibrate_exact_sequential(
    n_control_looks: Iterable[int],
    n_treatment_looks: Iterable[int],
    *,
    p_null: float,
    alpha: float = 0.05,
    method: str = "fisher",
    benefit_direction: str = "increase",
    simulations: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Калибрует единую границу exact p-value для многократных просмотров.

    Симулируется весь путь накопления данных под H0. Это важнее, чем точность
    одного Fisher-теста: повторное применение p<0.05 без калибровки ломает alpha.
    """
    nc = np.asarray(list(n_control_looks), dtype=int)
    nt = np.asarray(list(n_treatment_looks), dtype=int)
    if len(nc) != len(nt) or len(nc) == 0 or np.any(np.diff(nc) <= 0) or np.any(np.diff(nt) <= 0):
        raise ValueError("Размеры групп по interim должны возрастать и иметь одинаковую длину.")
    rng = np.random.default_rng(seed)
    min_p = np.ones(simulations)
    prev_nc = prev_nt = 0
    xc = np.zeros(simulations, dtype=int)
    xt = np.zeros(simulations, dtype=int)
    alternative = "greater" if benefit_direction == "increase" else "less"
    for cur_nc, cur_nt in zip(nc, nt):
        xc += rng.binomial(cur_nc - prev_nc, p_null, simulations)
        xt += rng.binomial(cur_nt - prev_nt, p_null, simulations)
        pvals = np.empty(simulations)
        for i in range(simulations):
            # Строки: treatment, control; столбцы: event, no event.
            table = np.array([[xt[i], cur_nt - xt[i]], [xc[i], cur_nc - xc[i]]])
            pvals[i] = _exact_pvalue(table, method, alternative)
        min_p = np.minimum(min_p, pvals)
        prev_nc, prev_nt = cur_nc, cur_nt
    threshold = float(np.quantile(min_p, alpha, method="higher"))
    actual_alpha = float(np.mean(min_p <= threshold))
    return {
        "p_threshold": threshold,
        "estimated_type_i_error": actual_alpha,
        "simulations": simulations,
        "method": method,
        "looks": pd.DataFrame({"look": np.arange(1, len(nc)+1), "n_control": nc, "n_treatment": nt}),
        "null_min_p_distribution": min_p,
    }


def evaluate_exact_sequential_path(
    cumulative: pd.DataFrame,
    *,
    x_control_col: str,
    n_control_col: str,
    x_treatment_col: str,
    n_treatment_col: str,
    p_threshold: float,
    method: str = "fisher",
    benefit_direction: str = "increase",
) -> pd.DataFrame:
    alternative = "greater" if benefit_direction == "increase" else "less"
    rows = []
    already_stopped = False
    for i, r in cumulative.reset_index(drop=True).iterrows():
        xc, nc = int(r[x_control_col]), int(r[n_control_col])
        xt, nt = int(r[x_treatment_col]), int(r[n_treatment_col])
        table = np.array([[xt, nt - xt], [xc, nc - xc]])
        p = _exact_pvalue(table, method, alternative)
        stop = (p <= p_threshold) and not already_stopped
        already_stopped = already_stopped or stop
        rows.append({"look": i+1, "exact_p_value": p, "calibrated_threshold": p_threshold, "stop_for_efficacy": stop})
    return pd.DataFrame(rows)

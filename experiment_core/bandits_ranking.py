from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats


def contextual_bandit_offline_evaluation(
    df: pd.DataFrame,
    *,
    action_col: str,
    reward_col: str,
    behavior_propensity_col: str,
    target_action_col: str | None = None,
    target_probability_cols: Mapping[str, str] | None = None,
    q_hat_cols: Mapping[str, str] | None = None,
    bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """IPS, SNIPS и DR offline evaluation для contextual bandit policy."""
    work = df.copy().reset_index(drop=True)
    action = work[action_col].astype(str).to_numpy()
    reward = pd.to_numeric(work[reward_col], errors="coerce").to_numpy(float)
    beh = np.clip(pd.to_numeric(work[behavior_propensity_col], errors="coerce").to_numpy(float), 1e-5, 1)
    actions = sorted(set(action))

    if target_probability_cols:
        target_prob = np.column_stack([pd.to_numeric(work[target_probability_cols[a]], errors="coerce").fillna(0) for a in actions])
        target_prob = target_prob / np.maximum(target_prob.sum(axis=1, keepdims=True), 1e-12)
        a_idx = np.array([actions.index(a) for a in action])
        eval_prob_observed = target_prob[np.arange(len(work)), a_idx]
    elif target_action_col:
        target = work[target_action_col].astype(str).to_numpy()
        eval_prob_observed = (target == action).astype(float)
        target_prob = np.column_stack([(target == a).astype(float) for a in actions])
        a_idx = np.array([actions.index(a) for a in action])
    else:
        raise ValueError("Укажите target_action_col или target_probability_cols.")

    w = eval_prob_observed / beh
    ips_contrib = w * reward
    ips = float(np.mean(ips_contrib))
    snips = float(np.sum(ips_contrib) / max(np.sum(w), 1e-12))
    ess = float(np.sum(w)**2 / max(np.sum(w**2), 1e-12))

    dr = np.nan
    dr_contrib = np.full(len(work), np.nan)
    if q_hat_cols:
        q = np.column_stack([pd.to_numeric(work[q_hat_cols[a]], errors="coerce") for a in actions])
        q_target = np.sum(target_prob * q, axis=1)
        q_obs = q[np.arange(len(work)), a_idx]
        dr_contrib = q_target + w * (reward - q_obs)
        dr = float(np.mean(dr_contrib))

    rng = np.random.default_rng(seed)
    values = {"IPS": [], "SNIPS": [], "DR": []}
    for _ in range(bootstrap):
        idx = rng.integers(0, len(work), len(work))
        values["IPS"].append(float(np.mean(ips_contrib[idx])))
        values["SNIPS"].append(float(np.sum(ips_contrib[idx]) / max(np.sum(w[idx]), 1e-12)))
        if q_hat_cols:
            values["DR"].append(float(np.mean(dr_contrib[idx])))

    rows = []
    for name, estimate in [("IPS", ips), ("SNIPS", snips), ("DR", dr)]:
        if name == "DR" and not q_hat_cols:
            continue
        rows.append({"estimator": name, "policy_value": estimate,
                     "ci_low": float(np.quantile(values[name], 0.025)), "ci_high": float(np.quantile(values[name], 0.975))})
    return {
        "summary": pd.DataFrame(rows),
        "diagnostics": pd.DataFrame([{
            "n": len(work), "weight_mean": float(np.mean(w)), "weight_max": float(np.max(w)),
            "effective_sample_size": ess, "overlap_match_or_probability_mean": float(np.mean(eval_prob_observed)),
        }]),
        "warnings": [
            "Большие importance weights и низкий ESS делают offline evaluation нестабильным.",
            "Для DR q_hat должны быть out-of-fold; policy не должна обучаться на тех же reward без честного split.",
        ],
    }


def interleaving_analysis(
    df: pd.DataFrame,
    *,
    winner_col: str,
    label_a: str = "A",
    label_b: str = "B",
    tie_label: str = "tie",
    cluster_col: str | None = None,
    bootstrap: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Анализ team-draft/interleaving по победителю каждой сессии или запроса."""
    work = df[[winner_col] + ([cluster_col] if cluster_col else [])].dropna().copy()
    labels = work[winner_col].astype(str)
    a = int((labels == str(label_a)).sum())
    b = int((labels == str(label_b)).sum())
    ties = int((labels == str(tie_label)).sum())
    non_ties = a + b
    if non_ties == 0:
        raise ValueError("Нет сессий, где победил A или B.")
    win_rate_a = a / non_ties
    test = stats.binomtest(a, non_ties, 0.5, alternative="two-sided")

    rng = np.random.default_rng(seed)
    rates = []
    if cluster_col:
        clusters = work[cluster_col].astype(str).unique()
        parts = {c: work[work[cluster_col].astype(str) == c] for c in clusters}
        for _ in range(bootstrap):
            sample = pd.concat([parts[c] for c in rng.choice(clusters, len(clusters), replace=True)], ignore_index=True)
            ls = sample[winner_col].astype(str)
            aa, bb = (ls == label_a).sum(), (ls == label_b).sum()
            if aa + bb:
                rates.append(aa / (aa + bb))
    else:
        arr = labels.to_numpy()
        for _ in range(bootstrap):
            ls = rng.choice(arr, len(arr), replace=True)
            aa, bb = np.sum(ls == label_a), np.sum(ls == label_b)
            if aa + bb:
                rates.append(aa / (aa + bb))

    return {
        "summary": pd.DataFrame([{
            "wins_a": a, "wins_b": b, "ties": ties, "non_tie_sessions": non_ties,
            "win_rate_a": win_rate_a, "ci_low": float(np.quantile(rates, 0.025)),
            "ci_high": float(np.quantile(rates, 0.975)), "exact_binomial_p_value": float(test.pvalue),
        }]),
        "warnings": ["Интерливинг измеряет относительное предпочтение ranking-систем, но не заменяет долгосрочные бизнес-метрики A/B."],
    }

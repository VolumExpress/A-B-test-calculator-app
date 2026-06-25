from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _propensity_array(df: pd.DataFrame, propensity: float | str | np.ndarray) -> np.ndarray:
    if isinstance(propensity, str):
        p = pd.to_numeric(df[propensity], errors="coerce").to_numpy(float)
    elif np.isscalar(propensity):
        p = np.full(len(df), float(propensity))
    else:
        p = np.asarray(propensity, dtype=float)
    return np.clip(p, 1e-4, 1 - 1e-4)


def _uplift_curve_on_grid(
    treatment: np.ndarray,
    outcome: np.ndarray,
    score: np.ndarray,
    propensity: np.ndarray,
    grid: np.ndarray,
) -> pd.DataFrame:
    order = np.argsort(-score)
    t, y, p = treatment[order], outcome[order], propensity[order]
    pseudo = t * y / p - (1 - t) * y / (1 - p)
    cumulative = np.cumsum(pseudo)
    n = len(y)
    idx = np.maximum(1, np.ceil(grid * n).astype(int)) - 1
    gain = cumulative[idx]
    # На 100% gain равен HT-оценке total incremental outcome.
    return pd.DataFrame({"fraction": grid, "cumulative_incremental_outcome": gain})


def qini_auuc_analysis(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    score_col: str,
    propensity: float | str | np.ndarray = 0.5,
    cluster_col: str | None = None,
    bootstrap: int = 500,
    grid_points: int = 101,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Qini/AUUC с IPW-коррекцией и bootstrap confidence intervals.

    Bootstrap выполняется по клиентам или по кластерам, если один клиент/менеджер
    представлен несколькими строками.
    """
    cols = [treatment_col, outcome_col, score_col] + ([cluster_col] if cluster_col else [])
    work = df[cols].dropna(subset=[treatment_col, outcome_col, score_col]).copy().reset_index(drop=True)
    t = pd.to_numeric(work[treatment_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(float)
    score = pd.to_numeric(work[score_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(score) & np.isin(t, [0, 1])
    work = work.loc[ok].reset_index(drop=True)
    t, y, score = t[ok], y[ok], score[ok]
    p = _propensity_array(df.loc[ok] if isinstance(propensity, str) else work, propensity)
    grid = np.linspace(0.01, 1.0, grid_points)
    curve = _uplift_curve_on_grid(t, y, score, p, grid)
    random_line = curve["fraction"] * curve["cumulative_incremental_outcome"].iloc[-1]
    curve["random_policy_line"] = random_line
    curve["qini_gain"] = curve["cumulative_incremental_outcome"] - random_line
    auuc = float(np.trapezoid(curve["cumulative_incremental_outcome"], curve["fraction"]))
    qini = float(np.trapezoid(curve["qini_gain"], curve["fraction"]))

    rng = np.random.default_rng(seed)
    boot_curves = np.empty((bootstrap, len(grid)))
    boot_auuc = np.empty(bootstrap)
    boot_qini = np.empty(bootstrap)
    if cluster_col:
        clusters = work[cluster_col].astype(str).unique()
        cluster_indices = {c: np.flatnonzero(work[cluster_col].astype(str).to_numpy() == c) for c in clusters}
    for b in range(bootstrap):
        if cluster_col:
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            idx = np.concatenate([cluster_indices[c] for c in sampled])
        else:
            idx = rng.integers(0, len(work), len(work))
        bc = _uplift_curve_on_grid(t[idx], y[idx], score[idx], p[idx], grid)
        rand = bc["fraction"] * bc["cumulative_incremental_outcome"].iloc[-1]
        bg = bc["cumulative_incremental_outcome"].to_numpy()
        bq = bg - rand.to_numpy()
        boot_curves[b] = bg
        boot_auuc[b] = np.trapezoid(bg, grid)
        boot_qini[b] = np.trapezoid(bq, grid)

    curve["ci_low"] = np.quantile(boot_curves, 0.025, axis=0)
    curve["ci_high"] = np.quantile(boot_curves, 0.975, axis=0)
    top10_idx = int(np.argmin(np.abs(grid - 0.10)))
    return {
        "curve": curve,
        "metrics": pd.DataFrame([{
            "n": len(work),
            "auuc": auuc,
            "auuc_ci_low": float(np.quantile(boot_auuc, 0.025)),
            "auuc_ci_high": float(np.quantile(boot_auuc, 0.975)),
            "qini_coefficient": qini,
            "qini_ci_low": float(np.quantile(boot_qini, 0.025)),
            "qini_ci_high": float(np.quantile(boot_qini, 0.975)),
            "uplift_top_10_fraction_total_gain": float(curve.iloc[top10_idx]["cumulative_incremental_outcome"]),
        }]),
        "warnings": [
            "Qini/AUUC оценивайте только на независимых данных или out-of-fold predictions.",
            "Для решения о внедрении дополнительно считайте incremental profit и policy value.",
        ],
    }


def uplift_calibration(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    score_col: str,
    propensity: float | str | np.ndarray = 0.5,
    bins: int = 10,
) -> pd.DataFrame:
    work = df[[treatment_col, outcome_col, score_col] + ([propensity] if isinstance(propensity, str) else [])].dropna().copy()
    work["_p"] = _propensity_array(work, propensity)
    work["_t"] = pd.to_numeric(work[treatment_col])
    work["_y"] = pd.to_numeric(work[outcome_col])
    work["_score"] = pd.to_numeric(work[score_col])
    work["_pseudo"] = work["_t"] * work["_y"] / work["_p"] - (1-work["_t"]) * work["_y"] / (1-work["_p"])
    work["bin"] = pd.qcut(work["_score"], q=min(bins, work["_score"].nunique()), duplicates="drop")
    return work.groupby("bin", observed=True).agg(
        n=("_y", "size"),
        predicted_uplift=("_score", "mean"),
        observed_uplift=("_pseudo", "mean"),
        score_min=("_score", "min"),
        score_max=("_score", "max"),
    ).reset_index()


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)


def crossfit_nuisance_models(
    df: pd.DataFrame,
    *,
    action_col: str,
    reward_col: str,
    feature_cols: Iterable[str],
    folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cross-fit propensity и outcome models для multi-action DR evaluation."""
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("Для cross-fitting укажите хотя бы один pre-treatment признак.")
    work = df[[action_col, reward_col, *feature_cols]].reset_index(drop=True)
    actions = sorted(work[action_col].astype(str).unique())
    action_to_idx = {a: i for i, a in enumerate(actions)}
    a = work[action_col].astype(str).map(action_to_idx).to_numpy()
    y = pd.to_numeric(work[reward_col], errors="coerce").to_numpy(float)
    X = work[list(feature_cols)]
    n, k = len(work), len(actions)
    p_hat = np.zeros((n, k))
    q_hat = np.zeros((n, k))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for train, test in cv.split(X, a):
        prop = Pipeline([("prep", _preprocessor(X)), ("model", LogisticRegression(max_iter=1500))])
        prop.fit(X.iloc[train], a[train])
        probs = prop.predict_proba(X.iloc[test])
        for j, cls in enumerate(prop.named_steps["model"].classes_):
            p_hat[test, int(cls)] = probs[:, j]
        for j in range(k):
            mask = train[a[train] == j]
            if len(mask) < 20:
                q_hat[test, j] = np.nanmean(y[train])
                continue
            model = Pipeline([("prep", _preprocessor(X)), ("model", Ridge(alpha=10.0))])
            model.fit(X.iloc[mask], y[mask])
            q_hat[test, j] = model.predict(X.iloc[test])
    return np.clip(p_hat, 1e-4, 1), q_hat, actions


def doubly_robust_policy_value(
    df: pd.DataFrame,
    *,
    action_col: str,
    reward_col: str,
    evaluation_action_col: str,
    behavior_propensity_col: str | None = None,
    q_hat_cols: Mapping[str, str] | None = None,
    feature_cols: Iterable[str] = (),
    bootstrap: int = 500,
    cluster_col: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """AIPW/DR оценка value новой дискретной политики по логам рандомизированной политики."""
    work = df.copy().reset_index(drop=True)
    observed = work[action_col].astype(str)
    target = work[evaluation_action_col].astype(str)
    reward = pd.to_numeric(work[reward_col], errors="coerce").to_numpy(float)

    if q_hat_cols is None:
        if not feature_cols:
            raise ValueError("Укажите q_hat_cols или feature_cols для cross-fitting.")
        p_matrix, q_matrix, actions = crossfit_nuisance_models(
            work, action_col=action_col, reward_col=reward_col, feature_cols=feature_cols
        )
    else:
        actions = sorted(q_hat_cols)
        q_matrix = np.column_stack([pd.to_numeric(work[q_hat_cols[a]], errors="coerce") for a in actions])
        p_matrix = np.full_like(q_matrix, np.nan, dtype=float)

    action_to_idx = {a: i for i, a in enumerate(actions)}
    a_idx = observed.map(action_to_idx).to_numpy()
    target_idx = target.map(action_to_idx).to_numpy()
    valid = np.isfinite(reward) & pd.notna(a_idx) & pd.notna(target_idx)
    a_idx = a_idx[valid].astype(int)
    target_idx = target_idx[valid].astype(int)
    reward = reward[valid]
    q_matrix = q_matrix[valid]
    if behavior_propensity_col:
        p_obs = np.clip(pd.to_numeric(work.loc[valid, behavior_propensity_col], errors="coerce").to_numpy(float), 1e-4, 1)
    else:
        if q_hat_cols is not None:
            counts = observed[valid].value_counts(normalize=True)
            p_obs = np.array([counts.get(actions[j], 1e-4) for j in a_idx])
        else:
            p_obs = p_matrix[valid][np.arange(valid.sum()), a_idx]

    q_target = q_matrix[np.arange(len(reward)), target_idx]
    q_observed = q_matrix[np.arange(len(reward)), a_idx]
    match = (a_idx == target_idx).astype(float)
    influence = q_target + match / p_obs * (reward - q_observed)
    ips = match / p_obs * reward
    value = float(np.mean(influence))
    ess = float((np.sum(match / p_obs)**2) / np.sum((match / p_obs)**2)) if np.sum(match) > 0 else 0

    rng = np.random.default_rng(seed)
    if cluster_col:
        clusters = work.loc[valid, cluster_col].astype(str).to_numpy()
        uniq = np.unique(clusters)
        idx_map = {c: np.flatnonzero(clusters == c) for c in uniq}
    vals = []
    for _ in range(bootstrap):
        if cluster_col:
            sampled = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([idx_map[c] for c in sampled])
        else:
            idx = rng.integers(0, len(influence), len(influence))
        vals.append(float(np.mean(influence[idx])))

    return {
        "summary": pd.DataFrame([{
            "dr_policy_value": value,
            "ips_policy_value": float(np.mean(ips)),
            "ci_low": float(np.quantile(vals, 0.025)),
            "ci_high": float(np.quantile(vals, 0.975)),
            "effective_sample_size_weights": ess,
            "policy_match_rate": float(np.mean(match)),
            "n": len(influence),
        }]),
        "influence_values": pd.DataFrame({"dr_contribution": influence, "ips_contribution": ips}),
        "warnings": [
            "Низкий match rate или ESS означает слабый overlap: offline-оценка политики ненадёжна.",
            "Все q_hat и propensity должны быть out-of-fold или рассчитаны на независимых данных.",
        ],
    }


def optimize_capacity_nba(
    df: pd.DataFrame,
    *,
    id_col: str,
    value_cols: Mapping[str, str],
    capacities: Mapping[str, int],
    no_action: str,
    cost_cols: Mapping[str, str | float] | None = None,
    fatigue_col: str | None = None,
    fatigue_penalty: float = 0.0,
    exact_limit: int = 3_000,
) -> dict[str, Any]:
    """Назначает максимум одно действие клиенту, соблюдая capacity каждого канала."""
    actions = list(value_cols)
    if no_action not in actions:
        raise ValueError("no_action должен присутствовать в value_cols.")
    n, k = len(df), len(actions)
    values = np.column_stack([pd.to_numeric(df[value_cols[a]], errors="coerce").fillna(-1e12) for a in actions]).astype(float)
    if cost_cols:
        for j, a in enumerate(actions):
            spec = cost_cols.get(a, 0.0)
            values[:, j] -= float(spec) if np.isscalar(spec) else pd.to_numeric(df[str(spec)], errors="coerce").fillna(0).to_numpy(float)
    if fatigue_col and fatigue_penalty:
        fatigue = pd.to_numeric(df[fatigue_col], errors="coerce").fillna(0).to_numpy(float)
        for j, a in enumerate(actions):
            if a != no_action:
                values[:, j] -= fatigue_penalty * fatigue

    action_idx = {a: j for j, a in enumerate(actions)}
    method = "greedy"
    assignment = np.full(n, action_idx[no_action], dtype=int)

    if n <= exact_limit:
        # Binary variables x_{i,a}. Minimize negative value.
        c = -values.ravel()
        integrality = np.ones(n*k)
        lb, ub = np.zeros(n*k), np.ones(n*k)
        constraints = []
        # Exactly one action per client.
        Aeq = np.zeros((n, n*k))
        for i in range(n):
            Aeq[i, i*k:(i+1)*k] = 1
        constraints.append(optimize.LinearConstraint(Aeq, np.ones(n), np.ones(n)))
        # Capacity per action, excluding no-action unless explicitly provided.
        Acap, lower, upper = [], [], []
        for a, cap in capacities.items():
            if a not in action_idx:
                continue
            row = np.zeros(n*k)
            row[action_idx[a]::k] = 1
            Acap.append(row); lower.append(0); upper.append(cap)
        if Acap:
            constraints.append(optimize.LinearConstraint(np.asarray(Acap), np.asarray(lower), np.asarray(upper)))
        res = optimize.milp(c=c, integrality=integrality, bounds=optimize.Bounds(lb, ub), constraints=constraints,
                            options={"time_limit": 30})
        if res.success and res.x is not None:
            assignment = res.x.reshape(n, k).argmax(axis=1)
            method = "MILP"
    if method == "greedy":
        base = values[:, action_idx[no_action]]
        candidates = []
        for j, a in enumerate(actions):
            if a == no_action:
                continue
            for i in range(n):
                candidates.append((values[i, j] - base[i], i, j, a))
        candidates.sort(reverse=True, key=lambda x: x[0])
        used = {a: 0 for a in actions}
        assigned = np.zeros(n, dtype=bool)
        for inc, i, j, a in candidates:
            if inc <= 0 or assigned[i] or used.get(a, 0) >= capacities.get(a, n):
                continue
            assignment[i] = j; assigned[i] = True; used[a] = used.get(a, 0) + 1

    assigned_action = np.array(actions, dtype=object)[assignment]
    chosen_value = values[np.arange(n), assignment]
    baseline_value = values[:, action_idx[no_action]]
    result = pd.DataFrame({id_col: df[id_col].to_numpy(), "assigned_action": assigned_action,
                           "assigned_net_value": chosen_value, "incremental_value_vs_no_action": chosen_value-baseline_value})
    usage = result.groupby("assigned_action").agg(clients=(id_col, "size"), total_net_value=("assigned_net_value", "sum"),
                                                   total_incremental_value=("incremental_value_vs_no_action", "sum")).reset_index()
    return {
        "assignments": result,
        "usage": usage,
        "summary": pd.DataFrame([{
            "optimization_method": method,
            "clients": n,
            "total_net_value": float(chosen_value.sum()),
            "incremental_value_vs_no_action": float((chosen_value-baseline_value).sum()),
        }]),
        "warnings": [] if method == "MILP" else ["Для большого файла использован быстрый greedy-алгоритм; он не гарантирует глобальный optimum."],
    }

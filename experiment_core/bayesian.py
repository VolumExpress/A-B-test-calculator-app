from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PriorSpec:
    name: str
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def ess(self) -> float:
        return self.alpha + self.beta


def beta_prior_from_mean_ess(mean: float, ess: float, name: str = "historical") -> PriorSpec:
    """Создаёт Beta prior через понятные пользователю среднее и условный ESS."""
    if not 0 < mean < 1:
        raise ValueError("Среднее prior должно быть между 0 и 1.")
    if ess <= 0:
        raise ValueError("ESS prior должен быть положительным.")
    return PriorSpec(name=name, alpha=mean * ess, beta=(1 - mean) * ess)


def prior_predictive_binary(prior: PriorSpec, n: int, draws: int = 50_000, seed: int = 42) -> dict[str, Any]:
    """Prior-predictive проверка: сколько событий prior ожидает до просмотра пилота."""
    rng = np.random.default_rng(seed)
    p = rng.beta(prior.alpha, prior.beta, draws)
    x = rng.binomial(n, p)
    return {
        "prior": prior.name,
        "prior_mean": float(np.mean(p)),
        "prior_ess": prior.ess,
        "event_count_mean": float(np.mean(x)),
        "event_count_p05": float(np.quantile(x, 0.05)),
        "event_count_p50": float(np.quantile(x, 0.50)),
        "event_count_p95": float(np.quantile(x, 0.95)),
        "probability_zero_events": float(np.mean(x == 0)),
    }


def bayesian_two_group_binary(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    *,
    prior_control: PriorSpec | None = None,
    prior_treatment: PriorSpec | None = None,
    minimum_relative_effect: float = 0.0,
    benefit_direction: str = "increase",
    draws: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Полный Beta-Binomial анализ двух групп.

    benefit_direction='increase': больше outcome лучше (например, продажа).
    benefit_direction='decrease': меньше outcome лучше (например, churn/default).
    """
    prior_control = prior_control or PriorSpec("Jeffreys", 0.5, 0.5)
    prior_treatment = prior_treatment or prior_control
    if min(n_control, n_treatment) <= 0:
        raise ValueError("Размеры групп должны быть положительными.")
    if not (0 <= x_control <= n_control and 0 <= x_treatment <= n_treatment):
        raise ValueError("Число событий должно быть от 0 до размера группы.")

    rng = np.random.default_rng(seed)
    pc = rng.beta(prior_control.alpha + x_control, prior_control.beta + n_control - x_control, draws)
    pt = rng.beta(prior_treatment.alpha + x_treatment, prior_treatment.beta + n_treatment - x_treatment, draws)
    absolute = pt - pc
    relative = absolute / np.maximum(pc, 1e-15)

    if benefit_direction == "decrease":
        benefit_abs = pc - pt
        benefit_rel = benefit_abs / np.maximum(pc, 1e-15)
    elif benefit_direction == "increase":
        benefit_abs = pt - pc
        benefit_rel = benefit_abs / np.maximum(pc, 1e-15)
    else:
        raise ValueError("benefit_direction: increase или decrease")

    return {
        "posterior_summary": pd.DataFrame([
            {
                "group": "control",
                "posterior_mean": float(pc.mean()),
                "ci_low": float(np.quantile(pc, 0.025)),
                "ci_high": float(np.quantile(pc, 0.975)),
                "prior": prior_control.name,
                "prior_ess": prior_control.ess,
            },
            {
                "group": "treatment",
                "posterior_mean": float(pt.mean()),
                "ci_low": float(np.quantile(pt, 0.025)),
                "ci_high": float(np.quantile(pt, 0.975)),
                "prior": prior_treatment.name,
                "prior_ess": prior_treatment.ess,
            },
        ]),
        "effect_summary": pd.DataFrame([{
            "posterior_mean_absolute_effect": float(benefit_abs.mean()),
            "absolute_effect_ci_low": float(np.quantile(benefit_abs, 0.025)),
            "absolute_effect_ci_high": float(np.quantile(benefit_abs, 0.975)),
            "posterior_mean_relative_effect": float(benefit_rel.mean()),
            "relative_effect_ci_low": float(np.quantile(benefit_rel, 0.025)),
            "relative_effect_ci_high": float(np.quantile(benefit_rel, 0.975)),
            "probability_treatment_better": float(np.mean(benefit_abs > 0)),
            "probability_effect_at_least_threshold": float(np.mean(benefit_rel >= minimum_relative_effect)),
            "minimum_relative_effect": minimum_relative_effect,
        }]),
        "draws": {
            "control": pc,
            "treatment": pt,
            "benefit_absolute": benefit_abs,
            "benefit_relative": benefit_rel,
            "raw_absolute": absolute,
            "raw_relative": relative,
        },
    }


def bayesian_prior_sensitivity(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    priors: Iterable[PriorSpec],
    *,
    minimum_relative_effect: float = 0.0,
    benefit_direction: str = "increase",
    draws: int = 30_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Сравнивает решение при нескольких разумных priors."""
    rows = []
    for i, prior in enumerate(priors):
        res = bayesian_two_group_binary(
            x_control, n_control, x_treatment, n_treatment,
            prior_control=prior, prior_treatment=prior,
            minimum_relative_effect=minimum_relative_effect,
            benefit_direction=benefit_direction,
            draws=draws, seed=seed + i,
        )
        row = res["effect_summary"].iloc[0].to_dict()
        row.update({"prior": prior.name, "prior_mean": prior.mean, "prior_ess": prior.ess})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["decision_sensitive"] = (
        result["probability_treatment_better"].max() - result["probability_treatment_better"].min() > 0.10
    )
    return result


def robust_mixture_posterior(
    x: int,
    n: int,
    historical: PriorSpec,
    *,
    historical_weight: float = 0.7,
    vague: PriorSpec | None = None,
    grid_size: int = 20_000,
) -> dict[str, Any]:
    """
    Robust mixture prior для одной доли.

    При конфликте текущих и исторических данных вес исторического компонента
    автоматически уменьшается через marginal likelihood.
    """
    vague = vague or PriorSpec("Jeffreys", 0.5, 0.5)
    if not 0 < historical_weight < 1:
        raise ValueError("Вес исторического компонента должен быть между 0 и 1.")

    def log_beta_binom_marginal(prior: PriorSpec) -> float:
        return (
            math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
            + math.lgamma(prior.alpha + x) + math.lgamma(prior.beta + n - x)
            - math.lgamma(prior.alpha + prior.beta + n)
            - (math.lgamma(prior.alpha) + math.lgamma(prior.beta) - math.lgamma(prior.alpha + prior.beta))
        )

    log_h = math.log(historical_weight) + log_beta_binom_marginal(historical)
    log_v = math.log(1 - historical_weight) + log_beta_binom_marginal(vague)
    mx = max(log_h, log_v)
    post_weight_h = math.exp(log_h - mx) / (math.exp(log_h - mx) + math.exp(log_v - mx))

    grid = np.linspace(1e-8, 1 - 1e-8, grid_size)
    density_h = stats.beta.pdf(grid, historical.alpha + x, historical.beta + n - x)
    density_v = stats.beta.pdf(grid, vague.alpha + x, vague.beta + n - x)
    density = post_weight_h * density_h + (1 - post_weight_h) * density_v
    density /= np.trapz(density, grid)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]

    def q(prob: float) -> float:
        return float(grid[np.searchsorted(cdf, prob)])

    mean = float(np.trapz(grid * density, grid))
    return {
        "posterior_historical_weight": post_weight_h,
        "posterior_mean": mean,
        "ci_low": q(0.025),
        "ci_high": q(0.975),
        "grid": grid,
        "density": density,
    }


def beta_binomial_predictive_probability(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    n_control_max: int,
    n_treatment_max: int,
    *,
    prior: PriorSpec | None = None,
    success_probability_threshold: float = 0.975,
    required_relative_effect: float = 0.0,
    benefit_direction: str = "increase",
    outer_simulations: int = 3_000,
    posterior_draws: int = 800,
    seed: int = 42,
) -> dict[str, float]:
    """Вероятность, что пилот завершится успехом после добора максимальной выборки."""
    prior = prior or PriorSpec("Jeffreys", 0.5, 0.5)
    rem_c = max(0, n_control_max - n_control)
    rem_t = max(0, n_treatment_max - n_treatment)
    rng = np.random.default_rng(seed)
    successes = np.zeros(outer_simulations, dtype=bool)

    ac, bc = prior.alpha + x_control, prior.beta + n_control - x_control
    at, bt = prior.alpha + x_treatment, prior.beta + n_treatment - x_treatment

    for i in range(outer_simulations):
        pc = rng.beta(ac, bc)
        pt = rng.beta(at, bt)
        fc = x_control + rng.binomial(rem_c, pc)
        ft = x_treatment + rng.binomial(rem_t, pt)
        pc_post = rng.beta(prior.alpha + fc, prior.beta + n_control_max - fc, posterior_draws)
        pt_post = rng.beta(prior.alpha + ft, prior.beta + n_treatment_max - ft, posterior_draws)
        if benefit_direction == "decrease":
            rel = (pc_post - pt_post) / np.maximum(pc_post, 1e-15)
        else:
            rel = (pt_post - pc_post) / np.maximum(pc_post, 1e-15)
        successes[i] = np.mean(rel >= required_relative_effect) >= success_probability_threshold

    p = float(successes.mean())
    mc_se = math.sqrt(max(p * (1 - p) / outer_simulations, 0))
    return {
        "predictive_probability": p,
        "mc_standard_error": mc_se,
        "simulations": outer_simulations,
        "success_probability_threshold": success_probability_threshold,
        "required_relative_effect": required_relative_effect,
    }

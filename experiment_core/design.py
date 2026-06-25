from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass(frozen=True)
class BinaryDesign:
    p_control: float
    p_treatment: float
    absolute_effect: float
    relative_effect: float
    alpha: float
    power: float
    sided: str
    treatment_share: float
    n_control: int
    n_treatment: int
    n_total: int
    clients_per_period: int
    periods: float
    expected_events_control: float
    expected_events_treatment: float
    expected_events_total: float


@dataclass(frozen=True)
class ContinuousDesign:
    mean_control: float
    mean_treatment: float
    absolute_effect: float
    std: float
    standardized_effect: float
    alpha: float
    power: float
    sided: str
    treatment_share: float
    n_control: int
    n_treatment: int
    n_total: int
    clients_per_period: int
    periods: float


def _alternative(sided: str) -> str:
    if sided == "one-sided":
        return "larger"
    if sided == "two-sided":
        return "two-sided"
    raise ValueError("sided должен быть 'one-sided' или 'two-sided'")


def _z_alpha(alpha: float, sided: str) -> float:
    return stats.norm.ppf(1 - alpha if sided == "one-sided" else 1 - alpha / 2)


def fixed_binary_design(
    p_control: float,
    p_treatment: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    treatment_share: float = 0.50,
    sided: str = "one-sided",
    clients_per_period: int = 10_000,
) -> BinaryDesign:
    """Классический fixed-horizon дизайн для двух независимых долей."""
    if not 0 < p_control < 1 or not 0 < p_treatment < 1:
        raise ValueError("Доли должны находиться между 0 и 1.")
    if p_control == p_treatment:
        raise ValueError("Ожидаемый эффект равен нулю.")
    if not 0 < treatment_share < 1:
        raise ValueError("Доля treatment должна находиться между 0 и 1.")
    if clients_per_period <= 0:
        raise ValueError("Трафик должен быть положительным.")

    ratio = treatment_share / (1 - treatment_share)
    effect_size = abs(proportion_effectsize(p_control, p_treatment))
    n_control_float = NormalIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=_alternative(sided),
    )
    n_control = math.ceil(float(n_control_float))
    n_treatment = math.ceil(n_control * ratio)
    n_total = n_control + n_treatment

    relative_effect = (p_treatment - p_control) / p_control
    return BinaryDesign(
        p_control=p_control,
        p_treatment=p_treatment,
        absolute_effect=p_treatment - p_control,
        relative_effect=relative_effect,
        alpha=alpha,
        power=power,
        sided=sided,
        treatment_share=treatment_share,
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_total,
        clients_per_period=clients_per_period,
        periods=n_total / clients_per_period,
        expected_events_control=n_control * p_control,
        expected_events_treatment=n_treatment * p_treatment,
        expected_events_total=n_control * p_control + n_treatment * p_treatment,
    )


def fixed_continuous_design(
    mean_control: float,
    mean_treatment: float,
    std: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    treatment_share: float = 0.50,
    sided: str = "two-sided",
    clients_per_period: int = 10_000,
) -> ContinuousDesign:
    """Дизайн для непрерывной метрики с общей плановой стандартной ошибкой."""
    if std <= 0:
        raise ValueError("Стандартное отклонение должно быть положительным.")
    if mean_control == mean_treatment:
        raise ValueError("Ожидаемый эффект равен нулю.")
    if not 0 < treatment_share < 1:
        raise ValueError("Доля treatment должна находиться между 0 и 1.")

    delta = abs(mean_treatment - mean_control)
    z_sum = _z_alpha(alpha, sided) + stats.norm.ppf(power)
    # Var(diff)=sigma^2/n_c + sigma^2/n_t. Выражаем через долю treatment.
    q = treatment_share
    n_total_float = (z_sum**2 * std**2 * (1 / q + 1 / (1 - q))) / delta**2
    n_total = math.ceil(n_total_float)
    n_treatment = math.ceil(n_total * q)
    n_control = n_total - n_treatment

    return ContinuousDesign(
        mean_control=mean_control,
        mean_treatment=mean_treatment,
        absolute_effect=mean_treatment - mean_control,
        std=std,
        standardized_effect=(mean_treatment - mean_control) / std,
        alpha=alpha,
        power=power,
        sided=sided,
        treatment_share=treatment_share,
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_total,
        clients_per_period=clients_per_period,
        periods=n_total / clients_per_period,
    )


def multiarm_binary_design(
    p_control: float,
    treatment_rates: dict[str, float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    sided: str = "one-sided",
    correction: str = "holm",
    clients_per_period: int = 10_000,
) -> pd.DataFrame:
    """
    Консервативный план multi-arm: каждая ветка сравнивается с общим контролем.

    Для Holm при планировании используется alpha/m, поскольку точная мощность Holm
    зависит от совместного распределения результатов. Это безопасная оценка сверху
    по требуемой выборке. В анализе фактические p-value корректируются методом Holm.
    """
    if not treatment_rates:
        raise ValueError("Нужна хотя бы одна treatment-ветка.")
    m = len(treatment_rates)
    correction = correction.lower()
    if correction not in {"holm", "bonferroni", "none"}:
        raise ValueError("correction: holm, bonferroni или none")
    alpha_each = alpha / m if correction in {"holm", "bonferroni"} else alpha

    per_arm = []
    for arm, p_t in treatment_rates.items():
        design = fixed_binary_design(
            p_control,
            p_t,
            alpha=alpha_each,
            power=power,
            treatment_share=0.50,
            sided=sided,
            clients_per_period=clients_per_period,
        )
        per_arm.append(
            {
                "arm": arm,
                "p_control": p_control,
                "p_treatment": p_t,
                "absolute_effect": p_t - p_control,
                "relative_effect": (p_t - p_control) / p_control,
                "alpha_for_planning": alpha_each,
                "n_control_pairwise": design.n_control,
                "n_treatment": design.n_treatment,
            }
        )

    df = pd.DataFrame(per_arm)
    shared_control = int(df["n_control_pairwise"].max())
    df["shared_control"] = shared_control
    df["recommended_arm_n"] = df["n_treatment"]
    total = shared_control + int(df["recommended_arm_n"].sum())
    df["recommended_total_n"] = total
    df["periods"] = total / clients_per_period
    df["correction"] = correction
    return df


def variance_reduction_scenarios(
    n_total: int,
    clients_per_period: int,
    reductions: Iterable[float] = (0.0, 0.10, 0.20, 0.30, 0.40),
) -> pd.DataFrame:
    rows = []
    for reduction in reductions:
        if not 0 <= reduction < 1:
            raise ValueError("Снижение дисперсии должно быть от 0 до 1.")
        adjusted = math.ceil(n_total * (1 - reduction))
        rows.append(
            {
                "variance_reduction": reduction,
                "sample_size": adjusted,
                "periods": adjusted / clients_per_period,
                "saved_clients": n_total - adjusted,
            }
        )
    return pd.DataFrame(rows)


def obrien_fleming_boundaries(
    alpha: float = 0.05,
    information_fractions: Iterable[float] = (0.33, 0.67, 1.0),
    sided: str = "one-sided",
) -> pd.DataFrame:
    """
    Простая O'Brien–Fleming-подобная таблица границ.

    Это планировочная аппроксимация для MVP. Для регуляторно значимого решения
    границы следует откалибровать симуляцией под фактическую метрику и график interim.
    """
    fractions = np.asarray(list(information_fractions), dtype=float)
    if np.any(fractions <= 0) or np.any(fractions > 1) or not np.all(np.diff(fractions) > 0):
        raise ValueError("Доли информации должны возрастать и быть в диапазоне (0,1].")
    final_z = _z_alpha(alpha, sided)
    raw = final_z / np.sqrt(fractions)
    return pd.DataFrame(
        {
            "look": np.arange(1, len(fractions) + 1),
            "information_fraction": fractions,
            "z_boundary_approx": raw,
            "nominal_p_one_sided": 1 - stats.norm.cdf(raw),
        }
    )


def design_to_frame(design: BinaryDesign | ContinuousDesign) -> pd.DataFrame:
    return pd.DataFrame([asdict(design)])

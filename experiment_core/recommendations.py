from __future__ import annotations

from typing import Any

import pandas as pd


def design_recommendations(
    *,
    task_type: str,
    metric_type: str,
    periods: float,
    max_periods: float,
    expected_events_min: float | None = None,
    use_sequential: bool = False,
    has_preperiod: bool = False,
    groups: int = 2,
) -> list[str]:
    recs: list[str] = []
    if periods <= max_periods:
        recs.append("Базовый дизайн укладывается в указанный бизнесом срок.")
    else:
        recs.append(
            "Базовый дизайн не укладывается в срок. Не завышайте MDE автоматически: "
            "проверьте трафик, CUPED/CUPAC, sequential design и фактическую доставку воздействия."
        )
    if expected_events_min is not None and expected_events_min < 5:
        recs.append(
            "Целевое событие экстремально редкое. Используйте exact-анализ и планирование по числу событий."
        )
    elif expected_events_min is not None and expected_events_min < 20:
        recs.append(
            "Число событий небольшое. Показывайте вместе асимптотический и exact-результат."
        )
    if has_preperiod:
        recs.append(
            "Есть исторические признаки: оцените потенциальное снижение дисперсии через A/A и CUPED/CUPAC."
        )
    if use_sequential:
        recs.append(
            "Interim-анализы должны быть зафиксированы до старта; обычное еженедельное сравнение p-value недопустимо."
        )
    if groups > 2:
        recs.append(
            "Несколько treatment-веток требуют общей коррекции множественных сравнений и единого контроля."
        )
    if task_type in {"Uplift / NBA", "Выбор канала"}:
        recs.append(
            "Оценивайте не только средний отклик, но и incremental profit/policy value при заданном capacity."
        )
    if metric_type == "continuous":
        recs.append(
            "Для денежной метрики проверьте тяжёлые хвосты и выбросы; при необходимости используйте winsorization по протоколу или bootstrap."
        )
    return recs


def result_interpretation(
    results: pd.DataFrame,
    warnings: list[str],
    *,
    alpha: float,
) -> list[str]:
    messages: list[str] = []
    if "significant_adjusted" in results.columns:
        winners = results.loc[results["significant_adjusted"], "arm"].astype(str).tolist()
        if winners:
            messages.append("После коррекции подтверждены ветки: " + ", ".join(winners) + ".")
        else:
            messages.append("После коррекции множественных сравнений ни одна ветка не подтверждена.")
    else:
        row = results.iloc[0]
        if float(row["p_value"]) < alpha:
            messages.append(
                "Нулевая гипотеза отклоняется на выбранном уровне alpha. "
                "Проверьте также доверительный интервал и бизнес-значимость эффекта."
            )
        else:
            messages.append(
                "Статистически значимый эффект не подтверждён. Это не доказывает отсутствие эффекта: "
                "сопоставьте накопленную выборку с планом и проверьте мощность."
            )
    messages.extend(warnings)
    return messages

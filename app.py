from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from experiment_core.analysis import analyze_experiment, analyze_uplift, validate_dataset
from experiment_core.design import (
    design_to_frame,
    fixed_binary_design,
    fixed_continuous_design,
    multiarm_binary_design,
    obrien_fleming_boundaries,
    variance_reduction_scenarios,
)
from experiment_core.excel_report import build_excel_report
from experiment_core.recommendations import design_recommendations, result_interpretation
from advanced_ui import render_advanced


APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"

st.set_page_config(
    page_title="Banking Experiment Calculator — Advanced MVP",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


TASK_OPTIONS = [
    "Churn / удержание",
    "Продажи / propensity",
    "Uplift / NBA",
    "Выбор канала",
    "Коллекшн",
    "Fraud / проверки",
    "Рекомендации / ranking",
    "AI-ассистент",
    "Другая продуктовая политика",
]

DESIGN_OPTIONS = [
    "A/B: бинарная метрика",
    "A/B: непрерывная метрика",
    "Multi-arm: бинарная метрика",
    "Анализ uplift-модели",
]


def initialize_state() -> None:
    defaults = {
        "mode": "Проектирование пилота",
        "design_step": 1,
        "analysis_step": 1,
        "passport": {},
        "design_result": None,
        "scenario_result": None,
        "design_recommendations": [],
        "analysis_result": None,
        "uplift_result": None,
        "uploaded_df": None,
        "use_sequential": False,
        "has_preperiod": False,
        "_use_sequential_widget": False,
        "_has_preperiod_widget": False,
        "design_type": DESIGN_OPTIONS[0],
        "_design_type_widget": DESIGN_OPTIONS[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Streamlit очищает состояние виджетов, которые временно не отображаются.
    # Самоприсваивание отделяет значения от жизненного цикла конкретного шага мастера.
    for key in ("_use_sequential_widget", "_has_preperiod_widget", "_design_type_widget"):
        st.session_state[key] = st.session_state[key]


def reset_flow(prefix: str) -> None:
    st.session_state[f"{prefix}_step"] = 1
    if prefix == "design":
        st.session_state.design_result = None
        st.session_state.scenario_result = None
        st.session_state.design_recommendations = []
    else:
        st.session_state.analysis_result = None
        st.session_state.uplift_result = None
        st.session_state.uploaded_df = None


def nav_buttons(prefix: str, max_step: int, *, disable_next: bool = False) -> None:
    left, _, right = st.columns([1, 5, 1])
    step_key = f"{prefix}_step"
    with left:
        if st.session_state[step_key] > 1 and st.button("← Назад", width="stretch"):
            st.session_state[step_key] -= 1
            st.rerun()
    with right:
        if st.session_state[step_key] < max_step and st.button(
            "Далее →", width="stretch", disabled=disable_next
        ):
            st.session_state[step_key] += 1
            st.rerun()


def read_uploaded_file(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    raw = uploaded.getvalue()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(raw))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(raw))
    raise ValueError("Поддерживаются CSV и XLSX.")


def render_header() -> None:
    st.title("🧪 Banking Experiment Calculator — Advanced MVP")
    st.caption(
        "Пошаговое проектирование и анализ банковских пилотов. "
        "Данные не сохраняются в БД: загрузка и выгрузка выполняются через Excel/CSV."
    )


def render_sidebar() -> None:
    st.sidebar.header("Режим работы")
    mode = st.sidebar.radio(
        "Выберите задачу",
        ["Проектирование пилота", "Анализ результатов", "Расширенные методы"],
        key="mode",
    )
    if mode == "Проектирование пилота":
        step = st.session_state.design_step
        labels = ["Паспорт", "Тип дизайна", "Параметры", "Результат"]
        st.sidebar.progress(step / len(labels))
        st.sidebar.write(f"Шаг {step} из {len(labels)}: **{labels[step-1]}**")
    elif mode == "Анализ результатов":
        step = st.session_state.analysis_step
        labels = ["Загрузка", "Настройка", "Результат"]
        st.sidebar.progress(step / len(labels))
        st.sidebar.write(f"Шаг {step} из {len(labels)}: **{labels[step-1]}**")
    else:
        st.sidebar.success("Расширенные методы: выберите модуль в основном окне.")
    st.sidebar.divider()
    st.sidebar.info(
        "Приложение не использует базу данных. Результаты и фоновые задачи живут только "
        "в текущей сессии; сохраняйте итоговые Excel/HTML/PDF-файлы."
    )
    if mode != "Расширенные методы" and st.sidebar.button("Начать заново", width="stretch"):
        reset_flow("design" if mode == "Проектирование пилота" else "analysis")
        st.rerun()


def design_step_1() -> None:
    st.subheader("Шаг 1. Паспорт пилота")
    st.write("Опишите бизнес-задачу. Эти поля попадут в итоговый Excel-отчёт.")
    current = st.session_state.passport
    with st.form("passport_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Название пилота *", value=current.get("Название пилота", ""))
            owner = st.text_input("Владелец", value=current.get("Владелец", ""))
            unit = st.text_input("Подразделение", value=current.get("Подразделение", ""))
        with c2:
            task = st.selectbox(
                "Тип банковской задачи",
                TASK_OPTIONS,
                index=TASK_OPTIONS.index(current.get("Тип задачи", TASK_OPTIONS[0]))
                if current.get("Тип задачи", TASK_OPTIONS[0]) in TASK_OPTIONS else 0,
            )
            start_date = st.date_input("Плановая дата старта")
            max_periods = st.number_input(
                "Максимальный срок пилота, периодов",
                min_value=0.1,
                value=float(current.get("Максимальный срок", 6.0)),
                step=0.5,
            )
        hypothesis = st.text_area(
            "Бизнес-гипотеза *",
            value=current.get("Гипотеза", ""),
            placeholder="Например: новая политика удержания снизит месячный churn без роста жалоб.",
        )
        action = st.text_area(
            "Что реально меняется для клиента или сотрудника?",
            value=current.get("Бизнес-действие", ""),
            placeholder="Например: uplift-модель выбирает клиентов для удерживающего звонка.",
        )
        submitted = st.form_submit_button("Сохранить паспорт", width="stretch")
    if submitted:
        if not name.strip() or not hypothesis.strip():
            st.error("Заполните название пилота и гипотезу.")
        else:
            st.session_state.passport = {
                "Название пилота": name.strip(),
                "Владелец": owner.strip(),
                "Подразделение": unit.strip(),
                "Тип задачи": task,
                "Плановая дата старта": str(start_date),
                "Максимальный срок": max_periods,
                "Гипотеза": hypothesis.strip(),
                "Бизнес-действие": action.strip(),
            }
            st.success("Паспорт сохранён.")
    nav_buttons("design", 4, disable_next=not bool(st.session_state.passport))


def design_step_2() -> None:
    st.subheader("Шаг 2. Выбор дизайна")
    st.write("Выберите простое описание эксперимента. Сложные термины знать не требуется.")
    design_type = st.radio(
        "Что сравниваем?",
        DESIGN_OPTIONS,
        key="_design_type_widget",
    )
    st.session_state.design_type = design_type
    descriptions = {
        "A/B: бинарная метрика": "Две группы; результат — событие 0/1: покупка, churn, default, отклик.",
        "A/B: непрерывная метрика": "Две группы; результат — сумма или число: выручка, маржа, LTV, время обработки.",
        "Multi-arm: бинарная метрика": "Несколько каналов или офферов сравниваются с одним контролем.",
        "Анализ uplift-модели": "Оценка ранжирования uplift по загруженным экспериментальным данным.",
    }
    st.info(descriptions[design_type])
    if design_type == "Анализ uplift-модели":
        st.warning(
            "Планирование размера uplift-эксперимента в MVP выполняется через будущий online A/B итоговой политики. "
            "Калибровка Qini/AUUC доступна в режиме анализа результатов."
        )
    use_sequential = st.checkbox(
        "Запланировать промежуточные анализы и возможность ранней остановки",
        key="_use_sequential_widget",
    )
    has_preperiod = st.checkbox(
        "Есть pre-period данные или исторический риск-скор для CUPED/CUPAC",
        key="_has_preperiod_widget",
    )
    # Храним значения отдельно от ключей виджетов. Иначе Streamlit может очистить
    # состояние checkbox, когда пользователь перейдёт на следующий шаг мастера.
    st.session_state.use_sequential = bool(use_sequential)
    st.session_state.has_preperiod = bool(has_preperiod)
    st.caption(
        "Sequential не означает ежедневное подглядывание. Точки анализа и правила остановки фиксируются до старта."
    )
    nav_buttons("design", 4)


def design_step_3() -> None:
    st.subheader("Шаг 3. Входные параметры")
    design_type = st.session_state.design_type
    task = st.session_state.passport.get("Тип задачи", "")
    max_periods = float(st.session_state.passport.get("Максимальный срок", 6.0))

    with st.form("design_parameters"):
        c1, c2, c3 = st.columns(3)
        with c1:
            alpha = st.number_input("Alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.005)
            power = st.number_input("Мощность", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
        with c2:
            sided_label = st.selectbox("Гипотеза", ["Односторонняя", "Двусторонняя"])
            sided = "one-sided" if sided_label == "Односторонняя" else "two-sided"
            clients = st.number_input("Доступный трафик за период", min_value=10, value=10_000, step=100)
        with c3:
            treatment_share = st.slider("Доля treatment", 0.10, 0.90, 0.50, 0.05)
            vr = st.slider("Ожидаемое снижение дисперсии", 0.0, 0.6, 0.0, 0.05)

        payload: dict[str, object] = {
            "alpha": alpha,
            "power": power,
            "sided": sided,
            "clients_per_period": int(clients),
            "treatment_share": treatment_share,
            "variance_reduction": vr,
        }

        if design_type == "A/B: бинарная метрика":
            p0_pct = st.number_input("Baseline в контроле, %", min_value=0.0001, max_value=99.0, value=0.5, step=0.1)
            effect_mode = st.radio("Как задать эффект?", ["Относительное изменение", "Значение treatment"], horizontal=True)
            if effect_mode == "Относительное изменение":
                rel_pct = st.number_input(
                    "Ожидаемое относительное изменение, %",
                    value=-20.0 if "Churn" in task else 20.0,
                    step=1.0,
                )
                p1 = p0_pct / 100 * (1 + rel_pct / 100)
            else:
                p1_pct = st.number_input("Ожидаемое значение treatment, %", min_value=0.0001, max_value=99.0, value=0.4, step=0.1)
                p1 = p1_pct / 100
            payload.update({"p_control": p0_pct / 100, "p_treatment": p1})

        elif design_type == "A/B: непрерывная метрика":
            mean0 = st.number_input("Среднее в контроле", value=1000.0, step=10.0)
            mean1 = st.number_input("Ожидаемое среднее в treatment", value=1050.0, step=10.0)
            std = st.number_input("Стандартное отклонение", min_value=0.0001, value=500.0, step=10.0)
            payload.update({"mean_control": mean0, "mean_treatment": mean1, "std": std})

        elif design_type == "Multi-arm: бинарная метрика":
            p0_pct = st.number_input("Baseline в контроле, %", min_value=0.0001, max_value=99.0, value=5.0, step=0.1)
            arms_count = st.number_input("Количество treatment-веток", min_value=2, max_value=8, value=3)
            correction = st.selectbox("Коррекция сравнений", ["holm", "bonferroni", "none"])
            treatment_rates: dict[str, float] = {}
            cols = st.columns(min(int(arms_count), 4))
            for idx in range(int(arms_count)):
                with cols[idx % len(cols)]:
                    value = st.number_input(
                        f"Ожидаемая метрика ветки {idx+1}, %",
                        min_value=0.0001,
                        max_value=99.0,
                        value=5.5 + idx * 0.2,
                        step=0.1,
                        key=f"arm_rate_{idx}",
                    )
                    treatment_rates[f"Ветка {idx+1}"] = value / 100
            payload.update({"p_control": p0_pct / 100, "treatment_rates": treatment_rates, "correction": correction})

        else:
            st.info("Для uplift выберите параметры будущего A/B итоговой политики или перейдите в анализ результатов.")
            p0_pct = st.number_input("Baseline итоговой метрики, %", min_value=0.0001, max_value=99.0, value=5.0, step=0.1)
            rel_pct = st.number_input("Ожидаемый прирост policy value/response, %", value=15.0, step=1.0)
            payload.update({"p_control": p0_pct / 100, "p_treatment": p0_pct / 100 * (1 + rel_pct / 100)})

        calculate = st.form_submit_button("Рассчитать дизайн", width="stretch")

    if calculate:
        try:
            if design_type in {"A/B: бинарная метрика", "Анализ uplift-модели"}:
                result = fixed_binary_design(
                    payload["p_control"], payload["p_treatment"],
                    alpha=alpha, power=power, treatment_share=treatment_share,
                    sided=sided, clients_per_period=int(clients),
                )
                design_df = design_to_frame(result)
                periods = result.periods * (1 - vr)
                expected_min = min(result.expected_events_control, result.expected_events_treatment)
                groups = 2
            elif design_type == "A/B: непрерывная метрика":
                result = fixed_continuous_design(
                    payload["mean_control"], payload["mean_treatment"], payload["std"],
                    alpha=alpha, power=power, treatment_share=treatment_share,
                    sided=sided, clients_per_period=int(clients),
                )
                design_df = design_to_frame(result)
                periods = result.periods * (1 - vr)
                expected_min = None
                groups = 2
            else:
                design_df = multiarm_binary_design(
                    payload["p_control"], payload["treatment_rates"],
                    alpha=alpha, power=power, sided=sided,
                    correction=payload["correction"], clients_per_period=int(clients),
                )
                periods = float(design_df["periods"].iloc[0]) * (1 - vr)
                expected_min = float((design_df["n_treatment"] * design_df["p_treatment"]).min())
                groups = len(payload["treatment_rates"]) + 1

            n_total = int(
                design_df["n_total"].iloc[0]
                if "n_total" in design_df.columns
                else design_df["recommended_total_n"].iloc[0]
            )
            scenarios = variance_reduction_scenarios(n_total, int(clients))
            recs = design_recommendations(
                task_type=task,
                metric_type="continuous" if "непрерывная" in design_type else "binary",
                periods=periods,
                max_periods=max_periods,
                expected_events_min=expected_min,
                use_sequential=st.session_state.use_sequential,
                has_preperiod=st.session_state.has_preperiod,
                groups=groups,
            )
            st.session_state.design_result = design_df
            st.session_state.scenario_result = scenarios
            st.session_state.design_recommendations = recs
            st.session_state.design_payload = payload
            st.success("Расчёт выполнен. Перейдите к результату.")
        except Exception as exc:
            st.error(f"Не удалось рассчитать дизайн: {exc}")
    nav_buttons("design", 4, disable_next=st.session_state.design_result is None)


def design_step_4() -> None:
    st.subheader("Шаг 4. Результат проектирования")
    if st.session_state.design_result is None:
        st.warning("Сначала рассчитайте параметры на предыдущем шаге.")
        nav_buttons("design", 4)
        return

    design_df = st.session_state.design_result
    scenarios = st.session_state.scenario_result
    if "n_total" in design_df.columns:
        row = design_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего клиентов", f"{int(row['n_total']):,}".replace(",", " "))
        c2.metric("Контроль", f"{int(row['n_control']):,}".replace(",", " "))
        c3.metric("Treatment", f"{int(row['n_treatment']):,}".replace(",", " "))
        c4.metric("Периодов", f"{row['periods']:.1f}")
    else:
        total = int(design_df["recommended_total_n"].iloc[0])
        periods = float(design_df["periods"].iloc[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего клиентов", f"{total:,}".replace(",", " "))
        c2.metric("Экспериментальных групп", len(design_df) + 1)
        c3.metric("Периодов", f"{periods:.1f}")

    st.markdown("### Расчёт")
    st.dataframe(design_df, width="stretch", hide_index=True)

    st.markdown("### Сценарии снижения дисперсии")
    chart_df = scenarios.set_index("variance_reduction")[["periods"]]
    st.line_chart(chart_df)
    st.dataframe(scenarios, width="stretch", hide_index=True)

    if st.session_state.use_sequential:
        st.markdown("### План промежуточных анализов")
        boundaries = obrien_fleming_boundaries(
            alpha=float(st.session_state.design_payload["alpha"]),
            sided=str(st.session_state.design_payload["sided"]),
        )
        st.dataframe(boundaries, width="stretch", hide_index=True)
        st.caption(
            "Границы являются планировочной аппроксимацией. Для финального протокола редких событий нужна симуляционная калибровка."
        )

    st.markdown("### Рекомендации")
    for item in st.session_state.design_recommendations:
        st.write("•", item)

    passport = dict(st.session_state.passport)
    passport["Тип дизайна"] = st.session_state.design_type
    excel_bytes = build_excel_report(
        passport=passport,
        design=design_df,
        scenarios=scenarios,
        recommendations=st.session_state.design_recommendations,
    )
    st.download_button(
        "⬇️ Скачать Excel-паспорт и расчёт",
        data=excel_bytes,
        file_name="pilot_design.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    nav_buttons("design", 4)


def analysis_step_1() -> None:
    st.subheader("Шаг 1. Загрузка результатов")
    st.write(
        "Загрузите обезличенный CSV/XLSX. Для базового анализа нужны колонки группы и результата."
    )
    template = ASSETS / "pilot_data_template.xlsx"
    sample = ASSETS / "pilot_sample_data.xlsx"
    c1, c2 = st.columns(2)
    if template.exists():
        c1.download_button(
            "Скачать шаблон Excel",
            template.read_bytes(),
            file_name=template.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    if sample.exists():
        c2.download_button(
            "Скачать демонстрационные данные",
            sample.read_bytes(),
            file_name=sample.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    uploaded = st.file_uploader("Файл результатов", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            df = read_uploaded_file(uploaded)
            st.session_state.uploaded_df = df
            st.success(f"Загружено строк: {len(df):,}".replace(",", " "))
            st.dataframe(df.head(50), width="stretch")
        except Exception as exc:
            st.error(f"Ошибка чтения файла: {exc}")
    nav_buttons("analysis", 3, disable_next=st.session_state.uploaded_df is None)


def analysis_step_2() -> None:
    st.subheader("Шаг 2. Настройка анализа")
    df = st.session_state.uploaded_df
    if df is None:
        st.warning("Сначала загрузите файл.")
        nav_buttons("analysis", 3)
        return
    columns = list(df.columns)
    with st.form("analysis_settings"):
        analysis_kind = st.selectbox(
            "Тип анализа",
            ["A/B или multi-arm", "Uplift-калибровка"],
        )
        if analysis_kind == "A/B или multi-arm":
            metric_type = st.selectbox("Тип метрики", ["binary", "continuous"])
            group_col = st.selectbox("Колонка группы", columns, index=columns.index("group") if "group" in columns else 0)
            outcome_col = st.selectbox("Колонка результата", columns, index=columns.index("outcome") if "outcome" in columns else min(1, len(columns)-1))
            id_options = ["— не использовать —"] + columns
            id_col_raw = st.selectbox("Идентификатор клиента", id_options)
            control_values = sorted(df[group_col].dropna().astype(str).unique())
            control_label = st.selectbox("Значение контрольной группы", control_values)
            correction = st.selectbox("Коррекция multi-arm", ["holm", "bonferroni", "fdr_bh"])
            alpha = st.number_input("Alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.005)
            settings = {
                "kind": analysis_kind, "metric_type": metric_type,
                "group_col": group_col, "outcome_col": outcome_col,
                "id_col": None if id_col_raw.startswith("—") else id_col_raw,
                "control_label": control_label, "correction": correction, "alpha": alpha,
            }
        else:
            treatment_col = st.selectbox("Колонка treatment 0/1", columns, index=columns.index("treatment") if "treatment" in columns else 0)
            outcome_col = st.selectbox("Колонка результата 0/1", columns, index=columns.index("outcome") if "outcome" in columns else min(1, len(columns)-1))
            score_col = st.selectbox("Колонка прогнозного uplift", columns, index=columns.index("predicted_uplift") if "predicted_uplift" in columns else min(2, len(columns)-1))
            bins = st.slider("Количество групп калибровки", 5, 20, 10)
            settings = {
                "kind": analysis_kind, "treatment_col": treatment_col,
                "outcome_col": outcome_col, "score_col": score_col, "bins": bins,
            }
        run = st.form_submit_button("Выполнить анализ", width="stretch")

    if run:
        try:
            if settings["kind"] == "A/B или multi-arm":
                validation = validate_dataset(
                    df, group_col=settings["group_col"], outcome_col=settings["outcome_col"],
                    id_col=settings["id_col"], metric_type=settings["metric_type"],
                )
                if validation.errors:
                    for error in validation.errors:
                        st.error(error)
                else:
                    result = analyze_experiment(
                        df,
                        group_col=settings["group_col"],
                        outcome_col=settings["outcome_col"],
                        control_label=str(settings["control_label"]),
                        metric_type=settings["metric_type"],
                        alpha=settings["alpha"],
                        correction=settings["correction"],
                    )
                    result["warnings"] = validation.warnings + result["warnings"]
                    st.session_state.analysis_result = result
                    st.session_state.uplift_result = None
                    st.session_state.analysis_settings = settings
                    st.success("Анализ выполнен.")
            else:
                uplift = analyze_uplift(
                    df,
                    treatment_col=settings["treatment_col"],
                    outcome_col=settings["outcome_col"],
                    score_col=settings["score_col"],
                    bins=settings["bins"],
                )
                st.session_state.uplift_result = uplift
                st.session_state.analysis_result = None
                st.session_state.analysis_settings = settings
                st.success("Uplift-анализ выполнен.")
        except Exception as exc:
            st.error(f"Не удалось выполнить анализ: {exc}")
    ready = st.session_state.analysis_result is not None or st.session_state.uplift_result is not None
    nav_buttons("analysis", 3, disable_next=not ready)


def analysis_step_3() -> None:
    st.subheader("Шаг 3. Результаты пилота")
    result = st.session_state.analysis_result
    uplift = st.session_state.uplift_result
    recommendations: list[str] = []

    if result is not None:
        st.markdown("### Сводка по группам")
        st.dataframe(result["group_summary"], width="stretch", hide_index=True)
        st.markdown("### Оценка эффекта")
        st.dataframe(result["results"], width="stretch", hide_index=True)
        st.markdown("### Качество данных")
        st.dataframe(result["quality"], width="stretch", hide_index=True)
        settings = st.session_state.analysis_settings
        recommendations = result_interpretation(
            result["results"], result["warnings"], alpha=settings["alpha"]
        )
        for message in recommendations:
            if "Mismatch" in message or "проверьте" in message.lower():
                st.warning(message)
            else:
                st.info(message)

    if uplift is not None:
        st.metric("Средний наблюдаемый uplift", f"{uplift['overall_uplift']:.2%}")
        calibration = uplift["calibration"]
        st.dataframe(calibration, width="stretch", hide_index=True)
        st.line_chart(calibration.set_index("uplift_bin")[["observed_uplift", "predicted_uplift_mean"]])
        recommendations = [
            "Проверяйте uplift на независимой выборке или через cross-fitting.",
            "Решение о внедрении принимайте по incremental profit/policy value, а не только по AUUC/Qini.",
        ] + list(uplift.get("warnings", []))
        for message in recommendations:
            st.info(message)

    if result is None and uplift is None:
        st.warning("Нет рассчитанного результата.")
        nav_buttons("analysis", 3)
        return

    passport = {
        "Название пилота": "Анализ загруженного пилота",
        "Источник": "Excel/CSV, загруженный в UI",
        "Тип анализа": st.session_state.analysis_settings.get("kind"),
    }
    excel_bytes = build_excel_report(
        passport=passport,
        analysis=result,
        uplift=uplift,
        recommendations=recommendations,
    )
    st.download_button(
        "⬇️ Скачать Excel-отчёт по пилоту",
        data=excel_bytes,
        file_name="pilot_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    nav_buttons("analysis", 3)


def main() -> None:
    initialize_state()
    render_header()
    render_sidebar()
    if st.session_state.mode == "Проектирование пилота":
        {1: design_step_1, 2: design_step_2, 3: design_step_3, 4: design_step_4}[st.session_state.design_step]()
    elif st.session_state.mode == "Анализ результатов":
        {1: analysis_step_1, 2: analysis_step_2, 3: analysis_step_3}[st.session_state.analysis_step]()
    else:
        render_advanced()


if __name__ == "__main__":
    main()

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from experiment_core.background import submit_job
from experiment_core.bayesian import (
    PriorSpec,
    bayesian_prior_sensitivity,
    bayesian_two_group_binary,
    beta_binomial_predictive_probability,
    beta_prior_from_mean_ess,
    prior_predictive_binary,
)
from experiment_core.sequential import (
    calibrate_exact_sequential,
    calibrate_gaussian_group_sequential,
    evaluate_exact_sequential_path,
    sequential_monitoring_table,
)
from experiment_core.variance_reduction import cupac_analysis, cuped_analysis
from experiment_core.survival import analyze_competing_risks, analyze_recurrent_events, analyze_survival
from experiment_core.causal_designs import (
    analyze_cluster_period_design,
    cluster_randomized_design,
    dose_response,
    generate_switchback_schedule,
    regression_discontinuity,
    stepped_wedge_schedule,
    synthetic_control,
)
from experiment_core.uplift_advanced import (
    doubly_robust_policy_value,
    optimize_capacity_nba,
    qini_auuc_analysis,
    uplift_calibration,
)
from experiment_core.bandits_ranking import contextual_bandit_offline_evaluation, interleaving_analysis
from experiment_core.reporting import (
    build_advanced_excel_report,
    build_html_protocol,
    build_pdf_protocol,
)


METHODS = [
    "Bayesian monitoring",
    "Sequential и exact-sequential",
    "CUPED / CUPAC по загруженным данным",
    "Survival, RMST и non-proportional hazards",
    "Competing risks",
    "Recurrent events",
    "Кластерный / stepped-wedge / switchback",
    "Synthetic control",
    "Regression discontinuity",
    "Interleaving для ranking",
    "Continuous treatment / dose-response",
    "Contextual bandits",
    "Qini / AUUC с bootstrap",
    "Doubly robust policy value",
    "Capacity-aware Next Best Action",
    "HTML / PDF / Excel протокол",
]


def initialize_advanced_state() -> None:
    defaults = {
        "advanced_results": {},
        "advanced_jobs": {},
        "advanced_passport": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _read_file(uploaded: Any, sheet_name: str | int = 0) -> pd.DataFrame:
    raw = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(raw))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(raw), sheet_name=sheet_name)
    raise ValueError("Поддерживаются CSV, XLSX и XLS.")


def _upload_df(key: str, help_text: str = "") -> pd.DataFrame | None:
    uploaded = st.file_uploader(
        "Загрузите CSV/XLSX",
        type=["csv", "xlsx", "xls"],
        key=f"upload_{key}",
        help=help_text or "Файл используется только в текущей сессии и не записывается в БД.",
    )
    if uploaded is None:
        return None
    try:
        df = _read_file(uploaded)
        st.caption(f"Загружено строк: {len(df):,}; колонок: {len(df.columns)}")
        with st.expander("Предпросмотр данных", expanded=False):
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        return df
    except Exception as exc:
        st.error(f"Не удалось прочитать файл: {exc}")
        return None


def _store(name: str, result: Mapping[str, Any]) -> None:
    st.session_state.advanced_results[name] = dict(result)


def _show_warnings(result: Mapping[str, Any]) -> None:
    for warning in result.get("warnings", []) if isinstance(result, Mapping) else []:
        st.warning(str(warning))


def _display_frames(result: Mapping[str, Any], *, exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    for key, value in result.items():
        if key in exclude or key in {"warnings", "draws"}:
            continue
        if isinstance(value, pd.DataFrame):
            st.markdown(f"#### {key.replace('_', ' ').title()}")
            st.dataframe(value, use_container_width=True, hide_index=True)
        elif isinstance(value, Mapping) and all(np.isscalar(v) or v is None for v in value.values()):
            st.markdown(f"#### {key.replace('_', ' ').title()}")
            st.json(dict(value))
    _show_warnings(result)


def _background_submit(job_key: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    jobs = st.session_state.advanced_jobs
    future = jobs.get(job_key)
    if future is None:
        if st.button("Запустить тяжёлый расчёт в фоне", key=f"launch_{job_key}", type="primary"):
            jobs[job_key] = submit_job(function, *args, **kwargs)
            st.info("Расчёт запущен в отдельном процессе. Нажмите «Обновить статус» через некоторое время.")
            st.rerun()
        return
    if future.done():
        try:
            result = future.result()
            jobs.pop(job_key, None)
            st.success("Фоновый расчёт завершён.")
            st.session_state[f"job_result_{job_key}"] = result
        except Exception as exc:
            jobs.pop(job_key, None)
            st.error(f"Фоновый расчёт завершился ошибкой: {exc}")
    else:
        st.info("Расчёт выполняется в фоне. Можно продолжать работать с другими разделами.")
        if st.button("Обновить статус", key=f"poll_{job_key}"):
            st.rerun()


def render_bayesian() -> None:
    st.subheader("Bayesian monitoring и predictive probability")
    st.write(
        "Используйте этот раздел, когда бизнесу важны вероятности понятного вида: "
        "«насколько вероятно, что treatment лучше» и «дойдёт ли пилот до успеха при максимальной выборке»."
    )
    with st.form("bayesian_form"):
        c1, c2 = st.columns(2)
        with c1:
            n_c = st.number_input("Control: клиентов", 1, value=30_000)
            x_c = st.number_input("Control: событий", 0, max_value=int(n_c), value=min(150, int(n_c)))
            n_t = st.number_input("Treatment: клиентов", 1, value=30_000)
            x_t = st.number_input("Treatment: событий", 0, max_value=int(n_t), value=min(120, int(n_t)))
        with c2:
            direction = st.selectbox("Что считается улучшением?", ["Снижение метрики", "Рост метрики"])
            min_effect = st.number_input("Минимальный относительный эффект", 0.0, 2.0, 0.10, 0.01)
            prior_mode = st.selectbox("Prior", ["Jeffreys", "Исторический mean + ESS"])
            historical_mean = st.number_input("Историческая частота", 0.000001, 0.999999, 0.005, format="%.6f")
            prior_ess = st.number_input("Prior ESS", 1.0, 1_000_000.0, 100.0, 10.0)
            draws = st.select_slider("Posterior draws", options=[10_000, 30_000, 100_000, 300_000], value=100_000)
        st.markdown("**Predictive probability**")
        c3, c4 = st.columns(2)
        with c3:
            n_c_max = st.number_input("Максимум control", int(n_c), value=max(int(n_c), 55_000))
            n_t_max = st.number_input("Максимум treatment", int(n_t), value=max(int(n_t), 55_000))
        with c4:
            posterior_success = st.number_input("Порог posterior success", 0.50, 0.9999, 0.975, 0.005)
            predictive_sims = st.select_slider("Predictive simulations", [500, 1_000, 3_000, 10_000], value=3_000)
        run = st.form_submit_button("Рассчитать Bayesian-анализ", use_container_width=True)
    if run:
        prior = PriorSpec("Jeffreys", 0.5, 0.5) if prior_mode == "Jeffreys" else beta_prior_from_mean_ess(historical_mean, prior_ess)
        direction_code = "decrease" if direction == "Снижение метрики" else "increase"
        res = bayesian_two_group_binary(
            int(x_c), int(n_c), int(x_t), int(n_t), prior_control=prior, prior_treatment=prior,
            minimum_relative_effect=min_effect, benefit_direction=direction_code, draws=int(draws),
        )
        sensitivity_priors = [
            PriorSpec("Jeffreys", 0.5, 0.5),
            beta_prior_from_mean_ess(historical_mean, min(100, prior_ess), "Historical ESS<=100"),
            beta_prior_from_mean_ess(historical_mean, prior_ess, f"Historical ESS={prior_ess:g}"),
        ]
        res["prior_sensitivity"] = bayesian_prior_sensitivity(
            int(x_c), int(n_c), int(x_t), int(n_t), sensitivity_priors,
            minimum_relative_effect=min_effect, benefit_direction=direction_code, draws=min(int(draws), 30_000),
        )
        res["prior_predictive"] = pd.DataFrame([
            prior_predictive_binary(prior, int(n_c)),
        ])
        res["predictive_probability"] = pd.DataFrame([
            beta_binomial_predictive_probability(
                int(x_c), int(n_c), int(x_t), int(n_t), int(n_c_max), int(n_t_max),
                prior=prior, success_probability_threshold=posterior_success,
                required_relative_effect=min_effect, benefit_direction=direction_code,
                outer_simulations=int(predictive_sims),
            )
        ])
        _store("Bayesian monitoring", res)
    result = st.session_state.advanced_results.get("Bayesian monitoring")
    if result:
        eff = result["effect_summary"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("P(treatment лучше)", f"{eff['probability_treatment_better']:.1%}")
        c2.metric("P(эффект ≥ порога)", f"{eff['probability_effect_at_least_threshold']:.1%}")
        pp = result["predictive_probability"].iloc[0]
        c3.metric("Predictive probability", f"{pp['predictive_probability']:.1%}")
        _display_frames(result)


def render_sequential() -> None:
    st.subheader("Sequential monitoring и exact-sequential калибровка")
    mode = st.radio("Режим", ["Gaussian group-sequential", "Exact-sequential для редких событий"], horizontal=True)
    if mode == "Gaussian group-sequential":
        st.caption("Подходит, когда событий достаточно для Z-аппроксимации. Точки анализа фиксируются заранее.")
        looks = st.slider("Число interim-анализов", 2, 6, 3)
        info = np.linspace(1/looks, 1, looks)
        c1, c2, c3 = st.columns(3)
        alpha = c1.number_input("Alpha", 0.001, 0.20, 0.05, 0.005, key="gs_alpha")
        family = c2.selectbox("Границы", ["obf", "pocock"])
        simulations = c3.select_slider("Калибровочные симуляции", [20_000, 50_000, 200_000, 500_000], value=200_000)
        if st.button("Калибровать границы", type="primary"):
            result = {"boundaries": calibrate_gaussian_group_sequential(info, alpha=alpha, family=family, simulations=simulations)}
            _store("Sequential analysis", result)
        result = st.session_state.advanced_results.get("Sequential analysis")
        if result:
            st.dataframe(result["boundaries"], use_container_width=True, hide_index=True)
            uploaded = _upload_df("sequential_path", "Накопленная таблица: x_control, n_control, x_treatment, n_treatment — по одному ряду на interim.")
            if uploaded is not None:
                cols = list(uploaded.columns)
                with st.form("seq_path_form"):
                    xc = st.selectbox("События control", cols)
                    nc = st.selectbox("Размер control", cols, index=min(1, len(cols)-1))
                    xt = st.selectbox("События treatment", cols, index=min(2, len(cols)-1))
                    nt = st.selectbox("Размер treatment", cols, index=min(3, len(cols)-1))
                    direction = st.selectbox("Улучшение", ["increase", "decrease"])
                    assumed_z = st.number_input("Ожидаемый финальный Z для conditional power (0 = не считать)", 0.0, 10.0, 2.5)
                    run = st.form_submit_button("Проанализировать накопленный путь")
                if run:
                    monitored = sequential_monitoring_table(
                        uploaded, x_control_col=xc, n_control_col=nc, x_treatment_col=xt, n_treatment_col=nt,
                        boundaries=result["boundaries"], benefit_direction=direction,
                        assumed_effect_z_final=assumed_z if assumed_z > 0 else None,
                    )
                    result["monitoring"] = monitored
                    _store("Sequential analysis", result)
                    st.dataframe(monitored, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Калибровка симулирует весь путь многократных exact-тестов под H0. Это необходимо: "
            "обычный Fisher p<0.05 на каждом interim не контролирует общий alpha."
        )
        with st.form("exact_seq_form"):
            looks = st.slider("Число interim", 2, 5, 3)
            final_nc = st.number_input("Финальный размер control", 20, value=2_000)
            final_nt = st.number_input("Финальный размер treatment", 20, value=2_000)
            p_null = st.number_input("Частота события под H0", 0.000001, 0.999999, 0.002, format="%.6f")
            alpha = st.number_input("Общий alpha", 0.001, 0.20, 0.05, 0.005, key="exact_alpha")
            method = st.selectbox("Exact test", ["fisher", "boschloo", "barnard"])
            direction = st.selectbox("Улучшение", ["increase", "decrease"], key="exact_direction")
            sims = st.select_slider("Симуляции", [2_000, 5_000, 20_000, 50_000], value=5_000)
            submit = st.form_submit_button("Подготовить фоновую калибровку")
        if submit:
            st.session_state["exact_config"] = {
                "nc": np.ceil(np.linspace(final_nc/looks, final_nc, looks)).astype(int),
                "nt": np.ceil(np.linspace(final_nt/looks, final_nt, looks)).astype(int),
                "p_null": p_null, "alpha": alpha, "method": method, "direction": direction, "sims": sims,
            }
        cfg = st.session_state.get("exact_config")
        if cfg:
            _background_submit(
                "exact_seq", calibrate_exact_sequential, cfg["nc"], cfg["nt"],
                p_null=cfg["p_null"], alpha=cfg["alpha"], method=cfg["method"],
                benefit_direction=cfg["direction"], simulations=cfg["sims"],
            )
        job_res = st.session_state.pop("job_result_exact_seq", None)
        if job_res is not None:
            cleaned = {k: v for k, v in job_res.items() if k != "null_min_p_distribution"}
            _store("Exact sequential", cleaned)
        result = st.session_state.advanced_results.get("Exact sequential")
        if result:
            c1, c2 = st.columns(2)
            c1.metric("Калиброванный exact p-threshold", f"{result['p_threshold']:.6f}")
            c2.metric("Оценённый Type I error", f"{result['estimated_type_i_error']:.2%}")
            st.dataframe(result["looks"], use_container_width=True, hide_index=True)
            path = _upload_df("exact_path", "Накопленные x_control, n_control, x_treatment, n_treatment.")
            if path is not None:
                cols = list(path.columns)
                with st.form("exact_path_cols"):
                    xc = st.selectbox("x_control", cols)
                    nc = st.selectbox("n_control", cols, index=min(1, len(cols)-1))
                    xt = st.selectbox("x_treatment", cols, index=min(2, len(cols)-1))
                    nt = st.selectbox("n_treatment", cols, index=min(3, len(cols)-1))
                    run = st.form_submit_button("Применить exact-границу")
                if run:
                    monitoring = evaluate_exact_sequential_path(
                        path, x_control_col=xc, n_control_col=nc, x_treatment_col=xt, n_treatment_col=nt,
                        p_threshold=result["p_threshold"], method=result["method"], benefit_direction=cfg["direction"],
                    )
                    result["monitoring"] = monitoring
                    _store("Exact sequential", result)
                    st.dataframe(monitoring, use_container_width=True, hide_index=True)


def render_cuped_cupac() -> None:
    st.subheader("Фактический CUPED / CUPAC по загруженным данным")
    st.write("Используйте только признаки, рассчитанные до назначения treatment. Post-treatment признаки создают leakage.")
    df = _upload_df("cuped")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("cuped_form"):
        c1, c2 = st.columns(2)
        outcome = c1.selectbox("Outcome", cols)
        treatment = c2.selectbox("Treatment 0/1", cols, index=min(1, len(cols)-1))
        method = st.radio("Метод", ["CUPED / ANCOVA", "CUPAC cross-fit"], horizontal=True)
        features = st.multiselect("Pre-period признаки", [c for c in cols if c not in {outcome, treatment}])
        metric = st.selectbox("Тип outcome", ["continuous", "binary"])
        folds = st.slider("Cross-fitting folds", 2, 10, 5)
        run = st.form_submit_button("Оценить снижение дисперсии", use_container_width=True)
    if run:
        if not features:
            st.error("Выберите хотя бы один pre-period признак.")
        else:
            result = cuped_analysis(df, outcome_col=outcome, treatment_col=treatment, preperiod_cols=features) if method.startswith("CUPED") else cupac_analysis(
                df, outcome_col=outcome, treatment_col=treatment, feature_cols=features, metric_type=metric, folds=folds
            )
            _store(method, result)
    result = st.session_state.advanced_results.get(method)
    if result:
        st.metric("Оценённое снижение дисперсии", f"{result['variance_reduction']:.1%}")
        st.metric("Ориентировочный multiplier выборки", f"{result['sample_size_multiplier']:.3f}")
        _display_frames(result, exclude={"predictions"})


def render_survival() -> None:
    st.subheader("Survival, RMST и non-proportional hazards")
    df = _upload_df("survival")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("survival_form"):
        duration = st.selectbox("Время до события / цензурирования", cols)
        event = st.selectbox("Event 0/1", cols, index=min(1, len(cols)-1))
        group = st.selectbox("Группа", cols, index=min(2, len(cols)-1))
        labels = sorted(df[group].dropna().astype(str).unique()) if group in df else []
        control = st.selectbox("Control", labels)
        treatment = st.selectbox("Treatment", [x for x in labels if x != control])
        tau = st.number_input("RMST horizon tau", 0.0001, value=float(pd.to_numeric(df[duration], errors="coerce").quantile(0.8)))
        milestone = st.number_input("Milestone", 0.0001, value=float(tau))
        bootstrap = st.select_slider("Bootstrap", [100, 300, 500, 1_000, 3_000], value=500)
        run = st.form_submit_button("Выполнить survival-анализ", use_container_width=True)
    if run:
        result = analyze_survival(df, duration_col=duration, event_col=event, group_col=group,
                                  control_label=control, treatment_label=treatment, tau=tau, milestone=milestone, bootstrap=bootstrap)
        _store("Survival RMST", result)
    result = st.session_state.advanced_results.get("Survival RMST")
    if result:
        fig = px.line(result["curves"], x="time", y="survival", color="group", title="Kaplan-Meier survival curves")
        st.plotly_chart(fig, use_container_width=True)
        _display_frames(result, exclude={"curves"})


def render_competing() -> None:
    st.subheader("Competing risks")
    st.write("Пример: churn интересующего типа конкурирует со смертью продукта, миграцией или другим взаимоисключающим исходом.")
    df = _upload_df("competing")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("competing_form"):
        duration = st.selectbox("Duration", cols)
        event_type = st.selectbox("Event type: 0=censor, 1/2/...=events", cols, index=min(1, len(cols)-1))
        group = st.selectbox("Group", cols, index=min(2, len(cols)-1))
        labels = sorted(df[group].dropna().astype(str).unique())
        control = st.selectbox("Control", labels)
        treatment = st.selectbox("Treatment", [x for x in labels if x != control])
        event_interest = st.number_input("Код события интереса", 1, value=1)
        tau = st.number_input("Горизонт CIF", 0.0001, value=float(pd.to_numeric(df[duration], errors="coerce").quantile(0.8)))
        bootstrap = st.select_slider("Bootstrap", [100, 300, 500, 1_000], value=400)
        run = st.form_submit_button("Рассчитать competing risks")
    if run:
        result = analyze_competing_risks(
            df, duration_col=duration, event_type_col=event_type, group_col=group,
            control_label=control, treatment_label=treatment, event_of_interest=int(event_interest), tau=tau, bootstrap=bootstrap
        )
        _store("Competing risks", result)
    result = st.session_state.advanced_results.get("Competing risks")
    if result:
        st.plotly_chart(px.line(result["cif_curves"], x="time", y="cif", color="group", title="Cumulative incidence"), use_container_width=True)
        _display_frames(result, exclude={"cif_curves"})


def render_recurrent() -> None:
    st.subheader("Recurrent events")
    df = _upload_df("recurrent", "Ожидается counting-process формат: id, start, stop, event, treatment.")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("recurrent_form"):
        id_col = st.selectbox("ID субъекта", cols)
        start = st.selectbox("Start", cols, index=min(1, len(cols)-1))
        stop = st.selectbox("Stop", cols, index=min(2, len(cols)-1))
        event = st.selectbox("Event count / 0-1", cols, index=min(3, len(cols)-1))
        treatment = st.selectbox("Treatment", cols, index=min(4, len(cols)-1))
        model = st.selectbox("Модель", ["andersen-gill", "poisson", "negative-binomial"])
        run = st.form_submit_button("Проанализировать повторные события")
    if run:
        result = analyze_recurrent_events(df, id_col=id_col, start_col=start, stop_col=stop, event_col=event, treatment_col=treatment, model=model)
        _store("Recurrent events", result)
    result = st.session_state.advanced_results.get("Recurrent events")
    if result:
        _display_frames(result)


def render_cluster_designs() -> None:
    st.subheader("Кластерные, stepped-wedge и switchback дизайны")
    sub = st.radio("Задача", ["Расчёт design effect", "Создать stepped-wedge schedule", "Создать switchback schedule", "Проанализировать cluster-period данные"], horizontal=False)
    if sub == "Расчёт design effect":
        with st.form("cluster_design"):
            n = st.number_input("Выборка при индивидуальной рандомизации", 10, value=10_000)
            m = st.number_input("Средний размер кластера", 1.01, value=25.0)
            icc = st.number_input("ICC", 0.0, 0.999, 0.01, 0.005)
            cv = st.number_input("CV размера кластеров", 0.0, 5.0, 0.2, 0.1)
            attrition = st.number_input("Потери", 0.0, 0.90, 0.05, 0.01)
            run = st.form_submit_button("Рассчитать")
        if run:
            result = {"summary": pd.DataFrame([cluster_randomized_design(individual_sample_size=int(n), mean_cluster_size=m, icc=icc, coefficient_of_variation=cv, attrition=attrition)])}
            _store("Cluster design", result)
    elif sub == "Создать stepped-wedge schedule":
        c1, c2 = st.columns(2)
        clusters = c1.number_input("Кластеры", 2, value=12)
        periods = c2.number_input("Периоды", 3, value=7)
        if st.button("Создать schedule", key="sw_schedule"):
            result = {"schedule": stepped_wedge_schedule(int(clusters), int(periods))}
            _store("Stepped-wedge schedule", result)
    elif sub == "Создать switchback schedule":
        c1, c2, c3 = st.columns(3)
        clusters = c1.number_input("Кластеры", 1, value=8, key="sb_clusters")
        periods = c2.number_input("Периоды", 2, value=20, key="sb_periods")
        block = c3.number_input("Длина блока", 1, value=2)
        washout = st.number_input("Washout periods после переключения", 0, value=0)
        if st.button("Создать switchback", key="sb_schedule"):
            result = {"schedule": generate_switchback_schedule(int(clusters), int(periods), block_length=int(block), washout_periods=int(washout))}
            _store("Switchback schedule", result)
    else:
        df = _upload_df("cluster_analysis")
        if df is not None:
            cols = list(df.columns)
            with st.form("cluster_analysis_form"):
                outcome = st.selectbox("Outcome", cols)
                treatment = st.selectbox("Treatment", cols, index=min(1, len(cols)-1))
                cluster = st.selectbox("Cluster", cols, index=min(2, len(cols)-1))
                period = st.selectbox("Period", cols, index=min(3, len(cols)-1))
                metric = st.selectbox("Metric", ["continuous", "binary"])
                carry = st.selectbox("Carryover column (optional)", ["<нет>"] + cols)
                run = st.form_submit_button("Выполнить GEE-анализ")
            if run:
                result = analyze_cluster_period_design(df, outcome_col=outcome, treatment_col=treatment, cluster_col=cluster,
                                                       period_col=period, metric_type=metric, carryover_col=None if carry == "<нет>" else carry)
                _store("Cluster-period analysis", result)
    for name in ["Cluster design", "Stepped-wedge schedule", "Switchback schedule", "Cluster-period analysis"]:
        if name in st.session_state.advanced_results:
            st.markdown(f"### {name}")
            _display_frames(st.session_state.advanced_results[name])


def render_synthetic() -> None:
    st.subheader("Synthetic control")
    df = _upload_df("synthetic", "Long format: unit, time, outcome. Один unit получает intervention.")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("synthetic_form"):
        unit = st.selectbox("Unit", cols)
        time = st.selectbox("Time", cols, index=min(1, len(cols)-1))
        outcome = st.selectbox("Outcome", cols, index=min(2, len(cols)-1))
        treated = st.selectbox("Treated unit", sorted(df[unit].dropna().astype(str).unique()))
        times = sorted(df[time].dropna().unique())
        intervention = st.selectbox("Intervention time", times, index=max(1, len(times)//2))
        placebo = st.checkbox("Рассчитать placebo donors", True)
        run = st.form_submit_button("Построить synthetic control")
    if run:
        work = df.copy(); work[unit] = work[unit].astype(str)
        result = synthetic_control(work, unit_col=unit, time_col=time, outcome_col=outcome, treated_unit=treated,
                                   intervention_time=intervention, placebo=placebo)
        _store("Synthetic control", result)
    result = st.session_state.advanced_results.get("Synthetic control")
    if result:
        plot = result["curve"].melt(id_vars="time", value_vars=["treated", "synthetic"], var_name="series", value_name="outcome")
        st.plotly_chart(px.line(plot, x="time", y="outcome", color="series"), use_container_width=True)
        _display_frames(result, exclude={"curve"})


def render_rdd() -> None:
    st.subheader("Regression discontinuity")
    df = _upload_df("rdd")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("rdd_form"):
        outcome = st.selectbox("Outcome", cols)
        running = st.selectbox("Running variable", cols, index=min(1, len(cols)-1))
        cutoff = st.number_input("Cutoff", value=float(pd.to_numeric(df[running], errors="coerce").median()))
        bandwidth = st.number_input("Bandwidth", min_value=0.000001, value=float(pd.to_numeric(df[running], errors="coerce").std() or 1.0))
        treatment = st.selectbox("Treatment для fuzzy RDD (optional)", ["<нет>"] + cols)
        order = st.selectbox("Полиномиальный порядок", [1, 2])
        run = st.form_submit_button("Выполнить RDD")
    if run:
        result = regression_discontinuity(df, outcome_col=outcome, running_col=running, cutoff=cutoff, bandwidth=bandwidth,
                                          treatment_col=None if treatment == "<нет>" else treatment, polynomial_order=order)
        _store("Regression discontinuity", result)
    result = st.session_state.advanced_results.get("Regression discontinuity")
    if result:
        st.plotly_chart(px.scatter(result["plot_data"], x="running_centered", y=outcome, color="above", opacity=0.45), use_container_width=True)
        _display_frames(result, exclude={"plot_data"})


def render_interleaving() -> None:
    st.subheader("Interleaving для ranking")
    df = _upload_df("interleaving", "Одна строка = сессия/запрос; winner содержит A, B или tie.")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("interleaving_form"):
        winner = st.selectbox("Winner", cols)
        a = st.text_input("Label A", "A")
        b = st.text_input("Label B", "B")
        tie = st.text_input("Tie label", "tie")
        cluster = st.selectbox("Cluster/user (optional)", ["<нет>"] + cols)
        bootstrap = st.select_slider("Bootstrap", [500, 1_000, 2_000, 5_000], value=2_000)
        run = st.form_submit_button("Проанализировать interleaving")
    if run:
        result = interleaving_analysis(df, winner_col=winner, label_a=a, label_b=b, tie_label=tie,
                                       cluster_col=None if cluster == "<нет>" else cluster, bootstrap=bootstrap)
        _store("Interleaving", result)
    result = st.session_state.advanced_results.get("Interleaving")
    if result:
        _display_frames(result)


def render_dose_response() -> None:
    st.subheader("Continuous treatment и dose-response")
    df = _upload_df("dose")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("dose_form"):
        outcome = st.selectbox("Outcome", cols)
        dose = st.selectbox("Dose / цена / лимит / интенсивность", cols, index=min(1, len(cols)-1))
        covars = st.multiselect("Pre-treatment covariates", [c for c in cols if c not in {outcome, dose}])
        grid = st.slider("Точек на кривой", 20, 100, 40)
        bootstrap = st.select_slider("Bootstrap", [50, 100, 200, 500, 1_000], value=200)
        run = st.form_submit_button("Оценить dose-response")
    if run:
        result = dose_response(df, outcome_col=outcome, dose_col=dose, covariate_cols=covars, grid_points=grid, bootstrap=bootstrap)
        _store("Dose-response", result)
    result = st.session_state.advanced_results.get("Dose-response")
    if result:
        fig = px.line(result["curve"], x="dose", y="expected_outcome", title="Adjusted dose-response")
        fig.add_scatter(x=result["curve"]["dose"], y=result["curve"]["ci_low"], mode="lines", name="CI low", line={"dash":"dot"})
        fig.add_scatter(x=result["curve"]["dose"], y=result["curve"]["ci_high"], mode="lines", name="CI high", line={"dash":"dot"})
        st.plotly_chart(fig, use_container_width=True)
        _display_frames(result, exclude={"curve"})


def _mapping_editor(actions: list[str], cols: list[str], prefix: str) -> dict[str, str]:
    mapping = {}
    for action in actions:
        mapping[action] = st.selectbox(f"{prefix}: {action}", cols, key=f"{prefix}_{action}")
    return mapping


def render_bandits() -> None:
    st.subheader("Contextual bandits: offline policy evaluation")
    df = _upload_df("bandits")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("bandit_form"):
        action = st.selectbox("Logged action", cols)
        reward = st.selectbox("Reward", cols, index=min(1, len(cols)-1))
        propensity = st.selectbox("Behavior propensity of logged action", cols, index=min(2, len(cols)-1))
        target = st.selectbox("Target policy chosen action", cols, index=min(3, len(cols)-1))
        actions = sorted(df[action].dropna().astype(str).unique())
        st.markdown("Q-hat columns для DR (optional, но рекомендуется)")
        q_cols = {a: st.selectbox(f"q_hat[{a}]", ["<нет>"] + cols, key=f"bandit_q_{a}") for a in actions}
        bootstrap = st.select_slider("Bootstrap", [100, 300, 500, 1_000, 3_000], value=500)
        run = st.form_submit_button("Оценить bandit policy")
    if run:
        q_map = {a: col for a, col in q_cols.items() if col != "<нет>"}
        result = contextual_bandit_offline_evaluation(
            df, action_col=action, reward_col=reward, behavior_propensity_col=propensity,
            target_action_col=target, q_hat_cols=q_map if len(q_map) == len(actions) else None, bootstrap=bootstrap,
        )
        _store("Contextual bandits", result)
    result = st.session_state.advanced_results.get("Contextual bandits")
    if result:
        _display_frames(result)


def render_qini() -> None:
    st.subheader("Qini / AUUC с доверительными интервалами")
    df = _upload_df("qini")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("qini_form"):
        treatment = st.selectbox("Treatment 0/1", cols)
        outcome = st.selectbox("Outcome", cols, index=min(1, len(cols)-1))
        score = st.selectbox("Predicted uplift", cols, index=min(2, len(cols)-1))
        propensity_mode = st.radio("Propensity", ["Константа", "Колонка"], horizontal=True)
        propensity_const = st.number_input("Propensity constant", 0.001, 0.999, 0.5)
        propensity_col = st.selectbox("Propensity column", cols)
        cluster = st.selectbox("Cluster/client for bootstrap (optional)", ["<нет>"] + cols)
        bootstrap = st.select_slider("Bootstrap", [100, 300, 500, 1_000, 3_000], value=500)
        run = st.form_submit_button("Подготовить Qini/AUUC расчёт")
    if run:
        st.session_state["qini_config"] = {
            "df": df, "treatment": treatment, "outcome": outcome, "score": score,
            "propensity": propensity_const if propensity_mode == "Константа" else propensity_col,
            "cluster": None if cluster == "<нет>" else cluster, "bootstrap": bootstrap,
        }
    cfg = st.session_state.get("qini_config")
    if cfg:
        _background_submit(
            "qini", qini_auuc_analysis, cfg["df"], treatment_col=cfg["treatment"], outcome_col=cfg["outcome"],
            score_col=cfg["score"], propensity=cfg["propensity"], cluster_col=cfg["cluster"], bootstrap=cfg["bootstrap"],
        )
    job_res = st.session_state.pop("job_result_qini", None)
    if job_res is not None:
        job_res["calibration"] = uplift_calibration(
            cfg["df"], treatment_col=cfg["treatment"], outcome_col=cfg["outcome"], score_col=cfg["score"], propensity=cfg["propensity"]
        )
        _store("Qini AUUC", job_res)
    result = st.session_state.advanced_results.get("Qini AUUC")
    if result:
        fig = px.line(result["curve"], x="fraction", y=["cumulative_incremental_outcome", "random_policy_line"], title="Uplift/Qini curve")
        st.plotly_chart(fig, use_container_width=True)
        _display_frames(result, exclude={"curve"})


def render_dr_policy() -> None:
    st.subheader("Doubly robust policy value")
    df = _upload_df("dr_policy")
    if df is None:
        return
    cols = list(df.columns)
    with st.form("dr_form"):
        action = st.selectbox("Observed action", cols)
        reward = st.selectbox("Reward / profit", cols, index=min(1, len(cols)-1))
        target = st.selectbox("Action chosen by evaluated policy", cols, index=min(2, len(cols)-1))
        propensity = st.selectbox("Behavior propensity of observed action (optional)", ["<оценить>"] + cols)
        features = st.multiselect("Pre-treatment features для cross-fitting", [c for c in cols if c not in {action, reward, target}])
        cluster = st.selectbox("Cluster/client bootstrap (optional)", ["<нет>"] + cols)
        bootstrap = st.select_slider("Bootstrap", [100, 300, 500, 1_000], value=500)
        run = st.form_submit_button("Рассчитать DR policy value")
    if run:
        if not features:
            st.error("Для cross-fit q_hat выберите хотя бы один pre-treatment признак.")
        else:
            result = doubly_robust_policy_value(
                df, action_col=action, reward_col=reward, evaluation_action_col=target,
                behavior_propensity_col=None if propensity == "<оценить>" else propensity,
                feature_cols=features, bootstrap=bootstrap, cluster_col=None if cluster == "<нет>" else cluster,
            )
            _store("DR policy value", result)
    result = st.session_state.advanced_results.get("DR policy value")
    if result:
        _display_frames(result, exclude={"influence_values"})


def render_capacity_nba() -> None:
    st.subheader("Capacity-aware Next Best Action")
    st.write("Для каждого действия нужна колонка ожидаемой ценности. Оптимизатор назначит не более одного действия клиенту.")
    df = _upload_df("capacity_nba")
    if df is None:
        return
    cols = list(df.columns)
    id_col = st.selectbox("Client ID", cols)
    actions_text = st.text_input("Действия через запятую", "no_action,sms,push,call")
    actions = [x.strip() for x in actions_text.split(",") if x.strip()]
    if not actions:
        return
    no_action = st.selectbox("No-action baseline", actions)
    st.markdown("#### Колонки ожидаемой value")
    value_cols = _mapping_editor(actions, cols, "value")
    st.markdown("#### Capacity")
    capacities = {}
    for action in actions:
        if action != no_action:
            capacities[action] = st.number_input(f"Capacity {action}", 0, value=max(1, len(df)//10), key=f"cap_{action}")
    fatigue = st.selectbox("Contact fatigue column (optional)", ["<нет>"] + cols)
    penalty = st.number_input("Penalty per fatigue unit", 0.0, value=0.0)
    if st.button("Оптимизировать назначения", type="primary"):
        result = optimize_capacity_nba(
            df, id_col=id_col, value_cols=value_cols, capacities=capacities, no_action=no_action,
            fatigue_col=None if fatigue == "<нет>" else fatigue, fatigue_penalty=penalty,
        )
        _store("Capacity-aware NBA", result)
    result = st.session_state.advanced_results.get("Capacity-aware NBA")
    if result:
        _display_frames(result, exclude={"assignments"})
        st.download_button(
            "Скачать назначения CSV", result["assignments"].to_csv(index=False).encode("utf-8-sig"),
            "nba_assignments.csv", "text/csv",
        )


def render_reports() -> None:
    st.subheader("Автоматический протокол: Excel / HTML / PDF")
    results = st.session_state.advanced_results
    if not results:
        st.info("Сначала выполните хотя бы один анализ в другом разделе.")
        return
    with st.form("advanced_passport_form"):
        title = st.text_input("Название протокола", "Протокол анализа пилота")
        pilot = st.text_input("Пилот / проект", st.session_state.advanced_passport.get("Пилот", ""))
        owner = st.text_input("Владелец", st.session_state.advanced_passport.get("Владелец", ""))
        unit = st.text_input("Подразделение", st.session_state.advanced_passport.get("Подразделение", ""))
        hypothesis = st.text_area("Гипотеза", st.session_state.advanced_passport.get("Гипотеза", ""))
        save = st.form_submit_button("Обновить паспорт")
    if save:
        st.session_state.advanced_passport = {"Пилот": pilot, "Владелец": owner, "Подразделение": unit, "Гипотеза": hypothesis}
    passport = st.session_state.advanced_passport
    st.write("В отчёт войдут разделы:", ", ".join(results.keys()))
    excel = build_advanced_excel_report(passport=passport, results=results)
    html_bytes = build_html_protocol(title=title, passport=passport, results=results)
    pdf_bytes = build_pdf_protocol(title=title, passport=passport, results=results)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Скачать Excel", excel, "advanced_pilot_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    c2.download_button("Скачать HTML", html_bytes, "pilot_protocol.html", "text/html", use_container_width=True)
    c3.download_button("Скачать PDF", pdf_bytes, "pilot_protocol.pdf", "application/pdf", use_container_width=True)
    if st.button("Очистить все advanced-результаты"):
        st.session_state.advanced_results = {}
        st.rerun()


def render_advanced() -> None:
    initialize_advanced_state()
    st.title("Расширенные методы")
    st.caption(
        "Методы для случаев, когда классического A/B недостаточно. Каждый раздел содержит отдельные предпосылки; "
        "результаты автоматически сохраняются в текущей сессии для общего отчёта."
    )
    template_path = Path(__file__).resolve().parent / "assets" / "advanced_input_templates.xlsx"
    if template_path.exists():
        st.download_button(
            "Скачать Excel-шаблоны продвинутых методов",
            data=template_path.read_bytes(),
            file_name="advanced_input_templates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    method = st.selectbox("Выберите метод", METHODS)
    renderers = {
        "Bayesian monitoring": render_bayesian,
        "Sequential и exact-sequential": render_sequential,
        "CUPED / CUPAC по загруженным данным": render_cuped_cupac,
        "Survival, RMST и non-proportional hazards": render_survival,
        "Competing risks": render_competing,
        "Recurrent events": render_recurrent,
        "Кластерный / stepped-wedge / switchback": render_cluster_designs,
        "Synthetic control": render_synthetic,
        "Regression discontinuity": render_rdd,
        "Interleaving для ranking": render_interleaving,
        "Continuous treatment / dose-response": render_dose_response,
        "Contextual bandits": render_bandits,
        "Qini / AUUC с bootstrap": render_qini,
        "Doubly robust policy value": render_dr_policy,
        "Capacity-aware Next Best Action": render_capacity_nba,
        "HTML / PDF / Excel протокол": render_reports,
    }
    renderers[method]()

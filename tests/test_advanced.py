import numpy as np
import pandas as pd

from experiment_core.bayesian import PriorSpec, bayesian_two_group_binary, beta_binomial_predictive_probability
from experiment_core.sequential import calibrate_exact_sequential, calibrate_gaussian_group_sequential
from experiment_core.variance_reduction import cuped_analysis, cupac_analysis
from experiment_core.survival import analyze_survival, analyze_competing_risks, analyze_recurrent_events
from experiment_core.causal_designs import (
    cluster_randomized_design, stepped_wedge_schedule, generate_switchback_schedule,
    regression_discontinuity, synthetic_control, dose_response,
)
from experiment_core.uplift_advanced import qini_auuc_analysis, doubly_robust_policy_value, optimize_capacity_nba
from experiment_core.bandits_ranking import contextual_bandit_offline_evaluation, interleaving_analysis
from experiment_core.reporting import build_html_protocol, build_pdf_protocol, build_advanced_excel_report


def test_bayesian_and_sequential():
    prior = PriorSpec("Jeffreys", 0.5, 0.5)
    result = bayesian_two_group_binary(50, 1000, 65, 1000, prior_control=prior, prior_treatment=prior, draws=3000)
    assert 0 <= result["effect_summary"].iloc[0]["probability_treatment_better"] <= 1
    pp = beta_binomial_predictive_probability(50, 1000, 65, 1000, 1500, 1500, outer_simulations=100, posterior_draws=100)
    assert 0 <= pp["predictive_probability"] <= 1
    boundaries = calibrate_gaussian_group_sequential([0.5, 1.0], simulations=5000)
    assert len(boundaries) == 2
    exact = calibrate_exact_sequential([50, 100], [50, 100], p_null=0.02, simulations=100)
    assert 0 < exact["p_threshold"] < 1


def test_variance_reduction():
    rng = np.random.default_rng(2)
    n = 800
    treatment = rng.binomial(1, 0.5, n)
    x = rng.normal(size=n)
    y = 1.5 * treatment + 2.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"t": treatment, "x": x, "y": y})
    cuped = cuped_analysis(df, outcome_col="y", treatment_col="t", preperiod_cols=["x"])
    cupac = cupac_analysis(df, outcome_col="y", treatment_col="t", feature_cols=["x"], folds=3)
    assert cuped["variance_reduction"] > 0
    assert cupac["variance_reduction"] > 0


def test_survival_competing_recurrent():
    rng = np.random.default_rng(3)
    n = 500
    treatment = rng.binomial(1, 0.5, n)
    duration = rng.exponential(10 + 2 * treatment)
    censor = rng.exponential(15, n)
    event = (duration <= censor).astype(int)
    observed = np.minimum(duration, censor)
    group = np.where(treatment == 1, "treatment", "control")
    df = pd.DataFrame({"duration": observed, "event": event, "group": group})
    survival = analyze_survival(df, duration_col="duration", event_col="event", group_col="group",
                                control_label="control", treatment_label="treatment", bootstrap=20)
    assert not survival["effect_summary"].empty
    df["event_type"] = np.where(event == 0, 0, rng.choice([1, 2], n))
    competing = analyze_competing_risks(df, duration_col="duration", event_type_col="event_type", group_col="group",
                                        control_label="control", treatment_label="treatment", bootstrap=20)
    assert not competing["effect_summary"].empty

    rows = []
    for i in range(60):
        tr = i % 2
        for j in range(3):
            rows.append((i, j, j + 1, rng.binomial(1, 0.15 + 0.05 * tr), tr))
    recurrent_df = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "treatment"])
    recurrent = analyze_recurrent_events(recurrent_df, id_col="id", start_col="start", stop_col="stop",
                                         event_col="event", treatment_col="treatment", model="andersen-gill")
    assert not recurrent["effect_summary"].empty


def test_causal_designs():
    cluster = cluster_randomized_design(individual_sample_size=1000, mean_cluster_size=20, icc=0.02)
    assert cluster["inflated_sample_size"] > 1000
    assert len(stepped_wedge_schedule(6, 5)) == 30
    assert len(generate_switchback_schedule(3, 6)) == 18

    rng = np.random.default_rng(4)
    running = rng.uniform(-2, 2, 800)
    outcome = 0.5 * (running >= 0) + 0.2 * running + rng.normal(size=800)
    rdd = regression_discontinuity(pd.DataFrame({"run": running, "y": outcome}), outcome_col="y", running_col="run", cutoff=0, bandwidth=1)
    assert not rdd["summary"].empty

    rows = []
    for time in range(16):
        base = np.sin(time / 3)
        for unit, shift in {"treated": 0, "d1": 0.1, "d2": -0.1, "d3": 0.05}.items():
            y = base + shift + (0.4 if unit == "treated" and time >= 8 else 0) + rng.normal(scale=0.03)
            rows.append((unit, time, y))
    panel = pd.DataFrame(rows, columns=["unit", "time", "y"])
    synth = synthetic_control(panel, unit_col="unit", time_col="time", outcome_col="y", treated_unit="treated", intervention_time=8)
    assert abs(synth["summary"].iloc[0]["average_post_effect"]) > 0.1

    d = rng.uniform(0, 1, 300)
    x = rng.normal(size=300)
    y = 2*d - d*d + x + rng.normal(size=300)
    dose = dose_response(pd.DataFrame({"d": d, "x": x, "y": y}), outcome_col="y", dose_col="d", covariate_cols=["x"], bootstrap=5)
    assert len(dose["curve"]) == 40


def test_uplift_policy_bandit_nba_and_reports():
    rng = np.random.default_rng(5)
    n = 600
    treatment = rng.binomial(1, 0.5, n)
    score = rng.normal(0.02, 0.01, n)
    outcome = rng.binomial(1, np.clip(0.05 + treatment * score, 0, 1))
    uplift_df = pd.DataFrame({"t": treatment, "y": outcome, "score": score})
    qini = qini_auuc_analysis(uplift_df, treatment_col="t", outcome_col="y", score_col="score", bootstrap=10)
    assert not qini["metrics"].empty

    action = rng.choice(["a", "b"], n)
    target = rng.choice(["a", "b"], n)
    x = rng.normal(size=n)
    reward = x + 0.2 * (action == "b") + rng.normal(size=n)
    policy_df = pd.DataFrame({"action": action, "target": target, "reward": reward, "x": x, "p": 0.5})
    dr = doubly_robust_policy_value(policy_df, action_col="action", reward_col="reward", evaluation_action_col="target",
                                    behavior_propensity_col="p", feature_cols=["x"], bootstrap=10)
    assert np.isfinite(dr["summary"].iloc[0]["dr_policy_value"])

    policy_df["q_a"] = x
    policy_df["q_b"] = x + 0.2
    bandit = contextual_bandit_offline_evaluation(policy_df, action_col="action", reward_col="reward",
                                                   behavior_propensity_col="p", target_action_col="target",
                                                   q_hat_cols={"a": "q_a", "b": "q_b"}, bootstrap=10)
    assert set(bandit["summary"]["estimator"]) == {"IPS", "SNIPS", "DR"}

    inter = interleaving_analysis(pd.DataFrame({"winner": rng.choice(["A", "B", "tie"], 200)}), winner_col="winner", bootstrap=20)
    assert not inter["summary"].empty

    nba_df = pd.DataFrame({"id": range(50), "none": 0.0, "sms": rng.normal(0.2, 0.2, 50), "call": rng.normal(0.4, 0.3, 50)})
    nba = optimize_capacity_nba(nba_df, id_col="id", value_cols={"none": "none", "sms": "sms", "call": "call"},
                                capacities={"sms": 10, "call": 5}, no_action="none")
    assert len(nba["assignments"]) == 50

    results = {"Qini": qini, "DR": dr}
    assert build_html_protocol(title="Тест", passport={"Проект": "X"}, results=results).startswith(b"<!doctype html>")
    assert build_pdf_protocol(title="Тест", passport={"Проект": "X"}, results=results).startswith(b"%PDF")
    assert build_advanced_excel_report(passport={"Проект": "X"}, results=results).startswith(b"PK")

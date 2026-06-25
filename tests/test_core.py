import numpy as np
import pandas as pd

from experiment_core.analysis import analyze_experiment, analyze_uplift, validate_dataset
from experiment_core.design import fixed_binary_design, fixed_continuous_design, multiarm_binary_design
from experiment_core.excel_report import build_excel_report


def test_binary_design():
    result = fixed_binary_design(0.005, 0.004, clients_per_period=10_000)
    assert result.n_total > 100_000
    assert 9 < result.periods < 13


def test_continuous_design():
    result = fixed_continuous_design(100, 105, 20, clients_per_period=1_000)
    assert result.n_total > 0
    assert result.n_control > 0


def test_multiarm_design():
    df = multiarm_binary_design(0.05, {"sms": 0.055, "push": 0.06})
    assert len(df) == 2
    assert df["recommended_total_n"].iloc[0] > 0


def test_binary_analysis_and_excel():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "client_id": np.arange(10_000),
        "group": np.where(np.arange(10_000) % 2 == 0, "control", "treatment"),
    })
    probs = np.where(df["group"].eq("control"), 0.05, 0.06)
    df["outcome"] = rng.binomial(1, probs)

    validation = validate_dataset(df, group_col="group", outcome_col="outcome", id_col="client_id")
    assert validation.is_valid

    analysis = analyze_experiment(
        df, group_col="group", outcome_col="outcome", control_label="control"
    )
    assert not analysis["results"].empty
    payload = build_excel_report(passport={"name": "test"}, analysis=analysis)
    assert payload[:2] == b"PK"


def test_uplift_analysis():
    rng = np.random.default_rng(10)
    n = 4_000
    score = rng.normal(0.02, 0.01, n)
    treatment = rng.binomial(1, 0.5, n)
    base = 0.05
    p = np.clip(base + treatment * score, 0.001, 0.5)
    outcome = rng.binomial(1, p)
    df = pd.DataFrame({"treatment": treatment, "outcome": outcome, "predicted_uplift": score})
    result = analyze_uplift(
        df, treatment_col="treatment", outcome_col="outcome", score_col="predicted_uplift"
    )
    assert not result["calibration"].empty

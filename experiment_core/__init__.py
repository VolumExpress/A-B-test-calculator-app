"""Расчётное ядро Banking Experiment Calculator."""

from .design import fixed_binary_design, fixed_continuous_design, multiarm_binary_design
from .analysis import analyze_experiment, analyze_uplift, validate_dataset
from .bayesian import bayesian_two_group_binary, beta_binomial_predictive_probability
from .sequential import calibrate_exact_sequential, calibrate_gaussian_group_sequential
from .variance_reduction import cuped_analysis, cupac_analysis
from .survival import analyze_survival, analyze_competing_risks, analyze_recurrent_events
from .causal_designs import synthetic_control, regression_discontinuity, dose_response
from .uplift_advanced import qini_auuc_analysis, doubly_robust_policy_value, optimize_capacity_nba
from .bandits_ranking import contextual_bandit_offline_evaluation, interleaving_analysis
from .excel_report import build_excel_report
from .reporting import build_advanced_excel_report, build_html_protocol, build_pdf_protocol

__all__ = [
    "fixed_binary_design", "fixed_continuous_design", "multiarm_binary_design",
    "analyze_experiment", "analyze_uplift", "validate_dataset",
    "bayesian_two_group_binary", "beta_binomial_predictive_probability",
    "calibrate_exact_sequential", "calibrate_gaussian_group_sequential",
    "cuped_analysis", "cupac_analysis",
    "analyze_survival", "analyze_competing_risks", "analyze_recurrent_events",
    "synthetic_control", "regression_discontinuity", "dose_response",
    "qini_auuc_analysis", "doubly_robust_policy_value", "optimize_capacity_nba",
    "contextual_bandit_offline_evaluation", "interleaving_analysis",
    "build_excel_report", "build_advanced_excel_report", "build_html_protocol", "build_pdf_protocol",
]

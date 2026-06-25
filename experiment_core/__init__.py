"""Расчётное ядро MVP банковского калькулятора экспериментов."""

from .design import (
    fixed_binary_design,
    fixed_continuous_design,
    multiarm_binary_design,
    variance_reduction_scenarios,
    obrien_fleming_boundaries,
)
from .analysis import analyze_experiment, analyze_uplift, validate_dataset
from .excel_report import build_excel_report

__all__ = [
    "fixed_binary_design",
    "fixed_continuous_design",
    "multiarm_binary_design",
    "variance_reduction_scenarios",
    "obrien_fleming_boundaries",
    "analyze_experiment",
    "analyze_uplift",
    "validate_dataset",
    "build_excel_report",
]

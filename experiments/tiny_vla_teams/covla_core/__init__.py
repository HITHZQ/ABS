"""Core experiment scaffold for Tiny VLA Teams.

The package is intentionally dependency-light.  It provides runnable mock
experiments plus stable interfaces for plugging in LIBERO/robosuite
environments and real VLA policies later.
"""

from .config import ExperimentConfig, baseline_sweep

__all__ = ["ExperimentConfig", "baseline_sweep"]

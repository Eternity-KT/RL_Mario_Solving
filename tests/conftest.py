"""Shared pytest configuration.

Tests that import ``src.env`` need the NES emulator (``nes-py``) and
``gym-super-mario-bros``. On machines or CI runners where these packages
could not be installed, those test modules are skipped at collection time
instead of failing with ImportError.
"""

import importlib.util

NES_MODULES = ("nes_py", "gym_super_mario_bros")
NES_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in NES_MODULES)

# Test modules that import src.env (which imports nes_py at package level).
_NES_DEPENDENT_TESTS = ["test_env.py", "test_wrappers.py"]

collect_ignore = [] if NES_AVAILABLE else list(_NES_DEPENDENT_TESTS)


def pytest_report_header(config):
    if NES_AVAILABLE:
        return "NES emulator: available"
    return "NES emulator: NOT available -> skipping " + ", ".join(_NES_DEPENDENT_TESTS)

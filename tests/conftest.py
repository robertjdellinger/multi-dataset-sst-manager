"""Pytest session setup for upstream tests with cwd-relative fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent


def pytest_configure(config):
    """Match the upstream tests' expected working directory from repo-root runs."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(TEST_ROOT)

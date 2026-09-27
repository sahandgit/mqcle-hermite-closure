"""Shared fixtures.  Adds ``src`` to ``sys.path`` when the package is not
installed, so the suite runs from a bare checkout as well as from ``pip -e``.

NEVER add ``src/mqcle`` itself: ``mqcle.types`` shadows the stdlib ``types``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mqcle.io import load_config                # noqa: E402
from mqcle.models import build_model            # noqa: E402
from mqcle.types import TraceConvention         # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config("tully_default.yaml")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260819)


@pytest.fixture(scope="session")
def model(cfg):
    return build_model(cfg["model"], mass=cfg["mass"])


@pytest.fixture(scope="session")
def dac(model):
    """Alias used by the Gate-E field tests: the Tully dual avoided crossing."""
    return model


@pytest.fixture(scope="session")
def convention() -> TraceConvention:
    return TraceConvention.TRACELESS

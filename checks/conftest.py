"""Shared set-up for the checks: find your code, and replay saved model replies instead of calling a model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GRADE_SRC", ROOT)).resolve()
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from replay import ReplayClient  # noqa: E402


@pytest.fixture
def src() -> Path:
    """The folder that holds the code being checked."""
    return SRC


@pytest.fixture
def fixtures() -> Path:
    return ROOT / "fixtures"


@pytest.fixture
def replay(fixtures: Path) -> ReplayClient:
    """A stand-in for anthropic.Anthropic() that answers from fixtures/replies.json."""
    return ReplayClient(json.loads((fixtures / "replies.json").read_text(encoding="utf-8")))

"""Pytest configuration helpers for the mcbo test suite."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_on_path() -> None:
    """Ensure the repository root is importable during tests."""
    project_root = Path(__file__).resolve().parents[1]
    project_str = str(project_root)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)


_ensure_project_on_path()

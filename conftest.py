"""
conftest.py — pytest shared fixtures and configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH for all tests
sys.path.insert(0, str(Path(__file__).parent / "src"))

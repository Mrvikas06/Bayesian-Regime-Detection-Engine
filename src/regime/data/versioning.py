"""
regime.data.versioning
======================
Utilities for dataset versioning, lineage tracking, and config hashing.
Every output row in this project should carry a lineage dictionary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 of parquet bytes → 16-hex-char version token."""
    buf = df.to_parquet(index=True)
    return hashlib.sha256(buf).hexdigest()[:16]


def hash_config(cfg: dict[str, Any]) -> str:
    """SHA-256 of JSON-serialised config dict."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_lineage_record(
    *,
    phase: int,
    data_hash: str,
    feature_hash: str | None = None,
    model_hash: str | None = None,
    config_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a lineage dictionary to attach to every output row.

    Parameters
    ----------
    phase        : phase number (1–15)
    data_hash    : hash of the input master parquet
    feature_hash : hash of the features parquet (Phase 2+)
    model_hash   : hash of the fitted model artefact (Phase 3+)
    config_hash  : hash of the YAML config used
    extra        : additional key-value pairs

    Returns
    -------
    dict suitable for JSON serialisation
    """
    rec: dict[str, Any] = {
        "phase": phase,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_version_hash": data_hash,
        "config_hash": config_hash,
    }
    if feature_hash is not None:
        rec["feature_set_version"] = feature_hash
    if model_hash is not None:
        rec["model_version"] = model_hash
    if extra:
        rec.update(extra)
    return rec


def attach_lineage(df: pd.DataFrame, lineage: dict[str, Any]) -> pd.DataFrame:
    """
    Add lineage columns to every row of a DataFrame.
    Columns: lineage_data_hash, lineage_config_hash, lineage_timestamp_utc, etc.
    """
    for k, v in lineage.items():
        df[f"lineage_{k}"] = str(v)
    return df


def save_lineage_json(lineage: dict[str, Any], path: str | Path) -> None:
    """Write lineage record to a JSON file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lineage, f, indent=2, default=str)

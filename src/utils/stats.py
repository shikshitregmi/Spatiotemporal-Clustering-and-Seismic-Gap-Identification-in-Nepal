"""Load, update, and persist pipeline statistics for the summary report."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import STATS_PATH, ensure_output_dirs


def load_stats() -> dict:
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return {}


def save_stats(stats: dict) -> Path:
    ensure_output_dirs()
    STATS_PATH.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return STATS_PATH


def update_stats(**kwargs) -> dict:
    stats = load_stats()
    stats.update(kwargs)
    save_stats(stats)
    return stats

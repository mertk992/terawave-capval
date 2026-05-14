"""
File-backed synthetic enterprise data loader.

The project deliberately uses fabricated records, but keeping them in a data
file makes tool calls traceable to a source corpus instead of opaque code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_SOURCE_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_enterprise_data.json"


@lru_cache(maxsize=1)
def load_synthetic_data() -> dict[str, Any]:
    """Load the synthetic source corpus used by tools and RAG."""
    with DATA_SOURCE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_source_metadata() -> dict[str, str]:
    raw = load_synthetic_data().get("metadata", {})
    return {
        "source_file": str(DATA_SOURCE_PATH),
        "dataset_id": raw.get("dataset_id", "unknown"),
        "synthetic": True,
    }


def get_approval_tiers() -> list[dict[str, Any]]:
    return deepcopy(load_synthetic_data()["approval_tiers"])


def get_budget_pools() -> dict[str, dict[str, Any]]:
    pools = {}
    for row in load_synthetic_data()["budget_pools"]:
        pools[row["name"]] = {
            "budget_m": row["budget_m"],
            "spent_m": row["spent_m"],
            "committed_m": row["committed_m"],
            "source_record_id": row["id"],
        }
    return pools


def get_historical_requests() -> list[dict[str, Any]]:
    return deepcopy(load_synthetic_data()["historical_requests"])


def get_documents() -> list[dict[str, Any]]:
    return deepcopy(load_synthetic_data()["documents"])

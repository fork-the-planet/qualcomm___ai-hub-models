# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Replay the geniex_perf_updates.json manifest from run_geniex_bench_benchmarks.py
onto a fresh checkout so the PR only edits geniex entries; genie is untouched.
"""

from __future__ import annotations

import argparse
import json

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.perf_collection import (
    clear_llm_metrics_for_profile_path,
    update_perf_yaml,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath


def apply_updates(manifest_path: str) -> int:
    with open(manifest_path) as f:
        updates = json.load(f)

    if not updates:
        print(f"No updates in {manifest_path}; nothing to apply.")
        return 0

    # Clear (model, profile_path) once before rewriting so dropped ctx
    # lengths / compute_units don't leave orphans behind.
    seen: set[tuple[str, str]] = set()
    for u in updates:
        key = (u["model_id"], u["profile_path"])
        if key in seen:
            continue
        seen.add(key)
        clear_llm_metrics_for_profile_path(
            model_id=u["model_id"],
            profile_path=ScorecardProfilePath(u["profile_path"]),
        )

    models: set[str] = set()
    for u in updates:
        update_perf_yaml(
            model_id=u["model_id"],
            device_name=u["device_name"],
            precision=Precision.parse(u["precision"]),
            context_length=u["context_length"],
            tps=u["tps"],
            ttft_ms=u["ttft_ms"],
            prefill_tps=u["prefill_tps"],
            ttft_max_ms=u["ttft_max_ms"],
            profile_path=ScorecardProfilePath(u["profile_path"]),
            desired_compute_unit=u["desired_compute_unit"],
        )
        models.add(u["model_id"])

    print(f"Applied {len(updates)} geniex perf updates across {len(models)} models:")
    for model_id in sorted(models):
        print(f"  {model_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", help="Path to geniex_perf_updates.json.")
    args = ap.parse_args()
    return apply_updates(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())

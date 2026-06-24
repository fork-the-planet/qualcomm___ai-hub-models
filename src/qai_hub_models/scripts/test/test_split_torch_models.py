# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest
import ruamel.yaml

from qai_hub_models.scorecard.artifacts import RUNTIME_STAGE_JOB_SUBMISSION
from qai_hub_models.scorecard.envvars import SpecialModelSetting
from qai_hub_models.scripts.split_torch_models import (
    _balance_lpt,
    _split_aot_jit,
    split_torch_models,
)


def _bin_loads(bins: list[list[str]], weights: dict[str, float]) -> list[float]:
    return [sum(weights[m] for m in b) for b in bins]


def test_lpt_balances_better_than_alphabetical() -> None:
    """LPT should produce a smaller makespan than even-sized alphabetical chunks."""
    # Heavy models clustered alphabetically — alphabetical chunking would
    # put them all in one bin; LPT spreads them.
    weights = {
        "aaa": 1000.0,
        "aab": 1000.0,
        "aac": 1000.0,
        "zzx": 10.0,
        "zzy": 10.0,
        "zzz": 10.0,
    }
    models = sorted(weights.keys())
    bins = _balance_lpt(models, [weights[m] for m in models], num_splits=3)
    loads = _bin_loads(bins, weights)
    # Each bin should have roughly 1010 — perfect balance is exactly that.
    assert max(loads) - min(loads) <= 0.01, loads
    # Each model placed exactly once.
    assert sorted(m for b in bins for m in b) == models


def test_lpt_handles_empty_input() -> None:
    assert _balance_lpt([], [], num_splits=3) == [[], [], []]
    assert _balance_lpt([], [], num_splits=0) == []


def test_lpt_more_bins_than_models() -> None:
    bins = _balance_lpt(["a", "b"], [1.0, 2.0], num_splits=4)
    populated = [b for b in bins if b]
    assert len(populated) == 2
    assert len(bins) == 4


def test_split_aot_jit_falls_back_when_no_estimates() -> None:
    """No estimates -> alphabetical chunks, AOT-first within each split."""
    aot = ["aot1", "aot2", "aot3"]
    jit = ["jit1", "jit2", "jit3"]
    bins = _split_aot_jit(
        aot, jit, num_splits=3, stage=RUNTIME_STAGE_JOB_SUBMISSION, estimates={}
    )
    assert len(bins) == 3
    # AOT models come first within each split.
    for split in bins:
        for i, m in enumerate(split):
            if m.startswith("jit"):
                assert all(x.startswith("jit") for x in split[i:])


def test_split_aot_jit_uses_lpt_with_estimates() -> None:
    """LPT balances over the union of AOT+JIT, then sorts AOT-first per split."""
    aot = ["aot_heavy", "aot_light_a", "aot_light_b"]
    jit = ["jit_x", "jit_y"]
    estimates = {
        "aot_heavy": {RUNTIME_STAGE_JOB_SUBMISSION: 1000.0},
        "aot_light_a": {RUNTIME_STAGE_JOB_SUBMISSION: 50.0},
        "aot_light_b": {RUNTIME_STAGE_JOB_SUBMISSION: 50.0},
        "jit_x": {RUNTIME_STAGE_JOB_SUBMISSION: 200.0},
        "jit_y": {RUNTIME_STAGE_JOB_SUBMISSION: 200.0},
    }
    bins = _split_aot_jit(
        aot, jit, num_splits=2, stage=RUNTIME_STAGE_JOB_SUBMISSION, estimates=estimates
    )
    # Across splits, the makespan should be balanced. With heavy=1000
    # and the other four totaling 500, optimal is 1000 vs 500.
    weights = {m: estimates[m][RUNTIME_STAGE_JOB_SUBMISSION] for m in aot + jit}
    loads = sorted(_bin_loads(bins, weights))
    assert loads == [500.0, 1000.0]
    # Within each split, AOT models come before JIT (sorted alphabetically inside each group).
    aot_set = set(aot)
    for split in bins:
        if split:
            seen_jit = False
            for m in split:
                if m in aot_set:
                    assert not seen_jit, f"AOT after JIT in split: {split}"
                else:
                    seen_jit = True


def test_split_aot_jit_union_balance_beats_per_group() -> None:
    """Union LPT must not be worse than per-group LPT for an adversarial input.

    Pathological case: AOT total = JIT total but they split unevenly when
    bin-packed separately. Union packing is free to mix-and-match.
    """
    aot = ["aot_a", "aot_b"]
    jit = ["jit_a", "jit_b"]
    # Per-group: AOT bins go [100][100], JIT bins go [100][100] -> [200, 200]
    # — coincidentally same as union. Use asymmetric weights so union wins.
    estimates = {
        "aot_a": {RUNTIME_STAGE_JOB_SUBMISSION: 100.0},
        "aot_b": {RUNTIME_STAGE_JOB_SUBMISSION: 30.0},
        "jit_a": {RUNTIME_STAGE_JOB_SUBMISSION: 70.0},
        "jit_b": {RUNTIME_STAGE_JOB_SUBMISSION: 30.0},
    }
    bins = _split_aot_jit(
        aot, jit, num_splits=2, stage=RUNTIME_STAGE_JOB_SUBMISSION, estimates=estimates
    )
    weights = {m: estimates[m][RUNTIME_STAGE_JOB_SUBMISSION] for m in aot + jit}
    loads = sorted(_bin_loads(bins, weights))
    # Total = 230, optimal = max(115, 100) = 115. Union LPT achieves it
    # by pairing aot_a(100) with one of jit_b/aot_b(30) vs aot_b+jit_a+jit_b.
    # Per-group LPT would force aot_a alone vs aot_b in AOT bins, then
    # jit_a alone vs jit_b in JIT bins -> [100+30, 30+70] = [130, 100],
    # makespan 130. Union should achieve 130 or better.
    assert max(loads) <= 130.0


def test_split_aot_jit_unknown_models_use_median() -> None:
    """Models missing from estimates are scheduled at the stage median."""
    estimates = {
        "known_heavy": {RUNTIME_STAGE_JOB_SUBMISSION: 100.0},
        "known_light": {RUNTIME_STAGE_JOB_SUBMISSION: 10.0},
    }
    # 'new_a' and 'new_b' aren't in the estimates at all. Stage median
    # is (10 + 100) / 2 = 55. So the LPT order is heavy(100), new(55),
    # new(55), light(10) — the new models should not be treated as zero.
    bins = _split_aot_jit(
        aot_models=[],
        jit_models=["known_heavy", "known_light", "new_a", "new_b"],
        num_splits=2,
        stage=RUNTIME_STAGE_JOB_SUBMISSION,
        estimates=estimates,
    )
    # Each bin gets two models (heavy+light vs new_a+new_b would be the
    # zero-weight failure mode). Heavy alone in one bin, all others in
    # the other, would also be wrong. Expect heavy paired with light
    # (one new in each bin) — load ~110 vs ~110.
    sizes = sorted(len(b) for b in bins)
    assert sizes == [2, 2]


def test_split_aot_jit_falls_back_when_stage_missing() -> None:
    """Estimates dict has no entries for the requested stage -> alphabetical."""
    estimates = {
        "m_a": {"some_other_stage": 100.0},
        "m_b": {"some_other_stage": 50.0},
    }
    bins = _split_aot_jit(
        aot_models=[],
        jit_models=["m_a", "m_b"],
        num_splits=2,
        stage=RUNTIME_STAGE_JOB_SUBMISSION,
        estimates=estimates,
    )
    assert len(bins) == 2
    # Even split alphabetically.
    assert sorted(m for b in bins for m in b) == ["m_a", "m_b"]


def test_load_runtime_estimates_missing_file_returns_empty(tmp_path: Path) -> None:
    from qai_hub_models.scripts.split_torch_models import _load_runtime_estimates

    assert _load_runtime_estimates(tmp_path / "does-not-exist.yaml") == {}


def test_load_runtime_estimates_reads_models_section(tmp_path: Path) -> None:
    from qai_hub_models.scripts.split_torch_models import _load_runtime_estimates

    yaml_path = tmp_path / "estimates.yaml"
    payload = {
        "source_action_id": "12345",
        "models": {
            "model_a": {"job_submission": 100.0, "accuracy": 50.0},
            "model_b": {"job_submission": 200.0},
        },
    }
    with open(yaml_path, "w") as f:
        ruamel.yaml.YAML().dump(payload, f)

    estimates = _load_runtime_estimates(yaml_path)
    assert estimates == {
        "model_a": {"job_submission": 100.0, "accuracy": 50.0},
        "model_b": {"job_submission": 200.0},
    }


def test_collapse_to_single_split_emits_one_torch_split() -> None:
    """Unit-only runs collapse all torch models (incl. LLM/PI0_5) into one split
    on the default runner -- no GPU custom splits.
    """
    splits = split_torch_models(
        {SpecialModelSetting.PYTORCH}, collapse_to_single_split=True
    )
    torch_splits = [s for s in splits if s["split_name"] != "static"]
    assert len(torch_splits) == 1
    assert torch_splits[0]["split_name"] == "torch"
    assert torch_splits[0]["runs_on"] is None
    # LLM models would normally land in their own GPU split; verify they're
    # folded into the single torch split here.
    models_str = torch_splits[0]["models"]
    assert isinstance(models_str, str)
    assert any(m.startswith("llama_v3") for m in models_str.split(","))


@pytest.mark.parametrize("num_splits", [1, 2, 4, 8])
def test_lpt_makespan_within_4_3_bound(num_splits: int) -> None:
    """LPT (4/3 - 1/(3n)) approximation: makespan <= (4/3) * optimal."""
    weights_list = [50, 50, 30, 30, 20, 20, 10, 10, 10, 10]
    models = [f"m{i}" for i in range(len(weights_list))]
    bins = _balance_lpt(models, [float(w) for w in weights_list], num_splits=num_splits)
    loads = [sum(float(weights_list[int(m[1:])]) for m in b) for b in bins]
    total = float(sum(weights_list))
    optimal_lower_bound = max(total / num_splits, *weights_list)
    assert max(loads) <= (4.0 / 3.0) * optimal_lower_bound + 1e-6

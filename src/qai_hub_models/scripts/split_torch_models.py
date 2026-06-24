# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Literal

import ruamel.yaml

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen, TestRunnerSplit
from qai_hub_models.scorecard.artifacts import (
    RUNTIME_ALL_STAGES,
    ScorecardArtifact,
)
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)

# Default max number of PyTorch models per split when auto-splitting
MAX_PT_MODELS_PER_SPLIT = 30

# Type for runs_on values: runner group/labels object, or null
RunsOnValue = dict[Literal["group", "labels"], str | list[str]] | None


def _load_runtime_estimates(
    yaml_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load per-model per-stage runtime estimates. Empty dict if missing/empty."""
    if yaml_path is None:
        yaml_path = ScorecardArtifact.MODEL_RUNTIME_ESTIMATES.intermediates_path
    if not yaml_path.exists() or yaml_path.stat().st_size == 0:
        return {}
    with open(yaml_path) as f:
        data = ruamel.yaml.YAML(typ="safe", pure=True).load(f) or {}
    models = data.get("models") or {}
    return {m: dict(v or {}) for m, v in models.items()}


def _stage_time_for_model(
    model_id: str,
    stage: str,
    estimates: dict[str, dict[str, float]],
    median_time: float,
) -> float:
    """Time for ``model_id`` at ``stage``, falling back to the cross-model median
    so newly added models aren't scheduled as zero-cost.
    """
    return estimates.get(model_id, {}).get(stage, median_time)


def _balance_lpt(
    models: list[str],
    weights: list[float],
    num_splits: int,
) -> list[list[str]]:
    """Greedy longest-processing-time bin packing into ``num_splits`` bins.

    LPT is bounded to (4/3 - 1/3n) of the optimal makespan, which is fine
    for CI balancing where exact optimality isn't needed. Bin contents come
    back in insertion (descending-weight) order; callers reorder as needed.
    """
    if num_splits <= 0:
        return []
    bins: list[list[str]] = [[] for _ in range(num_splits)]
    bin_loads = [0.0] * num_splits
    # Heaviest first; tie-break alphabetically for determinism.
    order = sorted(
        range(len(models)),
        key=lambda i: (-weights[i], models[i]),
    )
    for idx in order:
        target = min(range(num_splits), key=lambda b: bin_loads[b])
        bins[target].append(models[idx])
        bin_loads[target] += weights[idx]
    return bins


def _balance_alphabetical(models: list[str], num_splits: int) -> list[list[str]]:
    """Original split logic: even-sized alphabetical chunks."""
    if num_splits <= 0:
        return []
    chunk = math.ceil(len(models) / num_splits)
    out: list[list[str]] = []
    for i in range(num_splits):
        start = i * chunk
        end = min(start + chunk, len(models))
        out.append(models[start:end])
    return out


def _split_aot_jit(
    aot_models: list[str],
    jit_models: list[str],
    num_splits: int,
    stage: str | None,
    estimates: dict[str, dict[str, float]],
) -> list[list[str]]:
    """Balance AOT+JIT models across splits, AOT first within each split so
    slow compiles kick off early. Falls back to alphabetical chunks when
    no per-stage runtime data is available.
    """
    if num_splits <= 0:
        return []

    median_time: float = 0.0
    use_lpt = bool(stage and estimates)
    if use_lpt:
        # `estimates` can be non-empty yet have no entry for this specific
        # stage; fall back to alphabetical when that's the case.
        recorded = [
            v[stage] for v in estimates.values() if isinstance(v, dict) and stage in v
        ]
        if not recorded:
            use_lpt = False
        else:
            median_time = statistics.median(recorded)

    if use_lpt:
        assert stage is not None  # narrowed by `use_lpt = bool(stage and estimates)`
        all_models = aot_models + jit_models
        weights = [
            _stage_time_for_model(m, stage, estimates, median_time) for m in all_models
        ]
        bins = _balance_lpt(all_models, weights, num_splits)
        aot_set = set(aot_models)
        return [
            sorted(m for m in b if m in aot_set)
            + sorted(m for m in b if m not in aot_set)
            for b in bins
        ]

    aot_bins = _balance_alphabetical(aot_models, num_splits)
    jit_bins = _balance_alphabetical(jit_models, num_splits)
    return [aot_bins[i] + jit_bins[i] for i in range(num_splits)]


def split_torch_models(
    models: set,
    max_pt_splits: int | None = None,
    max_pt_models_per_split: int = MAX_PT_MODELS_PER_SPLIT,
    stage: str | None = None,
    runtime_estimates_path: Path | None = None,
    collapse_to_single_split: bool = False,
) -> list[dict[str, str | RunsOnValue]]:
    """
    Split models into chunks for parallel processing.

    Static models are all grouped into one split named "static".
    Torch models are split into multiple chunks.

    Parameters
    ----------
    models
        Set of model IDs or special settings (from EnabledModelsEnvvar.get())
    max_pt_splits
        Maximum number of default splits to create for torch models (does not
        include custom splits). If None, automatically calculate based on
        max_pt_models_per_split.
    max_pt_models_per_split
        Maximum number of models per default split when auto-calculating
        num_default_splits (does not include custom splits).
    stage
        Scorecard stage to balance for (``job_submission``, ``export_test``,
        or ``accuracy``). Falls back to alphabetical chunks when None or
        when no runtime data is available.
    runtime_estimates_path
        Override for the runtime estimates YAML (default: checked-in copy).
    collapse_to_single_split
        If True, emit a single torch split with all models on the default
        runner -- no custom GPU splits, no chunking. Use for unit-only runs
        where splitting buys nothing.

    Returns
    -------
    list[dict[str, str | RunsOnValue]]
        List of dicts with 'split_name', 'models', and 'runs_on' keys for each split.
    """
    torch_models, static_models = validate_and_split_enabled_models(models)

    splits: list[dict[str, str | RunsOnValue]] = []

    if collapse_to_single_split:
        if static_models:
            splits.append(
                {
                    "split_name": "static",
                    "models": ",".join(sorted(static_models)),
                    "runs_on": None,
                }
            )
        if torch_models:
            splits.append(
                {
                    "split_name": "torch",
                    "models": ",".join(sorted(torch_models)),
                    "runs_on": None,
                }
            )
        return splits

    # Add all static models as one split
    if static_models:
        splits.append(
            {
                "split_name": "static",
                "models": ",".join(sorted(static_models)),
            }
        )

    # Split torch models into chunks
    all_torch_models = sorted(torch_models)
    if all_torch_models:
        # Group models by test_split
        custom_splits: dict[TestRunnerSplit, list[str]] = {}
        # Track AOT and JIT separately so a split doesn't end up dominated by
        # one (AOT compiles take much longer than JIT).
        all_models_jit = []
        all_models_aot = []
        for model in all_torch_models:
            code_gen = QAIHMModelCodeGen.from_model(model)
            if code_gen.test_split != TestRunnerSplit.DEFAULT:
                custom_splits.setdefault(code_gen.test_split, []).append(model)
            elif code_gen.requires_aot_prepare:
                all_models_aot.append(model)
            else:
                all_models_jit.append(model)

        # Add custom splits
        for split_enum, split_models in sorted(
            custom_splits.items(), key=lambda x: x[0].value
        ):
            splits.append(
                {
                    "split_name": split_enum.name,
                    "models": ",".join(split_models),
                    "runs_on": split_enum.runs_on,
                }
            )

        # Split remaining torch models (JIT + AOT) into chunks
        num_default_splits = math.ceil(
            len(all_models_jit + all_models_aot) / max_pt_models_per_split
        )
        if max_pt_splits is not None:
            num_default_splits = min(num_default_splits, max_pt_splits)

        if num_default_splits > 0:
            estimates = _load_runtime_estimates(runtime_estimates_path) if stage else {}
            balanced = _split_aot_jit(
                all_models_aot,
                all_models_jit,
                num_default_splits,
                stage,
                estimates,
            )
            for i, models_in_split in enumerate(balanced):
                if models_in_split:
                    splits.append(
                        {
                            "split_name": f"torch_{i + 1}_of_{num_default_splits}",
                            "models": ",".join(models_in_split),
                        }
                    )

    # If there's only one split and it's an auto-generated default split, simplify the name to "torch"
    _custom_names = {s.name for s in TestRunnerSplit if s != TestRunnerSplit.DEFAULT}
    if len(splits) == 1 and splits[0]["split_name"] not in ("static", *_custom_names):
        splits[0]["split_name"] = "torch"

    for split in splits:
        if "runs_on" not in split:
            split["runs_on"] = None

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split models into chunks for parallel scorecard runs"
    )
    EnabledModelsEnvvar.add_arg(parser)
    parser.add_argument(
        "--max-num-pt-splits",
        type=int,
        default=None,
        help="Maximum number of default splits to create (does not include custom splits).",
    )
    parser.add_argument(
        "--max-models-per-pt-split",
        type=int,
        default=MAX_PT_MODELS_PER_SPLIT,
        help=f"Maximum models per default split (does not include custom splits, default: {MAX_PT_MODELS_PER_SPLIT})",
    )
    parser.add_argument(
        "--stage",
        choices=RUNTIME_ALL_STAGES,
        default=None,
        help=(
            "Stage to load-balance for. Splits are sized by recorded per-model "
            "runtime; falls back to alphabetical chunks if no data is available."
        ),
    )
    parser.add_argument(
        "--runtime-estimates-path",
        type=Path,
        default=None,
        help="Override path to model-runtime-estimates.yaml (default: checked-in intermediates copy).",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="json",
        help="Output format: 'json' for pretty JSON, 'github' for GitHub Actions matrix format",
    )
    parser.add_argument(
        "--collapse-to-single-split",
        action="store_true",
        help=(
            "Emit a single torch split with all models on the default runner. "
            "Use for unit-only runs where splitting buys nothing."
        ),
    )

    args = parser.parse_args()
    splits = split_torch_models(
        args.models,
        args.max_num_pt_splits,
        args.max_models_per_pt_split,
        stage=args.stage,
        runtime_estimates_path=args.runtime_estimates_path,
        collapse_to_single_split=args.collapse_to_single_split,
    )
    if args.output_format == "github":
        # Output as a single line JSON for GitHub Actions
        print(json.dumps(splits))
    else:
        # Pretty print JSON
        print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()

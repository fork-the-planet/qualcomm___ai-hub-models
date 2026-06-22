# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, TypeGuard

import numpy as np
import torch
from qai_hub import InputSpecs

from qai_hub_models import SampleInputsType, TargetRuntime
from qai_hub_models.configs.tensor_spec import (
    BboxFormat,
    BboxMetadata,
    ColorFormat,
    ImageMetadata,
    IoType,
    TensorSpec,
)

# ---------------------------------------------------------------------------
# InputSpec Type Definition
# ---------------------------------------------------------------------------
# PyTorch trace doesn't capture the input specs. Hence we need an additional
# InputSpec (name -> (shape, type)) when submitting profiling job to AI Hub.
# This is a subtype of qai_hub.InputSpecs
InputSpec = dict[str, TensorSpec]

# Output spec: maps output name -> TensorSpec. Structurally identical to
# InputSpec; defined here (rather than in configs.model_metadata) so that
# low-level utilities can reference it without importing model_metadata, which
# pulls in the scorecard package and would create an import cycle.
OutputSpec = dict[str, TensorSpec]

# Re-export for backwards compatibility
__all__ = [
    "BboxFormat",
    "BboxMetadata",
    "ColorFormat",
    "ImageMetadata",
    "InputSpec",
    "IoType",
    "OutputSpec",
    "TensorSpec",
    "broadcast_data_to_multi_batch",
    "get_batch_size",
    "make_torch_inputs",
    "str_to_torch_dtype",
    "to_hub_input_specs",
    "workbench_to_qihm_input_spec",
]


def workbench_to_qihm_input_spec(input_spec: InputSpecs) -> InputSpec:
    return {k: TensorSpec.from_workbench_spec(s) for k, s in input_spec.items()}


def to_hub_input_specs(
    input_spec: InputSpec,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """
    Convert InputSpec to hub-compatible format.

    This strips any TensorSpec metadata to produce a plain dict
    that can be passed to qai_hub APIs.

    Parameters
    ----------
    input_spec
        The InputSpec from model.get_input_spec()

    Returns
    -------
    dict[str, tuple[tuple[int, ...], str]]
        A hub-compatible input specification.
    """
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, entry in input_spec.items():
        if isinstance(entry, TensorSpec):
            result[name] = (entry.shape, entry.dtype)
        else:
            result[name] = entry
    return result


def is_input_spec(value: Any) -> TypeGuard[InputSpec]:
    """Check if value is an InputSpec (values are tuples or TensorSpecs, not dicts)."""
    if not isinstance(value, dict) or not value:
        return False
    return isinstance(next(iter(value.values())), (tuple, TensorSpec))


def is_input_spec_dict(value: Any) -> TypeGuard[dict[str, InputSpec]]:
    """Check if value is a dict of InputSpecs (values are dicts)."""
    if not isinstance(value, dict) or not value:
        return False
    return isinstance(next(iter(value.values())), dict)


def str_to_torch_dtype(s: str) -> torch.dtype:
    return dict(
        int32=torch.int32,
        int64=torch.int64,
        float32=torch.float32,
    )[s]


def make_torch_inputs(spec: InputSpec, seed: int | None = 42) -> list[torch.Tensor]:
    """Make sample torch inputs from input spec"""
    torch_input = []
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    for sp in spec.values():
        torch_dtype = str_to_torch_dtype(sp[1])
        if sp[1] in {"int32", "int64"}:
            t = torch.randint(10, sp[0], generator=generator).to(torch_dtype)
        else:
            t = torch.rand(sp[0], generator=generator).to(torch_dtype)
        torch_input.append(t)
    return torch_input


def get_batch_size(input_spec: InputSpec) -> int | None:
    """
    Derive the batch size from an input specification. Assumes the batch size
    is the first dimension in each shape. If two inputs differ in the value of the
    first dimension, return None, as we are unable to determine a batch size.
    """
    batch_size = None
    for spec in input_spec.values():
        if batch_size is None:
            batch_size = spec[0][0]
        elif batch_size != spec[0][0]:
            # Inputs differ in first dimension, so unable to determine a batch size
            return None
    return batch_size


def broadcast_data_to_multi_batch(
    spec: InputSpec, inputs: SampleInputsType
) -> SampleInputsType:
    """
    Attempts to broadcast the inputs to match the input spec if batch_size is > 1.
    If any samples do not match the specified input spec on any other dimension,
    the function throws an error.
    """
    batch_size = get_batch_size(spec)
    if batch_size == 1:
        return inputs
    return {
        name: [np.broadcast_to(sample, spec[name][0]) for sample in samples]
        for name, samples in inputs.items()
    }


def get_channel_last(
    spec: InputSpec | OutputSpec, runtime: TargetRuntime | None = None
) -> list[str]:
    return [
        k for k, v in spec.items() if v.apply_channel_last_during_compilation(runtime)
    ]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from qai_hub_models.utils.base_evaluator import BaseEvaluator, _DataLoader

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from qai_hub_models.models._shared.llm.generator import LLM_Generator


class LLMEvaluator(BaseEvaluator):
    """Base class for evaluators that run on an ``LLM_Generator``.

    Captures the contract the LLM evaluators share: they consume an
    ``LLM_Generator`` (not a plain ``torch.nn.Module``), iterate the dataset one
    sample at a time, and run a forward pass per sample. Subclasses implement
    ``add_batch`` to fold each sample's logits into their metric.
    """

    # Whether the generator should accumulate logits on the CPU rather than the
    # model's compute device. True for the forward-only evaluators here, which
    # retain full-sequence logits (batch, seq_len, vocab) across the whole
    # dataset and would otherwise exhaust GPU memory. Evaluators that run
    # autoregressive generation (and only need transient last-position logits)
    # set this False; see ``LLMResponseEvaluator``.
    accumulate_logits_on_cpu: bool = True

    # Set by every subclass __init__; the device inputs are moved to.
    device: torch.device

    def for_each_batch(
        self,
        generator: LLM_Generator,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: (
            Callable[[list[torch.Tensor], CausalLMOutputWithPast, torch.Tensor], None]
            | None
        ) = None,
    ) -> None:
        """Run the generator forward over each sample, invoking ``callback``."""
        total_samples = 0
        batch_size = 1
        num_samples = num_samples or len(data)
        with tqdm(
            total=num_samples,
            desc="Number of samples completed",
        ) as pbar:
            for sample in data:
                input_ids, attention_mask, ground_truth = sample  # type:ignore[misc]
                inputs = [input_ids, attention_mask]
                inputs = [inp.to(self.device) for inp in inputs]
                with torch.no_grad():
                    outputs = generator(*inputs)
                if callback:
                    callback(inputs, outputs, ground_truth)
                total_samples += 1
                pbar.update(batch_size)
                if total_samples >= num_samples:
                    break

    def add_from_dataset(
        self,
        model: torch.nn.Module,
        data: _DataLoader,
        eval_iterations: int | None = None,
    ) -> None:
        from qai_hub_models.models._shared.llm.generator import LLM_Generator

        assert isinstance(model, LLM_Generator), "This evaluator only works on LLMs"

        def _add_batch(
            _: list[torch.Tensor],
            outputs: CausalLMOutputWithPast,
            ground_truth: torch.Tensor,
        ) -> None:
            self.add_batch(outputs, ground_truth)

        self.for_each_batch(model, data, eval_iterations, _add_batch)

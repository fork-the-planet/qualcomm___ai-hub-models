# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.llm_helpers import (
    create_genie_config,
    log_evaluate_test_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    LLM_QNN,
)
from qai_hub_models.models.qwen3_4b import Model
from qai_hub_models.models.qwen3_4b.demo import qwen3_4b_chat_demo
from qai_hub_models.models.qwen3_4b.model import (
    HF_REPO_NAME,
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_PreSplit,
    Qwen3_4B_QuantizablePreSplit,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_8_elite_qrd
from qai_hub_models.scorecard.utils.testing_export_eval import run_llm_compile
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.dispatch import resolve_model, select_pipeline
from qai_hub_models.utils.export.result import MultiGraphCollectionExportResult

export_model = select_pipeline(resolve_model(MODEL_ID))

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [f"qwen3_4b_part_{i}_of_4.bin" for i in range(1, 5)]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 151936,
                "bos-token": 151643,
                "eos-token": 151645,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 128,
                        "pos-id-dim": 64,
                        "allow-async-init": False,
                        "rope-theta": 1000000,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                },
            },
        }
    }

    assert expected_config == actual_config


# Full model tests
@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT_W4A16", "wikitext", 13.72, 0, marks=pytest.mark.nightly),
        ("DEFAULT_W4A16", "mmlu", 0.646, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.76, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.72, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    dataset_cls = next(
        d
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    extra_kwargs = (
        {"_skip_quantsim_creation": False, "fp_model": None} if is_unquantized else {}
    )
    # Both FP and W4A16 run through the split-Parts wrapper so the reported
    # degradation isolates the pure quantization effect (matches the Llama 3.2
    # split models).
    actual_metric, _ = evaluate(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        kwargs=dict(
            checkpoint=checkpoint,
            **extra_kwargs,
        ),
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint=checkpoint,
        metric=task,
        value=actual_metric,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=0.03, atol=0)


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    # Calibrate on the PreSplit (monolithic QuantSim) like production; split
    # wrappers stack 4 Part sessions and OOM. Demo below still validates the split.
    checkpoint_path = test.setup_test_quantization(
        Qwen3_4B_QuantizablePreSplit,
        Qwen3_4B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
        use_dynamic_shapes=True,
    )
    qwen3_4b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    qwen3_4b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.nightly
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device", "checkpoint"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_8_elite_qrd, "DEFAULT_W4A16"),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    checkpoint: CheckpointSpec,
) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    result = run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            _skip_quantsim_creation=True,
            output_dir=test.GENIE_BUNDLES_ROOT,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()
    assert (genie_bundle_path / "sample_prompt.txt").exists()

    assert isinstance(result, MultiGraphCollectionExportResult)

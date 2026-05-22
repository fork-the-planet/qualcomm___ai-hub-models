# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from qai_hub_models.scripts import file_scorecard_regression_issue as mod
from qai_hub_models.scripts.file_scorecard_regression_issue import (
    build_issue_body,
)

# --- Fixtures ---
# Keys match PrettyTable column names from performance_diff.py and numerics_diff.py

PERF_REGRESSIONS = [
    {
        "Model ID": "resnet50",
        "Precision": "float",
        "Component": "resnet50",
        "Device": "Snapdragon 8 Gen 3",
        "Runtime": "TFLITE",
        "Prev Inference time": "10.0",
        "New Inference time": "20.0",
        "Kx slower": "2.0",
        "Job ID (prod)": "jnew789",
        "Previous Job ID (prod)": "jprev789",
    },
    {
        "Model ID": "mobilenet",
        "Precision": "w8a8",
        "Component": "mobilenet",
        "Device": "Snapdragon 8 Elite",
        "Runtime": "ONNX",
        "Prev Inference time": "5.0",
        "New Inference time": "15.0",
        "Kx slower": "3.0",
        "Job ID (prod)": "jnew012",
        "Previous Job ID (prod)": "jprev012",
    },
]

NUMERICS_REGRESSIONS = [
    {
        "Model ID": "yolov8_det",
        "Dataset Name": "coco-2017",
        "Metric Name": "mAP",
        "Device": "Snapdragon 8 Gen 3",
        "Precision": "float",
        "Runtime": "TFLITE",
        "FP Accuracy": "45.2 mAP",
        "Device Accuracy": "38.1 mAP",
        "Previous FP Accuracy": "45.2 mAP",
        "Previous Device Accuracy": "42.5 mAP",
    },
]


# --- Tests ---


def test_build_issue_body() -> None:
    """Full template rendering with both perf and numerics regressions."""
    body = build_issue_body(
        PERF_REGRESSIONS,
        NUMERICS_REGRESSIONS,
        "https://run",
        "https://perf",
        "https://num",
    )
    # Both sections present
    assert "## Performance Regressions" in body
    assert "## Numerics Regressions" in body
    # Perf data rendered
    assert "resnet50" in body
    assert "mobilenet" in body
    # Job IDs rendered as markdown links
    assert "[jnew789]" in body
    # Numerics data rendered
    assert "yolov8_det" in body
    # Links section
    assert "[Scorecard Run](https://run)" in body


def test_main_writes_output(tmp_path: Path) -> None:
    """End-to-end: main() loads JSON, renders template, writes output file."""
    perf_file = tmp_path / "perf-regressions-2x-2026-01-01.json"
    perf_file.write_text(json.dumps(PERF_REGRESSIONS))
    numerics_file = tmp_path / "numerics-regressions-2026-01-01.json"
    numerics_file.write_text(json.dumps(NUMERICS_REGRESSIONS))
    output_file = tmp_path / "regression-issue.json"

    with mock.patch(
        "sys.argv",
        [
            "file_scorecard_regression_issue.py",
            "--perf-regressions-json",
            str(perf_file),
            "--numerics-regressions-json",
            str(numerics_file),
            "--run-url",
            "https://run",
            "--perf-diff-url",
            "https://perf",
            "--numerics-diff-url",
            "https://num",
            "--output",
            str(output_file),
        ],
    ):
        mod.main()

    assert output_file.exists()
    issue = json.loads(output_file.read_text())
    assert "[Scorecard - Prod]" in issue["title"]
    assert "resnet50" in issue["body"]
    assert "yolov8_det" in issue["body"]
    assert issue["labels"] == ["p1", "scorecard"]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_models_cli._internal.aws import (
    DEFAULT_SESSION_DURATION,
    MAX_SESSION_DURATION,
    MIN_SESSION_DURATION,
    _get_session_duration,
)
from qai_hub_models_cli.envvars import AWS_SESSION_DURATION_ENVVAR


@pytest.mark.parametrize(
    ("envvar", "expected"),
    [
        (None, DEFAULT_SESSION_DURATION),
        ("7200", 7200),
        ("1000", MIN_SESSION_DURATION),
        ("50000", MAX_SESSION_DURATION),
        ("not-a-number", DEFAULT_SESSION_DURATION),
        ("", DEFAULT_SESSION_DURATION),
    ],
)
def test_get_session_duration(
    monkeypatch: pytest.MonkeyPatch, envvar: str | None, expected: int
) -> None:
    if envvar is None:
        monkeypatch.delenv(AWS_SESSION_DURATION_ENVVAR, raising=False)
    else:
        monkeypatch.setenv(AWS_SESSION_DURATION_ENVVAR, envvar)
    assert _get_session_duration() == expected

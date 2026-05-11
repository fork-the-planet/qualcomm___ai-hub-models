# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from qai_hub_models.utils.envvars import IsOnCIEnvvar


@pytest.fixture(scope="session", autouse=True)
def disable_evaluate_cache() -> Generator[None, None, None]:
    """This method should not be directly invoked in any unit test or scorecard."""
    with (
        patch(
            "qai_hub_models.utils.evaluate._populate_data_cache_impl",
            side_effect=RuntimeError(
                "Dataset cache should not be directly populated within scorecard."
            ),
        )
        if not IsOnCIEnvvar.get()
        else nullcontext()
    ):
        yield

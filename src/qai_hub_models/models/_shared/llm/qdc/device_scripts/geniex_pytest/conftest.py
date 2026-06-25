# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Appium session shim — QDC's APPIUM framework expects one; actual
device control happens via plain adb in test_geniex_bench_android.py.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Any

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

DEVICE_QDC_LOGS = "/data/local/tmp/QDC_logs"


def _make_options() -> AppiumOptions:
    options = AppiumOptions()
    options.set_capability("automationName", "UiAutomator2")
    options.set_capability("platformName", "Android")
    options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))
    return options


@pytest.fixture(scope="session", autouse=True)
def driver() -> Any:
    session = webdriver.Remote(
        command_executor="http://127.0.0.1:4723/wd/hub",
        options=_make_options(),
    )
    try:
        yield session
    finally:
        with contextlib.suppress(Exception):
            session.quit()


def _push_results_xml(xml_path: str) -> None:
    if not os.path.exists(xml_path):
        return
    subprocess.run(
        ["adb", "shell", f"mkdir -p {DEVICE_QDC_LOGS}"],
        check=False,
    )
    subprocess.run(
        ["adb", "push", xml_path, f"{DEVICE_QDC_LOGS}/results.xml"],
        check=False,
    )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    xml = getattr(session.config.option, "xmlpath", None) or "results.xml"
    _push_results_xml(xml)

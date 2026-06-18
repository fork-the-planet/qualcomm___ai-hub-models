# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.cli import _run_fetch, add_fetch_parser
from qai_hub_models_cli.common import (
    ASSET_FOLDER,
    STORE_URL,
)
from qai_hub_models_cli.fetch import (
    _asset_url,
    fetch,
    get_asset_url,
)

# ── _asset_url ───────────────────────────────────────────────────────


def test_asset_url_no_chipset() -> None:
    url, filename = _asset_url("mobilenet_v2", "tflite", "float", Version("0.45.0"))
    assert filename == "mobilenet_v2-tflite-float.zip"
    assert url == (
        f"{STORE_URL}/"
        f"{ASSET_FOLDER.format(model_id='mobilenet_v2', version='0.45.0')}/"
        f"mobilenet_v2-tflite-float.zip"
    )


def test_asset_url_with_chipset() -> None:
    url, filename = _asset_url(
        "mobilenet_v2",
        "qnn_context_binary",
        "w8a8",
        Version("0.45.0"),
        "qualcomm-snapdragon-8-gen-3",
    )
    assert (
        filename
        == "mobilenet_v2-qnn_context_binary-w8a8-qualcomm_snapdragon_8_gen_3.zip"
    )
    assert "qualcomm_snapdragon_8_gen_3" in url


def test_asset_url_with_enum_types() -> None:
    _, filename = _asset_url("mobilenet_v2", "tflite", "float", Version("0.45.0"))
    assert filename == "mobilenet_v2-tflite-float.zip"


# ── get_asset_url ────────────────────────────────────────────────────


def _mock_head(status_map: dict[str, int]) -> object:
    """Return a mock for requests.head that returns status codes based on URL substrings."""

    def _head(url: str, timeout: int = 10) -> MagicMock:
        resp = MagicMock()
        for pattern, status in status_map.items():
            if pattern in url:
                resp.status_code = status
                return resp
        resp.status_code = 404
        return resp

    return _head


def test_get_asset_url_found() -> None:
    with patch("qai_hub_models_cli.fetch.requests.head", _mock_head({"tflite": 200})):
        url = get_asset_url("mobilenet_v2", "tflite", "float", Version("0.45.0"))
    assert "mobilenet_v2-tflite-float.zip" in url


def test_get_asset_url_not_found() -> None:
    with (
        patch("qai_hub_models_cli.fetch.requests.head", _mock_head({})),
        pytest.raises(FileNotFoundError, match="No asset found"),
    ):
        get_asset_url("fake_model", "tflite", "float", Version("0.45.0"))


def test_get_asset_url_chipset_fallback() -> None:
    """When chipset URL returns 404, falls back to generic URL."""
    with patch(
        "qai_hub_models_cli.fetch.requests.head",
        _mock_head({"qualcomm_snapdragon": 404, "mobilenet_v2-tflite-float.zip": 200}),
    ):
        url = get_asset_url(
            "mobilenet_v2",
            "tflite",
            "float",
            Version("0.45.0"),
            chipset="qualcomm-snapdragon-8-gen-3",
        )
    assert "qualcomm_snapdragon" not in url
    assert "mobilenet_v2-tflite-float.zip" in url


def test_get_asset_url_chipset_found() -> None:
    """When chipset URL returns 200, uses it directly."""
    with patch(
        "qai_hub_models_cli.fetch.requests.head",
        _mock_head({"qualcomm_snapdragon": 200}),
    ):
        url = get_asset_url(
            "mobilenet_v2",
            "tflite",
            "float",
            Version("0.45.0"),
            chipset="qualcomm-snapdragon-8-gen-3",
        )
    assert "qualcomm_snapdragon" in url


def test_get_asset_url_unexpected_status() -> None:
    with (
        patch(
            "qai_hub_models_cli.fetch.requests.head",
            _mock_head({"tflite": 500}),
        ),
        pytest.raises(ConnectionError, match="Unexpected response"),
    ):
        get_asset_url("mobilenet_v2", "tflite", "float", Version("0.45.0"))


# ── fetch (integration with mocked network) ─────────────────────────


@patch("qai_hub_models_cli.fetch.get_asset_url")
@patch("qai_hub_models_cli.fetch.download")
def test_fetch_downloads_to_output_dir(
    mock_download: MagicMock,
    mock_get_url: MagicMock,
    tmp_path: Path,
) -> None:
    mock_get_url.return_value = "https://example.com/model-tflite-float.zip"
    mock_download.return_value = tmp_path / "model-tflite-float.zip"

    result = fetch(
        "model", "tflite", tmp_path, precision="float", version=Version("0.45.0")
    )
    assert result == tmp_path / "model-tflite-float.zip"
    mock_download.assert_called_once()
    mock_get_url.assert_called_once_with(
        "model", "tflite", "float", Version("0.45.0"), None
    )


@patch("qai_hub_models_cli.fetch.get_asset_url")
@patch("qai_hub_models_cli.fetch.download")
def test_fetch_extract_increments_dir(
    mock_download: MagicMock,
    mock_get_url: MagicMock,
    tmp_path: Path,
) -> None:
    mock_get_url.return_value = "https://example.com/model-tflite-float.zip"
    # Simulate existing extraction dir.
    (tmp_path / "model-tflite-float").mkdir()
    mock_download.return_value = tmp_path / "model-tflite-float-1"

    fetch("model", "tflite", tmp_path, extract=True, version=Version("0.45.0"))
    # download should have been called with the incremented path.
    call_args = mock_download.call_args
    assert "model-tflite-float-1" in str(call_args)


# ── _run_fetch (CLI arg handling) ────────────────────────────────────


def _make_args(overrides: dict[str, object] | None = None) -> MagicMock:
    """Create a mock argparse.Namespace with fetch defaults."""
    defaults: dict[str, object] = dict(
        model="mobilenet_v2",
        runtime="tflite",
        precision="float",
        chipset=None,
        qaihm_version=Version("0.45.0"),
        extract=True,
        output_dir=".",
        url_only=False,
        quiet=False,
    )
    if overrides:
        defaults.update(overrides)
    args = MagicMock()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


@patch(
    "qai_hub_models_cli.cli.get_asset_url",
    return_value="https://example.com/asset.zip",
)
def test_run_fetch_url_only_prints(
    mock_url: MagicMock,
) -> None:
    args = _make_args({"url_only": True})
    _run_fetch(args)
    mock_url.assert_called_once()


# ── add_fetch_parser ─────────────────────────────────────────────────


def test_add_fetch_parser_registers() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    fetch_parser = add_fetch_parser(subparsers)
    assert fetch_parser is not None

    args = parser.parse_args(["fetch", "mobilenet_v2", "--runtime", "tflite"])
    assert args.model == "mobilenet_v2"
    assert args.runtime == "tflite"
    assert args.precision == "float"
    assert args.extract is True
    assert args.url_only is False


def test_add_fetch_parser_output_dir() -> None:
    """Ensure -o/--output-dir is stored as output_dir on the namespace."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_fetch_parser(subparsers)

    args = parser.parse_args(["fetch", "model", "-r", "tflite", "-o", "/tmp/out"])
    assert args.output_dir == "/tmp/out"


@patch("qai_hub_models_cli.cli.get_model_asset_details")
@patch("qai_hub_models_cli.cli.fetch", return_value=Path("/out/model"))
@patch("qai_hub_models_cli.cli.print_upgrade_notice")
def test_run_fetch_with_parsed_args(
    _mock_notice: MagicMock,  # noqa: PT019
    mock_fetch: MagicMock,
    _mock_asset: MagicMock,  # noqa: PT019
) -> None:
    """Run _run_fetch with real parsed args to catch attribute name mismatches."""
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_fetch_parser(subparsers)
    args = parser.parse_args(["fetch", "mobilenet_v2", "-r", "tflite"])
    _run_fetch(args)
    mock_fetch.assert_called_once()


# ── upgrade notice ──────────────────────────────────────────────────


@patch("qai_hub_models_cli.cli.get_model_asset_details")
@patch("qai_hub_models_cli.cli.print_upgrade_notice")
@patch("qai_hub_models_cli.cli.fetch", return_value=Path("/out/model"))
@patch("qai_hub_models_cli.cli.__version__", "0.45.0")
def test_fetch_calls_upgrade_notice(
    mock_fetch: MagicMock,
    mock_notice: MagicMock,
    _mock_asset: MagicMock,  # noqa: PT019
) -> None:
    args = _make_args({"qaihm_version": "0.45.0"})
    _run_fetch(args)
    mock_notice.assert_called_once()


@patch("qai_hub_models_cli.cli.print_upgrade_notice")
@patch("qai_hub_models_cli.cli.fetch", return_value=Path("/out/model"))
@patch("qai_hub_models_cli.cli.__version__", "0.45.0")
def test_fetch_no_upgrade_when_quiet(
    mock_fetch: MagicMock,
    mock_notice: MagicMock,
) -> None:
    args = _make_args({"qaihm_version": "0.45.0", "quiet": True})
    _run_fetch(args)
    mock_notice.assert_not_called()

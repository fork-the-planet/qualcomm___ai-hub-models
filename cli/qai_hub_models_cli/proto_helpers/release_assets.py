# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.common import model_repo_url
from qai_hub_models_cli.proto import info_pb2
from qai_hub_models_cli.proto.platform_pb2 import ChipsetInfo, PlatformInfo
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry
from qai_hub_models_cli.proto_helpers.platform import (
    get_runtime_info,
    resolve_chipset,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    precision_str_to_proto,
    runtime_proto_to_str,
    runtime_str_to_proto,
)
from qai_hub_models_cli.utils import build_table
from qai_hub_models_cli.versions import CURRENT_VERSION


class AssetNotFoundError(FileNotFoundError):
    def __init__(self, *args: object, model_sharing_restricted: bool = False) -> None:
        self.model_sharing_restricted = model_sharing_restricted
        super().__init__(*args)


# Tool version proto fields paired with their display labels, in display order.
_TOOL_VERSION_LABELS: list[tuple[str, str]] = [
    ("qairt", "QAIRT"),
    ("onnx", "ONNX"),
    ("onnx_runtime", "ONNX Runtime"),
    ("tflite", "TFLite"),
    ("litert", "LiteRT"),
    ("ai_hub_models", "AI Hub Models"),
]


def format_tool_versions(tool_versions: ToolVersions) -> str:
    """Format the set tool versions as a comma-separated ``Label X.Y`` string."""
    parts = [
        f"{label} {value}"
        for field, label in _TOOL_VERSION_LABELS
        if (value := getattr(tool_versions, field))
    ]
    return ", ".join(parts) if parts else "—"


# Accepted tool names for an SDK version filter, mapped to their proto field.
# Includes the field name and a normalized form of the display label.
_SDK_FILTER_TOOLS: dict[str, str] = {
    **{field: field for field, _ in _TOOL_VERSION_LABELS},
    **{label.lower(): field for field, label in _TOOL_VERSION_LABELS},
}


def parse_sdk_version_filters(queries: list[str]) -> dict[str, str]:
    """Parse ``tool=version`` filter strings into a ``{tool: version}`` map.

    Each query uses ``tool=version`` syntax (e.g. ``"litert=1.4.4"``). Only the
    syntax is validated here; the tool name is normalized but resolved to a proto
    field later, in :func:`tool_versions_match`.

    Parameters
    ----------
    queries
        SDK version filter strings.

    Returns
    -------
    dict[str, str]
        Map of (normalized) tool name to (lower-cased) version substring.

    Raises
    ------
    ValueError
        If any query is not of the form ``tool=version``.
    """
    parsed: dict[str, str] = {}
    for query in queries:
        if "=" not in query:
            raise ValueError(
                f"Invalid SDK version filter {query!r}. "
                "Use 'tool=version' syntax, e.g. 'litert=1.4.4'."
            )
        tool, _, version = query.partition("=")
        tool = tool.strip().lower().replace("-", "_").replace(" ", "_")
        parsed[tool] = version.strip().lower()
    return parsed


def tool_versions_match(
    tool_versions: ToolVersions, sdk_versions: dict[str, str]
) -> bool:
    """Whether *tool_versions* satisfies every entry in *sdk_versions*.

    Parameters
    ----------
    tool_versions
        The asset's tool versions to test.
    sdk_versions
        Map of tool name (``qairt``, ``onnx``, ``onnx_runtime``, ``tflite``,
        ``litert``, ``ai_hub_models``, or a display label) to a version
        substring, as produced by :func:`parse_sdk_version_filters`. Each tool
        name is resolved to its proto field here, and its version is matched as
        a case-insensitive substring of the asset's value for that tool.

    Returns
    -------
    bool
        True if, for every entry, the asset's named tool version contains the
        requested substring.

    Raises
    ------
    ValueError
        If a tool name does not match a known tool.
    """
    for tool, version in sdk_versions.items():
        field = _SDK_FILTER_TOOLS.get(tool)
        if field is None:
            valid = ", ".join(name for name, _ in _TOOL_VERSION_LABELS)
            raise ValueError(f"Unknown SDK tool {tool!r}. Valid tools: {valid}.")
        if version not in getattr(tool_versions, field).lower():
            return False
    return True


@functools.lru_cache(maxsize=1)
def get_model_release_assets(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelReleaseAssets:
    """
    Fetch and cache the model release assets protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local release assets protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelReleaseAssets
        Parsed release assets protobuf containing download URLs for
        each available runtime, precision, and chipset combination.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    proto = fetch_model_proto(
        model,
        version,
        ModelReleaseAssets,
        cache_filename="release_assets.pb",
        manifest_url_field="release_assets",
        source_getter="get_release_assets_proto",
        local_path=local_path,
    )

    if not proto.assets:
        entry = get_manifest_entry(model, version)
        info_proto = get_model_info(model, version)

        if info_proto.HasField("llm_details"):
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_COMING_SOON
            ):
                raise AssetNotFoundError(
                    f"No pre-compiled model files are available for {entry.display_name}, but assets are coming soon. Stay tuned!"
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_US
            ):
                raise AssetNotFoundError(
                    f"If you have interest in downloading {entry.display_name}, reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_DOWNLOAD
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for download. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_PURCHASE
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for purchase. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )

        if info_proto.restrict_model_sharing:
            raise AssetNotFoundError(
                f"No pre-compiled model files for {entry.display_name} are available due to licensing"
                " restrictions. You can use the AI Hub Models package to manually"
                " export the model. See"
                f" {model_repo_url(entry.id, version)} for export instructions.",
                model_sharing_restricted=True,
            )

        raise AssetNotFoundError(
            f"No pre-compiled model files are available for {entry.display_name}. Reach out to us at"
            " qai-hub-support@qti.qualcomm.com."
        )

    return proto


def format_release_assets_table(
    release_assets: ModelReleaseAssets,
    chipsets: Iterable[ChipsetInfo],
    title: str | None = None,
) -> str:
    """Format a table of a model's download options.

    *chipsets* is used to display each asset's chipset by its marketing name.
    Returns only the table; use :func:`format_fetch_commands` for the
    accompanying ``fetch``/``devices`` command hints.
    """
    chipset_names = {c.name: c.marketing_name for c in chipsets}

    grouped: dict[tuple[str, str, str], list[str | None]] = {}
    for asset in release_assets.assets:
        prec = precision_proto_to_str(asset.precision)
        rt = runtime_proto_to_str(asset.runtime)
        sdk = (
            format_tool_versions(asset.tool_versions)
            if asset.HasField("tool_versions")
            else "—"
        )
        key = (prec, rt, sdk)
        chipset = (
            chipset_names.get(asset.chipset, asset.chipset)
            if asset.HasField("chipset")
            else None
        )
        grouped.setdefault(key, []).append(chipset)

    rows = []
    for (prec, rt, sdk), grouped_chipsets in grouped.items():
        if all(c is None for c in grouped_chipsets):
            chipset_str = "Universal"
        else:
            chipset_str = ", ".join(sorted(c for c in grouped_chipsets if c))
        rows.append([prec, rt, chipset_str, sdk])
    return build_table(
        ["Precision", "Runtime", "Chipsets", "SDK Versions"],
        rows,
        wrap_column="Chipsets",
        title=title,
        wrap_on_commas=True,
    )


def format_fetch_commands(
    release_assets: ModelReleaseAssets,
    model: str,
    subset: bool = False,
    runtime: str | None = None,
    precision: str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    sdk_versions: dict[str, str] | None = None,
    url_only: bool = False,
) -> str:
    """Format the ``fetch``/``devices`` command hints shown beneath a table.

    ``sdk_versions`` (a ``tool -> version`` map) is echoed into the suggested
    command as ``-s tool=version`` flags.
    """
    has_chipset_assets = any(
        asset.HasField("chipset") for asset in release_assets.assets
    )
    # Pre-fill the download command with the user's known values, falling back
    # to placeholders for anything they haven't specified.
    download_cmd = (
        f"qai_hub_models fetch {model} "
        f"-r {runtime or '<runtime>'} -p {precision or '<precision>'}"
    )
    if device is not None:
        download_cmd += f" -d '{device}'"
    elif chipset is not None:
        download_cmd += f" -c '{chipset}'"
    elif has_chipset_assets:
        download_cmd += " [ -c '<chipset>' || -d '<device>' ]"
    for tool, ver in (sdk_versions or {}).items():
        download_cmd += f" -s '{tool}={ver}'"
    if url_only:
        download_cmd += " --url-only"

    entries = []
    if subset:
        entries.append(("See all assets", f"qai_hub_models fetch {model} -i"))
    if has_chipset_assets:
        entries.append(("See devices per chipset", "qai_hub_models devices"))
    entries.append(
        ("Get an asset URL" if url_only else "Download an asset", download_cmd)
    )
    label_width = max(len(label) for label, _ in entries)
    return "\n".join(
        f"  {label + ':':<{label_width + 1}}  {cmd}" for label, cmd in entries
    )


def filter_release_assets(
    release_assets: ModelReleaseAssets,
    platform: PlatformInfo,
    runtime: Runtime.ValueType | str | None = None,
    precision: Precision.ValueType | str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    sdk_versions: dict[str, str] | None = None,
) -> ModelReleaseAssets:
    """
    Return a copy of *release_assets* keeping only assets matching the filters.

    Any filter left as ``None`` is not applied. *chipset* and *device* are
    mutually exclusive and both filter on the asset's chipset; universal assets
    (those without a chipset) always match the chipset/device filter, since they
    run on any chipset.

    Parameters
    ----------
    release_assets
        The model's release assets to filter.
    platform
        Platform registry used to resolve *chipset*/*device*.
    runtime
        Runtime enum value or string to filter on (e.g. ``"tflite"``).
    precision
        Precision enum value or string to filter on (e.g. ``"float"``).
    chipset
        Chipset reference (canonical ID, name, or alias) to filter on.
    device
        Device name to filter on; resolved to its chipset. Mutually exclusive
        with *chipset*.
    sdk_versions
        Map of tool proto field to version substring (see
        :func:`parse_sdk_version_filters`), e.g.
        ``{"litert": "1.4.4", "qairt": "2.20"}``. An asset must match *every*
        entry; each version is matched as a case-insensitive substring of the
        named tool's version. Assets without a named tool version set never match.

    Returns
    -------
    ModelReleaseAssets
        A new ``ModelReleaseAssets`` with the same metadata and only the
        assets that match every provided filter.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    KeyError
        If *runtime*, *precision*, *chipset*, or *device* is not known.
    """
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")

    runtime_val = runtime_str_to_proto(runtime) if runtime is not None else None
    precision_val = precision_str_to_proto(precision) if precision is not None else None
    chipset_name: str | None = None
    if chipset is not None or device is not None:
        chipset_name = resolve_chipset(platform, device=device, chipset=chipset).name

    filtered = ModelReleaseAssets(
        aihm_version=release_assets.aihm_version,
        model_id=release_assets.model_id,
    )
    for asset in release_assets.assets:
        if runtime_val is not None and asset.runtime != runtime_val:
            continue
        if precision_val is not None and asset.precision != precision_val:
            continue
        # Universal assets (no chipset) run on any chipset, so keep them when
        # filtering by chipset/device.
        if (
            chipset_name is not None
            and asset.HasField("chipset")
            and asset.chipset != chipset_name
        ):
            continue
        if sdk_versions and not (
            asset.HasField("tool_versions")
            and tool_versions_match(asset.tool_versions, sdk_versions)
        ):
            continue
        filtered.assets.add().CopyFrom(asset)
    return filtered


def get_model_asset_details(
    release_assets: ModelReleaseAssets,
    platform: PlatformInfo,
    runtime: Runtime.ValueType | str,
    precision: Precision.ValueType | str,
    chipset: str | None = None,
    device: str | None = None,
) -> ModelReleaseAssets.AssetDetails:
    """
    Look up a specific asset from a model's release assets.

    If the runtime is AOT-compiled, *chipset* (or *device*)
    is required and a chipset-specific asset is returned. Otherwise
    *chipset* is ignored and a universal asset is returned.

    Parameters
    ----------
    release_assets
        The model's release assets to search.
    platform
        Platform registry used to resolve *chipset*/*device* and the runtime.
    runtime
        Runtime enum value (e.g. ``RUNTIME_TFLITE``) or string
        (e.g. ``"tflite"``).
    precision
        Precision enum value (e.g. ``PRECISION_FLOAT``) or string
        (e.g. ``"float"``).
    chipset
        Chipset reference: canonical ID, name, or alias. Resolved to
        the canonical chipset ID. Required for AOT-compiled runtimes, ignored
        otherwise.
    device
        Device name to select the asset by; resolved to its chipset. Mutually
        exclusive with *chipset*.

    Returns
    -------
    ModelReleaseAssets.AssetDetails
        Matching asset entry with download URL, tool versions, etc.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    KeyError
        If no matching asset is found, if *chipset* is missing for an
        AOT-compiled runtime, or if *chipset*/*device* is not known.
    """
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")

    model = release_assets.model_id
    runtime_val = runtime_str_to_proto(runtime)
    is_aot = get_runtime_info(platform, runtime_val).is_aot_compiled

    errmsg: str | None = None
    if is_aot and chipset is None and device is None:
        errmsg = f"Chipset is required for AOT-compiled runtime {runtime!r}.\n\n"
    else:
        # Chipset/device only narrows AOT-compiled (chipset-specific) assets.
        matches = filter_release_assets(
            release_assets,
            platform,
            runtime,
            precision,
            chipset=chipset if is_aot else None,
            device=device if is_aot else None,
        )
        if matches.assets:
            return matches.assets[0]

    errmsg = errmsg or (
        f"No asset found for model={model!r} with runtime={runtime!r}, "
        f"precision={precision!r}, chipset={chipset!r}.\n"
        f"The model was found, but not with the requested runtime, precision, or chipset.\n\n"
    )
    errmsg += f"The following are valid fetch options for {model}:\n"
    errmsg += format_release_assets_table(release_assets, platform.chipsets)
    raise AssetNotFoundError(errmsg)

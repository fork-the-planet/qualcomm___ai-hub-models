# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version
from packaging.version import parse as parse_version
from prettytable import PrettyTable

from qai_hub_models_cli._internal.utils import is_internal_repo, use_internal_releases
from qai_hub_models_cli._version import __version__
from qai_hub_models_cli.common import (
    AIHUB_MODELS_URL,
    model_repo_url,
)
from qai_hub_models_cli.envvars import (
    VERBOSE_EXCEPTIONS_ENVVAR,
    bool_envvar_value,
)
from qai_hub_models_cli.fetch import fetch, get_asset_url
from qai_hub_models_cli.proto.info_pb2 import ModelDomain
from qai_hub_models_cli.proto.manifest_pb2 import ManifestModelEntry
from qai_hub_models_cli.proto.platform_pb2 import FormFactor, WebsiteWorld
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest
from qai_hub_models_cli.proto_helpers.platform import (
    filter_chipsets,
    filter_devices,
    format_chipsets_table,
    format_devices_table,
    format_similar_devices_table,
    get_platform,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    domain_proto_to_str,
    form_factor_proto_to_str,
    form_factor_str_to_proto,
    license_proto_to_str,
    os_str_to_proto,
    precision_proto_to_str,
    runtime_proto_to_str,
    tag_proto_to_str,
    use_case_proto_to_str,
    world_proto_to_str,
    world_str_to_proto,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    filter_release_assets,
    format_fetch_commands,
    format_release_assets_table,
    format_tool_versions,
    get_model_asset_details,
    get_model_release_assets,
    parse_sdk_version_filters,
)
from qai_hub_models_cli.utils import build_table, wrap_table_column
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    UnsupportedVersionError,
    get_supported_versions,
    normalize_version,
    print_upgrade_notice,
)


def _check_version_match() -> None:
    """Exit if qai_hub_models and qai_hub_models_cli versions differ."""
    try:
        cli_version = version("qai_hub_models_cli")
        models_version = version("qai_hub_models")
    except PackageNotFoundError:
        return
    if cli_version != models_version:
        print(
            f"Version mismatch: qai_hub_models_cli=={cli_version} "
            f"but qai_hub_models=={models_version}. "
            "Please reinstall both packages from the same version."
        )
        sys.exit(1)


def _parse_version(s: str) -> Version:
    """Argparse type function: normalize and parse a version string."""
    return parse_version(normalize_version(s))


def _add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-v/--version`` argument (stored as ``qaihm_version``)."""
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=_parse_version,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )


def _add_quiet_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the shared ``-q/--quiet`` flag with a command-specific help string."""
    parser.add_argument("-q", "--quiet", action="store_true", help=help_text)


def _add_chipset_attribute_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the chipset-attribute filters shared by the devices/chipsets commands."""
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Only show entries whose chipset supports fp16.",
    )
    parser.add_argument(
        "--htp-version",
        nargs="+",
        type=int,
        default=None,
        help="Filter by chipset HTP version(s).",
    )
    parser.add_argument(
        "--soc-model",
        nargs="+",
        type=int,
        default=None,
        help="Filter by chipset SoC model(s).",
    )


def _run_fetch(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])

    if args.info:
        all_assets = get_model_release_assets(args.model, args.qaihm_version)
        platform = get_platform(args.qaihm_version)
        release_assets = filter_release_assets(
            all_assets,
            platform,
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
            sdk_versions,
        )
        if not release_assets.assets:
            print("No release assets match the given filters.")
            return
        print(
            format_release_assets_table(
                release_assets,
                platform.chipsets,
                title="Release Assets",
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                # The user is already running -i, so don't suggest it again.
                subset=False,
                runtime=args.runtime,
                precision=args.precision,
                chipset=args.chipset,
                device=args.device,
                sdk_versions=sdk_versions,
            )
        )
        return

    try:
        if args.url_only:
            url = get_asset_url(
                model=args.model,
                runtime=args.runtime,
                precision=args.precision,
                version=args.qaihm_version,
                chipset=args.chipset,
                device=args.device,
                quiet=args.quiet,
                url_only=True,
                sdk_versions=sdk_versions,
            )
            print(url)
            return

        result = fetch(
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            device=args.device,
            version=args.qaihm_version,
            extract=args.extract,
            output_dir=args.output_dir,
            quiet=args.quiet,
            sdk_versions=sdk_versions,
        )
    except Exception as e:
        if args.quiet and not isinstance(
            e, (FileNotFoundError, UnsupportedVersionError)
        ):
            print(
                "Failed to fetch model. Consider excluding -q/--quiet from your command to reveal more logs."
            )
        raise

    if args.quiet:
        print(result)
        return

    if args.extract:
        print(f"Extracted to: {result}")
    else:
        print(f"Saved to: {result}")

    try:
        asset = get_model_asset_details(
            get_model_release_assets(args.model, args.qaihm_version),
            get_platform(args.qaihm_version),
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
        )
    except Exception:
        asset = None
    if asset is not None and asset.HasField("tool_versions"):
        print(
            f"\nThis download was verified with: "
            f"{format_tool_versions(asset.tool_versions)}\n"
            "Run the model with matching versions to match our reported numerics and performance. Other "
            "versions may behave differently or fail to run."
        )

    print_upgrade_notice()


def add_fetch_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fetch",
        help="Download a pre-compiled model asset.",
    )
    parser.add_argument("model", type=str.lower, help="Model ID (e.g. mobilenet_v2).")
    runtime_values = ", ".join(
        [
            runtime_proto_to_str(r)
            for r in Runtime.values()
            if r != Runtime.RUNTIME_UNSPECIFIED
        ]
    )
    parser.add_argument(
        "-r",
        "--runtime",
        default=None,
        type=str.lower,
        help=f"Target runtime. Known values: {runtime_values}. "
        "Older releases may support different values. "
        "Required unless -i/--info is given.",
    )
    precision_values = ", ".join(
        [
            precision_proto_to_str(p)
            for p in Precision.values()
            if p != Precision.PRECISION_UNSPECIFIED
        ]
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        type=str.lower,
        help=f"Model precision. Known values: {precision_values}. "
        "Older releases may support different values.",
    )
    # TODO(#18389): Add a list of valid chipsets
    # so the CLI can validate and suggest chipset names.
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Chipset name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target.add_argument(
        "-d",
        "--device",
        default=None,
        help="Device name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models devices` to see supported devices. Cannot be specified with chipset.",
    )
    parser.add_argument(
        "-s",
        "--sdk-version",
        nargs="+",
        default=None,
        type=str.lower,
        help="Filter by SDK/tool version using 'tool=version' syntax (e.g. "
        "'litert=1.4.4' or 'qairt=2.20'). Accepts multiple values; an asset "
        "must match all of them.",
    )
    _add_version_arg(parser)
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract the downloaded zip archive (default: true). Use --no-extract to skip.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Print the download URL only (do not download).",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="List the supported release assets and return without downloading. "
        "The runtime, precision, chipset, device, and sdk-version args act as filters.",
    )
    _add_quiet_arg(parser, "Suppress all output except the result path.")
    parser.set_defaults(func=_run_fetch)
    return parser


def _run_list_models(args: argparse.Namespace) -> None:
    manifest = get_manifest(args.qaihm_version)
    entries = sorted(manifest.models, key=lambda e: e.id)

    if args.domain:

        def _normalize_domain(s: str) -> str:
            return s.lower().replace("_", " ").replace("-", " ")

        domain_filter = _normalize_domain(args.domain)
        entries = [
            e
            for e in entries
            if _normalize_domain(domain_proto_to_str(e.domain)) == domain_filter
        ]

    if not entries:
        print("No models found.")
        return

    if args.quiet:
        for entry in entries:
            print(entry.id)
        return

    groups: dict[str, list[ManifestModelEntry]] = {}
    for entry in entries:
        domain = domain_proto_to_str(entry.domain)
        groups.setdefault(domain, []).append(entry)

    table = build_table(
        ["Name", "Domain"],
        [
            [f"{entry.display_name} ({entry.id})", domain]
            for domain, group in groups.items()
            for entry in group
        ],
        wrap_column="Name",
        title="Models",
    )
    print(table)

    print(f"Total: {len(entries)} models")
    print("Run `qai_hub_models info <model_id>` for details and download options.")
    print_upgrade_notice()


def add_list_models_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "models",
        help="List all available models.",
        description="List all models available in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    domain_values = ", ".join(
        domain_proto_to_str(d)
        for d in ModelDomain.values()
        if d != ModelDomain.MODEL_DOMAIN_UNSPECIFIED
    )
    parser.add_argument(
        "-d",
        "--domain",
        default=None,
        type=str.lower,
        help=f"Filter by domain. Known values: {domain_values}.",
    )
    _add_quiet_arg(parser, "Print model IDs only, one per line.")
    parser.set_defaults(func=_run_list_models)
    return parser


def _run_list_devices(args: argparse.Namespace) -> None:
    platform = get_platform(args.qaihm_version)
    devices = sorted(
        platform.devices,
        key=lambda d: (form_factor_proto_to_str(d.form_factor), d.name),
    )
    devices = filter_devices(
        devices,
        platform.chipsets,
        form_factor=[form_factor_str_to_proto(t) for t in args.type]
        if args.type
        else None,
        os=[os_str_to_proto(o) for o in args.os] if args.os else None,
        fp16=True if args.fp16 else None,
        htp_version=args.htp_version,
        soc_model=args.soc_model,
    )

    if not devices:
        print("No devices found.")
        return

    if args.quiet:
        for device in devices:
            print(device.name)
        return

    # Devices with a reference_chipset are "similar" devices whose perf numbers
    # are duplicated from another chipset; show them in a separate table.
    primary = [d for d in devices if not d.reference_chipset]
    similar = [d for d in devices if d.reference_chipset]

    print(format_devices_table(primary, platform.chipsets))
    print(f"Total: {len(primary)} devices.")

    if similar:
        print()
        print(format_similar_devices_table(similar, platform.chipsets))
        print(
            f"Total: {len(similar)} similar devices. NOTE: The similar devices table lists devices that have not "
            "been tested with AI Hub Models. However, the corresponding similar device / chipset "
            "serve as substitute compilation targets and have been tested. Assets built for the 'similar device' / 'similar chipset' "
            "are likely to run on the device, though performance and accuracy metrics may differ."
        )

    print(
        f"\nNOTE: This is a snapshot of devices tested with AI Hub Models v{args.qaihm_version}. AI Hub Workbench may support a different set of devices."
    )

    print("\nSee all supported chipsets using `qai-hub-models chipsets`.")

    print_upgrade_notice()


def add_list_devices_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "devices",
        help="List all supported devices.",
        description="List all devices supported in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    type_values = ", ".join(
        form_factor_proto_to_str(f)
        for f in FormFactor.values()
        if f != FormFactor.FORM_FACTOR_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        default=None,
        help=f"Filter by device type(s). Known values: {type_values}.",
    )
    parser.add_argument(
        "--os",
        nargs="+",
        default=None,
        help="Filter by operating system(s) (e.g. Android, Windows).",
    )
    _add_chipset_attribute_filter_args(parser)
    _add_quiet_arg(parser, "Print device names only, one per line.")
    parser.set_defaults(func=_run_list_devices)
    return parser


def _run_list_chipsets(args: argparse.Namespace) -> None:
    chipsets = sorted(
        get_platform(args.qaihm_version).chipsets,
        key=lambda c: (world_proto_to_str(c.world), c.marketing_name),
    )
    chipsets = filter_chipsets(
        chipsets,
        world=[world_str_to_proto(t) for t in args.type] if args.type else None,
        fp16=True if args.fp16 else None,
        htp_version=args.htp_version,
        soc_model=args.soc_model,
    )

    if not chipsets:
        print("No chipsets found.")
        return

    if args.quiet:
        for chipset in chipsets:
            print(chipset.marketing_name)
        return

    print(format_chipsets_table(chipsets))

    print(f"Total: {len(chipsets)} chipsets")
    print(
        f"\nNOTE: This is a snapshot of chipsets tested with AI Hub Models v{args.qaihm_version}. AI Hub Workbench may support a different set of devices."
    )
    print("\nSee all supported devices using `qai-hub-models devices`.")
    print_upgrade_notice()


def add_list_chipsets_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "chipsets",
        help="List all supported chipsets.",
        description="List all chipsets supported in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    type_values = ", ".join(
        world_proto_to_str(w)
        for w in WebsiteWorld.values()
        if w != WebsiteWorld.WEBSITE_WORLD_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        default=None,
        help=f"Filter by chipset type(s). Known values: {type_values}.",
    )
    _add_chipset_attribute_filter_args(parser)
    _add_quiet_arg(parser, "Print chipset names only, one per line.")
    parser.set_defaults(func=_run_list_chipsets)
    return parser


def _run_info(args: argparse.Namespace) -> None:
    info = get_model_info(args.model, args.qaihm_version)

    url = f"{AIHUB_MODELS_URL}/{info.id}"
    width = max(len(info.name), len(url)) + 4
    print("+" + "=" * width + "+")
    print(f"| {info.name:^{width - 2}} |")
    print(f"| {url:^{width - 2}} |")
    print("+" + "=" * width + "+")
    print()

    if info.description:
        desc_table = PrettyTable()
        desc_table.title = "Description"
        desc_table.header = False
        desc_table.align = "l"
        desc_table.add_row([info.description])
        wrap_table_column(desc_table, 0)
        print(desc_table)
        print()

    metadata_table = PrettyTable()
    metadata_table.header = False
    metadata_table.align = "l"
    if info.domain:
        metadata_table.add_row(["Domain", domain_proto_to_str(info.domain)])
    if info.use_case:
        metadata_table.add_row(["Use Case", use_case_proto_to_str(info.use_case)])
    if info.tags:
        metadata_table.add_row(
            ["Tags", ", ".join(tag_proto_to_str(t) for t in info.tags)]
        )
    if info.license_type:
        license_str = license_proto_to_str(info.license_type)
        if info.HasField("license_url"):
            license_str += f" ({info.license_url})"
        metadata_table.add_row(["License", license_str])
    if info.HasField("source_repo"):
        metadata_table.add_row(["Source Repo", info.source_repo])
    if info.HasField("research_paper"):
        title = (
            info.research_paper_title
            if info.HasField("research_paper_title")
            else "Paper"
        )
        metadata_table.add_row(["Paper", f"{title} ({info.research_paper})"])
    if metadata_table.rows:
        metadata_table.title = "Metadata"
        wrap_table_column(metadata_table, 1)
        print(metadata_table)
        print()

    if info.technical_details:
        details_table = PrettyTable()
        details_table.title = "Technical Details"
        details_table.header = False
        details_table.align = "l"
        for detail in info.technical_details:
            if detail.HasField("string_value"):
                val = detail.string_value
            elif detail.HasField("int_value"):
                val = str(detail.int_value)
            elif detail.HasField("float_value"):
                val = str(detail.float_value)
            else:
                val = ""
            details_table.add_row([detail.key, val])
        wrap_table_column(details_table, 1)
        print(details_table)
        print()

    try:
        release_assets = get_model_release_assets(args.model, args.qaihm_version)
        print(
            format_release_assets_table(
                release_assets,
                get_platform(args.qaihm_version).chipsets,
                title="Download Options",
            )
        )
        print()
        print(format_fetch_commands(release_assets, args.model))
        print()
        print(
            f"Most models can be further customized beyond what is offered by standard model downloads. Scripts that can export the model from source are available at {model_repo_url(info.id, args.qaihm_version)}"
        )
    except (FileNotFoundError, UnsupportedVersionError) as e:
        err_table = PrettyTable()
        err_table.title = "Download Options"
        err_table.header = False
        err_table.align = "l"
        err_table.add_row([str(e)])
        wrap_table_column(err_table, 0)
        print(err_table)


def add_info_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model.",
        description="Display model metadata including description, license, "
        "technical details, and available download options.",
    )
    parser.add_argument(
        "model",
        type=str.lower,
        help="Model ID or display name (e.g. mobilenet_v2).",
    )
    _add_version_arg(parser)
    parser.set_defaults(func=_run_info)
    return parser


def _run_versions(args: argparse.Namespace) -> None:
    supported = get_supported_versions(force_refresh=True)
    installed = CURRENT_VERSION

    if args.quiet:
        print(", ".join(str(v) for v in supported))
        return

    print("Supported AI Hub Models Versions:")
    labeled = [f"{v} (installed)" if v == installed else str(v) for v in supported]
    print(", ".join(labeled))
    print_upgrade_notice()


def add_versions_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "versions",
        help="List all AI Hub Models versions supported by this CLI.",
        description="List all AI Hub Models versions supported by this CLI. "
        "Shows which version is currently installed and whether newer versions are available.",
    )
    _add_quiet_arg(
        parser,
        "Print versions as a plain comma-separated list without the (installed) marker or upgrade notice.",
    )
    parser.set_defaults(func=_run_versions)
    return parser


def _run_validate_aws(args: argparse.Namespace) -> None:
    from qai_hub_models_cli._internal.aws import validate_credentials

    validate_credentials()


def add_validate_aws_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate_aws_credentials",
        help="Validate and refresh AWS credentials for internal release access.",
        description="Ensure the 'qaihm' AWS profile has valid credentials. "
        "If credentials are expired, refreshes them via saml2aws. "
        "Requires the [internal] extra (pip install qai_hub_models_cli[internal]).",
    )
    parser.set_defaults(func=_run_validate_aws)
    return parser


def main(args: list[str] | None = None) -> None:
    _check_version_match()

    parser = argparse.ArgumentParser(
        prog="qai_hub_models",
        description="Qualcomm AI Hub Models CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CURRENT_VERSION}",
    )
    subparsers = parser.add_subparsers()

    add_fetch_parser(subparsers)
    add_info_parser(subparsers)
    add_list_models_parser(subparsers)
    add_list_devices_parser(subparsers)
    add_list_chipsets_parser(subparsers)
    if use_internal_releases() or is_internal_repo():
        add_validate_aws_parser(subparsers)
    add_versions_parser(subparsers)

    parsed = parser.parse_args(args)
    if hasattr(parsed, "func"):
        try:
            parsed.func(parsed)
        except Exception as e:
            if bool_envvar_value(VERBOSE_EXCEPTIONS_ENVVAR):
                raise
            print(e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

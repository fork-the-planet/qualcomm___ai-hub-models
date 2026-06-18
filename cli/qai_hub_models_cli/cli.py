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
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest
from qai_hub_models_cli.proto_helpers.platform import (
    domain_proto_to_str,
    license_proto_to_str,
    precision_proto_to_str,
    runtime_proto_to_str,
    tag_proto_to_str,
    use_case_proto_to_str,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    format_release_assets_table,
    format_tool_versions,
    get_model_asset_details,
    get_model_release_assets,
)
from qai_hub_models_cli.utils import wrap_table_column
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


def _run_fetch(args: argparse.Namespace) -> None:
    try:
        if args.url_only:
            url = get_asset_url(
                args.model,
                args.runtime,
                args.precision,
                args.qaihm_version,
                args.chipset,
            )
            print(url)
            return

        result = fetch(
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            version=args.qaihm_version,
            extract=args.extract,
            output_dir=args.output_dir,
            quiet=args.quiet,
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
            args.model,
            args.runtime,
            args.precision,
            args.chipset,
            args.qaihm_version,
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
        required=True,
        type=str.lower,
        help=f"Target runtime. Known values: {runtime_values}. "
        "Older releases may support different values.",
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
        default="float",
        type=str.lower,
        help=f"Model precision. Known values: {precision_values}. "
        "Older releases may support different values. Default: float.",
    )
    # TODO(#18389): Add a list of valid chipsets
    # so the CLI can validate and suggest chipset names.
    parser.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Chipset name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub list-devices` (from the qai_hub package) to see valid names.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=_parse_version,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )
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
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except the result path.",
    )
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

    table = PrettyTable()
    table.field_names = ["Model", "Domain"]
    table.align = "l"
    for domain, group in groups.items():
        for entry in group:
            table.add_row([f"{entry.display_name} ({entry.id})", domain])
    wrap_table_column(table, 0)
    print(table)

    print(f"\nTotal: {len(entries)} models")
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
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=_parse_version,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )
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
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print model IDs only, one per line.",
    )
    parser.set_defaults(func=_run_list_models)
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
                args.model,
                title="Download Options",
            )
        )
        print(
            f"Most models can be further customized beyond what is offered by standard model downloads. Scripts that can export the model from source are available at {model_repo_url(info.id, args.qaihm_version)}"
        )
    except FileNotFoundError as e:
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
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=_parse_version,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )
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
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print versions as a plain comma-separated list without the (installed) marker or upgrade notice.",
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

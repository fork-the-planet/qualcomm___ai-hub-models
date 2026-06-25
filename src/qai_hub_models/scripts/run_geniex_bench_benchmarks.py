# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.models._shared.llm.perf_collection import (
    load_release_assets_for_model,
    update_perf_yaml,
)
from qai_hub_models.models._shared.llm.qdc.geniex_jobs import (
    GenieXBenchMetrics,
    submit_geniex_bench_to_qdc_device,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.fetch_prerelease_assets import download_prerelease_asset
from qai_hub_models.utils.path_helpers import MODEL_IDS

DEFAULT_DEVICES = "cs_x2_elite,cs_x_elite"

# One device per platform bucket in GenieXBenchQDCJobs.add_job_artifacts.
ALL_GENIEX_DEVICES = (
    "cs_x_elite",
    "cs_x2_elite",
    "cs_9075",
    "cs_8_elite_qrd",
    "cs_8_elite_gen_5_qrd",
)
LLAMACPP_DEVICE_ALIASES = ("cpu", "gpu", "npu")
LLAMACPP_CONTEXT_LENGTHS = [512, 1024, 4096]


def _qairt_precision(model_id: str) -> Precision:
    return QAIHMModelCodeGen.from_model(model_id).supported_precisions[0]


def _discover_models(filter_models: str | None, runtime: TargetRuntime) -> list[str]:
    if filter_models and filter_models.lower() != "all":
        candidates = [m.strip() for m in filter_models.split(",") if m.strip()]
    else:
        candidates = list(MODEL_IDS)

    out: list[str] = []
    for mid in candidates:
        cg = QAIHMModelCodeGen.from_model(mid)
        if any(cg.is_supported(p, runtime) for p in cg.supported_precisions):
            out.append(mid)
    return out


def discover_llamacpp_models(filter_models: str | None) -> list[str]:
    if filter_models and filter_models.lower() != "all":
        candidates = [m.strip() for m in filter_models.split(",") if m.strip()]
    else:
        candidates = list(MODEL_IDS)

    return [mid for mid in candidates if _llamacpp_asset(mid) is not None]


def _llamacpp_asset(model_id: str) -> tuple[Precision, str] | None:
    """Return (precision, gguf_url) for whichever precision keys the
    geniex_llamacpp asset in release-assets.yaml. Quants vary across models
    (q4_0 for most, mxfp4 for gpt_oss_20b, etc.) so we don't pin the key.
    """
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    for precision, prec_details in assets.precisions.items():
        asset = prec_details.universal_assets.get(ScorecardProfilePath.GENIEX_LLAMACPP)
        if asset and asset.download_url:
            return precision, asset.download_url
    return None


def discover_qairt_models(filter_models: str | None) -> list[str]:
    return _discover_models(filter_models, TargetRuntime.GENIEX_QAIRT)


def fetch_qairt_genie_bundle(
    model_id: str, precision: Precision, chipset: str, output_dir: Path
) -> tuple[Path, list[int]]:
    """Download/extract the CI-built genie bundle. Returns (bundle_dir, context_lengths)."""
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIE)
    if asset is None:
        available: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No genie release asset for {model_id!r} precision={precision} "
            f"chipset={chipset!r}. Available: {available or '<none>'}. "
            "Build release-assets.yaml before running geniex-bench QAIRT."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIE, precision, chipset
    )
    if not bundle_dir.exists():
        zip_path = download_prerelease_asset(
            asset,
            model_id=model_id,
            runtime=TargetRuntime.GENIE,
            precision=precision,
            chipset=chipset,
            output_folder=output_dir,
            verbose=True,
        )
        shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
        if not bundle_dir.exists():
            raise RuntimeError(
                f"Extracted genie bundle missing expected directory {bundle_dir}; "
                f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
            )

    metadata = ModelMetadata.from_json(bundle_dir / "metadata.json")
    if metadata is None or metadata.genie is None:
        raise RuntimeError(
            f"Genie bundle for {model_id!r} has no genie metadata at "
            f"{bundle_dir / 'metadata.json'}"
        )
    return bundle_dir, metadata.genie.context_lengths


def _scorecard_device(name: str) -> ScorecardDevice:
    return ScorecardDevice.get(name)


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Model",
                "Plugin",
                "Device",
                "Ctx",
                "Decode TPS",
                "Prefill TPS",
                "TTFT (ms)",
                "Status",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["model"],
                    r.get("plugin", ""),
                    r["device"],
                    r.get("ctx", ""),
                    r.get("decode_tps", ""),
                    r.get("prefill_tps", ""),
                    r.get("ttft_ms", ""),
                    r["status"],
                ]
            )


def write_summary(rows: list[dict]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return

    def _format_values(v: object, spec: str = ".2f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    with open(summary, "a") as f:
        f.write("## geniex-bench Benchmark Results\n\n")
        f.write(
            "| Model | Plugin | Device | Ctx | Decode TPS | Prefill TPS | TTFT (ms) | Status |\n"
        )
        f.write(
            "|-------|--------|--------|----:|-----------:|------------:|----------:|--------|\n"
        )
        f.writelines(
            f"| {r['model']} | {r.get('plugin', '-')} | {r['device']} | {r.get('ctx', '-')} | "
            f"{_format_values(r.get('decode_tps'))} | {_format_values(r.get('prefill_tps'))} | "
            f"{_format_values(r.get('ttft_ms'), '.1f')} | {r['status']} |\n"
            for r in rows
        )
        f.write("\n")


def run_geniex_bench_job(
    model_id: str,
    model_ref: str,
    device_token: str,
    api_token: str,
    context_lengths: list[int],
    save_dir_root: str,
    plugin: str,
) -> list[GenieXBenchMetrics]:
    sd = _scorecard_device(device_token)
    device_alias = ",".join(LLAMACPP_DEVICE_ALIASES) if plugin == "llama_cpp" else "npu"
    print(f"\n{'=' * 60}")
    print(f"Model:   {model_id}")
    print(f"Device:  {sd.name} ({sd.reference_device_name}, chipset={sd.chipset})")
    print(f"Plugin:  {plugin} (alias={device_alias})")
    print(f"Ref:     {model_ref}")
    print(f"Ctx:     {context_lengths}")
    print(f"{'=' * 60}")

    save_dir = os.path.join(save_dir_root, model_id, sd.name)
    return submit_geniex_bench_to_qdc_device(
        api_token=api_token,
        hub_device_name=sd.reference_device_name,
        chipset=sd.chipset,
        model_rows=[(model_id, model_ref)],
        context_lengths=context_lengths,
        plugin=plugin,
        device_alias=device_alias,
        job_name=f"geniex-bench {plugin} {model_id}",
        save_results_dir=save_dir,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run geniex-bench benchmarks on QDC.")
    ap.add_argument(
        "--models", default="all", help='Comma-separated model IDs or "all".'
    )
    ap.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="Comma-separated cs_* names or hub device names, "
        'or "all" for every geniex-bench-supported device.',
    )
    ap.add_argument(
        "--plugin",
        default="all",
        choices=["all", "llama_cpp", "qairt"],
    )
    ap.add_argument("--csv", default="geniex_bench_results.csv")
    ap.add_argument("--results-dir", default="geniex_bench_results")
    ap.add_argument("--skip-perf-update", action="store_true")
    ap.add_argument(
        "--perf-updates-json",
        default="geniex_perf_updates.json",
        help="JSON manifest of update_perf_yaml calls; replayed by run_geniex_perf_updates.py.",
    )
    args = ap.parse_args()

    api_token = os.environ.get("QDC_API_TOKEN")
    if not api_token:
        print("ERROR: QDC_API_TOKEN environment variable is not set", file=sys.stderr)
        return 1

    plugins = ["llama_cpp", "qairt"] if args.plugin == "all" else [args.plugin]

    if args.devices.strip().lower() == "all":
        devices = list(ALL_GENIEX_DEVICES)
    else:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    rows: list[dict] = []
    perf_updates: list[dict] = []
    for plugin in plugins:
        if plugin == "qairt":
            models = discover_qairt_models(args.models)
            if not models:
                print("No models support the GENIEX_QAIRT runtime.")
                continue
        else:
            models = discover_llamacpp_models(args.models)
            if not models:
                print("No models support the GENIEX_LLAMACPP runtime.")
                continue

        print(f"Plugin:  {plugin}")
        print(f"Models:  {models}")
        print(f"Devices: {devices}")

        for model_id in models:
            if plugin == "qairt":
                precision = _qairt_precision(model_id)
                llamacpp = None
            else:
                llamacpp = _llamacpp_asset(model_id)
                assert llamacpp is not None, (
                    f"{model_id} has no geniex_llamacpp asset in release-assets.yaml"
                )
                precision, _ = llamacpp
            for device_token in devices:
                sd = _scorecard_device(device_token)
                # Per-(model, device) failure must not abort the whole sweep.
                try:
                    if plugin == "qairt":
                        bundle_dir, ctx_list = fetch_qairt_genie_bundle(
                            model_id,
                            precision,
                            sd.chipset,
                            Path(args.results_dir) / "qairt_bundles",
                        )
                        metrics = run_geniex_bench_job(
                            model_id,
                            str(bundle_dir),
                            device_token,
                            api_token,
                            ctx_list,
                            args.results_dir,
                            plugin,
                        )
                    else:
                        assert llamacpp is not None
                        _, model_url = llamacpp
                        metrics = run_geniex_bench_job(
                            model_id,
                            model_url,
                            device_token,
                            api_token,
                            LLAMACPP_CONTEXT_LENGTHS,
                            args.results_dir,
                            plugin,
                        )
                except Exception as e:
                    print(
                        f"ERROR: geniex-bench job failed for {model_id} @ "
                        f"{sd.name} (plugin={plugin}): {e}",
                        file=sys.stderr,
                    )
                    rows.append(
                        {
                            "model": model_id,
                            "plugin": plugin,
                            "device": sd.name,
                            "status": "failed",
                        }
                    )
                    continue

                if not metrics:
                    rows.append(
                        {
                            "model": model_id,
                            "plugin": plugin,
                            "device": sd.name,
                            "status": "no_metrics",
                        }
                    )
                    continue

                for m in metrics:
                    base_plugin = m.plugin or plugin
                    plugin_label = (
                        f"{base_plugin}_{m.device_alias}"
                        if base_plugin == "llama_cpp" and m.device_alias
                        else base_plugin
                    )
                    rows.append(
                        {
                            "model": model_id,
                            "plugin": plugin_label,
                            "device": sd.name,
                            "ctx": m.context_length,
                            "decode_tps": m.decode_tps,
                            "prefill_tps": m.prefill_tps,
                            "ttft_ms": m.ttft_ms,
                            "status": "success",
                        }
                    )
                    if not args.skip_perf_update:
                        if plugin == "qairt":
                            profile_path = ScorecardProfilePath.GENIEX_QAIRT
                        else:
                            profile_path = ScorecardProfilePath.GENIEX_LLAMACPP
                        # ttft_max_ms=ttft_ms: geniex measures TTFT at full ctx.
                        update_kwargs = dict(
                            model_id=model_id,
                            device_name=sd.reference_device_name,
                            precision=str(precision),
                            context_length=m.context_length,
                            tps=m.decode_tps,
                            ttft_ms=m.ttft_ms,
                            prefill_tps=m.prefill_tps,
                            ttft_max_ms=m.ttft_ms,
                            profile_path=profile_path.value,
                            desired_compute_unit=m.device_alias,
                        )
                        perf_updates.append(update_kwargs)
                        update_perf_yaml(
                            model_id=model_id,
                            device_name=sd.reference_device_name,
                            precision=precision,
                            context_length=m.context_length,
                            tps=m.decode_tps,
                            ttft_ms=m.ttft_ms,
                            prefill_tps=m.prefill_tps,
                            ttft_max_ms=m.ttft_ms,
                            profile_path=profile_path,
                            desired_compute_unit=m.device_alias,
                        )

    print(f"\n{'=' * 60}\nRESULTS SUMMARY\n{'=' * 60}")
    for r in rows:
        if r["status"] == "success":
            print(
                f"  {r['model']} @ {r['device']} ctx={r['ctx']}: "
                f"decode={r['decode_tps']:.2f} prefill={r['prefill_tps']:.2f} "
                f"TTFT={r['ttft_ms']:.1f}ms"
            )
        else:
            print(f"  {r['model']} @ {r['device']}: {r['status']}")

    write_csv(rows, args.csv)
    print(f"\nResults saved to {args.csv}")
    with open(args.perf_updates_json, "w") as f:
        json.dump(perf_updates, f, indent=2)
    print(f"Wrote {len(perf_updates)} perf.yaml updates to {args.perf_updates_json}")
    write_summary(rows)
    failed = [r for r in rows if r["status"] != "success"]
    if not rows or failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

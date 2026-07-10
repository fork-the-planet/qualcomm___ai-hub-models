# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import multiprocessing
import os
import shutil
import sys
import traceback
from itertools import cycle
from pathlib import Path

import pandas as pd
import ruamel.yaml

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import load_similar_devices
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    BranchEnvvar,
    DateFormatEnvvar,
    DeploymentEnvvar,
    EnabledModelsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar,
    SpecialModelSetting,
    SpecialPrecisionSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.results.code_gen import (
    remove_numerics_failures,
    remove_perf_failures,
    update_code_gen_accuracy_failure_reasons,
    update_code_gen_failure_reasons,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff
from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff
from qai_hub_models.scorecard.results.scorecard_summary import (
    ModelTestConfig,
)
from qai_hub_models.scorecard.results.spreadsheet import ResultsSpreadsheet
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    ComponentNamesYaml,
    GraphNamesYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    PreQDQCompileScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
    ToolVersionsByPathYaml,
    get_model_component_and_graph_names,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.utils.numerics_yaml_helpers import (
    create_numerics_yaml,
    get_chipset_registry,
)
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    deployment_is_prod,
    get_default_hub_deployment,
    get_scorecard_client_or_raise,
    set_default_hub_client,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

# If the precision is any one of these two values, add it to the branch column
# to allow tableau to differentiate different types of scorecards
SPECIAL_PRECISIONS = ["bench", "default_quantized"]


def read_jobs_config(config_path: str) -> dict:
    """Read yaml files."""
    yaml = ruamel.yaml.YAML()
    with open(config_path) as file:
        return yaml.load(file)


def write_jobs_config(config: dict, path: str) -> None:
    """Write yaml files with special characters like copyright logo, etc."""
    yaml = ruamel.yaml.YAML()
    with open(path, "w") as file:
        yaml.dump(config, file)


def _merge_existing_accuracy_data(
    new_df: pd.DataFrame, models: set[str]
) -> pd.DataFrame:
    old_df = pd.read_csv(ScorecardArtifact.ACCURACY_CSV.intermediates_path)
    old_df = old_df[~old_df.model_id.isin(models)]
    return pd.concat([old_df, new_df])


def remove_failed_jobs(config: dict) -> None:
    """
    Failed jobs need to be in the config to get their job ids for summary but
    we want to delete them before writing to perf.yaml
    """
    for model_config in config["models"]:
        for perf_metrics in model_config["performance_metrics"]:
            for key in list(perf_metrics.keys()):
                if (
                    "job_id" in perf_metrics[key]
                    and perf_metrics[key]["inference_time"] == "null"
                ):
                    del perf_metrics[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser)
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar.add_arg(parser)
    StaticModelsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--gen-csv",
        action="store_true",
        help="Generate a csv that summarizes the profile and compile steps.",
    )
    parser.add_argument(
        "--gen-perf-summary",
        action="store_true",
        help="Generate a summary of the performance per model, and update perf yaml files.",
    )
    parser.add_argument(
        "--sync-code-gen",
        action="store_true",
        help="Sync code generation YAML with failures & successes in the scorecard YAML. If compile job fails for any device, that path is skipped. If the profile job for the default export device fails, that path is skipped.",
    )
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    BranchEnvvar.add_arg(parser)
    DateFormatEnvvar.add_arg_group(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--accuracy-csv-path",
        type=str,
        default=str(ScorecardArtifact.ACCURACY_CSV.path),
        help="Accuracy CSV input. When present and non-empty, drives numerics-yaml updates.",
    )
    return parser.parse_args()


def process_model(
    model_id: str,
    deployment: str,
    static_models_dir: Path,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
    pre_qdq_job_yamls: PreQDQCompileScorecardJobYaml,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None] | None:
    """
    Process results for a single model.

    Returns ``None`` on per-model failure (logged to stderr) so one bad model
    doesn't take down the whole multiprocessing batch.

    Parameters
    ----------
    model_id
        Model identifier.
    deployment
        Deployment environment.
    static_models_dir
        Directory containing static model configurations.
    component_names_yaml
        YAML containing component names for each model.
    graph_names_yaml
        YAML containing graph names for each model component.
    pre_qdq_job_yamls
        YAML containing pre qdq compile job information.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    result : tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None] | None
        ``None`` if processing this model raised an exception. Otherwise a
        3-tuple of:

        * spreadsheet — model spreadsheet, or empty spreadsheet if not gen_csv.
        * previous_perf — previous (on disk) perf yaml, or None if not
          gen_perf_summary. Always None for static models.
        * current_perf — current (from this scorecard) perf yaml, or None if
          not gen_perf_summary. Always None for static models.
    """
    try:
        if model_id in MODEL_IDS:
            # This model has an end to end pyTorch recipe.
            return process_e2e_recipe_model(
                model_id,
                component_names_yaml,
                graph_names_yaml,
                pre_qdq_job_yamls,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                gen_csv,
                sync_code_gen,
                gen_perf_summary,
                write_model_card,
            )
        # This model was uploaded statically (as a single file).
        if gen_csv:
            spreadsheet = process_static_file_model(
                model_id,
                deployment,
                static_models_dir,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
            )
        else:
            spreadsheet = ResultsSpreadsheet()
        return (spreadsheet, None, None)
    except Exception:
        # Skip this model so one bad input doesn't kill the multiprocessing
        # pool; the aggregate raise at the end of __main__ still fails the
        # job after assets land.
        print(
            f"{model_id} result processing failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None


def _get_pytorch_tags(model_info: QAIHMModelInfo) -> list[str]:
    tags = [tag.value for tag in model_info.tags]
    tags.append("pytorch")
    tags.append(model_info.status.value)
    return tags


def _get_static_tags(model_info: ScorecardModelConfig) -> list[str]:
    tags = [tag.name for tag in model_info.tags]
    tags.append("static")
    tags.append("private")
    tags.append(f"bu-{model_info.bu_owner.value}")
    return tags


def process_e2e_recipe_model(
    model_id: str,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
    pre_qdq_job_yamls: PreQDQCompileScorecardJobYaml,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None]:
    """
    Process results for a model with an end-to-end pyTorch recipe.

    Parameters
    ----------
    model_id
        Model identifier.
    component_names_yaml
        YAML containing component names for each model.
    graph_names_yaml
        YAML containing graph names for each model component.
    pre_qdq_job_yamls
        YAML containing pre qdq compile job information.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    spreadsheet : ResultsSpreadsheet
        Model spreadsheet, or empty spreadsheet if not gen_csv.
    previous_perf : QAIHMModelPerf | None
        Previous (on disk) perf yaml, or None if not gen_perf_summary.
    current_perf : QAIHMModelPerf | None
        Current (from this scorecard) perf yaml, or None if not gen_perf_summary.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load configs
    model_info = QAIHMModelInfo.from_model(model_id)
    cj = model_info.code_gen_config

    # Skip certain models
    if cj.is_precompiled or cj.skip_hub_tests_and_scorecard or cj.skip_scorecard:
        return ResultsSpreadsheet(), None, None

    # Get enabled test paths for this model
    component_names, graph_names, component_graph_names = (
        get_model_component_and_graph_names(
            model_id, component_names_yaml, graph_names_yaml
        )
    )
    test_params = ModelTestConfig.from_recipe_model(
        model_info, component_names, graph_names, component_graph_names
    )

    # Get summaries for this model and its components.
    print_with_id("Loading summary")
    summaries = test_params.get_all_export_test_summaries(
        pre_qdq_job_yamls,
        quantize_job_yamls,
        compile_job_yamls,
        link_job_yamls,
        profile_job_yamls,
        inference_job_yamls,
    )

    entries: ResultsSpreadsheet = ResultsSpreadsheet()
    code_gen_config = QAIHMModelCodeGen.from_model(model_id)
    entries.set_model_metadata(
        model_id,
        model_info.domain,
        model_info.use_case,
        _get_pytorch_tags(model_info),
        known_failure_reasons=model_info.code_gen_config.disabled_paths,
        default_quantized_precision=code_gen_config.default_quantized_precision,
        default_device=ScorecardDevice.get(code_gen_config.default_device),
    )
    if gen_csv:
        print_with_id("Adding to Spreadsheet")
        for export_test_summary in summaries:
            entries.append_export_test_summary(export_test_summary)

    if sync_code_gen and not cj.freeze_perf_yaml and not cj.skips_profile_and_inference:
        # Enable or disable runtimes on this model depending on whether the default device has passing jobs
        update_code_gen_failure_reasons(summaries, test_params.enabled_paths, cj)
        code_gen_path = cj.to_model_yaml(model_id)
        print_with_id(f"Updated Runtime Failure Reasons in {code_gen_path}")

        # Update model status & reason, if applicable
        if update_model_publish_status(model_info):
            info_yaml_path, _ = model_info.to_model_yaml(write_code_gen=False)
            print_with_id(pstr=f"Updated publish status at {info_yaml_path}")

    model_card = QAIHMModelPerf()
    prev_model_card = QAIHMModelPerf()
    if gen_perf_summary:
        print_with_id("Writing Performance YAML")

        # Build model card
        model_card = QAIHMModelPerf()
        model_card_without_failures = QAIHMModelPerf() if write_model_card else None
        for summary in summaries:
            summary.add_to_perf(model_card, include_failures=True)
            if model_card_without_failures and summary.params.path.is_published:
                summary.add_to_perf(model_card_without_failures, include_failures=False)

        if model_card_without_failures:
            model_card_without_failures.apply_similar_devices(load_similar_devices())

        # Load old model card and write new model card
        prev_model_card = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        if (
            not cj.freeze_perf_yaml
            and not cj.skips_profile_and_inference
            and model_card_without_failures
        ):
            card_path = model_card_without_failures.to_model_yaml(model_id)
            print_with_id(f"Wrote {card_path}")

    return entries, prev_model_card, model_card


def process_static_file_model(
    model_id: str,
    deployment: str,
    models_dir: Path,
    compile_job_yamls: CompileScorecardJobYaml | None,
    link_job_yamls: LinkScorecardJobYaml | None,
    profile_job_yamls: ProfileScorecardJobYaml | None,
    inference_job_yamls: InferenceScorecardJobYaml | None,
) -> ResultsSpreadsheet:
    """
    Process results for a static model (uploaded onnx or traced pyTorch file).

    Returns model spreadsheet.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load config
    model_info = ScorecardModelConfig.from_yaml(models_dir / (model_id + ".yaml"))
    test_params = ModelTestConfig.from_static_model(model_info)

    # Get summaries for this model and its components.
    with default_hub_client_as(
        get_scorecard_client_or_raise(deployment, model_info.restrict_access)
    ):
        summaries = test_params.get_all_export_test_summaries(
            None,
            None,
            compile_job_yamls,
            link_job_yamls,
            profile_job_yamls,
            inference_job_yamls,
        )

        print_with_id("Adding to Spreadsheet")
        entries = ResultsSpreadsheet()
        entries.set_model_metadata(
            model_id,
            model_info.domain,
            model_info.use_case,
            _get_static_tags(model_info),
            default_quantized_precision=None,
            default_device=model_info.devices[0],
        )
        for export_test_summary in summaries:
            entries.append_export_test_summary(export_test_summary)

        return entries


if __name__ == "__main__":
    args = parse_args()
    static_model_dir: Path = args.static_models_dir

    # Verify args are compatible with the chosen deployment.
    using_prod_hub = deployment_is_prod(args.deployment)
    if not using_prod_hub and args.sync_code_gen:
        print("Warning: Can't sync code gen if deployment is not prod.")
        args.sync_code_gen = False

    os.makedirs(args.artifacts_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # List of models for which to generate perf.
    pytorch_models, static_models = validate_and_split_enabled_models(
        args.models, static_model_dir
    )
    all_models = SpecialModelSetting.ALL in args.models
    model_list = sorted(pytorch_models.union(static_models))

    # Load datestr
    date = DateFormatEnvvar.parse(args.date, args.date_format)

    # Set client to use target deployment
    set_default_hub_client(get_scorecard_client_or_raise(args.deployment))

    # Load Base YAMLs
    if using_prod_hub:
        # Load previous scorecard state
        component_names_yaml = ComponentNamesYaml.from_intermediates()
        graph_names_yaml = GraphNamesYaml.from_intermediates()
        pre_qdq_job_yamls = PreQDQCompileScorecardJobYaml.from_intermediates()
        quantize_job_yamls = QuantizeScorecardJobYaml.from_intermediates()
        compile_job_yamls = CompileScorecardJobYaml.from_intermediates()
        link_job_yamls = LinkScorecardJobYaml.from_intermediates()
        profile_job_yamls = ProfileScorecardJobYaml.from_intermediates()
        inference_job_yamls = InferenceScorecardJobYaml.from_intermediates()

        # Erase jobs for models we're collecting results for, if applicable
        if args.ignore_existing_intermediate_jobs:
            if all_models:
                component_names_yaml.clear()
                graph_names_yaml.clear()
                pre_qdq_job_yamls.clear()
                quantize_job_yamls.clear()
                compile_job_yamls.clear()
                link_job_yamls.clear()
                profile_job_yamls.clear()
                inference_job_yamls.clear()
            else:
                for model in model_list:
                    component_names_yaml.clear(model)
                    graph_names_yaml.clear(model)
                    pre_qdq_job_yamls.clear(model)
                    quantize_job_yamls.clear(model)
                    compile_job_yamls.clear(model)
                    link_job_yamls.clear(model)
                    profile_job_yamls.clear(model)
                    inference_job_yamls.clear(model)
    else:
        # Previous scorecard state is applicable only on prod
        component_names_yaml = ComponentNamesYaml()
        graph_names_yaml = GraphNamesYaml()
        pre_qdq_job_yamls = PreQDQCompileScorecardJobYaml()
        quantize_job_yamls = QuantizeScorecardJobYaml()
        compile_job_yamls = CompileScorecardJobYaml()
        link_job_yamls = LinkScorecardJobYaml()
        profile_job_yamls = ProfileScorecardJobYaml()
        inference_job_yamls = InferenceScorecardJobYaml()

    # Capture the previous (pre-merge) compile- and inference-job yamls for
    # the regression reports' "Previous *" columns before the in-memory merge
    # below makes current and previous indistinguishable.
    previous_compile_jobs = CompileScorecardJobYaml(dict(compile_job_yamls.mapping))
    previous_inference_jobs = InferenceScorecardJobYaml(
        dict(inference_job_yamls.mapping)
    )

    # Append job results from test artifacts
    component_names_yaml.mapping.update(
        ComponentNamesYaml.from_test_artifacts().mapping
    )
    graph_names_yaml.mapping.update(GraphNamesYaml.from_test_artifacts().mapping)
    pre_qdq_job_yamls.update(PreQDQCompileScorecardJobYaml.from_test_artifacts())
    quantize_job_yamls.update(QuantizeScorecardJobYaml.from_test_artifacts())
    compile_job_yamls.update(CompileScorecardJobYaml.from_test_artifacts())
    link_job_yamls.update(LinkScorecardJobYaml.from_test_artifacts())
    profile_job_yamls.update(ProfileScorecardJobYaml.from_test_artifacts())
    inference_job_yamls.update(InferenceScorecardJobYaml.from_test_artifacts())
    current_compile_jobs = CompileScorecardJobYaml.from_test_artifacts()
    current_inference_jobs = InferenceScorecardJobYaml.from_test_artifacts()

    # Extract Data from Models
    if len(model_list) > 1:
        # Use multiprocessing for multiple models because getting jobs from Hub is slow
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.starmap(
            process_model,
            zip(
                model_list,
                cycle([args.deployment]),
                cycle([static_model_dir]),
                cycle([component_names_yaml]),
                cycle([graph_names_yaml]),
                cycle([pre_qdq_job_yamls]),
                cycle([quantize_job_yamls]),
                cycle([compile_job_yamls]),
                cycle([link_job_yamls]),
                cycle([profile_job_yamls]),
                cycle([inference_job_yamls]),
                cycle([args.gen_csv]),
                cycle([args.sync_code_gen]),
                cycle([args.gen_perf_summary]),
                cycle([using_prod_hub]),
            ),
        )
        pool.close()
        # join() ensures worker stderr is fully flushed before we print the
        # failure summary at the end of __main__.
        pool.join()
    else:
        # Single model option for that allows breakpoints
        model_summaries = [
            process_model(
                model_list[0],
                args.deployment,
                static_model_dir,
                component_names_yaml,
                graph_names_yaml,
                pre_qdq_job_yamls,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                args.gen_csv,
                args.sync_code_gen,
                args.gen_perf_summary,
                using_prod_hub,
            )
        ]

    perf_report: PerformanceDiff | None = None
    if args.gen_perf_summary:
        perf_report = PerformanceDiff(
            current_compile_jobs=current_compile_jobs,
            previous_compile_jobs=previous_compile_jobs,
        )
    spreadsheet = ResultsSpreadsheet() if args.gen_csv else None
    if spreadsheet is not None:
        spreadsheet.set_date(date)
        # Tableau wants to differentiate between different types of scorecards
        # So mark them as such in the branch column.
        branch = args.branch
        precisions = args.precisions
        for precision in precisions:
            if isinstance(precision, SpecialPrecisionSetting):
                branch += f" - {precision.value}"
        spreadsheet.set_branch(branch)

    # Numerics setup. Skipped (no-op) when the accuracy CSV is missing/empty —
    # this is the perf-only run path.
    accuracy_path = Path(args.accuracy_csv_path)
    accuracy_csv_present = accuracy_path.exists() and accuracy_path.stat().st_size > 0
    accuracy_df = pd.read_csv(accuracy_path) if accuracy_csv_present else None
    chipset_registry = get_chipset_registry() if accuracy_csv_present else None
    global_numerics_diff: NumericsDiff | None = (
        NumericsDiff(
            current_inference_jobs=current_inference_jobs,
            previous_inference_jobs=previous_inference_jobs,
        )
        if accuracy_csv_present
        else None
    )

    failed_model_ids: list[str] = []
    for model_id, model_summary in zip(model_list, model_summaries, strict=False):
        if model_summary is None:
            failed_model_ids.append(model_id)
            continue
        model_spreadsheet, prev_model_card, curr_model_card = model_summary

        # Combine model spreadsheet with group spreadsheet
        if spreadsheet is not None:
            spreadsheet.combine(model_spreadsheet)

        # Update performance report with model card diff
        if perf_report is not None:
            # Summary is made between the existing perf.yaml and the newly
            # created model card.
            perf_report.update_summary(
                model_id,
                previous_report=prev_model_card,
                new_report=curr_model_card,
            )

        # Numerics is pytorch-only; static models don't have inference jobs.
        if not (accuracy_csv_present and model_id in pytorch_models):
            continue
        assert accuracy_df is not None
        assert chipset_registry is not None
        assert global_numerics_diff is not None
        try:
            model_info = QAIHMModelInfo.from_model(model_id)
            if (
                model_info.code_gen_config.skip_hub_tests_and_scorecard
                or model_info.code_gen_config.skip_scorecard
                or model_info.code_gen_config.freeze_perf_yaml
            ):
                continue

            model_diff = NumericsDiff(
                current_inference_jobs=current_inference_jobs,
                previous_inference_jobs=previous_inference_jobs,
            )
            numerics = create_numerics_yaml(
                model_id,
                accuracy_df,
                chipset_registry,
                model_diff,
                benchmark=model_info.numerics_benchmark,
                threshold_override=model_info.code_gen_config.numerics_threshold_override,
            )
            global_numerics_diff.merge_from(model_diff)
            if numerics is None:
                QAIHMModelNumerics().to_model_yaml(model_id)  # deletes existing file
                continue

            if numerics.metrics:
                # Update failure reasons according to what NumericsDiff says is
                # above the acceptable accuracy threshold.
                update_code_gen_accuracy_failure_reasons(
                    model_id, model_info.code_gen_config, model_diff
                )

                # Update numerics.yaml to remove failing paths
                numerics = remove_numerics_failures(
                    numerics, model_info.code_gen_config.disabled_paths
                )

                if args.sync_code_gen and using_prod_hub:
                    # If sync-code-gen is on, save the updated failure reasons to disk.
                    model_info.code_gen_config.to_model_yaml(model_id)

                    # Do not remove failing paths if frozen or LLM
                    # LLMs because it is handled by apply_llm_perf_updates.
                    if (
                        not model_info.code_gen_config.freeze_perf_yaml
                        and not model_info.code_gen_config.is_llm
                    ):
                        perf = remove_perf_failures(
                            perf=QAIHMModelPerf.from_model(
                                model_id, not_exists_ok=True
                            ),
                            failure_reason=model_info.code_gen_config.disabled_paths,
                        )
                        perf.apply_similar_devices(load_similar_devices())
                        perf.to_model_yaml(model_id)

                    # Un-publish or re-publish the model if needed by updating info.yaml.
                    if update_model_publish_status(model_info):
                        model_info.to_model_yaml(write_code_gen=False)

            numerics.to_model_yaml(model_id)
            print(f"{model_id} numerics update complete")
        except Exception:
            # Skip this model so one bad input doesn't kill the batch; the
            # aggregate raise at the end of __main__ still fails the job
            # after assets land.
            print(
                f"{model_id} numerics update failed:\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            failed_model_ids.append(model_id)

    # Write spreadsheet to disk
    if spreadsheet is not None:
        summary_path = os.path.join(args.artifacts_dir, "export-summary.csv")
        spreadsheet.to_csv(summary_path)
        print(f"Spreadsheet written to {os.path.realpath(summary_path)}")

    # Write performance summary to disk
    if perf_report is not None:
        report_path = os.path.join(
            args.artifacts_dir, f"performance-summary-{now_str}.txt"
        )
        # Diff toolchain versions between this run's intermediates (current)
        # and the checked-in intermediates from the previous results branch (previous).
        current_tool_versions = ToolVersionsByPathYaml.from_yaml(
            ScorecardArtifact.TOOL_VERSIONS.path,
            create_empty_if_no_file=True,
        )
        previous_tool_versions = ToolVersionsByPathYaml.from_yaml(
            ScorecardArtifact.TOOL_VERSIONS.intermediates_path,
            create_empty_if_no_file=True,
        )
        toolchain_changes = current_tool_versions.diff(previous_tool_versions)
        perf_report.dump_summary(report_path, toolchain_changes=toolchain_changes)

        regressions_path = os.path.join(
            args.artifacts_dir, f"perf-regressions-2x-{now_str}.json"
        )
        perf_report.dump_severe_regressions_json(regressions_path)

    # Write numerics summary to disk
    if global_numerics_diff is not None:
        numerics_summary_path = os.path.join(
            args.artifacts_dir, f"numerics-summary-{now_str}.txt"
        )
        global_numerics_diff.dump_summary(numerics_summary_path)

        numerics_regressions_path = os.path.join(
            args.artifacts_dir, f"numerics-regressions-{now_str}.json"
        )
        global_numerics_diff.dump_regressions_json(numerics_regressions_path)

        # Write accuracy to intermediates folder
        if args.sync_code_gen and using_prod_hub:
            assert accuracy_df is not None
            if args.models not in (
                {SpecialModelSetting.PYTORCH},
                {SpecialModelSetting.ALL},
            ):
                accuracy_df = _merge_existing_accuracy_data(accuracy_df, pytorch_models)
            accuracy_df.to_csv(
                ScorecardArtifact.ACCURACY_CSV.intermediates_path, index=False
            )
    else:
        print("No accuracy CSV found. Skipping numerics-yaml updates.")

    # Write jobs and environment to intermediates folder.
    if using_prod_hub:
        component_names_yaml.to_file()
        graph_names_yaml.to_file()
        quantize_job_yamls.to_file()
        compile_job_yamls.to_file()
        link_job_yamls.to_file()
        profile_job_yamls.to_file()
        inference_job_yamls.to_file()
        print(f"Component Names written to {component_names_yaml.path}")
        print(f"Graph Names written to {graph_names_yaml.path}")
        print(f"Quantize Job IDs written to {quantize_job_yamls.path}")
        print(f"Compile Job IDs written to {compile_job_yamls.path}")
        print(f"Link Job IDs written to {link_job_yamls.path}")
        print(f"Profile Job IDs written to {profile_job_yamls.path}")
        print(f"Inference Job IDs written to {inference_job_yamls.path}")

        try:
            shutil.copy(
                ScorecardArtifact.TOOL_VERSIONS.path,
                ScorecardArtifact.TOOL_VERSIONS.intermediates_path,
            )
            print(
                f"Tool versions written to {ScorecardArtifact.TOOL_VERSIONS.intermediates_path}"
            )
        except (shutil.SameFileError, FileNotFoundError):
            pass

        try:
            shutil.copy(
                ScorecardArtifact.ENVIRONMENT_FILE.path,
                ScorecardArtifact.ENVIRONMENT_FILE.intermediates_path,
            )
            print(
                f"Test envvars written to {ScorecardArtifact.ENVIRONMENT_FILE.intermediates_path}"
            )
        except (shutil.SameFileError, FileNotFoundError):
            pass

    # Fail loudly only AFTER all assets are on disk for downstream uploads.
    if failed_model_ids:
        raise RuntimeError(
            f"{len(failed_model_ids)} model(s) failed during result "
            f"collection (assets were still written; see stderr above for "
            f"per-model tracebacks): {', '.join(failed_model_ids)}"
        )

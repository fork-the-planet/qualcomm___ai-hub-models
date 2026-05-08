# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Generic, TypeVar

import ruamel.yaml
from pydantic import Field
from qai_hub import JobType
from typing_extensions import Self

from qai_hub_models.configs.release_assets_yaml import (
    QAIHMModelReleaseAssets,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.errors import CachedScorecardJobError
from qai_hub_models.scorecard.params import JobTypeVar, ScExportTestParams, ScJobParams
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    LinkScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
    ScorecardJobTypeVar,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.export_result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)

ScorecardJobYamlTypeVar = TypeVar("ScorecardJobYamlTypeVar", bound="ScorecardJobYaml")


# Schema for sdk versions dumped to Hugging Face / Scorecard Intermediates in YAML format.
class ToolVersionsByPathYaml(BaseQAIHMConfig):
    tool_versions: dict[ScorecardProfilePath, ToolVersions] = Field(
        default_factory=dict
    )

    @staticmethod
    def from_profile_paths(
        paths: list[ScorecardProfilePath] | None = None,
    ) -> ToolVersionsByPathYaml:
        """
        Get a tool versions YAML object, with all paths in the list populated with tool versions.
        This will fetch versions for AI Hub Workbench deployment used by scorecard (set by envvars).

        If paths is None, populates all enabled scorecard profile paths.
        """
        out = ToolVersionsByPathYaml()
        for path in paths or ScorecardProfilePath:
            out.tool_versions[path] = path.tool_versions
        return out

    @staticmethod
    def from_dir(
        dirpath: str | os.PathLike, filename: str = "tool-versions.yaml"
    ) -> ToolVersionsByPathYaml:
        return ToolVersionsByPathYaml.from_yaml(
            Path(dirpath) / filename,
            create_empty_if_no_file=True,
        )

    def to_dir(
        self,
        dirpath: str | os.PathLike,
        filename: str = "tool-versions.yaml",
    ) -> bool:
        return self.to_yaml(Path(dirpath) / filename, write_if_empty=False)


class ScorecardJobYaml(Generic[ScorecardJobTypeVar]):
    ARTIFACT_TYPE: ScorecardArtifact
    SCORECARD_JOB_TYPE: type[ScorecardJobTypeVar]

    def __init__(
        self,
        job_id_mapping: dict[str, str] | None = None,
        path: str | os.PathLike | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.job_id_mapping = job_id_mapping or {}
        # ScorecardJob classes are expensive to create
        # (workbench API calls), so we cache them.
        self.job_cache: dict[str, ScorecardJobTypeVar] = {}

    @classmethod
    def from_file(
        cls, config_path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> Self:
        """Read yaml files."""
        if not os.path.exists(config_path):
            if create_empty_if_no_file:
                return cls({}, config_path)
            raise FileNotFoundError(f"File not found with job ids at {config_path}")

        yaml = ruamel.yaml.YAML()
        with open(config_path) as file:
            return cls(yaml.load(file), config_path)

    @classmethod
    def from_intermediates(cls) -> Self:
        return cls.from_file(
            cls.ARTIFACT_TYPE.intermediates_path, create_empty_if_no_file=True
        )

    @classmethod
    def from_test_artifacts(cls) -> Self:
        return cls.from_file(cls.ARTIFACT_TYPE.path, create_empty_if_no_file=True)

    def to_file(self, path: str | Path | None = None, append: bool = False) -> None:
        path = path or self.path
        assert path is not None
        if len(self.job_id_mapping) > 0:
            with open(path, "a" if append else "w") as yaml_file:
                ruamel.yaml.YAML().dump(self.job_id_mapping, yaml_file)
        elif not append:
            # If the dict is empty, ruamel dumps "{}" (which is not YAML) and breaks the file
            Path(path).touch()

    def clear_jobs(self, model_id: str | None = None) -> None:
        if not model_id:
            self.job_id_mapping.clear()
        else:
            # find jobs to delete
            # catch "model", ignore "model_quantized"
            keys_to_delete = [
                key
                for key in self.job_id_mapping
                if (
                    key.startswith(f"{model_id}_")
                    and not key.startswith(f"{model_id}_quantized_")
                )
            ]

            # Delete keys
            for key in keys_to_delete:
                del self.job_id_mapping[key]

    def get_job_key(self, params: ScJobParams) -> str:
        return params.job_id(self.SCORECARD_JOB_TYPE.job_type)

    def set_job_id(self, job_id: str, params: ScJobParams) -> None:
        """
        Set the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

        Parameters
        ----------
        job_id
            Job ID to associate with the other parameters in the YAML
        params
            Job identification parameters
        """
        self.job_id_mapping[self.get_job_key(params)] = job_id

    def update(self, other: ScorecardJobYaml) -> None:
        """Merge the other YAML into this YAML, overwriting any existing jobs with the same job name"""
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot merge scorecard YAMLS of types {type(other)} and {type(self)}"
            )
        self.job_id_mapping.update(other.job_id_mapping)

    def get_job(
        self,
        params: ScJobParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = False,
    ) -> ScorecardJobTypeVar | None:
        """
        Get the scorecard job from the YAML associated with these parameters.

        Parameters
        ----------
        params
            Job identification parameters
        wait_for_job
            If false, running jobs are treated like they were "skipped"
        wait_for_job_max_seconds
            Allow the job this many seconds after creation to complete
        raise_if_not_successful
            If true, raise an error if the job is not successful

        Returns
        -------
        job : ScorecardJobTypeVar | None
            The scorecard job matching these parameters, or None if the job does not exist.
        """
        # Get the job.
        key = self.get_job_key(params)
        job = self.job_cache.get(key)
        if job is None:
            if job_id := self.job_id_mapping.get(key):
                job = self.SCORECARD_JOB_TYPE(job_id)
                # ScorecardJob classes are expensive to create
                # (workbench API calls), so we cache them.
                self.job_cache[key] = job
            else:
                return None

        # Wait for the job to finish and cache the results.
        if wait_for_job:
            job.wait(wait_for_job_max_seconds)
        else:
            job.cache_results()

        # Verify the job succeeded.
        if raise_if_not_successful and not job.success:
            if job.running:
                error_str = f"still running after max allowed job duration of {wait_for_job_max_seconds / 60 if wait_for_job_max_seconds else '__'} minutes"
            else:
                error_str = job.job_status

            raise CachedScorecardJobError(
                params.str_with_description(
                    f"Prerequisite {job.job._job_type.display_name.title()} job {error_str}: {job.job.url}"
                )
            )

        return job

    def _get_all_job_params(self, params: ScExportTestParams) -> list[ScJobParams]:
        return params.all_job_params(self.SCORECARD_JOB_TYPE.job_type)

    def get_all_jobs(
        self,
        params: ScExportTestParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = False,
        raise_if_jobs_are_missing: bool = False,
    ) -> dict[ScJobParams, ScorecardJobTypeVar | None]:
        """Get all cached jobs that should exist for the given test paramaters. If a job should exist but is not in the cache, it is returned as None."""
        all_jobs = {
            pp: self.get_job(
                pp, wait_for_job, wait_for_job_max_seconds, raise_if_not_successful
            )
            for pp in self._get_all_job_params(params)
        }

        if raise_if_jobs_are_missing and sum(
            x is not None for x in all_jobs.values()
        ) != len(all_jobs):
            raise CachedScorecardJobError(
                params.str_with_description(
                    f"Could not find all cached {self.SCORECARD_JOB_TYPE.job_type.name} jobs."
                )
            )

        return all_jobs

    def update_from_export_output(
        self,
        export_output: None
        | JobTypeVar
        | MultiGraphGroup[JobTypeVar]
        | ComponentGroup[JobTypeVar]
        | MultiGraphComponentGroup[JobTypeVar],
        test_params: ScExportTestParams,
    ) -> None:
        """From the output of a step in export.py, populate this cache."""
        if export_output is None:
            raise ValueError("Export output is missing.")

        if isinstance(export_output, MultiGraphComponentGroup):
            for (
                component,
                graph_name,
            ), job in export_output.component_graph_names.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.precision,
                        test_params.path,
                        test_params.device,
                        component,
                        graph_name=graph_name,
                    ),
                )
        elif isinstance(export_output, ComponentGroup):
            for component, job in export_output.components.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.precision,
                        test_params.path,
                        test_params.device,
                        component,
                        graph_name=None,
                    ),
                )
        elif isinstance(export_output, MultiGraphGroup):
            for graph_name, job in export_output.graph_names.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.precision,
                        test_params.path,
                        test_params.device,
                        component=None,
                        graph_name=graph_name,
                    ),
                )
        else:
            self.set_job_id(
                export_output.job_id,
                ScJobParams(
                    test_params.model_id,
                    test_params.precision,
                    test_params.path,
                    test_params.device,
                    component=None,
                    graph_name=None,
                ),
            )

    def get_export_output(
        self,
        test_params: ScExportTestParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = True,
        raise_if_jobs_are_missing: bool = True,
    ) -> (
        JobTypeVar
        | MultiGraphGroup[JobTypeVar]
        | ComponentGroup[JobTypeVar]
        | MultiGraphComponentGroup[JobTypeVar]
        | None
    ):
        """Load the output of a step in export.py that would have generated this cache."""
        all_jobs = self.get_all_jobs(
            test_params,
            wait_for_job,
            wait_for_job_max_seconds,
            raise_if_not_successful,
            raise_if_jobs_are_missing,
        )
        has_components = test_params.component_names is not None
        has_graph_names = (
            self.SCORECARD_JOB_TYPE.job_type not in {JobType.QUANTIZE, JobType.LINK}
            and test_params.graph_names is not None
        ) or test_params.component_graph_names is not None

        if has_graph_names:
            if has_components:
                mgcg_components: dict[tuple[str, str | None], JobTypeVar] = {}
                for job_params, sc_job in all_jobs.items():
                    assert job_params.component is not None
                    if sc_job is None:
                        continue
                    mgcg_components[(job_params.component, job_params.graph_name)] = (
                        sc_job.job
                    )
                return MultiGraphComponentGroup(component_graph_names=mgcg_components)
            out_gn: dict[str, JobTypeVar] = {}
            for job_params, sc_job in all_jobs.items():
                assert (
                    job_params.component is None and job_params.graph_name is not None
                )
                if sc_job is not None:
                    out_gn[job_params.graph_name] = sc_job.job
            return MultiGraphGroup(graph_names=out_gn)

        if has_components:
            out_comp: dict[str, JobTypeVar] = {}
            for job_params, sc_job in all_jobs.items():
                assert (
                    job_params.component is not None and job_params.graph_name is None
                )
                if sc_job is not None:
                    out_comp[job_params.component] = sc_job.job
            return ComponentGroup(components=out_comp)

        if len(all_jobs) == 0:
            return None
        assert len(all_jobs) == 1
        sc_job = next(iter(all_jobs.values()))
        assert sc_job is not None
        return sc_job.job


class PreQDQCompileScorecardJobYaml(ScorecardJobYaml[CompileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.COMPILE_YAML
    SCORECARD_JOB_TYPE = CompileScorecardJob

    def get_job_key(self, params: ScJobParams) -> str:
        return params.pre_quantize_compile_job_id

    def _get_all_job_params(self, params: ScExportTestParams) -> list[ScJobParams]:
        return params.all_pre_qdq_compile_job_params


class QuantizeScorecardJobYaml(ScorecardJobYaml[QuantizeScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.QUANTIZE_YAML
    SCORECARD_JOB_TYPE = QuantizeScorecardJob


class CompileScorecardJobYaml(ScorecardJobYaml[CompileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.COMPILE_YAML
    SCORECARD_JOB_TYPE = CompileScorecardJob


class LinkScorecardJobYaml(ScorecardJobYaml[LinkScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.LINK_YAML
    SCORECARD_JOB_TYPE = LinkScorecardJob


class ProfileScorecardJobYaml(ScorecardJobYaml[ProfileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.PROFILE_YAML
    SCORECARD_JOB_TYPE = ProfileScorecardJob


class InferenceScorecardJobYaml(ScorecardJobYaml[InferenceScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.INFERENCE_YAML
    SCORECARD_JOB_TYPE = InferenceScorecardJob


class ScorecardAssetYaml(BaseQAIHMConfig):
    models: dict[str, QAIHMModelReleaseAssets] = Field(default_factory=dict)

    def add_asset(
        self,
        details: QAIHMModelReleaseAssets.AssetDetails,
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> None:
        if model_id not in self.models:
            self.models[model_id] = QAIHMModelReleaseAssets()
        self.models[model_id].add_asset(
            details,
            precision,
            device.chipset if path.runtime.is_aot_compiled else None,
            path,
        )

    def get_asset(
        self,
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> QAIHMModelReleaseAssets.AssetDetails | None:
        if model_id not in self.models:
            return None
        return self.models[model_id].get_asset(
            precision, device.chipset if path.runtime.is_aot_compiled else None, path
        )

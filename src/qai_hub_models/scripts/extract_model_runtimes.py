# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Extract per-model wall-clock runtimes from scorecard JUnit XMLs.

Walks the three combined JUnit XMLs produced by a weekly scorecard run
(job submission, export-script test, on-device accuracy) and aggregates
``<testsuite time="...">`` seconds per model. The model id is recovered
from each testsuite's ``classname``, which pytest sets to the dotted
module path (e.g. ``qai_hub_models.models.yolov7.test_export``).

Output is a YAML file consumed by ``split_torch_models`` to load-balance
splits across CI runners. By default it's written to the checked-in
intermediates location, where the scorecard collection workflow's
existing ``git add src/qai_hub_models/scorecard/intermediates/*.yaml``
step picks it up automatically — no separate commit logic needed here.

Two ways to source the XMLs:

* ``--action-id <run_id>`` (and optional ``--repo``) downloads the
  ``test-results-scorecard`` artifact from a previous scorecard run via
  the ``gh`` CLI and uses the XMLs inside.
* ``--job-submission-xml`` / ``--export-test-xml`` / ``--accuracy-xml``
  point at local files (used for local bootstrapping or testing).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import ruamel.yaml

from qai_hub_models.scorecard.artifacts import (
    RUNTIME_ALL_STAGES,
    RUNTIME_STAGE_ACCURACY,
    RUNTIME_STAGE_EXPORT_TEST,
    RUNTIME_STAGE_JOB_SUBMISSION,
    ScorecardArtifact,
)

# Filenames inside the test-results-scorecard artifact, written by
# scorecard.yml's combine_test_results job.
SCORECARD_ARTIFACT_NAME = "test-results-scorecard"
JOB_SUBMISSION_XML_NAME = "qaihm-model-tests-junit.xml"
EXPORT_TEST_XML_NAME = "qaihm-export-zip-tests-junit.xml"
ACCURACY_XML_NAME = "qaihm-device-accuracy-tests-junit.xml"

# Matches dotted classnames pytest emits for tests under
# qai_hub_models/models/<model_id>/. Captures the model id segment.
_MODEL_CLASSNAME_RE = re.compile(r"^qai_hub_models\.models\.([^.]+)\.")


def _extract_model_id(classname: str, name: str) -> str | None:
    match = _MODEL_CLASSNAME_RE.match(classname)
    if match:
        return match.group(1)
    # Some pytest configurations put the dotted path in 'name' instead.
    match = _MODEL_CLASSNAME_RE.match(name)
    if match:
        return match.group(1)
    return None


def parse_junit_per_model_seconds(xml_path: Path) -> dict[str, float]:
    """Sum ``time`` per model across all testsuites/testcases in one JUnit XML.

    A combined scorecard XML contains many ``<testsuite>`` elements (one
    per per-model pytest invocation). We attribute each testcase's time
    to the model parsed from its classname so an unknown ``<testsuite>``
    element doesn't leak time into the wrong bucket.
    """
    if not xml_path.exists():
        return {}

    totals: dict[str, float] = defaultdict(float)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"Warning: could not parse {xml_path}: {exc}", file=sys.stderr)
        return {}
    root = tree.getroot()

    for testcase in root.iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        model_id = _extract_model_id(classname, name)
        if model_id is None:
            continue
        time_attr = testcase.get("time", "0")
        try:
            totals[model_id] += float(time_attr)
        except ValueError:
            continue

    return dict(totals)


def build_runtime_estimates(
    job_submission_xml: Path | None,
    export_test_xml: Path | None,
    accuracy_xml: Path | None,
) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {stage: seconds}}`` aggregated across the three XMLs."""
    stage_inputs: dict[str, Path | None] = {
        RUNTIME_STAGE_JOB_SUBMISSION: job_submission_xml,
        RUNTIME_STAGE_EXPORT_TEST: export_test_xml,
        RUNTIME_STAGE_ACCURACY: accuracy_xml,
    }

    per_stage: dict[str, dict[str, float]] = {}
    for stage, xml_path in stage_inputs.items():
        if xml_path is None:
            per_stage[stage] = {}
            continue
        per_stage[stage] = parse_junit_per_model_seconds(xml_path)

    all_models = set().union(*(s.keys() for s in per_stage.values()))
    estimates: dict[str, dict[str, float]] = {}
    for model_id in sorted(all_models):
        entry: dict[str, float] = {}
        for stage in RUNTIME_ALL_STAGES:
            if model_id in per_stage[stage]:
                entry[stage] = round(per_stage[stage][model_id], 1)
        estimates[model_id] = entry
    return estimates


def write_runtime_estimates_yaml(
    output_path: Path,
    estimates: dict[str, dict[str, float]],
    source_action_id: str | None = None,
) -> None:
    """Write the runtime estimates YAML in the canonical schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if source_action_id:
        payload["source_action_id"] = source_action_id
    payload["models"] = estimates

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(output_path, "w") as f:
        yaml.dump(payload, f)


_ACTION_ID_RE = re.compile(r"[0-9]+")
_REPO_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")


def _download_scorecard_artifact(
    action_id: str,
    repo: str | None,
    dest: Path,
) -> None:
    """Download the ``test-results-scorecard`` artifact via ``gh run download``.

    ``action_id`` and ``repo`` come from CI inputs (workflow dispatch),
    so they're validated against tight regexes before being shelled out
    to ``gh``. Without that, a value starting with ``-`` would be parsed
    as a flag rather than a positional run id.
    """
    if shutil.which("gh") is None:
        raise SystemExit("gh CLI not found on PATH; cannot download artifacts.")
    if not _ACTION_ID_RE.fullmatch(action_id):
        raise ValueError(f"action_id must be numeric, got: {action_id!r}")
    if repo is not None and not _REPO_RE.fullmatch(repo):
        raise ValueError(f"repo must be 'owner/name', got: {repo!r}")
    cmd = [
        "gh",
        "run",
        "download",
        "-n",
        SCORECARD_ARTIFACT_NAME,
        "-D",
        str(dest),
    ]
    if repo:
        cmd.extend(["-R", repo])
    cmd.append("--")
    cmd.append(action_id)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-id",
        default=None,
        help=(
            "Scorecard kickoff GitHub Actions run id. When set, downloads the "
            f"'{SCORECARD_ARTIFACT_NAME}' artifact via gh CLI and reads the JUnit "
            "XMLs inside it. Mutually exclusive with the per-XML path flags."
        ),
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo (owner/name) to pass to gh. Defaults to gh's auto-detection.",
    )
    parser.add_argument(
        "--job-submission-xml",
        type=Path,
        default=None,
        help="Path to combined model-tests JUnit XML (qaihm-model-tests-junit.xml).",
    )
    parser.add_argument(
        "--export-test-xml",
        type=Path,
        default=None,
        help="Path to combined export JUnit XML (qaihm-export-zip-tests-junit.xml).",
    )
    parser.add_argument(
        "--accuracy-xml",
        type=Path,
        default=None,
        help="Path to combined accuracy JUnit XML (qaihm-device-accuracy-tests-junit.xml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ScorecardArtifact.MODEL_RUNTIME_ESTIMATES.intermediates_path,
        help=(
            "Output YAML path. Defaults to the checked-in intermediates location, "
            "which the scorecard collection workflow auto-commits via its existing "
            "'git add scorecard/intermediates/*.yaml' step."
        ),
    )
    parser.add_argument(
        "--source-action-id",
        default=None,
        help=(
            "Override for the source_action_id field stamped into the YAML. "
            "Defaults to --action-id when that's set."
        ),
    )

    args = parser.parse_args()

    explicit_xmls = (args.job_submission_xml, args.export_test_xml, args.accuracy_xml)
    if args.action_id and any(explicit_xmls):
        parser.error("--action-id cannot be combined with --*-xml path flags.")
    if not args.action_id and not any(explicit_xmls):
        parser.error(
            "Pass --action-id to download from a scorecard run, or pass at least "
            "one of --job-submission-xml/--export-test-xml/--accuracy-xml."
        )

    source_action_id = args.source_action_id or args.action_id

    if args.action_id:
        with tempfile.TemporaryDirectory(prefix="runtime-estimates-") as tmp:
            tmp_path = Path(tmp)
            _download_scorecard_artifact(args.action_id, args.repo, tmp_path)
            job_xml = tmp_path / JOB_SUBMISSION_XML_NAME
            export_xml = tmp_path / EXPORT_TEST_XML_NAME
            accuracy_xml = tmp_path / ACCURACY_XML_NAME
            estimates = build_runtime_estimates(
                job_xml if job_xml.exists() else None,
                export_xml if export_xml.exists() else None,
                accuracy_xml if accuracy_xml.exists() else None,
            )
    else:
        estimates = build_runtime_estimates(*explicit_xmls)

    if not estimates:
        print("No model runtimes recovered; not writing YAML.")
        return

    write_runtime_estimates_yaml(args.output, estimates, source_action_id)
    print(f"Wrote runtime estimates for {len(estimates)} models to {args.output}")


if __name__ == "__main__":
    main()

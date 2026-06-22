# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

r"""CLI for configuring datasets that require manual file downloads.

Usage:

    python -m qai_hub_models.scripts.configure_dataset \
        --class qai_hub_models.datasets.kitti.kitti.KittiDataset \
        --files /path/to/images.zip /path/to/labels.zip /path/to/calibs.zip

The class can be any importable `BaseDataset` subclass that overrides
`configure()`. This includes classes shipped with the repo, classes in
model folders, and user-defined classes on `sys.path` (which includes the
current working directory when invoked as a script).

Trying to configure a dataset whose class does not override `configure()`
will raise `NotImplementedError` — that means the dataset auto-downloads
and you do not need this CLI for it.
"""

from __future__ import annotations

import argparse
import importlib

from qai_hub_models.utils.base_dataset import BaseDataset


def _resolve_class(import_path: str) -> type[BaseDataset]:
    """Resolve a dotted ``module.path.ClassName`` import path to a class."""
    if "." not in import_path:
        raise ValueError(
            f"--class must be a dotted import path like "
            f"'package.module.ClassName' (got {import_path!r})."
        )
    module_path, _, class_name = import_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Could not import module {module_path!r} for --class {import_path!r}."
        ) from e
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise ValueError(
            f"Module {module_path!r} has no attribute {class_name!r}."
        ) from e
    if not isinstance(cls, type) or not issubclass(cls, BaseDataset):
        raise TypeError(
            f"{import_path} must be a subclass of "
            "qai_hub_models.utils.base_dataset.BaseDataset."
        )
    return cls


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Configure a dataset that needs to be downloaded externally. "
            "Instructions on how to use this script are typically printed "
            "when trying to quantize or evaluate a model that requires one "
            "of these datasets."
        )
    )
    parser.add_argument(
        "--class",
        dest="cls",
        type=str,
        required=True,
        help=(
            "Dotted import path to the dataset class, e.g. "
            "qai_hub_models.datasets.kitti.kitti.KittiDataset"
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        required=True,
        help="Local filepaths needed to set up this dataset.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    cls = _resolve_class(args.cls)
    cls.configure(args.files)


if __name__ == "__main__":
    main()

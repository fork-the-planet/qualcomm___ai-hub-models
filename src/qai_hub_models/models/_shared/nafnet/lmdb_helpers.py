# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""LMDB image helpers shared by NAFNet denoise/deblur datasets."""

from __future__ import annotations

from os import path as osp

import cv2
import lmdb
import numpy as np


def paired_paths_from_lmdb(folders: list[str], keys: list[str]) -> list[dict[str, str]]:
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        raise ValueError(
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, "meta_info.txt")) as fin:
        input_lmdb_keys = [line.split(".")[0] for line in fin]
    with open(osp.join(gt_folder, "meta_info.txt")) as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f"Keys in {input_key}_folder and {gt_key}_folder are different."
        )
    return [
        {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
        for lmdb_key in sorted(input_lmdb_keys)
    ]


def imfrombytes(content: bytes, float32: bool = False) -> np.ndarray:
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def get_image_from_lmdb(
    txn: lmdb.Transaction, path: str, key_name: str, float32: bool = True
) -> np.ndarray:
    img_bytes = txn.get(path.encode("ascii"))
    if img_bytes is None:
        raise ValueError(f"Key not found: {path}")
    try:
        return imfrombytes(img_bytes, float32=float32)
    except ValueError as exc:
        raise RuntimeError(f"{key_name} path {path} not working") from exc

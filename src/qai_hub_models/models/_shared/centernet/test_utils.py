# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Detections below this score are padding rows the decoders always emit up to
# max_dets; they are unused downstream and unstable across environments.
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2


def _filter_by_score(
    dets: np.ndarray, score_index: int, score_threshold: float
) -> np.ndarray:
    dets = np.asarray(dets)
    dets = dets.reshape(-1, dets.shape[-1])
    return dets[dets[:, score_index] >= score_threshold]


def _greedy_match(
    actual: np.ndarray, expected: np.ndarray, stable_columns: Sequence[int]
) -> np.ndarray:
    # Pair each actual row with the nearest unclaimed expected row on the
    # stable columns. Sorting-based matching is unsafe here because two
    # detections can have near-equal sort keys and swap order under drift.
    cols = list(stable_columns)
    a = actual[:, cols]
    e = expected[:, cols]
    dists = np.linalg.norm(a[:, None, :] - e[None, :, :], axis=-1)

    n = a.shape[0]
    order = np.argsort(dists.min(axis=1))
    matched = np.empty(n, dtype=np.intp)
    claimed = np.zeros(expected.shape[0], dtype=bool)
    for i in order:
        candidates = np.where(~claimed)[0]
        j = candidates[np.argmin(dists[i, candidates])]
        matched[i] = j
        claimed[j] = True
    return expected[matched]


def assert_detections_close(
    actual: np.ndarray,
    expected: np.ndarray,
    score_index: int,
    stable_columns: Sequence[int],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    actual_f = _filter_by_score(actual, score_index, score_threshold)
    expected_f = _filter_by_score(expected, score_index, score_threshold)

    assert actual_f.shape[0] == expected_f.shape[0], (
        f"Different number of detections with score >= {score_threshold}: "
        f"actual has {actual_f.shape[0]} (scores={actual_f[:, score_index]}), "
        f"expected has {expected_f.shape[0]} (scores={expected_f[:, score_index]})."
    )

    expected_matched = _greedy_match(actual_f, expected_f, stable_columns)

    cols = list(stable_columns)
    np.testing.assert_allclose(
        actual_f[:, cols], expected_matched[:, cols], rtol=rtol, atol=atol
    )

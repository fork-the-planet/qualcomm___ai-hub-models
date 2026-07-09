# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

# The mediapipe app tests used to compare the *rendered* output image pixel for
# pixel against a golden PNG. That is brittle: a handful of overlay pixels
# (landmarks / boxes) flip by up to 128 across environments while the model
# itself is unchanged, so the pixel comparison fails even though the detections
# are effectively identical. Instead we compare the structured output
# (bounding box + landmark coordinates), which is what the model actually
# predicts, with a tolerance that is robust to that cross-environment drift.

# Landmark / box coordinates live in input-image pixel space (hundreds to a few
# thousand), so a couple of pixels of drift is negligible.
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1.0

# The landmark tensor's last column is a confidence score in [0, 1] (its last
# dim is (x, y, confidence)). The pixel-scale tolerance above would make any
# confidence check vacuous, so confidence gets its own tight tolerance.
CONF_RTOL = 1e-2
CONF_ATOL = 1e-2


def _only_image_tensor(batched: Any) -> np.ndarray:
    # Apps run on a single image, so each output list has one element: the
    # tensor of all detections for that image (leading dim = num detections).
    assert len(batched) >= 1, "Expected at least one batch element."
    tensor = batched[0]
    assert isinstance(tensor, torch.Tensor), "Expected a detection tensor."
    assert tensor.numel() > 0, "Expected a non-empty detection tensor."
    return tensor.detach().cpu().numpy()


def landmarks_from_raw_output(raw_output: Sequence[Any]) -> dict[str, np.ndarray]:
    """Extract the comparable structured output from an app's raw_output tuple.

    All mediapipe apps return
        (batched_selected_boxes, batched_selected_keypoints,
         batched_roi_4corners, batched_selected_landmarks, ...)
    We compare the selected box and the selected landmarks, which together
    determine everything drawn on the output image.
    """
    assert len(raw_output) >= 4, (
        "Expected raw_output layout (boxes, keypoints, roi_4corners, landmarks, "
        f"...); got only {len(raw_output)} element(s)."
    )
    boxes = _only_image_tensor(raw_output[0])
    landmarks = _only_image_tensor(raw_output[3])
    return {"boxes": boxes, "landmarks": landmarks}


def assert_landmarks_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    conf_rtol: float = CONF_RTOL,
    conf_atol: float = CONF_ATOL,
) -> None:
    """Assert two structured mediapipe outputs match within tolerance.

    The first two columns are pixel-space coordinates and use the (loose) pixel
    tolerance (``rtol``/``atol``); any further column (the landmarks' confidence
    score) is compared with the tight ``conf_rtol``/``conf_atol`` tolerance so a
    regression there is not masked.
    """
    for key in ("boxes", "landmarks"):
        a = np.asarray(actual[key])
        e = np.asarray(expected[key])
        assert a.shape == e.shape, (
            f"{key}: shape mismatch, actual {a.shape} vs expected {e.shape}."
        )
        np.testing.assert_allclose(
            a[..., :2],
            e[..., :2],
            rtol=rtol,
            atol=atol,
            err_msg=f"{key} coordinates mismatch",
        )
        if a.shape[-1] > 2:
            np.testing.assert_allclose(
                a[..., 2:],
                e[..., 2:],
                rtol=conf_rtol,
                atol=conf_atol,
                err_msg=f"{key} confidence mismatch",
            )

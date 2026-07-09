# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest
import torch

from qai_hub_models.models._shared.mediapipe.test_utils import (
    assert_landmarks_close,
    landmarks_from_raw_output,
)
from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_face.demo import main as demo_main
from qai_hub_models.models.mediapipe_face.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeFace,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

# Golden structured output (bounding box + landmark coordinates) for the demo
# image. Comparing these directly is robust to the cross-environment pixel
# drift that made the old rendered-image comparison flaky.
LANDMARKS_GOLDEN_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_landmarks_golden.npz"
)


# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.
def test_face_app() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    expected = load_numpy(LANDMARKS_GOLDEN_ADDRESS)
    app = MediaPipeFaceApp.from_pretrained(
        MediaPipeFace.from_pretrained(include_detector_postprocessing=False)
    )
    actual = landmarks_from_raw_output(
        app.predict_landmarks_from_image(image, raw_output=True)
    )
    assert_landmarks_close(actual, expected)


def test_face_app_with_det_postprocessing() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    expected = load_numpy(LANDMARKS_GOLDEN_ADDRESS)
    app = MediaPipeFaceApp.from_pretrained(
        MediaPipeFace.from_pretrained(include_detector_postprocessing=True)
    )
    actual = landmarks_from_raw_output(
        app.predict_landmarks_from_image(image, raw_output=True)
    )
    assert_landmarks_close(actual, expected)


def test_landmarks_from_raw_output_rejects_short_tuple() -> None:
    # raw_output must have at least the four leading elements; a shorter tuple
    # should fail loudly rather than silently compare the wrong tensors.
    short = ([torch.zeros(1, 2, 2)], [torch.zeros(1, 4, 2)], [torch.zeros(1, 4, 2)])
    with pytest.raises(AssertionError):
        landmarks_from_raw_output(short)


def test_assert_landmarks_close_catches_confidence_drift() -> None:
    # Coordinates match exactly but the landmark confidence column drifts well
    # beyond CONF_ATOL; the split tolerance must catch it (the old single
    # pixel-scale tolerance would have masked this).
    expected = {
        "boxes": np.zeros((1, 2, 2), dtype=np.float32),
        "landmarks": np.array([[[10.0, 20.0, 0.9]]], dtype=np.float32),
    }
    actual = {
        "boxes": expected["boxes"].copy(),
        "landmarks": expected["landmarks"].copy(),
    }
    actual["landmarks"][..., 2] = 0.1  # confidence regression, coords unchanged
    with pytest.raises(AssertionError):
        assert_landmarks_close(actual, expected)


def test_assert_landmarks_close_tolerates_subpixel_drift() -> None:
    # A sub-pixel coordinate wobble (and a tiny confidence wobble) is within
    # tolerance and must not fail.
    expected = {
        "boxes": np.array([[[0.0, 0.0], [100.0, 100.0]]], dtype=np.float32),
        "landmarks": np.array([[[10.0, 20.0, 0.9]]], dtype=np.float32),
    }
    actual = {
        "boxes": expected["boxes"] + 0.3,
        "landmarks": expected["landmarks"] + np.array([0.3, 0.3, 0.001], np.float32),
    }
    assert_landmarks_close(actual, expected)


def test_demo() -> None:
    demo_main(is_test=True)

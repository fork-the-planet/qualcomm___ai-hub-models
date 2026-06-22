# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.datasets.kinetics400 import preprocess_video_224
from qai_hub_models.models._shared.video_classifier.app import KineticsClassifierApp


class VideoMAEApp(KineticsClassifierApp):
    def preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        return preprocess_video_224(clip, short_side_size=320)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

__all__ = ["MODEL_ASSET_VERSION", "MODEL_ID", "App", "Model"]

from qai_hub_models.models.deepspeech2.app import DeepSpeech2App as App

from .model import MODEL_ASSET_VERSION, MODEL_ID
from .model import DeepSpeech2 as Model

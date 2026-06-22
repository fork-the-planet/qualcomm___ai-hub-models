# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models._shared.pipertts.model import (
    PiperTTS,
)

MODEL_ID = __name__.split(".")[-2]


class PiperTTS_IT(PiperTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ITALIAN

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.deepspeech2.app import DeepSpeech2App
from qai_hub_models.models.deepspeech2.demo import main as demo_main
from qai_hub_models.models.deepspeech2.model import DEFAULT_AUDIO, DeepSpeech2
from qai_hub_models.utils.testing import skip_clone_repo_check

EXPECTED_TRANSCRIPTION = "AND SO MY FELLOW AMERICA ASK NOT WHYT YOUR COUNTRY CAN DO FOR YOU AN WHAT YOU CAN DO WOR YOUR GUNDRINK AS"


def _test_impl(app: DeepSpeech2App) -> None:
    audio_path = str(DEFAULT_AUDIO.fetch())
    transcription = app.predict(audio_path)
    assert transcription == EXPECTED_TRANSCRIPTION


@skip_clone_repo_check
def test_task() -> None:
    _test_impl(DeepSpeech2App(DeepSpeech2.from_pretrained()))


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)

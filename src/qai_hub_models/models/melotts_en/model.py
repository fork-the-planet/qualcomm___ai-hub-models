# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import nltk
from typing_extensions import Self

from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models._shared.melotts.model import (
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    T5Decoder,
    T5Encoder,
    get_tts_object,
)

MODEL_ID = __name__.split(".")[-2]


class Encoder_EN(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH), speed_adjustment=0.85)


class Flow_EN(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH))


class Decoder_EN(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH))


class BertWrapper_EN(BertWrapper):
    @classmethod
    def from_pretrained(cls) -> Self:
        return super().from_pretrained(TTSLanguage.ENGLISH)


class MeloTTS_EN(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ENGLISH

    @classmethod
    def from_pretrained(cls) -> Self:
        nltk.download("averaged_perceptron_tagger_eng")
        return cls(
            Encoder_EN.from_pretrained(),
            Flow_EN.from_pretrained(),
            Decoder_EN.from_pretrained(),
            bert_wrapper=BertWrapper_EN.from_pretrained(),
            t5_encoder=T5Encoder.from_pretrained(),
            t5_decoder=T5Decoder.from_pretrained(),
        )

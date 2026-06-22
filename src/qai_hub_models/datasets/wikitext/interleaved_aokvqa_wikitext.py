# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.datasets.aokvqa import AOKVQA
from qai_hub_models.datasets.wikitext.wikitext import WikiText
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
    InterleavedDataset,
)


class InterleavedAOKVQAWikitext(InterleavedDataset):
    """Interleaves AOKVQA and Wikitext for mixed VLM calibration."""

    @classmethod
    def dataset_name(cls) -> str:
        return "interleaved_aokvqa_wikitext"

    def load_datasets(self, split: DatasetSplit, **kwargs: Any) -> list[BaseDataset]:
        return [
            AOKVQA(
                split=split,
                tokenizer=kwargs.get("tokenizer"),
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=kwargs.get("num_samples", 0) // 2,
                processor=kwargs.get("processor"),
                image_size=kwargs.get("image_size"),
            ),
            WikiText(
                split=split,
                tokenizer=kwargs["tokenizer"],
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=kwargs.get("num_samples", 0) // 2,
            ),
        ]

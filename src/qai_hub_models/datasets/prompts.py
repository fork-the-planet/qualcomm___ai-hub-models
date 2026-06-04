# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from transformers import PreTrainedTokenizerBase

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit

PROMPTS_DATASET_ID = "prompts"
PROMPTS_VERSION = 1
TEXT_PROMPTS_FILENAME = "prompts.yaml"

MULTIMODAL_DATASET_ID = "multimodal_prompts"
MULTIMODAL_VERSION = 1
MULTIMODAL_PROMPTS_FILENAME = "multimodal_prompts.yaml"
SAMPLE_IMAGES_SUBDIR = "sample_images"


@dataclass
class PromptLabel:
    """Per-sample metadata used by ``LLMResponseEvaluator``.

    Carried as the dataset ``label`` so it survives the standard
    ``(input_ids, attention_mask, label, ...)`` collate convention.
    """

    index: int
    prompt: str
    image_path: str | None = None


def _format_text_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    is_vlm: bool,
) -> str:
    """Apply the model's chat template to a raw user prompt."""
    content: Any = prompt
    if is_vlm:
        # VLM processors (Qwen2.5-VL) require a list-of-content-parts. This form
        # is NOT safe for plain LLMs: their chat templates concatenate content
        # as a string, so the list either raises (Qwen2.5 text) or leaks a
        # literal "[{'type': 'text', ...}]" into the prompt (Llama-3.2). Keep
        # the bare string for non-VLMs.
        content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(formatted_prompt, str)
    return formatted_prompt


class TextPrompts(BaseDataset):
    """Text-only prompts spanning factual, math, reasoning, code, etc.

    Yields items shaped like the other LLM eval datasets so they collate the
    same way and the evaluator can drop them straight into the generator.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
        # processor accepted (and ignored) for API symmetry with VLM datasets
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if split != DatasetSplit.TEST:
            raise ValueError("TextPrompts only supports the `test` split")
        if tokenizer is None:
            raise ValueError("TextPrompts requires a tokenizer.")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.num_samples = num_samples
        self.is_vlm = processor is not None

        self._yaml_asset = CachedWebDatasetAsset.from_asset_store(
            PROMPTS_DATASET_ID, PROMPTS_VERSION, TEXT_PROMPTS_FILENAME
        )
        super().__init__(self._yaml_asset.path, split)
        with open(self._yaml_asset.path) as f:
            prompts = yaml.safe_load(f)
        assert isinstance(prompts, list)
        self.prompts: list[str] = prompts

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
        item = batch[0]
        return item["input_ids"], item["attention_mask"], item["label"]

    def __len__(self) -> int:
        if self.num_samples and self.num_samples > 0:
            return min(self.num_samples, len(self.prompts))
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]
        formatted = _format_text_prompt(self.tokenizer, prompt, self.is_vlm)
        tokenized = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        input_ids = tokenized["input_ids"][:, -self.context_length :]
        attention_mask = tokenized["attention_mask"][:, -self.context_length :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": PromptLabel(index=idx, prompt=prompt),
        }

    def _download_data(self) -> None:
        self._yaml_asset.fetch()

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @classmethod
    def dataset_name(cls) -> str:
        return "prompts"


def _multimodal_image_asset(filename: str) -> CachedWebDatasetAsset:
    """Asset descriptor for one sample image in the multimodal_prompts bucket."""
    return CachedWebDatasetAsset.from_asset_store(
        MULTIMODAL_DATASET_ID,
        MULTIMODAL_VERSION,
        f"{SAMPLE_IMAGES_SUBDIR}/{filename}",
    )


class MultimodalPrompts(BaseDataset):
    """Image + question pairs for qualitative VLM evaluation.

    Tokenization, image preprocessing, and vision token insertion are
    delegated to the VLM processor (same approach as ``AOKVQA``).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if split != DatasetSplit.TEST:
            raise ValueError("MultimodalPrompts only supports the `test` split")
        if processor is None:
            raise ValueError(
                "MultimodalPrompts requires a VLM processor "
                "(pass one through the evaluator's vlm_processor argument)."
            )
        self.processor = processor
        self.context_length = context_length
        self.num_samples = num_samples
        self.image_size: tuple[int, int] | None = (
            (int(image_size[0]), int(image_size[1])) if image_size is not None else None
        )

        self._yaml_asset = CachedWebDatasetAsset.from_asset_store(
            MULTIMODAL_DATASET_ID, MULTIMODAL_VERSION, MULTIMODAL_PROMPTS_FILENAME
        )
        super().__init__(self._yaml_asset.path, split)

        with open(self._yaml_asset.path) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, list)
        items: list[dict[str, str]] = []
        for entry in raw:
            if (
                not isinstance(entry, dict)
                or "image" not in entry
                or "prompt" not in entry
            ):
                raise TypeError(
                    f"Each {self._yaml_asset.path} entry must be a mapping with "
                    f"'image' and 'prompt'."
                )
            items.append({"image": str(entry["image"]), "prompt": str(entry["prompt"])})
        self.items = items

        # Resolve each unique image filename to its on-disk path. Doing it up
        # front gives a nicer error if S3 access is broken; fetch() is cheap
        # when the file is already cached.
        self._image_paths: dict[str, Path] = {}
        for filename in {item["image"] for item in items}:
            asset = _multimodal_image_asset(filename)
            self._image_paths[filename] = Path(asset.fetch())

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
        item = batch[0]
        result: tuple[Any, ...] = (
            item["input_ids"],
            item["attention_mask"],
            item["label"],
        )
        for key in ("pixel_values", "image_grid_thw"):
            if key in item:
                result = (*result, item[key])
        return result

    def __len__(self) -> int:
        if self.num_samples and self.num_samples > 0:
            return min(self.num_samples, len(self.items))
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        filename = item["image"]
        prompt = item["prompt"]
        image_path = self._image_paths[filename]

        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            image = image.resize(self.image_size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        assert inputs["input_ids"].shape[1] <= self.context_length
        inputs.pop("mm_token_type_ids", None)
        inputs["label"] = PromptLabel(
            index=idx, prompt=prompt, image_path=str(image_path)
        )
        return inputs

    def _download_data(self) -> None:
        self._yaml_asset.fetch()
        # Image fetches happen in __init__ once the YAML is parsed.

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @classmethod
    def dataset_name(cls) -> str:
        return "multimodal_prompts"


__all__ = [
    "MultimodalPrompts",
    "PromptLabel",
    "TextPrompts",
]

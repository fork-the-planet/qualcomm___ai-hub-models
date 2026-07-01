# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

# Original source: Zeyde et al., "On Single Image Scale-Up Using Sparse-Representations"
# https://huggingface.co/datasets/eugenesiow/Set14
SET14_URL = (
    "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz"
)
SET14_FOLDER_NAME = "Set14"
SET14_VERSION = 1
SET14_ASSET = CachedWebDatasetAsset(
    SET14_URL,
    SET14_FOLDER_NAME,
    SET14_VERSION,
    "Set14_HR.tar.gz",
    private_s3_key="qai-hub-models/datasets/set14/Set14_HR.tar.gz",
)
NUM_IMAGES = 14


class Set14Dataset(BaseDataset):
    """Set14 super-resolution dataset.

    Preprocessing follows the QuickSRNet dataloader convention:
      - HR image: original resolution, center-cropped to dimensions divisible
        by the scaling factor.
      - LR image: downsampled from HR using PIL bicubic resize, then clipped
        to uint8.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        scaling_factor: int = 4,
    ) -> None:
        self.set14_path = SET14_ASSET.extracted_path
        BaseDataset.__init__(self, self.set14_path, split, input_spec)
        self.scaling_factor = scaling_factor
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 128, 128))}
        self.input_height: int = input_spec["image"][0][2]
        self.input_width: int = input_spec["image"][0][3]
        self.image_files: list[str] = sorted(
            f for f in os.listdir(self.set14_path) if f.endswith(".png")
        )

    def _validate_data(self) -> bool:
        if not self.set14_path.exists():
            return False
        images = [f for f in self.set14_path.iterdir() if f.suffix == ".png"]
        return len(images) == NUM_IMAGES

    def __len__(self) -> int:
        return NUM_IMAGES

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single (low-resolution input, high-resolution ground truth) pair.

          - Reads image with OpenCV (BGR -> RGB).
          - Center-crops HR to largest dimensions divisible by scaling_factor.
          - Downsamples to LR using PIL bicubic resize, then clips to uint8.

        Parameters
        ----------
        item
            Dataset index (0 to 13).

        Returns
        -------
        lr_img_tensor : torch.Tensor
            Shape (3, H, W), dtype float32, range [0, 1], RGB.
        hr_img_tensor : torch.Tensor
            Shape (3, H*scale, W*scale), dtype float32, range [0, 1], RGB.
        """
        raw = cv2.imread(str(self.set14_path / self.image_files[item]))
        if raw is None:
            raise FileNotFoundError(self.image_files[item])
        img_rgb: np.ndarray = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # Resize to fixed HR size
        hr_h = self.input_height * self.scaling_factor
        hr_w = self.input_width * self.scaling_factor
        img_rgb = cv2.resize(img_rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        lr, hr = self._create_hr_lr_pair(img_rgb)
        return lr, hr

    def _create_hr_lr_pair(self, img: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Create LR/HR pair.

        Parameters
        ----------
        img
            HxWx3 uint8 RGB numpy array at original resolution.

        Returns
        -------
        lr_img_tensor : torch.Tensor
            Shape (3, H, W), dtype float32, range [0, 1].
        hr_img_tensor : torch.Tensor
            Shape (3, H*scale, W*scale), dtype float32, range [0, 1].
        """
        height, width = img.shape[0:2]

        # Center-crop to largest dimensions divisible by scaling_factor.
        x_remainder = width % (
            2 * self.scaling_factor
            if self.scaling_factor == 1.5
            else self.scaling_factor
        )
        y_remainder = height % (
            2 * self.scaling_factor
            if self.scaling_factor == 1.5
            else self.scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = img[top:bottom, left:right]

        hr_img = np.array(hr_img, dtype="float64")

        # Downsample using PIL bicubic resize, then clip to uint8
        hr_pil = Image.fromarray(hr_img.astype(np.uint8))
        lr_w = int(hr_pil.width / self.scaling_factor)
        lr_h = int(hr_pil.height / self.scaling_factor)
        lr_img = np.asarray(hr_pil.resize((lr_w, lr_h), Image.BICUBIC))
        lr_img = np.clip(lr_img, 0.0, 255.0).astype(np.uint8)
        hr_img = hr_img.astype(np.uint8)

        lr_img_tensor = torch.from_numpy(
            np.asarray(lr_img).transpose((2, 0, 1))
        ).contiguous()
        lr_img_tensor = lr_img_tensor.to(dtype=torch.float32).div(255)

        hr_img_tensor = torch.from_numpy(
            np.asarray(hr_img).transpose((2, 0, 1))
        ).contiguous()
        hr_img_tensor = hr_img_tensor.to(dtype=torch.float32).div(255)

        return lr_img_tensor, hr_img_tensor

    def _download_data(self) -> None:
        SET14_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 14

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/eugenesiow/Set14",
            split_description="all 14 standard SR benchmark images",
        )

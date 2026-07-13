# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.coco import CocoBodyDatasetBase
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

FACEMAP3DMM_DATASET_VERSION = 2
FACEMAP3DMM_DATASET_ID = "facemap3dmm_dataset"
FACEMAP3DMM_DATASET_DIR_NAME = "facemap3dmm_trainvaltest"

FACEMAP3DMM_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/facemap_3dmm/facemap3dmm_trainvaltest.zip",
    FACEMAP3DMM_DATASET_ID,
    FACEMAP3DMM_DATASET_VERSION,
    f"data/{FACEMAP3DMM_DATASET_DIR_NAME}.zip",
)


class CocoFaceDataset(CocoBodyDatasetBase):
    """
    Wrapper class around CocoFace dataset
    http://images.cocodataset.org/

    COCO keypoints::
        0-16 : 'jawline',
        17-21: 'right eyebrow',
        22-26: 'left eyebrow',
        27-30: 'nose bridge',
        31-35: 'nose bottom',
        36-41: 'right eye',
        42-47: 'left eye',
        48-59: 'outer lips'
        60-67: 'inner lips'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        super().__init__(split, input_spec, num_samples)
        self.kpt_db: list[tuple[Path, int, int, torch.Tensor]]

    def _load_kpt_db(self) -> list[tuple[Path, int, int, torch.Tensor]]:
        kpt_db: list[tuple[Path, int, int, torch.Tensor]] = []
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            annotations = self.cocoGt.loadAnns(ann_ids)

            for ann in annotations:
                if ann.get("face_valid", 0) is False:
                    continue  # Keep only persons with valid face

                x1, y1, w, h = ann["face_box"]
                if ann.get("area", 0) > 0 and x1 >= 0 and y1 >= 0:
                    x2 = x1 + w
                    y2 = y1 + h
                    bbox = (x1, y1, x2, y2)

                    img_path = self.image_dir / cast(str, img_info["file_name"])

                    if not img_path.exists():
                        raise FileNotFoundError(f"Image file not found at {img_path}")

                    kpt_db.append(
                        (
                            img_path,
                            img_id,
                            ann.get("category_id", 0),
                            torch.tensor(bbox, dtype=torch.float32),
                        )
                    )
                    break
        return kpt_db

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, int, torch.Tensor]]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            RGB, range [0-1] network input image.

        ground_truth : tuple[int, int, torch.Tensor]
            imageId
                The ID of the image.
            category_id
                The ground truth category ID
            bbox
                The ground truth face bounding box in xyxy format.
                This box is in pixel space.
        """
        file_name, image_id, category_id, bbox = self.kpt_db[index]
        img_path = file_name

        x0, y0, x1, y1 = bbox
        image_array = cv2.imread(cast(str, img_path))
        assert image_array is not None, f"Failed to read image {img_path}"
        image_array = cv2.resize(
            image_array[int(y0) : int(y1 + 1), int(x0) : int(x1 + 1)],
            (self.target_h, self.target_w),
            interpolation=cv2.INTER_LINEAR,
        )

        image = torch.from_numpy(image_array).float().permute(2, 0, 1)

        return image, (image_id, category_id, bbox)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000


class FaceMap3DMMDataset(BaseDataset):
    """FaceMap 3DMM Dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.data_path = FACEMAP3DMM_PRIVATE_ASSET.extracted_path
        self.images_path = self.data_path
        self.gt_path = self.data_path

        self.input_data_zip = input_data_zip
        if input_spec is not None:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]
        else:
            self.input_height = 128
            self.input_width = 128
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        input_image : torch.Tensor
            shape  - [3, H, W]
            range  - [0, 1]
            channel layout - [RGB]

        ground_truth : tuple[int, torch.Tensor, torch.Tensor]
            image_id_tensor
                integer value to represent image id, not used
            gt_landmarks_tensor
                the ground truth x, y positions of facial landmarks, for evaluation only - [68,2]
            bbox_tensor
                the location of the face bounding box, represented as a tensor with shape [4] and layout [left, right, top, bottom]. It is used to crop the face from the original image, for evaluation only.
        """
        image_path = self.image_list[index]
        image_array = cv2.imread(cast(str, image_path))
        assert image_array is not None, f"Failed to load image: {image_path}"

        bbox = [-1, -1, -1, -1]
        landmark_position = np.zeros([76, 2], dtype=np.float32)
        if self.split_str == "val":
            gt_path = self.gt_list[index]
            gt = np.loadtxt(gt_path).astype("int")

            landmark_position[:, :] = gt[6:158].astype("float").reshape(-1, 2)

            image_width, image_height = gt[0], gt[1]
            x0, x1, y0, y1 = gt[-4:]
            width = x1 - x0 + 1
            height = y1 - y0 + 1

            adjusted_x0 = x0 - int(width * 0.1)
            adjusted_y0 = y0 - int(height * 0.1)
            adjusted_width = int(width * 1.2)
            adjusted_height = int(height * 1.2)

            if (
                adjusted_x0 >= 0
                and adjusted_y0 >= 0
                and adjusted_x0 + adjusted_width - 1 < image_width
                and adjusted_y0 + adjusted_height - 1 < image_height
            ):
                x0, y0 = adjusted_x0, adjusted_y0
                x1 = x0 + adjusted_width - 1
                y1 = y0 + adjusted_height - 1

            image_array = image_array[y0 : y1 + 1, x0 : x1 + 1, :]
            bbox = [x0, y0, x1, y1]

        image_array = cv2.resize(
            image_array,
            (self.input_height, self.input_width),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1)
        image_tensor_rgb = torch.flip(image_tensor, dims=[0])
        image_tensor_rgb_norm = image_tensor_rgb / 255  # [0-1] -> [0-255]

        image_id = abs(hash(str(image_path.name[:-4]))) % (10**8)

        return image_tensor_rgb_norm, (
            image_id,
            torch.tensor(landmark_position[:68, :], dtype=torch.float32),
            torch.tensor(bbox, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "images" / self.split_str
        self.gt_path = self.gt_path / "labels" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        for img_path in self.images_path.iterdir():
            self.image_list.append(img_path)
            if self.split_str == "val":
                gt_filename = img_path.name.replace(".png", ".txt")
                gt_path = self.gt_path / gt_filename
                if not gt_path.exists():
                    print(f"Ground truth file not found: {gt_path!s}")
                    return False
                self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        FACEMAP3DMM_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(input_data_zip=str(files[0]))

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 400

    @staticmethod
    def default_num_calibration_samples() -> int:
        """The default value for how many samples to run in each inference job."""
        return 530

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.pose_evaluator import CocoKeypointsPoseEvaluator
from qai_hub_models.evaluators.utils.pose import get_final_preds


class HRNetPoseEvaluator(CocoKeypointsPoseEvaluator):
    """Evaluator for HRNet pose estimation models."""

    def add_batch(
        self,
        output: tuple[torch.Tensor, ...] | torch.Tensor,
        gt: tuple,
    ) -> None:
        """Process a batch of HRNet model outputs and ground truth data.

        Parameters
        ----------
        output
            Model heatmaps, shape (B, J, H, W), or a tuple whose first
            element is the heatmap tensor.
        gt
            Tuple of (image_ids, category_ids, centers, scales,
            box_scores, areas):

            image_ids : torch.Tensor
                COCO image IDs, shape (B,).
            category_ids : torch.Tensor
                COCO category IDs, shape (B,).
            centers : torch.Tensor
                Bounding box centres (cx, cy) in pixels, shape (B, 2).
            scales : torch.Tensor
                HRNet-convention scales [w, h] * 1.25, shape (B, 2).
            box_scores : torch.Tensor
                Detector confidence scores, shape (B,).
            areas : torch.Tensor
                Bounding box areas in pixels^2, shape (B,).
        """
        heatmaps = output[0] if isinstance(output, tuple) else output
        image_ids, category_ids, centers, scales, box_scores, areas = gt
        preds, maxvals = get_final_preds(
            heatmaps.detach().cpu().numpy(),
            centers.numpy(),
            scales.numpy(),
        )
        self._store_predictions(
            preds, maxvals, image_ids, category_ids, box_scores, areas
        )

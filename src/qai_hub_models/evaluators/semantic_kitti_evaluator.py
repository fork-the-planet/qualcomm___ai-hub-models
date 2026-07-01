# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    MEAN_IOU,
    MetricMetadata,
)


class SemanticKittiEvaluator(BaseEvaluator):
    """Evaluator for SemanticKITTI LiDAR semantic segmentation.

    Parameters
    ----------
    n_classes
        Number of output classes.
    learning_map
        Mapping from raw SemanticKITTI label IDs to training class IDs.
    learning_ignore
        Mapping from class ID to bool; ``True`` means the class is ignored
        when computing mIoU.
    knn_params
        Optional KNN post-processing parameters (``knn``, ``search``,
        ``sigma``, ``cutoff``).  When provided, per-point labels are refined
        by voting among range-image neighbours before updating the confusion
        matrix.
    """

    def __init__(
        self,
        n_classes: int,
        learning_map: dict,
        learning_ignore: dict,
        knn_params: dict | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.learning_map = learning_map
        self.knn_params = knn_params
        self.include: list[int] = []
        self.ignore: list[int] = []
        for key, value in learning_ignore.items():
            if value:
                self.ignore.append(key)
            else:
                self.include.append(key)
        self.reset()

    def reset(self) -> None:
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes)).long()
        self.ones: torch.Tensor | None = None
        self.last_scan_size: int | None = None

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[torch.Tensor, ...],
    ) -> None:
        """

        Parameters
        ----------
        output
            Float logits of shape ``[B, NUM_CLASSES, H, W]`` or a
            pre-argmaxed class-index mask of shape ``[B, H, W]``.
        gt
            A tuple with at least three tensors:

            proj_x : torch.Tensor
                Column indices of lidar points, shape ``[B, max_points]``.
                Padding positions are filled with ``-1``.
            proj_y : torch.Tensor
                Row indices of lidar points, shape ``[B, max_points]``.
                Padding positions are filled with ``-1``.
            labels : torch.Tensor
                Raw SemanticKITTI semantic labels, shape ``[B, max_points]``.
            proj_range : torch.Tensor, optional
                Projected range image, shape ``[B, H, W]``.
                Required for KNN post-processing.
            unproj_range : torch.Tensor, optional
                Per-point range values, shape ``[B, max_points]``.
                Required for KNN post-processing.
        """
        p_x = gt[0]
        p_y = gt[1]
        label = gt[2]
        proj_range_batch = gt[3] if len(gt) > 3 else None
        unproj_range_batch = gt[4] if len(gt) > 4 else None

        unproj_argmax_list = []
        for i in range(output.shape[0]):
            out_i = output[i]
            # Support both logits [C, H, W] and pre-argmaxed [H, W]
            if out_i.dim() == 3:
                pred_mask = torch.argmax(out_i, dim=0).to(torch.int32)
            else:
                pred_mask = out_i.to(torch.int32)

            npoints = int((p_x[i] >= 0).sum())
            px_i = p_x[i, :npoints]
            py_i = p_y[i, :npoints]

            if (
                self.knn_params is not None
                and proj_range_batch is not None
                and unproj_range_batch is not None
            ):
                unproj_argmax_i = self._knn_postprocess(
                    proj_range_batch[i],
                    unproj_range_batch[i, :npoints],
                    pred_mask,
                    px_i,
                    py_i,
                )
            else:
                unproj_argmax_i = pred_mask[py_i, px_i]

            unproj_argmax_list.append(unproj_argmax_i.reshape(-1))

        x_row = torch.cat(unproj_argmax_list)
        y_row = torch.cat(
            [label[i, : int((p_x[i] >= 0).sum())] for i in range(output.shape[0])]
        )

        temp = []
        for val in y_row:
            key = 0 if val == -1 else int(val)
            temp.append(self.learning_map.get(key, 0))
        y_row = torch.tensor(temp)

        idxs = torch.stack([x_row, y_row], dim=0)
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones(idxs.shape[-1]).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True
        )

    def _knn_postprocess(
        self,
        proj_range: torch.Tensor,
        unproj_range: torch.Tensor,
        proj_argmax: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
    ) -> torch.Tensor:
        """Refine per-point labels by voting among range-image neighbours.

        Parameters
        ----------
        proj_range
            Projected range image of shape ``[H, W]``.
        unproj_range
            Per-point range values of shape ``[npoints]``.
        proj_argmax
            Predicted class-index mask of shape ``[H, W]``.
        px
            Column indices of shape ``[npoints]``.
        py
            Row indices of shape ``[npoints]``.

        Returns
        -------
        torch.Tensor
            Refined per-point class indices of shape ``[npoints]``.
        """
        assert self.knn_params is not None
        knn = self.knn_params["knn"]
        search = self.knn_params["search"]
        sigma = self.knn_params["sigma"]
        cutoff = self.knn_params["cutoff"]

        if search % 2 == 0:
            raise ValueError("KNN search kernel size must be odd")

        _, W = proj_range.shape
        P = unproj_range.shape[0]
        pad = (search - 1) // 2

        # Unfold range-image neighbourhood: [1, search*search, H*W]
        proj_unfold = F.unfold(
            proj_range[None, None].float(),
            kernel_size=(search, search),
            padding=(pad, pad),
        )
        unproj_unfold = proj_unfold[:, :, py * W + px]  # [1, search*search, P]
        unproj_unfold[unproj_unfold < 0] = float("inf")
        unproj_unfold[:, (search * search - 1) // 2, :] = unproj_range.float()

        # Inverse Gaussian spatial weight
        coords = torch.arange(search, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1)
        mean = (search - 1) / 2.0
        gauss = torch.exp(-((grid - mean) ** 2).sum(-1) / (2 * sigma**2))
        inv_gauss = (1 - gauss / gauss.sum()).view(1, -1, 1)

        k2_dist = torch.abs(unproj_unfold - unproj_range.float()) * inv_gauss
        _, knn_idx = k2_dist.topk(knn, dim=1, largest=False, sorted=False)

        # Unfold argmax and gather top-k neighbours
        argmax_unfold = F.unfold(
            proj_argmax[None, None].float(),
            kernel_size=(search, search),
            padding=(pad, pad),
        ).long()
        knn_argmax = torch.gather(argmax_unfold[:, :, py * W + px], 1, knn_idx)

        if cutoff > 0:
            knn_argmax[torch.gather(k2_dist, 1, knn_idx) > cutoff] = self.n_classes

        # Majority vote (skip class 0 = unlabelled, last = invalid)
        knn_onehot = torch.zeros(1, self.n_classes + 1, P, dtype=torch.float32)
        knn_onehot.scatter_add_(
            1, knn_argmax, torch.ones_like(knn_argmax, dtype=torch.float32)
        )
        votes = knn_onehot[0, 1:-1]
        vote_sum = votes.sum(dim=0)
        knn_result = votes.argmax(dim=0) + 1
        fallback = proj_argmax[py, px]
        return torch.where(vote_sum > 0, knn_result, fallback.to(knn_result.dtype))

    def getIoU(self) -> float:
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        union = tp + fp + fn + 1e-15
        iou_mean = (tp[self.include] / union[self.include]).mean()
        return iou_mean.item() * 100

    def get_accuracy_score(self) -> float:
        return self.getIoU()

    def formatted_accuracy(self) -> str:
        return f"{self.getIoU():.3f} mIOU"

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_IOU

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import Sampler

__all__ = ["EveryNSampler"]


class EveryNSampler(Sampler):
    """Samples every N items deterministically from a torch dataset.

    Picks indices 0, N, 2N, 3N, ... so that num_samples items are returned
    evenly spread across the full dataset.
    """

    def __init__(self, n: int, num_samples: int) -> None:
        self.n = n
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        return iter(range(0, self.num_samples * self.n, self.n))

    def __len__(self) -> int:
        return self.num_samples

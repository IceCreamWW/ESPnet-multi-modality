#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Preencoder."""

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from typeguard import check_argument_types
from typing import Tuple

import torch


class Embedding(AbsPreEncoder):
    """Embedding Preencoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        padding_idx: int = 2,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = output_size
        self.embed = torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        output = self.embed(input)
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim

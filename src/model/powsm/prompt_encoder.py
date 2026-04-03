# Copyright 2019 Shigeki Karita  (Apache 2.0)
"""Prompt encoder: TransformerEncoder with input_layer=None."""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from src.espnet_import.attention import MultiHeadedAttention
from src.espnet_import.encoder_layer import EncoderLayer
from src.espnet_import.layer_norm import LayerNorm
from src.espnet_import.nets_utils import make_pad_mask
from src.espnet_import.positionwise_feed_forward import PositionwiseFeedForward
from src.espnet_import.repeat import repeat


class PromptEncoder(torch.nn.Module):
    """Transformer encoder for prompt conditioning (input_layer=None)."""

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        use_flash_attn: bool = False,
        **_,
    ):
        super().__init__()
        self._output_size = output_size
        # linear projection only when sizes differ
        self.embed = (
            None if input_size == output_size
            else torch.nn.Linear(input_size, output_size)
        )
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    False,          # qk_norm
                    use_flash_attn,
                    False,          # causal
                    False,          # cross_attn
                ),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
        self.normalize_before = normalize_before
        if normalize_before:
            self.after_norm = LayerNorm(output_size)
        self.interctc_use_conditioning = False  # required by PowsmCTCModel

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
        **_,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        if self.embed is not None:
            xs_pad = self.embed(xs_pad)
        for layer in self.encoders:
            xs_pad, masks = layer(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None

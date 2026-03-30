#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from src.core.utils import download_hf_snapshot
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from src.model.sentencepieces_tokenizer import SentencepiecesTokenizer
from src.model.zipa.zipformer_crctc.model import Conv2dSubsampling
from src.model.zipa.zipformer_crctc.model import (
    AsrModel,
    AttentionDecoderModel,
    Decoder,
    Joiner,
    Zipformer2,
)
from src.model.zipa.zipformer_crctc.model.scaling import ScheduledFloat
from src.model.zipa.zipformer_crctc.zipa_ctc_model import ZipaCtcModel

from src.model.zipa.zipformer_crctc.model.utils import AttributeDict
from src.model.zipa.zipformer_crctc.zipa_ctc_inference import ZipaCtcInference

import warnings

warnings.filterwarnings("ignore")


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=1,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=params.attention_decoder_dim,
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(params.encoder_dim)),
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
    )
    return model


small_params = AttributeDict(
    {
        # Fixed parameters
        "feature_dim": 80,
        "subsampling_factor": 4,
        "vocab_size": 127,
        # Zipformer encoder stack parameters
        "num_encoder_layers": "2,2,3,4,3,2",
        "downsampling_factor": "1,2,4,8,4,2",
        "feedforward_dim": "512,768,1024,1536,1024,768",
        "num_heads": "4,4,4,8,4,4",
        "encoder_dim": "192,256,384,512,384,256",
        "query_head_dim": "32",
        "value_head_dim": "12",
        "pos_head_dim": "4",
        "pos_dim": 48,
        "encoder_unmasked_dim": "192, 192, 256, 256, 256, 192",
        "cnn_module_kernel": "31,31,15,15,15,31",
        # Decoder and joiner
        "decoder_dim": 512,
        "joiner_dim": 512,
        # Attention decoder
        "attention_decoder_dim": 512,
        "attention_decoder_num_layers": 6,
        "attention_decoder_attention_dim": 512,
        "attention_decoder_num_heads": 8,
        "attention_decoder_feedforward_dim": 2048,
        # Training details
        "causal": False,
        "chunk_size": "16,32,64,-1",
        "left_context_frames": "64,128,256,-1",
        # Loss and decoding heads
        "use_transducer": False,
        "use_ctc": True,
        "use_attention_decoder": False,
        "use_cr_ctc": True,
        "use_unsup_cr_ctc": False,
    }
)


large_params = AttributeDict(
    {
        # Fixed parameters
        "feature_dim": 80,
        "subsampling_factor": 4,
        "vocab_size": 127,
        # Zipformer encoder stack parameters
        "num_encoder_layers": "4,3,4,5,4,4",
        "downsampling_factor": "1,2,4,8,4,2",
        "feedforward_dim": "768,768,1536,2048,1536,768",
        "num_heads": "6,6,6,8,6,6",
        "encoder_dim": "512,512,768,1024,768,512",
        "query_head_dim": "64",
        "value_head_dim": "48",
        "pos_head_dim": "4",
        "pos_dim": 48,
        "encoder_unmasked_dim": "192,192,256,320,256,192",
        "cnn_module_kernel": "31,31,15,15,15,31",
        # Decoder and joiner
        "decoder_dim": 1024,
        "joiner_dim": 1024,
        # Attention decoder
        "attention_decoder_dim": 512,
        "attention_decoder_num_layers": 6,
        "attention_decoder_attention_dim": 512,
        "attention_decoder_num_heads": 8,
        "attention_decoder_feedforward_dim": 2048,
        # Training details
        "causal": False,
        "chunk_size": "16,32,64,-1",
        "left_context_frames": "64,128,256,-1",
        # Loss and decoding heads
        "use_transducer": False,
        "use_ctc": True,
        "use_attention_decoder": False,
        "use_cr_ctc": True,
        "use_unsup_cr_ctc": False,
    }
)


def build_zipactc_model(
    *,
    work_dir: str,
    hf_repo: str,
    force: bool = False,
):
    """Build ZIPA-CTC model from local files or huggingface repo.
    Args:
        work_dir: Directory to store downloaded files from hf repo.
        hf_repo: Huggingface repo name. If None, load from local files.
        force: Whether to force re-download from hf repo.
    Returns: ZipaCtcModel instance.
    """
    download_dir = f"{work_dir}/{hf_repo.replace('/', '_')}"
    download_hf_snapshot(
        repo_id=hf_repo,
        force_download=force,
        work_dir=download_dir,
    )
    root = Path(download_dir)
    model_path = list(root.glob("*.pth"))
    if len(model_path) == 0:
        raise FileNotFoundError(f"No model file found under {root}")
    if len(model_path) > 1:
        raise RuntimeError(f"Multiple model files found under {root}: {model_path}")
    model_path = str(model_path[0])

    if "small" in model_path:
        params = small_params
    elif "large" in model_path:
        params = large_params
    else:
        raise ValueError(
            f"model_name must contain 'small' or 'large' but got {model_path}"
        )

    asr_model = get_model(params)

    zipa_ctc = ZipaCtcModel(encoder=asr_model, blank_id=0)
    zipa_ctc.encoder.load_state_dict(
        torch.load(model_path, map_location="cpu"), strict=True
    )
    return zipa_ctc


def build_zipactc_inference(
    *, work_dir: str, hf_repo: str, bpe_model, force: bool = False, device="cpu"
):
    model = build_zipactc_model(work_dir=work_dir, hf_repo=hf_repo, force=force)
    tokenizer = SentencepiecesTokenizer(model=bpe_model)
    inference = ZipaCtcInference(
        inference_model=model, tokenizer=tokenizer, device=device
    )
    return inference

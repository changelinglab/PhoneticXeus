import copy
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import yaml
import json
import torch

from src.model.powsm.specaug import SpecAug
from src.model.powsm.e_branchformer import EBranchformerEncoder
from src.model.xeusphoneme.cnn_frontend import CNNFrontend as Wav2VecCNN
from src.model.xeusphoneme.linear_layer import LinearProjection
from src.core.utils import download_hf_snapshot
from src.model.xeusphoneme.xeuspr_model import XeusPRModel
from src.model.xeusphoneme.xeuspr_inference import XeusPRInference
from src.model.powsm.ctc import CTC
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=False)


class XeusPRTokenizer:
    """Tokenizer that maps IPA phones to IDs using the xeuspr ipa_vocab.json."""

    def __init__(self, vocab_file: str):
        with open(vocab_file) as f:
            self.vocab: Dict[str, int] = json.load(f)
        self.unk_id = self.vocab.get("<unk>", 0)

    def tokens2ids(self, tokens) -> list:
        return [self.vocab.get(t, self.unk_id) for t in tokens]



def build_xeus_pr(
    config_file: str,
    checkpoint: Optional[str] = None,
    vocab_file: Optional[str] = None,
    ctc_config: Optional[dict] = None,
    weighted_sum: bool = False,
    interctc_layer_idx: Optional[list] = None,
    interctc_weight: float = 0.0,
    interctc_use_conditioning: bool = False,
    interctc_ctc_type: str = "phone",
    ctc_aux_config: Optional[dict] = None,
    decoder_config: Optional[dict] = None,
    ctc_weight: float = 1.0,
) -> XeusPRModel:
    """Build Xeus PR model from config and optional checkpoint.

    Args:
        config_file: Path to config yaml file
        checkpoint: Path to model checkpoint (pretrained or fully trained)
        vocab_file: Path to vocabulary file. If None, use vocab in config.
        ctc_config: Optional dict of CTC config
        weighted_sum: Whether to use weighted sum of transformer layers

    Returns:
        XeusPRModel
    """
    with open(config_file, "r", encoding="utf-8") as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    if vocab_file is not None:
        with open(vocab_file) as f:
            tok2id = json.load(f)
            id2tok = {v: k for k, v in tok2id.items()}
            token_list = [id2tok[i] for i in range(len(id2tok))]
    elif isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
    else:
        token_list = list(args.token_list)
    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")

    assert (
        getattr(args, "frontend") == "wav2vec_cnn"
    ), "Config must specify wav2vec_cnn frontend"
    frontend = Wav2VecCNN(**args.frontend_conf)
    input_size = frontend.output_size()

    specaug = None
    if hasattr(args, "specaug") and args.specaug == "specaug":
        specaug = SpecAug(**args.specaug_conf)

    normalize = None
    assert (
        getattr(args, "preencoder") == "linear"
    ), "Config must specify linear preencoder"
    preencoder = LinearProjection(input_size=input_size, **args.preencoder_conf)
    input_size = preencoder.output_size()
    assert (
        args.encoder == "e_branchformer"
    ), f"Only e_branchformer supported, got {args.encoder}"
    encoder_conf = dict(args.encoder_conf)
    if interctc_layer_idx:
        encoder_conf["interctc_layer_idx"] = interctc_layer_idx
    if interctc_use_conditioning:
        encoder_conf["interctc_use_conditioning"] = True
    encoder = EBranchformerEncoder(input_size=input_size, **encoder_conf)

    ctc_config = ctc_config or getattr(args, "ctc_conf", {})
    ctc_config_orig = copy.deepcopy(ctc_config)
    # Build CTC
    ctc = CTC(
        odim=vocab_size,
        encoder_output_size=encoder.output_size(),
        **ctc_config,
    )

    # Build optional aux CTC (orthographic vocabulary)
    ctc_aux = None
    if ctc_aux_config is not None:
        import sentencepiece as spm

        ctc_aux_config = dict(ctc_aux_config)  # copy to avoid mutating caller's dict
        sp = spm.SentencePieceProcessor()
        sp.load(ctc_aux_config.pop("vocab_file"))
        aux_vocab_size = sp.get_piece_size()
        ctc_aux = CTC(
            odim=aux_vocab_size,
            encoder_output_size=encoder.output_size(),
            ctc_type="builtin",
            **ctc_aux_config,
        )
        log.info(f"Built aux CTC with vocab size {aux_vocab_size}")

    # Build optional attention decoder
    decoder = None
    if decoder_config:
        from src.model.powsm.transformer_decoder import TransformerDecoder

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **decoder_config,
        )

    # Build model
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=preencoder,
        ignore_id=getattr(args, "ignore_id", -1),
        sym_blank=getattr(args, "sym_blank", "<blank>"),
        freeze_frontend=checkpoint is not None,
        weighted_sum=weighted_sum,
        interctc_weight=interctc_weight,
        interctc_use_conditioning=interctc_use_conditioning,
        interctc_ctc_type=interctc_ctc_type,
        ctc_aux=ctc_aux,
        decoder=decoder,
        ctc_weight=ctc_weight,
    )

    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            # convert to standard xeus style checkpoint
            state_dict = state_dict["state_dict"]  # for finetuned lightning checkpoints
            state_dict = {
                k.replace("net.", ""): v
                for k, v in state_dict.items()
                if k.startswith("net.")
            }
        load_info = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded checkpoint: {checkpoint} with load info: {load_info}")
        print(f"Loaded checkpoint: {checkpoint} with load info: {load_info}")

    model.training_args = args
    model._net_config = {
        "ctc_config": ctc_config_orig,
        "weighted_sum": weighted_sum,
        "interctc_layer_idx": interctc_layer_idx,
        "interctc_weight": interctc_weight,
        "interctc_use_conditioning": interctc_use_conditioning,
        "interctc_ctc_type": interctc_ctc_type,
        "ctc_aux_config": ctc_aux_config,
        "decoder_config": decoder_config,
        "ctc_weight": ctc_weight,
    }
    return model


def build_xeus_pr_from_hf(
    *,
    work_dir: str,
    hf_repo: Optional[str] = None,
    force: bool = False,
    config_file: Optional[str] = None,
    checkpoint: Optional[str] = None,
    vocab_file: Optional[str] = None,
    ctc_config: Optional[dict] = None,
    load_ckpt: bool = True,
    weighted_sum: bool = False,
    interctc_layer_idx: Optional[list] = None,
    interctc_weight: float = 0.0,
    interctc_use_conditioning: bool = False,
    interctc_ctc_type: str = "phone",
    ctc_aux_config: Optional[dict] = None,
    decoder_config: Optional[dict] = None,
    ctc_weight: float = 1.0,
) -> XeusPRModel:
    """Build Xeus PR model from local files or HuggingFace repo.

    Args:
        work_dir: Directory to store downloaded files from HF repo
        hf_repo: HuggingFace repo name (e.g., "username/xeus-pr")
            If None, load from local files only
        force: Whether to force re-download from HF repo
        config_file: Path to config file. If None, use default path in work_dir.
            Takes precedence over hf_repo download.
        checkpoint: Path to checkpoint file. If None, use default path in work_dir.
            Takes precedence over hf_repo download.
        vocab_file: Path to vocabulary file. If None, use path in config.
        ctc_config: Optional dict of CTC config
        load_ckpt: Whether to load checkpoint weights
        weighted_sum: Whether to use weighted sum of transformer layers
    Returns:
        XeusPRModel
    """
    # Default relative paths in HF repo
    REL_CONFIG = "model/config.yaml"
    REL_CKPT = "model/xeus_checkpoint_new.pth"

    # Download from HF if repo specified
    if hf_repo:
        log.info(f"Downloading snapshot from HuggingFace: {hf_repo}")
        download_hf_snapshot(
            repo_id=hf_repo,
            force_download=force,
            work_dir=work_dir,
        )

    # Resolve file paths
    root = Path(work_dir)
    cfg = config_file or str(root / REL_CONFIG)
    ckpt = checkpoint or str(root / REL_CKPT)

    # Verify files exist
    assert Path(cfg).exists(), f"Config file not found: {cfg}"
    if not load_ckpt:
        ckpt = None
    else:
        assert Path(ckpt).exists(), f"Checkpoint file not found: {ckpt}"

    log.info(f"Building model from config: {cfg}")
    log.info(f"Loading checkpoint: {ckpt}")

    return build_xeus_pr(
        config_file=cfg,
        checkpoint=ckpt,
        vocab_file=vocab_file,
        ctc_config=ctc_config,
        weighted_sum=weighted_sum,
        interctc_layer_idx=interctc_layer_idx,
        interctc_weight=interctc_weight,
        interctc_use_conditioning=interctc_use_conditioning,
        interctc_ctc_type=interctc_ctc_type,
        ctc_aux_config=ctc_aux_config,
        decoder_config=decoder_config,
        ctc_weight=ctc_weight,
    )


def build_xeus_pr_inference(
    work_dir: str,
    checkpoint: str,
    vocab_file: str,
    device,
    config_file: Optional[str] = None,
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    dtype: str = "float32",
    ctc_config: Optional[dict] = None,
    weighted_sum: bool = False,
    interctc_layer_idx: Optional[list] = None,
    interctc_weight: float = 0.0,
    interctc_use_conditioning: bool = False,
    interctc_ctc_type: str = "phone",
    ctc_aux_config: Optional[dict] = None,
    decoder_config: Optional[dict] = None,
) -> XeusPRInference:
    model = build_xeus_pr_from_hf(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force_download,
        config_file=config_file,
        checkpoint=checkpoint,
        vocab_file=vocab_file,
        ctc_config=ctc_config,
        load_ckpt=True,
        weighted_sum=weighted_sum,
        interctc_layer_idx=interctc_layer_idx,
        interctc_weight=interctc_weight,
        interctc_use_conditioning=interctc_use_conditioning,
        interctc_ctc_type=interctc_ctc_type,
        ctc_aux_config=ctc_aux_config,
        decoder_config=decoder_config,
    )
    inference_obj = XeusPRInference(model, device=device, dtype=dtype)
    return inference_obj

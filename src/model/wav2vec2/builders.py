import json
from typing import Optional
import torch
from src.model.wav2vec2.tokenizer import Wav2Vec2Tokenizer
from src.model.wav2vec2.wav2vec2_model import Wav2Vec2Model
from src.model.wav2vec2.wav2vec2_inference import Wav2Vec2Inference
from src.model.wav2vec2.wav2vec2pr_model import Wav2Vec2PRModel
from src.model.powsm.ctc import CTC
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_wav2vec2_tokenizer(
    hf_repo: str = "facebook/mms-300m",
):
    """Build Wav2Vec2 tokenizer

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2 tokenizer
    """
    tokenizer = Wav2Vec2Tokenizer(hf_repo=hf_repo)
    log.info(f"Wav2Vec2 tokenizer loaded from {hf_repo}")
    return tokenizer


def build_wav2vec2_model(
    hf_repo: str = "facebook/mms-300m",
    output_vocabsz: Optional[int] = None,
    blank_id: int = 0,
    freeze_frontend: bool = False,
    weighted_sum: bool = False,
):
    """Build Wav2Vec2 model

    Args:
        hf_repo: HuggingFace repository ID
        output_vocabsz: Optional output vocabulary size
        blank_id: Blank token ID for CTC
        freeze_frontend: Whether to freeze the encoder layers
        weighted_sum: Whether to use a weighted sum of encoder layers for CTC

    Returns:
        Wav2Vec2 model
    """
    model = Wav2Vec2Model(
        hf_repo=hf_repo,
        output_vocabsz=output_vocabsz,
        blank_id=blank_id,
        weighted_sum=weighted_sum,
    )
    log.info(f"Wav2Vec2 model loaded from {hf_repo}")
    log.info(f"Model vocab size: {model.vocab_size}")
    return model


def build_wav2vec2pr_inference(
    checkpoint: str,
    vocab_file: str,
    hf_repo: str = "facebook/mms-300m",
    device: str = "cpu",
):
    """Build Wav2Vec2 inference module

    Returns:
        Wav2Vec2 inference module
    """
    net = build_wav2vec2pr(
        hf_repo=hf_repo,
        vocab_file=vocab_file,
        ctc_config=None,
        freeze_frontend=False,
        checkpoint=checkpoint,
    )
    inference_module = Wav2Vec2Inference(net, device=device)
    log.info("Wav2Vec2 inference module built")
    return inference_module


def build_wav2vec2pr(
    hf_repo: str = "facebook/mms-300m",
    vocab_file: Optional[str] = None,
    ctc_config: Optional[dict] = None,
    freeze_frontend: bool = True,
    checkpoint: Optional[str] = None,
    weighted_sum: bool = True,
    interctc_weight: float = 0.0,
    interctc_layer_idx: Optional[list] = None,
    interctc_use_conditioning: bool = False,
) -> Wav2Vec2PRModel:
    """Build Wav2Vec2 Phone Recognition model.

    Args:
        hf_repo: HuggingFace repository ID for the pretrained Wav2Vec2 model
        vocab_file: Path to vocabulary JSON file (token -> id mapping)
        ctc_config: Optional dict of CTC configuration
        freeze_frontend: Whether to freeze the feature extraction layers
        checkpoint: Optional path to a checkpoint file to load model weights
        weighted_sum: Whether to use a weighted sum of encoder layers for CTC

    Returns:
        Wav2Vec2PRModel instance
    """
    # Load vocabulary
    if vocab_file is not None:
        with open(vocab_file) as f:
            tok2id = json.load(f)
            id2tok = {v: k for k, v in tok2id.items()}
            token_list = [id2tok[i] for i in range(len(id2tok))]
    else:
        raise ValueError("vocab_file is required for Wav2Vec2PRModel")

    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")

    # Build encoder (without output vocab, CTC handles projection)
    encoder = Wav2Vec2Model(
        hf_repo=hf_repo,
        output_vocabsz=None,
        weighted_sum=weighted_sum,
    )
    log.info(f"Wav2Vec2 encoder loaded from {hf_repo}")

    # Build CTC module
    ctc_config = ctc_config or {}
    ctc = CTC(
        odim=vocab_size,
        encoder_output_size=encoder.encoder_output_size(),
        **ctc_config,
    )

    # Build model
    model = Wav2Vec2PRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=token_list,
        freeze_frontend=freeze_frontend,
        interctc_weight=interctc_weight,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    log.info("Wav2Vec2PRModel built successfully")
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            # convert to standard w2v2 checkpoint
            state_dict = state_dict["state_dict"]  # for finetuned lightning checkpoints
            state_dict = {
                k.replace("net.", ""): v
                for k, v in state_dict.items()
                if k.startswith("net.")
            }
        load_info = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded checkpoint: {checkpoint} with load info: {load_info}")
        print(f"Loaded checkpoint: {checkpoint} with load info: {load_info}")

    return model


if __name__ == "__main__":
    # python -m src.model.wav2vec2.builders
    import torch

    model = build_wav2vec2_model(hf_repo="facebook/mms-300m")
    wav = torch.randn(1, 16000 * 5)
    inputs = {"speech": wav, "speech_lengths": torch.tensor([wav.shape[1]])}
    with torch.no_grad():
        outputs = model.encode(**inputs)
    print(outputs)
    print(f"Output shape: {outputs[0].shape}, Lengths shape: {outputs[1].shape}")

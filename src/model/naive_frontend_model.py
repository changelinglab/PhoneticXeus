import torch
import torch.nn as nn
from src.model.powsm.frontend import DefaultFrontend
from typing import Tuple
import torchaudio
import torch.nn.functional as F
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FrontendModel(torch.nn.Module):
    """
    A naive model that only computes features using powsm's frontend,
    without any neural network. Can act as a naive baseline.

    Expected inputs
    ---------------
    x : torch.Tensor
        Shape (B, T) float tensor of mono waveforms.
    x_lengths : torch.Tensor
        Shape (B,) lengths in samples.

    Returns
    -------
    h : torch.Tensor
        Log-Mel features of shape (B, T_frames, n_mels).
    h_lens : torch.Tensor
        Shape (B,) feature lengths in frames.
    """

    def __init__(
        self,
        fs=16_000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        hidden_sizes=[],
        output_vocabsz=1,
        blank_id=0,
    ):
        super().__init__()
        # Use the requested config for the *default* frontend
        self.sampling_rate = fs
        self.frontend = DefaultFrontend(
            fs=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            apply_stft=True,
            frontend_conf=None,  # no WPE/MVDR; just plain STFT -> LogMel
        )
        frontend_dim = self.frontend.n_mels
        self.blank_id = blank_id
        self.hidden_net = nn.Identity()
        if hidden_sizes:
            layers = []
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(
                    nn.Linear(
                        frontend_dim if i == 0 else hidden_sizes[i - 1], hidden_size
                    )
                )
            self.hidden_net = nn.Sequential(*layers)

        self.encoder_dim = frontend_dim if not hidden_sizes else hidden_sizes[-1]
        self.ctc_lo = torch.nn.Linear(
            frontend_dim if not hidden_sizes else hidden_sizes[-1], output_vocabsz
        )

    # needed for hidden representation based classification models
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ):
        """
        Compute features with the default frontend.

        Parameters
        ----------
        x : Tensor (B, T)
        x_lengths : Tensor (B,)

        Returns
        -------
        h : Tensor (B, T_frames, n_mels)
        h_lens : Tensor (B,)
        """
        if x.dim() != 2:
            raise ValueError(
                f"x must be (B, T) mono waveforms; got shape {tuple(x.shape)}"
            )
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()

        # DefaultFrontend.forward returns (features, feature_lengths)
        h, h_lens = self.frontend(x, x_lengths)
        h = self.hidden_net(h)
        return h, h_lens

    def encode(self, x: torch.Tensor, x_lengths: torch.Tensor):
        """
        Alias for forward, to match expected interface.

        Parameters
        ----------
        x : Tensor (B, T)
        x_lengths : Tensor (B,)

        Returns
        -------
        h : Tensor (B, T_frames, n_mels)
        h_lens : Tensor (B,)
        """
        return self(x, x_lengths)

    def encoder_output_size(self) -> int:
        return self.encoder_dim

    ## needed for forced alignment model
    def points_by_frames(self) -> int:
        """Get the ratio of input points to output frames."""
        return self.frontend.hop_length

    def ctc_logits(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits from encoder output"""
        h, hlen = self(speech, speech_lengths)
        h = self.ctc_lo(h)
        return h, hlen

    def get_blank_id(self) -> int:
        """Get the blank symbol ID for CTC."""
        return self.blank_id

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """Calculate frame-wise alignment from CTC probabilities.
        Only an inference function that uses the ctc posteriors.

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            utt_id: str, identifier for the utterance
        Returns:
            Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Log probability scores of the labels for each time
                step.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]
        assert batch_size == 1, "Forced alignment needs batch size 1."

        # -1 is used as padding index in collate fn
        text = text[:, : text_lengths.max()]  # for data-parallel
        logits, logit_lengths = self.ctc_logits(speech, speech_lengths)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, Tmax, odim)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        if log_probs.shape[1] < text.shape[1]:
            log.error(
                f"Logits length {log_probs.shape} is shorter than "
                f"text length {text.shape}, for utt_id: {utt_id}"
            )
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs,
            text,
            logit_lengths,
            text_lengths,
            blank=self.blank_id,
        )
        return align_label, align_prob

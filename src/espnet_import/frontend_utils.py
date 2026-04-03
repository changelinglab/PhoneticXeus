# Merged from:
# (1) espnet2/asr/frontend/abs_frontend.py
# (2) espnet2/layers/log_mel.py
# (3) espnet2/layers/stft.py
# (4) espnet/nets/pytorch_backend/frontends/frontend.py
# (5) espnet2/utils/get_default_kwargs.py
# Helpers inlined from:
#   espnet2/layers/inversible_interface.py
#   espnet2/enh/layers/complex_utils.py

import inspect
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import librosa
import numpy
import numpy as np
import torch
import torch.nn as nn
from packaging.version import parse as V
from typeguard import typechecked

from src.espnet_import.nets_utils import make_pad_mask


is_torch_1_10_plus = V(torch.__version__) >= V("1.10.0")


# --- espnet2/layers/inversible_interface.py ---

class InversibleInterface(ABC):
    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        

# --- espnet2/enh/layers/complex_utils.py ---

def to_complex(c):
    # Convert to torch native complex
    if torch.is_complex(c):
        return c
    elif hasattr(c, "real") and hasattr(c, "imag"):
        return torch.complex(c.real, c.imag)
    else:
        return torch.view_as_complex(c)


# --- espnet2/asr/frontend/abs_frontend.py ---

class AbsFrontend(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# --- espnet2/layers/log_mel.py ---

class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens


# --- espnet2/layers/stft.py ---

class Stft(torch.nn.Module, InversibleInterface):
    @typechecked
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or
                    (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            stft_kwargs["return_complex"] = True
            # NOTE(Jinchuan) CuFFT is not compatible with bfloat16
            output = torch.stft(input.float(), **stft_kwargs)
            output = torch.view_as_real(output).type(input.dtype)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which "
                    "does not support the training mode."
                )

            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = torch.cat(
                    [
                        torch.zeros(n_pad_left),
                        window,
                        torch.zeros(n_pad_right),
                    ],
                    0,
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = torch.ones(win_length)

            output = []
            for i, instance in enumerate(input):
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                output.append(
                    torch.tensor(np.stack([stft.real, stft.imag], -1))
                )
            output = torch.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = torch.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(
                bs, -1, output.size(1), output.size(2), 2
            ).transpose(1, 2)

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(
                    ilens - self.n_fft, self.hop_length, rounding_mode="trunc"
                )
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or complex Tensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(
                self.win_length, dtype=datatype, device=input.device
            )
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens


# --- espnet/nets/pytorch_backend/frontends/frontend.py ---

class Frontend(nn.Module):
    """Frontend class."""

    def __init__(
        self,
        idim: int,
        # WPE options
        use_wpe: bool = False,
        wtype: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        # Beamformer options
        use_beamformer: bool = False,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        bnmask: int = 2,
        badim: int = 320,
        ref_channel: int = -1,
        bdropout_rate=0.0,
    ):
        """Initialize frontend."""
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data,
        # e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2

        self.wpe = None # not used in this codebase
        self.beamformer = None # not used in this codebase

    def forward(
        self,
        x: torch.Tensor,
        ilens: Union[torch.LongTensor, numpy.ndarray, List[int]],
    ) -> Tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor]]:
        """Calculate frontend forward propagation."""
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        mask = None
        h = x
        if h.dim() == 4:
            if self.training:
                choices = [(False, False)] if not self.use_frontend_for_all else []
                if self.use_wpe:
                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = choices[
                    numpy.random.randint(len(choices))
                ]

            else:
                use_wpe = self.use_wpe
                use_beamformer = self.use_beamformer

            if use_wpe:
                raise NotImplementedError("WPE is not implemented in this codebase.")
            if use_beamformer:
                raise NotImplementedError("Beamformer is not implemented in this codebase.")

        return h, ilens, mask


def frontend_for(args, idim):
    """Instantiate an frontend module given the program arguments.."""
    return Frontend(
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,
        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        bnmask=args.bnmask,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate,
    )


# --- espnet2/utils/get_default_kwargs.py ---

class _Invalid:
    """Marker object for not serializable-object"""


def get_default_kwargs(func):
    """Get the default values of the input function.

    Examples:
        >>> def func(a, b=3):  pass
        >>> get_default_kwargs(func)
        {'b': 3}

    """

    def yaml_serializable(value):
        # isinstance(x, tuple) includes namedtuple, so type is used here
        if type(value) is tuple:
            return yaml_serializable(list(value))
        elif isinstance(value, set):
            return yaml_serializable(list(value))
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                return _Invalid
            retval = {}
            for k, v in value.items():
                v2 = yaml_serializable(v)
                # Register only valid object
                if v2 not in (_Invalid, inspect.Parameter.empty):
                    retval[k] = v2
            return retval
        elif isinstance(value, list):
            retval = []
            for v in value:
                v2 = yaml_serializable(v)
                # If any elements in the list are invalid,
                # the list also becomes invalid
                if v2 is _Invalid:
                    return _Invalid
                else:
                    retval.append(v2)
            return retval
        elif value in (inspect.Parameter.empty, None):
            return value
        elif isinstance(value, (float, int, complex, bool, str, bytes)):
            return value
        else:
            return _Invalid

    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: p.default for p in params.values()}
    # Remove not yaml-serializable object
    data = yaml_serializable(data)
    return data

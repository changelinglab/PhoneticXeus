"""Datamodule to read kaldi style powsm datasets using scp indices."""

import torch
import torchaudio
import kaldiio
from torch.utils.data import Dataset
import lightning as L
import yaml
from typing import Optional, Dict, List
from src.utils import RankedLogger
from pathlib import Path

log = RankedLogger(__name__, rank_zero_only=True)


class KaldiDataset(Dataset):
    def __init__(
        self,
        wav_scp_file,
        text_file,
        lang_file,
        data_dir: Path,
        sampling_rate=16000,
        vocab_file: Optional[str] = None,
        ignore_id: int = -1,
        portable_wavscp=True,
        max_time_sec: Optional[float] = None,
    ):
        self.sampling_rate = sampling_rate
        self.ignore_id = ignore_id
        self.data_dir = data_dir
        self.wav_scp = self._load_wav_scp(wav_scp_file, portable_wavscp)
        self.text = self._load_text(text_file)
        self.key2lang = self._extract_language(lang_file)
        self.max_time_sec = max_time_sec

        # Load vocabulary for tokenization
        self.vocab = self._load_vocab(vocab_file) if vocab_file else None

        assert set(self.wav_scp.keys()).issubset(
            set(self.text.keys())
        ), "Extra key in wav.scp"
        self.keys = list(self.wav_scp.keys())
        assert all(
            k in self.key2lang for k in self.keys
        ), "Missing language tags for some keys"
        log.info(
            f"Loaded dataset: {len(self.key2lang)} lang keys, {len(self.keys)} samples"
        )
        if vocab_file:
            log.info(
                f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}"
            )

    def _load_wav_scp(self, path, portable_wavscp=True):
        def _create_env_specific_path(wav_path, portable_wavscp):
            wav_path = wav_path.strip()
            ark_or_wav, element_index = (
                wav_path.split(":", 1) if ":" in wav_path else (wav_path, None)
            )
            ark_or_wav = Path(ark_or_wav)
            if ark_or_wav.is_absolute():
                abs_wav_path = ark_or_wav
            else:
                if not portable_wavscp:
                    ark_or_wav = Path(*ark_or_wav.parts[2:])  # skip "dump/raw"
                abs_wav_path = self.data_dir / ark_or_wav
            if element_index is not None:
                abs_wav_path = f"{abs_wav_path}:{element_index}"
            return str(abs_wav_path)

        wav_scp = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key, wav_path = parts[0], parts[1]
                    wav_path = _create_env_specific_path(wav_path, portable_wavscp)
                    wav_scp[key] = str(wav_path)
        return wav_scp

    def _load_text(self, path):
        text_dict = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # some examples have spaces in between
                    # we must retain full length of transcript
                    text_dict[parts[0]] = " ".join(parts[1:])
        return text_dict

    def _extract_language(self, path):
        key2lang = {}
        with open(path) as f:
            for line in f:
                key, tag = line.strip().split()[:2]
                if key.endswith("_pr"):
                    key = key[:-3]
                key2lang[key] = tag.split("><")[0][1:].strip()
        return key2lang

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary mapping from file.

        Args:
            vocab_file: Path to vocabulary file. Each line should contain a token.

        Returns:
            Dictionary mapping token string to token ID.
        """
        vocab = {}
        with open(vocab_file) as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                vocab[token] = idx
        return vocab

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text into token IDs using loaded vocabulary.

        Args:
            text: Text string (typically phonetic transcription).

        Returns:
            List of token IDs. Unknown tokens are replaced with ignore_id.
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not loaded. Provide vocab_file parameter.")

        tokens = []
        for token in text.split():
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                # Replace unknown tokens with ignore_id
                log.warning(f"Unknown token: {token}, replacing with ignore_id")
                tokens.append(self.ignore_id)
        return tokens

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = self.wav_scp[key]
        transcription = self.text[key]

        if ".ark" in wav_path:
            sr, wav = kaldiio.load_mat(wav_path)
            waveform = torch.from_numpy(wav).float().unsqueeze(0)
        else:
            waveform, sr = torchaudio.load(wav_path)

        if self.max_time_sec:  # trim
            max_samples = int(self.max_time_sec * sr)
            waveform = waveform[:, :max_samples]

        # to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        waveform = waveform.squeeze(0)  # (1, T) -> (T,)
        # Tokenize text if vocabulary is loaded
        text_tokens = self._tokenize_text(transcription) if self.vocab else None

        return {
            "key": key,
            "utt_id": key,
            "speech": waveform.to(torch.float32),
            "speech_length": waveform.shape[-1],
            "text": transcription,
            "text_tokens": text_tokens,
            "wavpath": wav_path,
            # powsm lang sym. default is <unk> if missing in vocab
            "lang_sym": self.key2lang[key],
            "split": "test",
            "metadata_idx": idx,
            "target": transcription,
            "text": transcription,
        }


class KaldiDataModule(L.LightningDataModule):
    def __init__(
        self,
        wav_scp_file,
        text_file,
        lang_file,
        data_dir: Path,
        sampling_rate=16000,
        batch_size=16,
        num_workers=4,
        vocab_file: Optional[str] = None,
        ignore_id: int = -1,
        portable_wavscp: bool = True,
    ):
        super().__init__()
        log.info(
            f"Initializing KaldiDataModule with {wav_scp_file}, {text_file}, {lang_file}"
        )
        self.wav_scp_file = wav_scp_file
        self.text_file = text_file
        self.lang_file = lang_file
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.ignore_id = ignore_id
        self.data_dir = data_dir
        self.portable_wavscp = portable_wavscp

    def setup(self, stage=None):
        self.dataset = KaldiDataset(
            wav_scp_file=self.wav_scp_file,
            text_file=self.text_file,
            lang_file=self.lang_file,
            data_dir=self.data_dir,
            sampling_rate=self.sampling_rate,
            vocab_file=self.vocab_file,
            ignore_id=self.ignore_id,
            portable_wavscp=self.portable_wavscp,
        )

    def train_dataloader(self):
        raise ValueError("This datamodule is not intended for training use.")

    def val_dataloader(self):
        raise ValueError("This datamodule is not intended for validation use.")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        keys = [item["key"] for item in batch]
        speeches = [item["speech"] for item in batch]
        speech_lengths = torch.tensor([item["speech_length"] for item in batch])
        texts = [item["text"] for item in batch]
        wavpaths = [item["wavpath"] for item in batch]
        languages = [item["lang_sym"] for item in batch]

        # Pad speeches to the max length in the batch
        max_speech_length = max(speech_lengths)
        padded_speeches = torch.zeros(len(batch), max_speech_length)
        for i, speech in enumerate(speeches):
            padded_speeches[i, : speech.shape[-1]] = speech

        # Handle text tokenization for CTC-based training
        text_data = {"text": texts}

        if batch[0].get("text_tokens") is not None:
            # Pad tokenized text to max length in batch
            text_tokens_list = [item["text_tokens"] for item in batch]
            max_text_length = max(len(tokens) for tokens in text_tokens_list)

            padded_texts = torch.full(
                (len(batch), max_text_length),
                self.dataset.ignore_id,
                dtype=torch.long,
            )
            text_lengths = torch.zeros(len(batch), dtype=torch.long)

            for i, tokens in enumerate(text_tokens_list):
                padded_texts[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                text_lengths[i] = len(tokens)

            text_data["text"] = padded_texts
            text_data["text_length"] = text_lengths

        return {
            "keys": keys,
            "speech": padded_speeches,
            "speech_length": speech_lengths,
            "text": text_data["text"],
            "text_length": text_data.get("text_length"),
            "wavpath": wavpaths,
            "lang_sym": languages,
        }


def build_kaldi_datamodule(
    dataset_name,
    data_dir,
    dataset_config_path="configs/data/prism_pr_evalsets.yaml",
    sampling_rate=16000,
    batch_size=16,
    num_workers=4,
    vocab_file: Optional[str] = None,
    ignore_id: int = -1,
    portable_wavscp: bool = True,
):
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(data_dir)
    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_config = config["datasets"][dataset_name]
    #############################################
    wav_scp_file = Path(ds_config["wav_scp"])
    text_file = Path(ds_config["text_phoneme"])
    lang_file = Path(ds_config["language"])
    #############################################
    if not wav_scp_file.is_absolute():
        wav_scp_file = data_dir / wav_scp_file
    if not text_file.is_absolute():
        text_file = data_dir / text_file
    if not lang_file.is_absolute():
        lang_file = data_dir / lang_file
    #############################################

    return KaldiDataModule(
        wav_scp_file=wav_scp_file,
        text_file=text_file,
        lang_file=lang_file,
        data_dir=data_dir,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        vocab_file=vocab_file,
        ignore_id=ignore_id,
        portable_wavscp=portable_wavscp,
    )


if __name__ == "__main__":
    # Test with: IPAPACK_DATA_ROOT=/path/to/data python -m src.data.kaldi_dataset
    import os

    datamodule = build_kaldi_datamodule(
        "doreco",
        data_dir=os.environ.get("IPAPACK_DATA_ROOT", "") + "/dump/raw",
        dataset_config_path="configs/data/prism_pr_evalsets.yaml",
        portable_wavscp=False,
        sampling_rate=16000,
        batch_size=2,
        num_workers=1,
    )
    datamodule.setup()
    for batch in datamodule.predict_dataloader().dataset:
        key = batch["key"]
        speech = batch["speech"].cpu().numpy()
        print(speech.shape)
        break

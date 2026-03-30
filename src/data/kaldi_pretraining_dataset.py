"""Datamodule to read kaldi style powsm datasets using scp indices."""

import json
import os
import random

import torch
import torchaudio
import kaldiio
import yaml
from torch.utils.data import Dataset
import lightning as L
from lightning.pytorch.utilities import CombinedLoader
from typing import Optional, Dict, List, Union
from tqdm import tqdm
import panphon  # 0.22.2
import json

from src.data.text_tokenizer import build_text_tokenizer
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class KaldiPretrainingDataset(Dataset):
    def __init__(
        self,
        wav_scp_file: Union[str, Dict[str, Union[str, Dict[str, float]]]],
        text_file,
        lang_file,
        sampling_rate=16000,
        split="test",
        limit_samples=None,
        filter_langs=None,
        read_asr_text=False,
        vocab_file: Optional[str] = None,
        task_set: Optional[List[str]] = None,
        accent_file: Optional[Dict[str, str]] = None,
        max_duration_sec: Optional[int] = 20,
        aux_tokenizer=None,
    ):
        self.sampling_rate = sampling_rate
        self.task_set = task_set  # set of tasks to filter on, e.g., ['pr', 'asr']
        self.wav_scp = self._load_wav_scp(wav_scp_file, limit_samples)
        self.text = self._load_text(text_file, limit_samples)
        self.key2lang = self._extract_language(lang_file, limit_samples)
        self.key2accent = self._extract_accent(accent_file)
        self.ipa_segmenter = panphon.FeatureTable()
        self.aux_tokenizer = aux_tokenizer
        if read_asr_text or aux_tokenizer is not None:
            # lang file also has asr text
            self.asr_text = self._load_asr_text(lang_file, limit_samples)
        else:
            self.asr_text = None
        if filter_langs:
            self.filter_by_langs(filter_langs)
        self.split = split
        self.max_duration_sec = max_duration_sec

        # Load vocabulary for tokenization
        self.vocab = self._load_vocab(vocab_file) if vocab_file else None
        self.unk_id = -1 if not self.vocab else self.vocab.get("<unk>", -1)

        self.keys = list(set(self.wav_scp.keys()).intersection(set(self.text.keys())))
        assert (
            self.keys
        ), "Keyset is empty after loading wav_scp and text files. WAV keys: {}, TEXT keys: {}".format(
            list(self.wav_scp.keys())[:2], list(self.text.keys())[:2]
        )
        assert all(
            k in self.key2lang for k in self.keys
        ), "Missing language tags for some keys, check the task_set in config."
        log.info(
            f"Loaded dataset: {len(self.key2lang)} lang keys, {len(self.keys)} samples"
        )
        log.info(f"Number of unique languages: {len(set(self.key2lang.values()))}")
        if vocab_file:
            log.info(
                f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}."
                f" Unk ID: {self.unk_id}"
            )

    def filter_by_langs(self, filter_langs: List[str]):
        """Filter dataset to only include samples from specified languages."""
        original_sz = len(self.key2lang)
        filtered_keys = set(
            [k for k in self.key2lang if self.key2lang[k] in filter_langs]
        )
        self.wav_scp = {k: v for k, v in self.wav_scp.items() if k in filtered_keys}
        self.text = {k: v for k, v in self.text.items() if k in filtered_keys}
        if self.asr_text:
            self.asr_text = {
                k: v for k, v in self.asr_text.items() if k in filtered_keys
            }
        log.info(
            f"Filtering dataset by languages {filter_langs}. "
            f"Reduced samples from {original_sz} to {len(filtered_keys)}."
        )
        self.key2lang = {k: v for k, v in self.key2lang.items() if k in filtered_keys}
        self.key2accent = {
            k: v for k, v in self.key2accent.items() if k in filtered_keys
        }

    def _keep_key(self, key: str) -> bool:
        """Check if a key should be kept based on task_set."""
        if not self.task_set:
            return True
        return any(key.endswith(f"_{task}") for task in self.task_set)

    def _load_wav_scp(self, path_or_pathcnt, limit_samples: Optional[int] = None):
        wav_scp = {}
        if isinstance(path_or_pathcnt, str):
            path_cnt = [(path_or_pathcnt, 0)]  # <=0 means load all
        else:
            path_cnt = [(p, c) for p, c in path_or_pathcnt.items()]
        for path, cnt in path_cnt:
            wav_scp_ = {}
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key, wav_path = parts[0], parts[1]
                        if not self._keep_key(key):
                            continue
                        if not os.path.isabs(wav_path):
                            data_root = os.environ.get(
                                "IPAPACK_DATA_ROOT", ""
                            )
                            wav_path = f"{data_root}/{wav_path}"
                        wav_scp_[key] = wav_path
                    if limit_samples and len(wav_scp_) >= limit_samples:
                        break
            if cnt > 0:
                wav_scp_ = dict(
                    random.choices(list(wav_scp_.items()), k=cnt)
                    if cnt > len(wav_scp_)
                    else random.sample(list(wav_scp_.items()), k=cnt)
                )
            log.info(
                f"Loaded {len(wav_scp_)} samples from wav.scp: {path} with count {cnt}"
            )
            print(
                f"Loaded {len(wav_scp_)} samples from wav.scp: {path} with count {cnt}"
            )
            wav_scp.update(wav_scp_)
        return wav_scp

    def _load_text(self, path, limit_samples: Optional[int] = None):
        text_dict = {}
        with open(path) as f:
            for line in tqdm(f, desc="Reading text"):
                key, *remaining = line.strip().split()
                if not self._keep_key(key):
                    continue
                if len(remaining) >= 1:
                    # some examples have spaces in between
                    # we must retain full length of transcript
                    text_dict[key] = " ".join(remaining)
                if limit_samples and len(text_dict) >= limit_samples:
                    break
        return text_dict

    def _load_asr_text(self, path, limit_samples: Optional[int] = None):
        # TODO(shikhar): ADHOC ANALYSIS FUNCTION, REMOVE LATER
        asr_text_dict = {}
        with open(path) as f:
            for line in tqdm(f, desc="Reading ASR text"):
                key, *remaining = line.strip().split()
                if not key.endswith("_asr"):
                    continue
                if len(remaining) >= 1:
                    # some examples have spaces in between
                    # we must retain full length of transcript
                    key = key[:-4] + "_pr"
                    asr_text_dict[key] = " ".join(remaining[1:])
                if limit_samples and len(asr_text_dict) >= limit_samples:
                    break
        log.info("Loaded ASR text for %d samples", len(asr_text_dict))
        return asr_text_dict

    def _extract_accent(self, path):
        key2accent = {}
        if not path:
            return key2accent
        with open(path) as f:
            for line in tqdm(f, desc="Reading accent"):
                accent_jsonl = json.loads(line.strip())
                utt_id = accent_jsonl["utt_id"]
                accent = accent_jsonl["tag"]
                key2accent[utt_id] = accent
        return key2accent

    def _extract_language(self, path, limit_samples: Optional[int] = None):
        key2lang = {}
        with open(path) as f:
            for line in tqdm(f, desc="Reading language"):
                key, tag = line.strip().split()[:2]
                if not self._keep_key(key):
                    continue
                if key.endswith("_pr") and "pr" not in (self.task_set or []):
                    # remove _pr suffix for some datasets in evalset index.
                    # but do not remove if we are using pr task in training.
                    # TODO(shikhar): Bad design, should be modified at source to make it generic.
                    key = key[:-3]
                key2lang[key] = tag.split("><")[0][1:].strip()
                if limit_samples and len(key2lang) >= limit_samples:
                    break
        return key2lang

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Return a Dictionary that maps token string to token ID.
        vocab_file: Path to vocabulary file which is a json of token to id mapping.
        """
        with open(vocab_file) as f:
            vocab = json.load(f)
        return vocab

    def _tokenize_text(self, text: str) -> List[int]:
        """Return List of token IDs. Unknown tokens are replaced with ignore_id.
        text: Text string (typically phonetic transcription).
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not loaded. Provide vocab_file parameter.")
        tokens = []
        for ipastr in text.split("/"):
            if not ipastr:
                continue
            for ipasym in self.ipa_segmenter.ipa_segs(ipastr):
                tokens.append(self.vocab.get(ipasym, self.unk_id))
        return tokens

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = self.wav_scp[key]
        transcription = self.text[key]
        asr_text = self.asr_text[key] if self.asr_text else None
        lang = self.key2lang[key]
        accent = self.key2accent.get(key, "<unk>")

        if ".ark" in wav_path:
            sr, wav = kaldiio.load_mat(wav_path)
            waveform = torch.from_numpy(wav).float().unsqueeze(0)
        else:
            waveform, sr = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:  # to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        waveform = waveform.squeeze(0)  # (1, T) -> (T,)
        # Trim if longer than max duration
        waveform = waveform[: self.max_duration_sec * self.sampling_rate]
        wavlen = waveform.shape[-1]

        # Tokenize text if vocabulary is loaded
        text_tokens = self._tokenize_text(transcription) if self.vocab else None

        # Tokenize ASR text with aux tokenizer if available
        asr_text_tokens = None
        if self.aux_tokenizer is not None and self.asr_text:
            raw = self.asr_text.get(key)
            if raw:
                asr_text_tokens = self.aux_tokenizer.tokenize(raw)

        return {
            "key": key,
            "utt_id": key,
            "speech": waveform.to(torch.float32),
            "speech_length": wavlen,
            "sr": self.sampling_rate,
            "text_tokens": text_tokens,
            "wavpath": wav_path,
            # powsm lang sym. default is <unk> if missing in vocab
            "lang_sym": lang,
            "accent_sym": accent,
            "split": self.split,
            "metadata_idx": idx,
            "target": transcription,
            "text": transcription,
            "asr_text": asr_text,
            "asr_text_tokens": asr_text_tokens,
        }


class KaldiPretrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        wav_scp_file: Dict[str, Union[str, Dict[str, float]]],
        text_file: Dict[str, str],
        lang_file: Dict[str, str],
        train_split: str,
        dev_splits: List[str],
        predict_split: Optional[str] = None,
        sampling_rate=16000,
        max_duration_sec=20,
        batch_size=16,
        num_workers=4,
        limit_samples: Optional[int] = None,
        filter_langs: Optional[List[str]] = None,
        read_asr_text: bool = False,
        task_set: Dict[str, List[str]] = None,
        accent_file: Dict[str, str] = None,
        vocab_file: Optional[str] = None,
        aux_vocab_file: Optional[str] = None,
        aux_tokenizer_type: str = "sentencepiece",
    ):
        super().__init__()
        self.wav_scp_file = wav_scp_file
        self.text_file = text_file
        self.lang_file = lang_file
        self.train_split = train_split
        self.dev_splits = dev_splits
        self.predict_split = predict_split
        self.sampling_rate = sampling_rate
        self.max_duration_sec = max_duration_sec
        self.limit_samples = limit_samples
        self.filter_langs = filter_langs
        self.read_asr_text = read_asr_text
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.task_set = task_set
        self.accent_file = accent_file
        self.splits = list(wav_scp_file.keys())
        self.ignore_id = -1
        self.aux_tokenizer = (
            build_text_tokenizer(aux_vocab_file, aux_tokenizer_type)
            if aux_vocab_file is not None
            else None
        )
        log.info(f"Splits: {self.splits}")

    def _ds(self, split):
        return KaldiDataset(
            self.wav_scp_file[split],
            self.text_file[split],
            self.lang_file[split],
            self.sampling_rate,
            split=split,
            limit_samples=self.limit_samples,
            filter_langs=self.filter_langs,
            vocab_file=self.vocab_file,
            task_set=self.task_set[split],
            read_asr_text=self.read_asr_text,
            max_duration_sec=self.max_duration_sec,
            accent_file=self.accent_file[split] if self.accent_file else None,
            aux_tokenizer=self.aux_tokenizer,
        )

    def setup(self, stage=None):
        for split in self.splits:
            setattr(self, f"{split}_dataset", self._ds(split=split))

    def _dl(self, split):
        return torch.utils.data.DataLoader(
            getattr(self, f"{split}_dataset"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            # persistent_workers=True,
            # prefetch_factor=2,
        )

    def train_dataloader(self):
        return self._dl(split=self.train_split)

    def val_dataloader(self):
        """Return dataloader(s) for validation splits."""
        loaders = [self._dl(split=s) for s in self.dev_splits]
        assert len(loaders) > 0, "No validation splits found."
        return CombinedLoader(loaders, mode="sequential")

    def test_dataloader(self):
        if self.predict_split is None:
            raise ValueError("No predict_split specified.")
        return self._dl(split=self.predict_split)

    def predict_dataloader(self):
        if self.predict_split is None:
            raise ValueError("No predict_split specified.")
        return self._dl(split=self.predict_split)

    def collate_fn(self, batch):
        keys = [item["key"] for item in batch]
        speeches = [item["speech"] for item in batch]
        speech_lengths = torch.tensor([item["speech_length"] for item in batch])
        texts = [item["text"] for item in batch]
        asr_texts = [item["asr_text"] for item in batch]
        wavpaths = [item["wavpath"] for item in batch]
        languages = [item["lang_sym"] for item in batch]
        accents = [item["accent_sym"] for item in batch]

        # Pad speeches to the max length in the batch
        max_speech_length = max(speech_lengths)
        max_speech_length = (
            self.max_duration_sec * self.sampling_rate
        )  # enforce max length
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
                self.ignore_id,
                dtype=torch.long,
            )
            text_lengths = torch.zeros(len(batch), dtype=torch.long)

            for i, tokens in enumerate(text_tokens_list):
                padded_texts[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                text_lengths[i] = len(tokens)

            text_data["text"] = padded_texts
            text_data["text_length"] = text_lengths

        result = {
            "keys": keys,
            "speech": padded_speeches,
            "speech_length": speech_lengths,
            "text": text_data["text"],
            "text_length": text_data.get("text_length"),
            "wavpath": wavpaths,
            "lang_sym": languages,
            "accent_sym": accents,
            "asr_text": asr_texts,
        }

        # Pad tokenized ASR text (aux vocabulary) if present
        if any(item.get("asr_text_tokens") is not None for item in batch):
            tokens_list = [item.get("asr_text_tokens") or [] for item in batch]
            max_len = max(len(t) for t in tokens_list) or 1
            padded_asr = torch.full((len(batch), max_len), self.ignore_id, dtype=torch.long)
            asr_lengths = torch.zeros(len(batch), dtype=torch.long)
            for i, t in enumerate(tokens_list):
                if t:
                    padded_asr[i, : len(t)] = torch.tensor(t, dtype=torch.long)
                    asr_lengths[i] = len(t)
            result["asr_text_tokens"] = padded_asr      # (B, T_asr)
            result["asr_text_length"] = asr_lengths     # (B,)

        return result


def build_kaldi_pretraining_datamodule(
    train_split: str,
    dev_splits: List[str],
    dataset_config_path: str = "configs/data/ipapack_index.yaml",
    predict_split: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    limit_samples: Optional[int] = None,
    filter_langs: Optional[List[str]] = None,
    read_asr_text: bool = False,
    vocab_file: Optional[str] = None,
    aux_vocab_file: Optional[str] = None,
    aux_tokenizer_type: str = "sentencepiece",
):
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)

    all_splits = [train_split] + dev_splits + ([predict_split] if predict_split else [])
    wav_scp_file, text_file, lang_file, task_set, accent_file = {}, {}, {}, {}, {}
    for split_key in all_splits:
        if split_key not in config["datasets"]:
            raise ValueError(f"Split '{split_key}' not found in dataset config")
        ds_config = config["datasets"][split_key]
        wav_scp_file[split_key] = ds_config["wav_scp"]
        text_file[split_key] = ds_config["text_phoneme"]
        lang_file[split_key] = ds_config["language"]
        task_set[split_key] = ds_config.get("task_set", None)
        accent_file[split_key] = ds_config.get("accent", None)

    log.info(f"Loaded splits: {all_splits}")

    return KaldiPretrainingDataModule(
        wav_scp_file=wav_scp_file,
        text_file=text_file,
        lang_file=lang_file,
        train_split=train_split,
        dev_splits=dev_splits,
        predict_split=predict_split,
        sampling_rate=config.get("sampling_rate", 16000),
        batch_size=batch_size,
        num_workers=num_workers,
        limit_samples=limit_samples,
        filter_langs=filter_langs,
        read_asr_text=read_asr_text,
        task_set=task_set,
        accent_file=accent_file,
        vocab_file=vocab_file,
        aux_vocab_file=aux_vocab_file,
        aux_tokenizer_type=aux_tokenizer_type,
    )


if __name__ == "__main__":
    # Test with: python -m src.data.kaldi_pretraining_dataset
    datamodule = build_kaldi_pretraining_datamodule(
        train_split="train_accentmix_multi",  # "train_accentmix_multi",
        dev_splits=[
            "dev_1k",
            "dev_gmuaccent",
            "dev_buckeye",
            "dev_epadb",
            "dev_speechoceanotth",
            "dev_l2arctic",
        ],
        predict_split="predict",
        dataset_config_path="configs/data/ipapack_index.yaml",
        batch_size=1,
        num_workers=1,
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
        # limit_samples=2,
    )
    datamodule.setup()
    print("Train dataloader:")
    c = 0
    for i in datamodule.train_dataloader():
        print(i)
        c += 1
        if c > 10:
            break

    # print("--" * 20)
    # print("Dev dataloader:")
    # for i in datamodule.val_dataloader():
    #     print(i)

    # print("--" * 20)
    # print("Test dataloader:")
    # for i in datamodule.test_dataloader():
    #     print(i)

    # print("--" * 20)
    # print("Predict dataloader:")
    # for i in datamodule.predict_dataloader():
    #     print(i)

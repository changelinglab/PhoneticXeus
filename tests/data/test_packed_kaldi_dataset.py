import torch
import numpy as np
import json
import pytest
from unittest.mock import patch
from src.data.packed_kaldi_dataset import (
    PackedKaldiDataset,
    PackedKaldiDataModule,
    DistributedDynamicSampler,
)

# ==========================================
# Fixtures
# ==========================================


@pytest.fixture
def mock_env(tmp_path):
    """Creates a temporary environment with mock ASR data files."""
    wav_scp, txt, lng, l_npy, vocab = (
        tmp_path / "wav.scp",
        tmp_path / "text",
        tmp_path / "lang",
        tmp_path / "len.npy",
        tmp_path / "vocab.json",
    )
    # Define 4 utterances with varying lengths
    wav_scp.write_text("u1 p1.wav\nu2 p2.ark:10\nu3 p3.wav\nu4 p4.wav")
    txt.write_text("u1 PHONEME_A\nu2 PHONEME_B\nu3 PHONEME_C\nu4 PHONEME_D")
    lng.write_text("u1 <en>\nu2 <en>\nu3 <en>\nu4 <en>")

    # u1+u3+u4 = 5+2+3 = 10s. u2 = 10s.
    np.save(l_npy, {"u1": 5.0, "u2": 10.0, "u3": 2.0, "u4": 3.0})

    with open(vocab, "w") as f:
        json.dump(
            {
                "PHONEME_A": 1,
                "PHONEME_B": 2,
                "PHONEME_C": 3,
                "PHONEME_D": 4,
                "<unk>": 0,
                "<sep>": 5,
            },
            f,
        )

    return {
        "wav": str(wav_scp),
        "txt": str(txt),
        "lng": str(lng),
        "len": str(l_npy),
        "vocab": str(vocab),
    }


# ==========================================
# Dataset Tests
# ==========================================


def test_packing_ratio_behavior(mock_env):
    """Tests that use_packing=0.0 results in no packing (one item per pack)."""
    ds_no_pack = PackedKaldiDataset(**mock_env, use_packing=0.0, max_duration=20.0)
    assert len(ds_no_pack.packs) == 4

    ds_full_pack = PackedKaldiDataset(**mock_env, use_packing=1.0, max_duration=12.0)
    # u2(10s) is alone. u1+u4+u3 (5+3+2 + overhead) fits in 12s.
    # Total packs should be 2.
    assert len(ds_full_pack.packs) == 2


def test_tokenization_and_sep(mock_env):
    """Validates phonetic tokenization and <sep> insertion."""
    ds = PackedKaldiDataset(**mock_env, use_packing=1.0, max_duration=20.0)

    # Force a pack manually for testing token logic
    ds.packs = [["u1", "u3"]]
    with patch("torchaudio.load", return_value=(torch.randn(1, 16000), 16000)):
        item = ds[0]
        tokens = item["tokens"].tolist()
        # Should be [u1_token, sep_token, u3_token] -> [1, 5, 3]
        assert tokens == [1, 5, 3]


@patch("kaldiio.load_mat")
@patch("torchaudio.load")
def test_hybrid_loading(mock_audio, mock_kaldi, mock_env):
    """Ensures both standard wav and Kaldi ark files are loaded correctly."""
    mock_audio.return_value = (torch.randn(1, 16000), 16000)
    mock_kaldi.return_value = (16000, np.random.randn(16000))

    ds = PackedKaldiDataset(**mock_env, use_packing=0.0)

    # u1 is .wav (p1.wav)
    _ = ds[0]
    assert mock_audio.called

    # u2 is .ark (p2.ark:10)
    _ = ds[1]
    assert mock_kaldi.called


# ==========================================
# Sampler Tests
# ==========================================


def test_sampler_epoch_shuffling():
    """Checks that setting epoch changes order but remains deterministic."""
    durations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    s1 = DistributedDynamicSampler(durations, max_units=10.0, num_replicas=1, rank=0)
    s1.set_epoch(1)
    order1 = list(s1)

    s2 = DistributedDynamicSampler(durations, max_units=10.0, num_replicas=1, rank=0)
    s2.set_epoch(1)
    order2 = list(s2)

    s3 = DistributedDynamicSampler(durations, max_units=10.0, num_replicas=1, rank=0)
    s3.set_epoch(2)
    order3 = list(s3)

    assert order1 == order2
    assert order1 != order3


def test_sampler_oversized_item():
    """Ensures an item longer than max_units is still yielded in its own batch."""
    durations = [50.0]  # max_units is 20.0
    sampler = DistributedDynamicSampler(
        durations, max_units=20.0, num_replicas=1, rank=0
    )
    batches = list(sampler)
    assert len(batches) == 1
    assert batches[0] == [0]


# ==========================================
# DataModule Tests
# ==========================================


@patch("torchaudio.load")
def test_static_vs_dynamic_modes(mock_load, mock_env):
    """Tests switching between standard static batching and dynamic duration batching."""
    mock_load.return_value = (torch.randn(1, 16000), 16000)

    # Static mode: batch_size=2
    dm_static = PackedKaldiDataModule(
        wav_dict={"train": mock_env["wav"]},
        txt_dict={"train": mock_env["txt"]},
        lng_dict={"train": mock_env["lng"]},
        len_dict={"train": mock_env["len"]},
        vocab_file=mock_env["vocab"],
        batch_size=2,
        dynamic_batching=False,
    )
    dm_static.setup()
    batch = next(iter(dm_static.train_dataloader()))
    assert batch["speech"].shape[0] == 2

    # Dynamic mode: max_units=11.0 (u2 is 10s, so it should be alone)
    dm_dynamic = PackedKaldiDataModule(
        wav_dict={"train": mock_env["wav"]},
        txt_dict={"train": mock_env["txt"]},
        lng_dict={"train": mock_env["lng"]},
        len_dict={"train": mock_env["len"]},
        vocab_file=mock_env["vocab"],
        max_units=11.0,
        dynamic_batching=True,
    )
    dm_dynamic.setup()
    # Find the batch containing u2 (index 1)
    loader = dm_dynamic.train_dataloader()
    batches = list(loader)

    # One of the batches should contain exactly 1 item (u2) because 10s + next item > 11s
    assert any(len(b["keys"]) == 1 for b in batches)


def test_collate_padding():
    """Tests collate_fn handles padding values for audio and tokens correctly."""
    dm = PackedKaldiDataModule(None, None, None, None, "dummy.json")
    mock_batch = [
        {"speech": torch.ones(100), "tokens": torch.tensor([1, 2]), "key": "k1"},
        {"speech": torch.ones(50), "tokens": torch.tensor([3]), "key": "k2"},
    ]

    collated = dm.collate_fn(mock_batch)

    # Audio padding (0.0)
    assert collated["speech"][1, 75] == 0.0
    # Token padding (-1)
    assert collated["tokens"][1, 1] == -1
    # Lengths
    assert torch.equal(collated["speech_length"], torch.tensor([100, 50]))
    assert torch.equal(collated["token_length"], torch.tensor([2, 1]))

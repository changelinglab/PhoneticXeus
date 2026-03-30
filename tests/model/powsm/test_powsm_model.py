import torch

from src.model.powsm import powsm_model
from src.model.powsm.ctc import CTC


class DummyEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)
        self.interctc_use_conditioning = False

    def forward(self, feats, feats_lengths, *args, **kwargs):
        out = self.proj(feats)
        return out, feats_lengths, None

    def output_size(self):
        return self.proj.out_features


class DummyDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)
        self.linear = torch.nn.Linear(vocab_size, vocab_size)

    def forward(self, encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens):
        embedded = self.embedding(ys_in_pad)
        logits = self.linear(embedded)
        return logits, None


def test_powsm_model_forward_backward_pass():
    vocab = ["<blank>", "<sos>", "<eos>", "<sop>", "<na>", "<space>", "a", "b"]
    encoder = DummyEncoder(input_dim=4, output_dim=6)
    decoder = DummyDecoder(vocab_size=len(vocab))
    ctc_module = CTC(odim=len(vocab), encoder_output_size=encoder.output_size())

    model = powsm_model.PowsmModel(
        vocab_size=len(vocab),
        token_list=vocab,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc_module,
        sym_space="<space>",
        sym_blank="<blank>",
        sym_sos="<sos>",
        sym_eos="<eos>",
        sym_sop="<sop>",
        sym_na="<na>",
    )
    model.train()

    batch_size = 2
    max_frames = 6
    feat_dim = 4

    speech = torch.randn(batch_size, max_frames, feat_dim, requires_grad=True)
    speech_lengths = torch.tensor([6, 5], dtype=torch.long)
    text = torch.tensor([[6, 7, 5, -1], [7, 6, -1, -1]], dtype=torch.long)
    text_lengths = torch.tensor([3, 2], dtype=torch.long)
    text_prev = torch.tensor(
        [[vocab.index("<na>"), -1, -1, -1], [6, 5, -1, -1]], dtype=torch.long
    )
    text_prev_lengths = torch.tensor([1, 2], dtype=torch.long)
    text_ctc = torch.tensor([[6, 7, 5, -1], [7, 6, 5, -1]], dtype=torch.long)
    text_ctc_lengths = torch.tensor([3, 3], dtype=torch.long)

    loss = model(
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
        text_prev=text_prev,
        text_prev_lengths=text_prev_lengths,
        text_ctc=text_ctc,
        text_ctc_lengths=text_ctc_lengths,
    )["loss"]

    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(
        param.grad is not None for param in model.parameters() if param.requires_grad
    )

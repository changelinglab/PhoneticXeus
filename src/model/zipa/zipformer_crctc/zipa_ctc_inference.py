"""ZIPA-CTC inference class.
Usage:
    python -m src.model.zipa.zipformer_crctc.zipa_ctc_inference
"""

import torch


class ZipaCtcInference:
    def __init__(self, inference_model, tokenizer, device="cpu"):
        self.device = device
        self.inference_model = inference_model.to(device)
        self.tokenizer = tokenizer
        self.inference_model.to(device)
        self.inference_model.eval()

    def batchify(self, speech):
        speech = speech.unsqueeze(0).to(self.device)
        speech_length = torch.tensor(
            [speech.size(1)], dtype=torch.long, device=self.device
        )
        return speech, speech_length

    def ctc_collapse(self, predicted_ids):
        blank_id = self.inference_model.get_blank_id()
        collapsed = []
        prev = None
        for idx in predicted_ids:
            if idx == blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            collapsed.append(idx)
            prev = idx
        return collapsed

    @torch.no_grad()
    def __call__(self, speech, *args, **kwargs):
        """
        Args:
            speech: (Length,) raw waveform
        Returns:
            List[Dict] containing predicted strings
        """
        speech, speech_length = self.batchify(speech)  # -> (1, T), (1,)
        assert speech.size(0) == 1, "Only batch_size=1 is supported in inference."
        logits, _ = self.inference_model.ctc_logits(speech, speech_length)
        predicted_ids = torch.argmax(logits, dim=-1)  # (B, T)
        collapsed_preds = []
        for b in range(predicted_ids.size(0)):
            collapsed = self.ctc_collapse(predicted_ids[b].cpu().tolist())
            collapsed_preds.append(collapsed)
        transcription = self.tokenizer.ids2text(collapsed_preds[0])  # list of tokens
        return [
            {
                "processed_transcript": transcription,
            }
        ]


if __name__ == "__main__":
    from src.model.zipa.zipformer_crctc.builders import build_zipactc_inference

    work_dir = "path/to/exp/zipactc_cache"
    hf_repo = "anyspeech/zipa-large-crctc-500k"
    bpe_model = "src/model/zipa/resources/unigram_127.model"

    inference = build_zipactc_inference(
        work_dir=work_dir, hf_repo=hf_repo, bpe_model=bpe_model, force=False
    )

    # Example usage
    import torchaudio

    waveform, sr = torchaudio.load("path/to/test_audio.wav")
    waveform = waveform.squeeze(0)  # (Length,)
    print("Read waveform with shape:", waveform.shape)
    results = inference(waveform)
    print(results)

import torch


class Wav2Vec2PhonemeInference:
    def __init__(self, inference_model, tokenizer, device="cpu"):
        self.device = device
        self.inference_model = inference_model.to(device)
        self.tokenizer = tokenizer
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
            Dict containing list of predicted strings
        """
        speech, speech_length = self.batchify(speech)
        logits, _ = self.inference_model.ctc_logits(speech, speech_length)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = self.ctc_collapse(predicted_ids[0].cpu().numpy())
        transcription = self.tokenizer.ids2tokens(predicted_ids)
        transcription = "".join(transcription)
        processed_transcription = transcription.replace(
            self.tokenizer.unk_symbol, ""
        ).replace(self.tokenizer.pad_symbol, "")
        return [
            {
                "processed_transcript": processed_transcription,
                "predicted_transcript": transcription,
            }
        ]

import os
import sys

# Ensure vendored src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download

from src.model.xeusphoneme.builders import build_xeus_pr_inference

MAX_SECONDS = 60
SAMPLE_RATE = 16000

inference = None


def load_model():
    ckpt = hf_hub_download(
        "changelinglab/PhoneticXeus", "checkpoint-22000.ckpt"
    )
    resources = os.path.join(
        os.path.dirname(__file__),
        "src", "model", "xeusphoneme", "resources",
    )
    vocab = os.path.join(resources, "ipa_vocab.json")
    config = os.path.join(resources, "xeus_config.yaml")
    return build_xeus_pr_inference(
        work_dir="/tmp/cache/xeus",
        checkpoint=ckpt,
        vocab_file=vocab,
        config_file=config,
        device="cpu",
        interctc_use_conditioning=True,
    )


def transcribe(audio_path):
    """Run phone recognition on uploaded/recorded audio."""
    global inference
    if audio_path is None:
        return "", ""

    if inference is None:
        inference = load_model()

    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform[: SAMPLE_RATE * MAX_SECONDS]

    if waveform.numel() == 0:
        return "", ""

    results = inference(waveform)

    processed = results[0]["processed_transcript"]
    predicted = results[0]["predicted_transcript"]
    spaced = " ".join(
        t for t in predicted.split("/")
        if not (t.startswith("<") and t.endswith(">"))
    )
    return spaced, processed


with gr.Blocks(title="PhoneticXeus") as demo:
    gr.Markdown(
        "# PhoneticXeus\n"
        "Multilingual phone recognition -- record or upload the multilingual speech "
        "to get an IPA transcription.\n\n"
        "Model: [changelinglab/PhoneticXeus]"
        "(https://huggingface.co/changelinglab/PhoneticXeus) "
        "| Paper: [arXiv 2603.29042]"
        "(https://arxiv.org/abs/2603.29042)"
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Input Audio",
        )

    btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        phones_output = gr.Textbox(
            label="IPA Phones (space-separated)",
            lines=3,
            show_copy_button=True,
        )
        raw_output = gr.Textbox(
            label="Raw output (concatenated)",
            lines=3,
            show_copy_button=True,
        )

    btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[phones_output, raw_output],
    )

    gr.Markdown(
        "---\n"
        f"Max audio length: {MAX_SECONDS}s. "
        "Audio is resampled to 16 kHz mono."
    )

demo.launch()

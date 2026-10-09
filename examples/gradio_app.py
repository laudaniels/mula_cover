"""Minimal local GUI for MuLaCover reference-audio cover generation.

Install the extra dependency once, then launch:

    python -m pip install -e '.[audio,gui]'
    python examples/gradio_app.py --model_path ./ckpt

Opens a local Gradio page (http://127.0.0.1:7860 by default) where you can
upload a reference track, fill in lyrics/tags, and generate a cover without
typing a CLI command each time. On an 8 GB GPU, keep "Low VRAM mode" on.
"""

import argparse
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import torch

from mulacover import MuLaCoverGenPipeline

_pipe = None
_pipe_key = None


def _build_pipeline(model_path: str, low_vram: bool) -> MuLaCoverGenPipeline:
    global _pipe, _pipe_key
    key = (model_path, low_vram)
    if _pipe is not None and _pipe_key == key:
        return _pipe

    has_cuda = torch.cuda.is_available()
    main_device = torch.device("cuda:0") if has_cuda else torch.device("cpu")
    backbone_dtype = torch.bfloat16 if has_cuda else torch.float32

    if low_vram and has_cuda:
        devices = {
            "mulacover": main_device,
            "codec": torch.device("cpu"),
            "qwen": torch.device("cpu"),
            "transcriptor": torch.device("cpu"),
        }
    else:
        devices = main_device

    dtypes = {
        "mulacover": backbone_dtype,
        "codec": torch.float32,
        "qwen": torch.float32,
        "transcriptor": torch.float32,
    }
    _pipe = MuLaCoverGenPipeline.from_pretrained(
        model_path, device=devices, dtype=dtypes, lazy_load=True
    )
    _pipe_key = key
    return _pipe


def generate(
    model_path,
    ref_audio,
    lyrics,
    tags,
    bpm,
    cfg_scale,
    temperature,
    topk,
    max_audio_length_ms,
    seed,
    low_vram,
    save_symbolic,
):
    if not ref_audio:
        return None, [], "Upload a reference audio file first."
    if not lyrics or not lyrics.strip():
        return None, [], "Lyrics must not be empty."
    if not tags or not tags.strip():
        return None, [], "Style tags must not be empty."

    try:
        pipe = _build_pipeline(model_path, low_vram)
        if seed not in (None, ""):
            torch.manual_seed(int(seed))

        out_dir = Path(tempfile.mkdtemp(prefix="mulacover_gui_"))
        save_path = out_dir / "cover.wav"
        symbolic_dir = out_dir / "transcribed" if save_symbolic else None

        inputs = {"ref_audio": ref_audio, "lyrics": lyrics, "tags": tags}
        if bpm not in (None, ""):
            inputs["bpm"] = float(bpm)

        pipe(
            inputs,
            save_path=str(save_path),
            symbolic_save_dir=str(symbolic_dir) if symbolic_dir else None,
            cfg_scale=float(cfg_scale),
            temperature=float(temperature),
            topk=int(topk),
            max_audio_length_ms=int(max_audio_length_ms),
        )

        midi_files = []
        if symbolic_dir is not None and symbolic_dir.is_dir():
            midi_files = sorted(str(p) for p in symbolic_dir.iterdir())

        return str(save_path), midi_files, "Done."
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return None, [], (
                "CUDA out of memory. Try: enable 'Low VRAM mode', lower "
                "max_audio_length_ms, close other GPU apps, or restart this "
                "app to release GPU memory."
            )
        return None, [], f"Error: {exc}"
    except Exception as exc:  # surface any other failure in the UI
        traceback.print_exc()
        return None, [], f"Error: {exc}"


def build_demo(default_model_path: str) -> gr.Blocks:
    with gr.Blocks(title="MuLaCover") as demo:
        gr.Markdown(
            "# MuLaCover — reference-audio cover generator\n"
            "Upload a reference track, provide new lyrics and a style "
            "description, and generate a cover."
        )
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(
                    value=default_model_path, label="Checkpoint directory"
                )
                ref_audio = gr.Audio(type="filepath", label="Reference audio")
                lyrics = gr.Textbox(
                    lines=12,
                    label="Lyrics",
                    placeholder="[Verse]\nline one\nline two\n\n[Chorus]\n...",
                )
                tags = gr.Textbox(
                    label="Style tags",
                    placeholder=(
                        "topic:[Longing]; genre:[country]; "
                        "instrument:[Strings,acoustic guitar]; mood:[hopeful]"
                    ),
                )
                bpm = gr.Number(label="BPM override (optional)", value=None)
                with gr.Accordion("Advanced", open=False):
                    cfg_scale = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="cfg_scale")
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="temperature")
                    topk = gr.Slider(1, 1000, value=250, step=1, label="topk")
                    max_len = gr.Slider(
                        1000, 300_000, value=30_000, step=1000,
                        label="max_audio_length_ms (start small on 8 GB GPUs)",
                    )
                    seed = gr.Number(label="seed (optional)", value=None)
                    low_vram = gr.Checkbox(
                        value=True,
                        label="Low VRAM mode (codec/Qwen/transcriptor on CPU) — recommended for 8 GB GPUs",
                    )
                    save_symbolic = gr.Checkbox(
                        value=True, label="Save transcribed MIDI (melody/chord/drums)"
                    )
                run_btn = gr.Button("Generate cover", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                audio_out = gr.Audio(label="Generated cover", type="filepath")
                midi_out = gr.File(label="Transcribed MIDI files", file_count="multiple")

        run_btn.click(
            generate,
            inputs=[
                model_path, ref_audio, lyrics, tags, bpm, cfg_scale,
                temperature, topk, max_len, seed, low_vram, save_symbolic,
            ],
            outputs=[audio_out, midi_out, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the MuLaCover GUI")
    parser.add_argument("--model_path", default="./ckpt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.model_path)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

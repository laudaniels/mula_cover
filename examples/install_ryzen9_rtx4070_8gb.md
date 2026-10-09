# MuLaCover install — Ryzen 9 / RTX 4070 (8 GB VRAM)

Tested combination for a desktop with an AMD Ryzen 9 CPU and an NVIDIA RTX
4070 with 8 GB VRAM, running Ubuntu (native or WSL2), using **reference-audio**
conditioning (not MIDI input). Read the "8 GB VRAM notes" section before your
first run — the MuLaCover backbone checkpoint is close to this card's VRAM
budget by itself, and reference-audio mode adds a transcription step
(YourMT3 + ChordNet) on top.

## 0. Check the GPU and driver

```bash
nvidia-smi
```

Note the "CUDA Version" shown top-right — that's the maximum CUDA toolkit
your driver supports, not the installed toolkit. Use it to pick the PyTorch
wheel in step 2.

## 1. OS packages

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv git curl ffmpeg libsndfile1
```

## 2. Clone and set up the environment

```bash
git clone https://github.com/laudaniels/mula_cover.git
cd mula_cover
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pick ONE wheel, matching the CUDA version nvidia-smi reported in step 0:
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130      # driver reports CUDA 13.x
# python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
#   --index-url https://download.pytorch.org/whl/cu128    # driver reports CUDA 12.x

python -m pip install -e '.[audio]'
python -c "import torch; from mulacover import MuLaCoverGenPipeline; assert torch.cuda.is_available(); print('Ready')"
```

## 3. Download checkpoints

```bash
hf auth login
hf download HeartMuLa/MuLaCover --local-dir ckpt/MuLaCover
(cd ckpt/MuLaCover && sha256sum -c SHA256SUMS)
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ckpt/Qwen3-Embedding-0.6B
hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ckpt/HeartCodec-oss
```

Reference-audio conditioning (used here instead of MIDI input) additionally
needs the YourMT3 and ChordNet checkpoints:

```bash
mkdir -p ckpt/SymbolicTranscriptor/yourmt3 ckpt/SymbolicTranscriptor/chord
curl -fL \
  'https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt' \
  -o ckpt/SymbolicTranscriptor/yourmt3/last.ckpt
for fold in 0 1 2 3 4; do
  chord_file="joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s${fold}.best.sdict"
  curl -fL \
    "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/$chord_file" \
    -o "ckpt/SymbolicTranscriptor/chord/$chord_file"
done
```

## 4. 8 GB VRAM notes — read before your first run

The MuLaCover backbone checkpoint is roughly **8.2 GB on disk in bfloat16** —
at or above the 4070's 8 GB VRAM budget once you add activations and the
autoregressive KV cache. Expect `CUDA out of memory` to be a real
possibility, especially for longer generations. `--lazy_load` is on by
default and keeps the codec/Qwen/transcriptor off the GPU except when each is
actually used, but it doesn't shrink the backbone itself.

If you hit an OOM, try these in order:

1. Free VRAM first: close browsers/other GPU apps, check what's already
   used with `nvidia-smi` right before running.
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running,
   to reduce allocator fragmentation.
3. Test with a short clip first to validate the pipeline before a full-length
   run: add `--max_audio_length_ms 30000` (30 s) to the CLI command below.
4. Push the lighter components to CPU via the Python API, leaving the GPU
   free for just the backbone:

   ```python
   import torch
   from mulacover import MuLaCoverGenPipeline

   pipe = MuLaCoverGenPipeline.from_pretrained(
       "./ckpt",
       device={"mulacover": "cuda:0", "codec": "cpu", "qwen": "cpu", "transcriptor": "cpu"},
       dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
              "qwen": torch.float32, "transcriptor": torch.float32},
       lazy_load=True,
   )
   ```

   This makes the codec/embedding/transcription steps run on CPU (slower)
   but leaves the full 8 GB free for the backbone.
5. Last resort: `--device cpu` for a correctness check — much slower, but
   has no VRAM ceiling.

## 5. Run a test generation

```bash
source .venv/bin/activate
env -u LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  mulacover \
  --model_path ./ckpt --ref_audio /path/to/reference.mp3 \
  --lyrics ./assets/lyrics.txt --tags ./assets/tags.txt \
  --symbolic_save_dir ./transcribed --save_path ./assets/cover.wav \
  --device cuda:0 --seed 42 \
  --max_audio_length_ms 30000
```

`--symbolic_save_dir` saves the transcribed melody/chord/drum MIDI so you can
reuse them (via `--melody_midi`/`--chord_midi`) without re-running
transcription on the same reference track. Drop `--max_audio_length_ms 30000`
once this short run succeeds, to generate full-length audio.

## 6. Optional: simple local GUI

Instead of typing a CLI command each time, `examples/gradio_app.py` gives you
a local web page to upload reference audio, fill in lyrics/tags, and play
back the result. It wraps the same pipeline and respects the same VRAM
trade-offs — leave "Low VRAM mode" enabled on the 4070.

```bash
python -m pip install -e '.[audio,gui]'
python examples/gradio_app.py --model_path ./ckpt
```

Then open http://127.0.0.1:7860 in a browser.

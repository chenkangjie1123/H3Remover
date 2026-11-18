<div align="center">

<h2>🎬 H3Remover: Reinforcing Video Object Removal without Paired Data</h2>

<p><strong>✨ Remove Anything with H3Remover.</strong></p>

<p>
  <a href="https://chenkangjie1123.github.io/H3Remover/"><img src="docs/assets/project-badge.svg" alt="Project Page" height="36"></a>
  &nbsp;
  <a href="https://github.com/chenkangjie1123/H3Remover/blob/main/docs/H3Remover.pdf"><img src="docs/assets/paper-badge.svg" alt="Paper" height="36"></a>
  &nbsp;
  <a href="https://huggingface.co/chenkangjie1123/H3Remover"><img src="docs/assets/weights-badge.svg" alt="Model Weights" height="36"></a>
</p>
<p>
  <a href="#overview">🔎 Overview</a> &nbsp;·&nbsp;
  <a href="#quickstart">🚀 Quickstart</a> &nbsp;·&nbsp;
  <a href="examples/README.md">Examples</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/chenkangjie1123/H3Remover">Weights</a> &nbsp;·&nbsp;
  <a href="#license">License</a>
</p>
<p>
  <a href="https://chenkangjie1123.github.io/H3Remover/#demo"><img src="assets/showcase-100.gif" width="100%" alt="100 public-dataset examples: masked inputs transition to object-removal results"></a>
</p>
<p><sub>🌍 100 examples. Many scenes. Masked inputs → removal results. <a href="https://chenkangjie1123.github.io/H3Remover/#demo">▶ Watch the full video ↗</a></sub></p>

</div>

## Overview

**Remove objects. Preserve the scene.** H3Remover uses reinforcement learning to remove mask-selected objects and their associated visual effects while preserving surrounding scene details, without paired training data.

- 🎯 **Mask-guided editing.** Select one or multiple objects with a video mask.
- 🪶 **Lightweight release.** A LoRA adapter and a minimal video inference pipeline.

## Quickstart

### 1. 🛠️ Install

Use Linux, Python 3.13, FFmpeg, and a CUDA-capable PyTorch installation. The reference environment uses PyTorch 2.10.0 with CUDA 13. Install PyTorch for your system, then:

```bash
git clone https://github.com/chenkangjie1123/H3Remover.git
cd H3Remover
pip install -r requirements.txt
```

### 2. 📦 Download weights

```bash
hf download chenkangjie1123/H3Remover --include '*.safetensors' '*.json' --local-dir weights/h3remover
```

Download the full-precision **Ref2VA** base and shared components from [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3):

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --include 'modular_model_index.json' 'transformer_ref/*' 'text_encoder/*' \
  'tokenizer/*' 'processor/*' 'vae/*' 'audio_vae/*' 'scheduler/*' 'audio_scheduler/*' \
  --local-dir weights/MiniMax-H3
```

### 3. ✨ Remove objects

```bash
python infer.py \
  --base weights/MiniMax-H3 \
  --adapter weights/h3remover \
  --video examples/street/source.mp4 \
  --mask examples/street/mask.mp4 \
  --output outputs/result.mp4
```

Use a grayscale mask: **white marks target objects; black marks non-target regions**. Source and mask must share the same timing, frame count, and dimensions. The script creates the red reference overlay automatically from the binary mask.

The pipeline processes the first 5.17 seconds as 124 frames at 24 FPS, with aspect-preserving letterboxing to 960 × 544. Provide at least 5.17 seconds of synchronized video and mask. Output is a silent MP4. Use `--seed` to set the random seed and `--steps` to set the inference steps (default: 50).

Component CPU offloading and tiled VAE decoding are enabled to reduce GPU memory usage.

## License

The model is distributed under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Download the base model separately from its official repository.

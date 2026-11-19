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
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="#gradio-demo">Gradio Demo</a> &nbsp;·&nbsp;
  <a href="docs/inference.md">Inference Guide</a> &nbsp;·&nbsp;
  <a href="examples/README.md">Examples</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/chenkangjie1123/H3Remover">Weights</a> &nbsp;·&nbsp;
  <a href="#license">License</a>
</p>
<p>
  <a href="https://chenkangjie1123.github.io/H3Remover/#demo"><img src="assets/showcase-100.gif" width="100%" alt="100 public-dataset examples: masked inputs transition to object-removal results"></a>
</p>
<p><sub>100 examples across diverse scenes. Masked inputs → removal results. <a href="https://chenkangjie1123.github.io/H3Remover/promo.html">Watch the full video ↗</a></sub></p>

</div>

## Overview

**Remove objects. Preserve the scene.** H3Remover (H3 Remover) uses reinforcement learning for video object removal: removing mask-selected objects and their associated visual effects while preserving surrounding scene details, without paired training data.

- **Mask-guided editing.** Select one or multiple objects with a video mask.
- **Simple inference.** Download the adapter and base weights with one command, then run the included example.

## Quickstart

### 1. Install

Use **Linux**, an **NVIDIA GPU**, and **FFmpeg**. The reference environment uses Python 3.13 and PyTorch 2.10.0 with CUDA 13:

```bash
git clone https://github.com/chenkangjie1123/H3Remover.git
cd H3Remover
conda create -n h3remover python=3.13 -y
conda activate h3remover
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Install FFmpeg with your system package manager. See the [inference guide](docs/inference.md) for environment checks and input details.

### 2. Download weights

```bash
python download_weights.py
```

This downloads the pinned adapter and full Ref2VA base to `weights/`, verifies the adapter checksum, and reuses completed downloads. If prompted for access, accept the license on the [adapter](https://huggingface.co/chenkangjie1123/H3Remover) and [base](https://huggingface.co/MiniMaxAI/MiniMax-H3) pages, then run `hf auth login` and retry.

### 3. Remove objects

```bash
python infer.py \
  --video examples/street/source.mp4 \
  --mask examples/street/mask.mp4 \
  --output outputs/result.mp4
```

Replace the example paths with your own video and mask. **White selects objects to remove; black keeps the rest.** Source and mask must have matching dimensions, frame count, and FPS. The red reference overlay is created automatically.

The output is a silent **124-frame, 24 FPS, 960 × 544** MP4. Provide at least **5.17 seconds**; longer inputs use the first clip. CPU offloading and tiled decoding are enabled.

<a id="gradio-demo"></a>

## 🏃‍♂️ Gradio Demo

After the [quickstart installation](#quickstart), launch the local editor:

```bash
python -m pip install -r requirements-demo.txt
python demo.py
```

Open **http://127.0.0.1:7860**, load an example or upload a video and binary mask, then **Preview targets → Remove selected objects**. Previewing an uploaded mask does not need model weights. Live editing uses your local CUDA GPU and the weights from the quickstart. The **Watch a finished example** tab plays a recorded comparison without GPU inference.

For automatic street masks, install the [optional models](docs/automatic-removal.md) and set `PI4_PYTHON` and `SAM31_PYTHON` to their Python executables before launching. The interface binds to localhost and does not create a public share link.

## Select people, vehicles and other moving objects in street videos

Have a driving clip but no mask? [`auto_remove.py`](auto_remove.py) builds one mask from two sources: **SAM 3.1 selects people and vehicles whether moving or parked**, and **Pi4 proposes other moving regions for SAM to segment** without a category prompt. H3Remover then edits the combined mask. After [installing Pi4 and SAM in their own environments](docs/automatic-removal.md#1-install-optional-models-separately), run:

```bash
python auto_remove.py --video driving.mp4 --output-dir outputs/driving \
  --prompts vehicle person bicycle motorcycle \
  --pi4-python "$(conda run -n pi4 python -c 'import sys; print(sys.executable)')" \
  --sam-python "$(conda run -n sam31 python -c 'import sys; print(sys.executable)')"
```

Open `outputs/driving/edited.mp4` for the edit. To inspect the targets first, add **`--prepare-only`**: this writes a binary mask and a video overlay without loading H3. Check small or occluded objects before editing. To keep the previous rule (only motion-confirmed instances of the prompted classes), pass `--policy motion-only`. [Edit using a reviewed mask →](docs/automatic-removal.md#3-inspect-the-result)

## Go Further

- [Inference guide](docs/inference.md) — preview masks, use local weights, change seeds, and troubleshoot.
- [Example videos](examples/README.md) — ready-to-run public video and mask pairs.
- [No mask yet? Generate one with Pi4 and SAM 3.1](docs/automatic-removal.md#try-the-street-example) — try the street example, inspect the target overlay, then pass the binary mask to H3Remover.

## License

The model is distributed under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Download the base model separately from its official repository.

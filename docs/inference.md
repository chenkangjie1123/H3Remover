# Inference Guide

[← Overview](../README.md)

Run commands from the H3Remover repository root, with the `h3remover` environment activated.

## Check the environment

```bash
ffmpeg -version
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.version.cuda)"
```

The reference setup is Linux, Python 3.13, PyTorch 2.10.0, and CUDA 13.0. It requires a compatible NVIDIA driver. See [official PyTorch builds](https://pytorch.org/get-started/previous-versions/) for other CUDA versions. The models run on one GPU with component CPU offloading and tiled VAE decoding; sufficient system RAM is also needed to load the full base. This release does not claim a measured minimum GPU memory requirement.

## Prepare inputs

Provide a source MP4 and a grayscale mask MP4 with identical dimensions, FPS, and frame count. White pixels select targets; black pixels select background. Colored overlays are previews, not binary masks.

The first 124 output frames are sampled at 24 FPS with nearest-frame selection. Both streams use the same indices and aspect-preserving letterboxing to 960 × 544. Masks use nearest-neighbor spatial resizing. Inputs must cover at least 5.17 seconds. This entry point does not stitch long clips or retain audio.

Preview the exact reference overlay without downloading weights or loading a GPU model:

```bash
python infer.py --preview-only \
  --video examples/street/source.mp4 --mask examples/street/mask.mp4 \
  --output outputs/street-mask-preview.mp4
```

Then run removal:

```bash
python infer.py \
  --video examples/street/source.mp4 --mask examples/street/mask.mp4 \
  --output outputs/street.mp4
```

Output directories are created automatically. Existing output files are not overwritten; choose a new filename for another run.

## Weights and settings

`python download_weights.py` downloads the adapter and required full-precision Ref2VA components. It does not download other H3 transformers. An interrupted download can be rerun; Hugging Face reuses its download cache.

Already have the base? Download only the adapter and point inference to your local model:

```bash
python download_weights.py --adapter-only
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --base /path/to/MiniMax-H3 --adapter weights/h3remover \
  --video examples/street/source.mp4 --mask examples/street/mask.mp4 \
  --output outputs/street-seed42.mp4 --seed 42 --steps 50
```

Use the full Ref2VA base, not a text-to-video or quantized checkpoint. The default seed is `20261104`, with 50 sampling points. The adapter rank and alpha are both 16. A fixed editing instruction tells the model to remove the highlighted objects and their associated shadows, reflections, and emitted light while preserving the remaining scene.

## Troubleshooting

| Message | Action |
| --- | --- |
| Missing weights | Run `python download_weights.py`, or supply `--base` and `--adapter`. |
| Gated repository / access denied | Accept the model's license in your Hugging Face account and run `hf auth login`. |
| FFmpeg not found | Install FFmpeg and make sure it is available on your PATH. |
| Source and mask mismatch | Export both from the same frame sequence with identical timing and dimensions. |
| Input is too short | Use a longer synchronized clip; the script does not silently repeat frames. |
| CUDA out of memory | Free GPU memory or use a larger GPU. Offloading is already enabled. |

## Automatic masks

For moving objects, see the optional [Pi4 + SAM 3.1 workflow](automatic-removal.md). You can inspect its binary mask and target preview before invoking removal.

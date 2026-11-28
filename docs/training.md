# Train H3Remover

Start from the original Ref2VA base with a fresh LoRA. Each training example is a source video plus a mask for each selected object; no clean target video is needed. The paper adapter is the 100-update checkpoint. The commands below use your own data, not a distributed copy of the paper's training split.

### 1. Install the models

Follow the [quickstart](../README.md#quickstart) for the H3 environment and base weights. The downloaded H3Remover adapter is not used to initialize training. Add the pinned [DiffusionART](https://github.com/vanzll/DiffusionART) source to the H3 environment:

```bash
git clone https://github.com/vanzll/DiffusionART.git third_party/DiffusionART
git -C third_party/DiffusionART checkout 58e2052278f935a90fffbef5624d93e458085c07
export PYTHONPATH="$PWD/third_party/DiffusionART/src${PYTHONPATH:+:$PYTHONPATH}"
```

Install the three frozen reward models from their upstream repositories, in separate environments if necessary. Run all commands from the H3Remover repository root. Make the depth and HPS packages importable in their respective environments. Obtain third-party code and weights under their upstream terms.

| Teacher | Setup |
| --- | --- |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Use the **relative-depth Small (ViT-S)** weights, not the metric variant. The adapter uses FP32, input size 518, at 24 FPS. |
| [OSOR](https://github.com/Zhouqm-Git/osor/tree/main/osor-sdxlinpainting) | Use the SDXL inpainting base and the **Phase II** checkpoint. In the upstream SDXL directory generate the fixed embedding with `python scripts/cache_prompts.py --config configs/phase2.yaml`. Only the predicted alpha is used. |
| [HPSv3](https://github.com/MizzenAI/HPSv3) | Use the official 7B reward checkpoint and Qwen2-VL-7B-Instruct. Start with the upstream `hpsv3/config/HPSv3_7B.yaml`; set `model_name_or_path` to the local Qwen weights, `min_pixels` and `max_pixels` to 200704 and `disable_flash_attn2` to true. For relative model/config paths in the upstream YAML, use absolute paths. |

For the paper configuration, use OSOR code revision `d5b68e822996d61aae6ba612e5e9b5a14f576879` with [Phase II weights](https://huggingface.co/QinmingZhou/OSOR/tree/9288156766b1df44e5f50fb697876a6f2fd8c062); use HPSv3 code revision `bd0c5fcb5f587617b0169c07222ab78d01e2f3c2` with the [7B reward weights](https://huggingface.co/MizzenAI/HPSv3/tree/4f81e3e09edd82fe3c5f636444c721b592a735ca).

Create `teachers.json` with local paths (replace the examples):

```json
{
  "depth": {"python": "/envs/depth/bin/python", "repo": "/repos/Video-Depth-Anything", "checkpoint": "/models/video_depth_anything_vits.pth"},
  "osor": {"python": "/envs/osor/bin/python", "scripts": "/repos/osor/osor-sdxlinpainting/scripts", "base": "/models/sdxl-inpainting", "checkpoint": "/models/sdxlinpainting_phase2.pth", "embeddings": "/models/fixed_prompt_embeds.pt"},
  "hps": {"python": "/envs/hps/bin/python", "config": "/models/hps-inference.yaml", "checkpoint": "/models/HPSv3.safetensors"}
}
```

### 2. Prepare the source videos

Use JSON Lines: one object per line, paths relative to the manifest. For example, a manifest placed in the repository root can reference the bundled public inputs:

```json
{"video":"examples/bicycle/source.mp4","masks":["examples/bicycle/mask.mp4"]}
```

Use **one grayscale binary video per target instance** for multi-object clips. Do not merge several instances if you want an instance-balanced removal reward. All mask videos must match the source dimensions, frame count and FPS. The adapter uses the same 124-frame, 24 FPS, 960 × 544 canvas as inference. Review masks before preparation. Optional fields on the **aligned output grid**:

- `eligible`: 124 booleans; set unreliable frames to false. Default: all frames reviewed. Empty-target frames and frames with a visible instance smaller than 32 pixels are excluded automatically.
- `roi`: `[x, y, width, height]` of valid image content. Defaults to the fitted source rectangle. Specify it for footage with pre-existing black bars.
- `void`: path to a boolean `.npy` array of shape `[124,544,960]`; true pixels are excluded from background supervision and depth calibration.

```bash
python -m pipeline.training.prepare --manifest train.jsonl --teachers teachers.json --output outputs/train-cache --gpu 0
```

This freezes the target support, source inverse depth, source HPS scores, and the OSOR protection region **before** generating any candidates. Use a new output folder, keep the teacher weights/configuration fixed, and allow disk space for uncompressed per-frame arrays. Prepare a separate, scene-disjoint evaluation manifest the same way if desired.

### 3. Train and use the adapter

```bash
torchrun --standalone --nproc-per-node=8 -m pipeline.training \
  --base weights/MiniMax-H3 --data outputs/train-cache \
  --teachers teachers.json --output outputs/full --updates 100
```

Prepare at least eight distinct training clips. Each rank scores its candidates on its own GPU, running the depth and preference teachers in separate, short-lived processes. An eight-GPU setup must have sufficient memory for H3 and the active teacher on each device. This portable runner favors dependency isolation over throughput.

Pass `--eval-data outputs/eval-cache` to generate fixed-seed, 50-point evaluation videos and reward breakdowns every 20 optimizer updates. Every 10 updates and at the requested final step, training exports inference-ready adapters (not optimizer-resume checkpoints):

```bash
python infer.py --adapter outputs/full/step-0100 --video examples/street/source.mp4 \
  --mask examples/street/mask.mp4 --output outputs/trained-result.mp4
```

### Full recipe

The reward is **T − E − 5C + 0.05ΔHPS**. T measures the fraction of selected pixels whose inverse depth decreases beyond calibration noise, averaged across visible instances. E counts background near/tied/far ordering disagreements. C compares original and edited RGB only on valid background where source-side OSOR alpha is **below 0.1**. ΔHPS compares the same source/edited frames, not the red overlay. The first three terms use a fixed denominator of 124; HPS averages up to eight predetermined eligible frames. A failed gauge or insufficient support gives an unknown reward, disabling the entire four-candidate group.

Each round uses eight sources × four candidates, ten sampling points (nine native forwards), four replay times, and two disjoint 16-trajectory gradient updates. Only rank-16 attention LoRA parameters are trained. AdamW uses a peak LR of 0.0005, 20 warmup updates, gradient clipping at 0.3, and a 300-update cosine schedule even when exporting at step 100.

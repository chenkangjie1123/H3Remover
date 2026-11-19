# Automatic Moving-Object Removal

[← Overview](../README.md)

**Mask generation was tested on one public MOSE street clip.** Pi4 and SAM 3.1 ran sequentially on one GPU for all 124 frames. The test stopped at the binary mask; it did not run H3. Use the [video + mask quickstart](../README.md#quickstart) to edit after checking the preview.

The optional workflow is:

**Video → Pi4 motion probabilities → SAM 3.1 object masks → H3Remover edited video.**

Pi4 supplies learned motion evidence, including scenes with a moving camera. SAM supplies complete instance contours. An instance is selected when at least half its pixels have Pi4 motion probability ≥ 0.5 in at least two frames (minimum area: 32 pixels). Once selected, all masks returned for that instance are included, including frames in which it pauses. These are configurable selection heuristics, not guarantees that every moving object is detected.

## 1. Install optional models separately

Keep the H3Remover environment from the quickstart. Pi4 and SAM use separate environments because their official dependency versions differ. They run sequentially and release GPU memory between stages.

### Pi4

```bash
git clone https://github.com/yyfz/Pi4.git third_party/Pi4
git -C third_party/Pi4 checkout 32dfbea9ce6268ea17446208a804710df9475c32
conda create -n pi4 python=3.11 -y
conda activate pi4
cd third_party/Pi4
python -m pip install -r requirements-cu126.txt
cd ../..
python -c "import sys; print(sys.executable)"
```

Save the printed Python path for `--pi4-python`. Follow [Pi4's installation guide](https://github.com/yyfz/Pi4/blob/main/docs/inference.md) for driver and compiler requirements. Its official weights are downloaded automatically at a pinned revision.

### SAM 3.1

```bash
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 31cb7e624e4fe95301b1be1d4e809111593d2e45
conda create -n sam31 python=3.12 -y
conda activate sam31
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e 'third_party/sam3[notebooks]'
python -c "import sys; print(sys.executable)"
```

Save this path for `--sam-python`. Obtain access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) and log in with `hf auth login`. See the [official SAM 3.1 guide](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) for its checkpoint and dependency requirements. A local checkpoint can be passed with `--sam-checkpoint`.

## Try the street example

After installing the optional environments above, generate **only the vehicle mask**. This command does not require the H3 weights:

```bash
conda activate h3remover
python auto_remove.py --prepare-only --prompts vehicle \
  --video examples/street/source.mp4 --output-dir outputs/street-mask \
  --pi4-python "$(conda run -n pi4 python -c 'import sys; print(sys.executable)')" \
  --sam-python "$(conda run -n sam31 python -c 'import sys; print(sys.executable)')"
```

Open `outputs/street-mask/mask_preview.mp4` to see which vehicles were selected. Use `mask.mp4`, **not the overlay**, with `source.mp4` for removal.

On MOSE `b0240233`, 16 vehicle track IDs passed the motion filter. Cars crossing the road and the bus entering late were selected; a partially visible car at the left border was missed at the first frame. All 16 candidate tracks passed, so this clip does not test rejection of a parked car. The GPU test used local SAM 3.1 weights (FP16 repack), RGB frame input and detector batch size 4. The standard upstream checkpoint installation above is not a bit-for-bit reproduction of that runtime.

## Run removal directly

Reactivate `h3remover`, then use the actual Python paths printed above:

```bash
conda activate h3remover
python auto_remove.py \
  --video examples/street/source.mp4 --output-dir outputs/automatic-street \
  --pi4-python /path/to/pi4/bin/python \
  --sam-python /path/to/sam31/bin/python
```

The default categories are person, vehicle, bicycle, motorcycle, and animal. To limit the candidate categories, add e.g. `--prompts vehicle person`. Category names generate candidates; Pi4 motion evidence decides which instances to retain. Categories outside the prompt list are not exhaustively covered.

Use `--gpu 1` to select another physical GPU. Add `--prepare-only` to stop after the mask preview, without loading H3. Use a new output directory for every preparation run; existing artifacts are never overwritten.

## 3. Inspect the result

| File | Contents |
| --- | --- |
| `source.mp4` | Shared 124-frame, 24 FPS, 960 × 544 input for all stages. |
| `mask.mp4` | Binary union of selected moving-instance masks. |
| `mask_preview.mp4` | Red target overlay with white contours. |
| `edited.mp4` | H3Remover result; omitted with `--prepare-only`. |
| `selection.json` | Instance selection evidence and per-frame mask areas. |
| `motion.npy` / `motion.json` | Pi4 probabilities and source identity. |
| `run.json` / `*.log` | Stage durations, status, and diagnostics. |

All stages share the same sampled frames and canvas. Missing SAM frames, invalid probabilities, or reaching the instance capacity stop the run rather than silently passing incomplete masks to H3. An empty selection stops before H3 loads. Inspect the preview for missed objects or incorrect identities.

After checking a prepare-only run, or if only the H3 stage failed, reuse the saved masks without rerunning Pi4 or SAM:

```bash
python infer.py \
  --video outputs/automatic-street/source.mp4 \
  --mask outputs/automatic-street/mask.mp4 \
  --output outputs/automatic-street/edited.mp4
```

## Licenses

Pi4 code and weights use [CC BY-NC 4.0](https://github.com/yyfz/Pi4/blob/main/LICENSE). SAM and H3 retain their upstream license terms. Optional model code and weights are downloaded from their official repositories, not redistributed here.

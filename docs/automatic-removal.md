# Automatic Moving-Object Removal

[← Overview](../README.md)

Use `--prepare-only` to generate the targets without loading H3. Then follow the [video + mask quickstart](../README.md#quickstart) if the overlay matches your edit.

The optional workflow is:

**Video → Pi4 motion probabilities → SAM 3.1 object masks → H3Remover edited video.**

The default policy, `street-union`, merges two masks:

1. **People and vehicles:** all visible masks returned by SAM for the prompted categories are retained, without testing motion. Parked vehicles and standing pedestrians are intentional targets.
2. **Other motion:** threshold Pi4 at 0.5. For sufficiently uncovered components (at least 32 pixels and 25% unexplained), send boxes around the remaining connected regions to SAM's image detector. A candidate must overlap its motion seed over ≥50% of its own area and ≥15% of the seed area; at least half of the candidate must be outside the direct mask. Add only newly selected pixels. This path uses no class names. It is per-frame and can miss small objects or flicker.

`--policy motion-only` retains the earlier behavior: select a prompted track when enough of its mask overlaps Pi4 motion in at least two frames; keep that track wherever SAM finds it. It does not run the category-free branch.

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

After installing the optional environments above, generate **only the combined mask**, without the H3 weights:

```bash
conda activate h3remover
python auto_remove.py --prepare-only \
  --video examples/street/source.mp4 --output-dir outputs/street-mask \
  --pi4-python "$(conda run -n pi4 python -c 'import sys; print(sys.executable)')" \
  --sam-python "$(conda run -n sam31 python -c 'import sys; print(sys.executable)')"
```

Open `outputs/street-mask/mask_preview.mp4` to see the selection. Use `mask.mp4`, **not the overlay**, with `source.mp4` for removal.

On one Waymo driving clip, the combined mask included the van ahead, a parked car and sidewalk pedestrians. Pi4 + SAM added pixels in 41 of 124 output frames, including a false positive on a roadside sign. These counts measure coverage, not detection accuracy. Preview the overlay before removal.

With the earlier `--policy motion-only --prompts vehicle` setting, 16 MOSE `b0240233` vehicle track IDs passed the motion filter; one partial vehicle at the first frame was missed. The GPU checks used a local SAM 3.1 FP16 repack, RGB frame input and detector batch size 4.

## Run removal directly

Reactivate `h3remover`, then use the actual Python paths printed above:

```bash
conda activate h3remover
python auto_remove.py \
  --video examples/street/source.mp4 --output-dir outputs/automatic-street \
  --pi4-python /path/to/pi4/bin/python \
  --sam-python /path/to/sam31/bin/python
```

The default direct categories are person, vehicle, bicycle, and motorcycle. Override with `--prompts` to change which categories bypass motion filtering. This does not restrict the extra motion branch.

Use `--gpu 1` to select another physical GPU. Add `--prepare-only` to stop after the mask preview, without loading H3. Use a new output directory for every preparation run; existing artifacts are never overwritten.

## 3. Inspect the result

| File | Contents |
| --- | --- |
| `source.mp4` | Shared 124-frame, 24 FPS, 960 × 544 input for all stages. |
| `mask.mp4` | Final binary mask. |
| `branches.npz` | Street-union only: horizontally bit-packed `direct` and `extra` masks and image `width`. Unpack with `np.unpackbits(arr, axis=-1, count=width)`. |
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

"""Frozen reward-model adapters. Each command runs in the teacher's own environment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .rewards import hps_indices

QUALITY = (
    "A photorealistic scene with natural lighting, clear photographic detail, "
    "coherent geometry and intact background textures, free of visual artifacts."
)


def call(kind, config, sample, rgb, output, gpu):
    settings = json.loads(config.read_text())[kind]
    env = os.environ.copy()
    # A reward model is a single-device inference process, not a training rank.
    for key in list(env):
        if key.startswith(("TORCHELASTIC_", "PET_")) or key in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(
        [
            settings["python"],
            "-m",
            "pipeline.training.teachers",
            kind,
            str(config),
            str(sample),
            str(rgb),
            str(output),
        ],
        env=env,
        check=True,
    )


def predict(kind, cfg, sample, rgb):
    import torch

    if "repo" in cfg:
        sys.path.insert(0, str(Path(cfg["repo"]).resolve()))

    if kind == "depth":
        from video_depth_anything.video_depth import VideoDepthAnything

        model = VideoDepthAnything(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=False
        )
        model.load_state_dict(
            torch.load(cfg["checkpoint"], weights_only=True, map_location="cpu"),
            strict=True,
        )
        model.eval().requires_grad_(False).cuda()
        values, fps = model.infer_video_depth(
            rgb, 24, input_size=518, device="cuda", fp32=True
        )
        if fps != 24 or values.shape != rgb.shape[:-1] or not np.isfinite(values).all():
            raise ValueError("Invalid inverse-depth prediction")
        return np.asarray(values, np.float32)

    eligible = np.load(sample / "eligible.npy")
    if kind == "hps":
        from hpsv3 import HPSv3RewardInferencer
        from PIL import Image

        model = HPSv3RewardInferencer(
            config_path=cfg["config"], checkpoint_path=cfg["checkpoint"], device="cuda"
        )
        model.model.eval().requires_grad_(False)
        x, y, w, h = np.load(sample / "roi.npy")
        scores = []
        for i in hps_indices(eligible):
            image = Image.fromarray(rgb[i, y : y + h, x : x + w])
            pred = (
                model.reward(prompts=[QUALITY], image_paths=[image])
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            if pred.shape != (1, 2) or not np.isfinite(pred).all():
                raise ValueError("Invalid HPSv3 output")
            scores.append(float(pred[0, 0]))
        return np.array(scores)

    sys.path.insert(0, str(Path(cfg["scripts"]).resolve()))
    from inference_enhance import SDXLInferenceEnhance

    modules = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "proj_in",
        "proj_out",
        "ff.net.2",
        "ff.net.0.proj",
    ]
    model = SDXLInferenceEnhance(
        cfg["base"],
        cfg["checkpoint"],
        cfg["embeddings"],
        modules,
        256,
        400,
        device="cuda",
        weight_dtype=torch.bfloat16,
    )
    union = np.load(sample / "masks.npy", mmap_mode="r").any(axis=0)
    protected = np.zeros_like(union)
    captured = {}

    def capture(module, args, output):
        captured["alpha"] = output[1].detach().sigmoid().float()

    handle = model.G.register_forward_hook(capture)
    try:
        for i in np.flatnonzero(eligible):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            image = (
                torch.from_numpy(rgb[i].copy()).permute(2, 0, 1)[None].cuda().float()
                / 255
            )
            mask = torch.from_numpy(union[i])[None, None].cuda().float()
            model.infer(image, mask, return_type="pt")
            alpha = torch.nn.functional.interpolate(
                captured.pop("alpha"),
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if not torch.isfinite(alpha).all():
                raise ValueError("Invalid effect alpha")
            protected[i] = (alpha < 0.1).cpu().numpy()
    finally:
        handle.remove()
    return protected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["osor", "depth", "hps"])
    for name in ("config", "sample", "rgb", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    import torch

    with torch.inference_mode():
        result = predict(
            a.kind,
            json.loads(a.config.read_text())[a.kind],
            a.sample,
            np.load(a.rgb, mmap_mode="r"),
        )
    np.save(a.output, result)


if __name__ == "__main__":
    main()

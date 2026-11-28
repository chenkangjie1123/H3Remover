"""Train the Full H3Remover adapter with native Ref2VA rollouts and DiffusionART."""

import argparse
import hashlib
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .backend import Backend, canonical_velocity
from .prepare import condition
from .rewards import full_reward
from .teachers import call

SEED = 20261019
COV = dict(
    reduction="unbiased",
    direction_mode="historical",
    scale_mode="unit_normalized_covariance",
    guidance_gain=1,
    zero_signal_mode="exact",
)
LOSS = dict(
    loss_space="velocity_linear_sigma",
    direction_normalization="rms",
    mask_space="velocity",
    velocity_mask_metric="scheduler_reference_metric",
    corrective_mask_enabled=True,
    q2_reference_sde_noise_level=0.8,
    q2_reference_kl_threshold=1e-5,
)


def schedule(i):
    return (
        (i + 1) / 20
        if i < 20
        else 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (i - 20) / 280)))
    )


def sample_paths(folder, config):
    index = json.loads((folder / "index.json").read_text())
    if index["teachers"] != hashlib.sha256(config.read_bytes()).hexdigest():
        raise ValueError(
            "Use the same teacher configuration for preparation and training"
        )
    return [folder / f"{i:06d}" for i in range(index["count"])]


def evidence(folder):
    names = ("rgb", "masks", "valid", "eligible", "depth", "osor", "hps")
    return {name: np.load(folder / (name + ".npy"), mmap_mode="r") for name in names}


def decode(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, list):
        if isinstance(value[0], list):
            value = value[0]
        value = np.stack([np.asarray(frame) for frame in value])
    if value.ndim == 5:
        value = value[0]
    if value.shape == (3, 124, 544, 960):
        value = value.transpose(1, 2, 3, 0)
    if value.shape != (124, 544, 960, 3) or not np.isfinite(value).all():
        raise ValueError("Invalid decoded video")
    return (
        value
        if value.dtype == np.uint8
        else np.rint(255 * np.clip(value, 0, 1)).astype(np.uint8)
    )


def score(folder, source, edit, teachers, gpu, work):
    path = work / "candidate.npy"
    np.save(path, edit)
    for kind in ("depth", "hps"):
        call(kind, teachers, folder, path, work / (kind + ".npy"), gpu)
    return full_reward(
        source["rgb"],
        edit,
        source["masks"],
        source["valid"],
        source["eligible"],
        source["depth"],
        np.load(work / "depth.npy"),
        source["osor"],
        source["hps"],
        np.load(work / "hps.npy"),
    )


def export(backend, target):
    from safetensors.torch import save_file

    params = {
        k: p.detach().cpu().contiguous()
        for k, p in backend.model.named_parameters()
        if p.requires_grad
    }
    if len(params) != 400 or not all("lora_" in k for k in params):
        raise ValueError("Unexpected adapter layout")
    target.mkdir()
    save_file(params, str(target / "h3remover.safetensors"), metadata={"format": "pt"})
    (target / "h3remover_config.json").write_text(
        json.dumps(
            dict(
                format_version=1,
                rank=16,
                alpha=16,
                base_model="MiniMaxAI/MiniMax-H3",
                workflow="ref2va",
            )
        )
        + "\n"
    )


def replay_window(backend, rolls, direction, magnitude):
    from visual_rl.algorithms.art import art_loss

    backend.model.train()
    for roll in rolls:
        for c in roll["cache"]:
            n = c["condition_rows"]
            now = canonical_velocity(backend.replay(c)[0][:, n:])
            old = canonical_velocity(c["old_video"][:, n:].to(now.device))
            loss = art_loss(
                now,
                old,
                now.new_tensor([c["sigma"]]),
                direction,
                magnitude,
                ode_velocity_coefficient=now.new_tensor([c["dt"]]),
                **LOSS,
            )
            (
                loss["per_item_loss"].mean() / (len(rolls) * len(roll["cache"]))
            ).backward()


def sync_gradients(params, world):
    import torch
    import torch.distributed as dist

    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)
        p.grad.div_(world)
    return float(torch.nn.utils.clip_grad_norm_(params, 0.3, error_if_nonfinite=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "teachers", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--eval-data", type=Path)
    p.add_argument("--base", default="weights/MiniMax-H3")
    p.add_argument(
        "--updates",
        type=int,
        default=100,
        help="Stop at this update; LR horizon stays 300",
    )
    a = p.parse_args()
    if not 1 <= a.updates <= 300:
        p.error("--updates must be in [1, 300]")
    import torch
    import torch.distributed as dist
    from visual_rl.algorithms.covariance import art_scalar_covariance
    from visual_rl.sampling import select_timesteps

    from pipeline.video import save_video

    rank, local, world = (
        int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if world != 8:
        raise ValueError("The Full recipe uses eight ranks with four candidates each")
    torch.cuda.set_device(local)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = visible.split(",")[local] if visible else local
    data = sample_paths(a.data, a.teachers)
    if len(data) < world:
        raise ValueError("Prepare at least eight training clips")
    evaluation = sample_paths(a.eval_data, a.teachers) if a.eval_data else []
    random.Random(SEED).shuffle(data)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    try:
        if rank == 0:
            a.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        backend = Backend(a.base, f"cuda:{local}")
        torch.manual_seed(SEED)
        backend.add_lora()
        params = backend.parameters()
        for param in params:
            dist.broadcast(param.detach(), src=0)
        opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
        lr = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        pool = backend.replay_pool(10, 9)
        step = round_id = stalls = 0
        with TemporaryDirectory(dir=a.output) as temp:
            work = Path(temp)
            while step < a.updates:
                before = step
                folder = data[(8 * round_id + rank) % len(data)]
                source = evidence(folder)
                ref = condition(source["rgb"], source["masks"])
                slots = select_timesteps(
                    pool, 4, torch.Generator().manual_seed(SEED + 10000000 + round_id)
                )
                rolls, scores = [], []
                for j in range(4):
                    roll = backend.rollout(
                        source["rgb"],
                        ref,
                        SEED + round_id * 100000 + rank * 100 + j,
                        slots,
                    )
                    edit = decode(roll.pop("videos"))
                    scores.append(score(folder, source, edit, a.teachers, gpu, work))
                    rolls.append(roll)
                known = all(s is not None for s in scores)
                batches = [None] * world
                dist.all_gather_object(
                    batches,
                    [
                        (r["endpoint"], s["reward"] if known else 0.0)
                        for r, s in zip(rolls, scores)
                    ],
                )
                entries = [v for batch in batches for v in batch]
                cov = art_scalar_covariance(
                    torch.cat([x for x, _ in entries]),
                    torch.tensor([v for _, v in entries]),
                    torch.arange(world).repeat_interleave(4),
                    world,
                    **COV,
                )
                direction = cov["group_directions"][rank : rank + 1].to(backend.device)
                mag = cov["group_guidance_magnitudes"][rank : rank + 1].to(
                    backend.device
                )
                if cov["group_active"].any():
                    for start in (0, 2):
                        opt.zero_grad(set_to_none=True)
                        replay_window(backend, rolls[start : start + 2], direction, mag)
                        norm = sync_gradients(params, world)
                        if norm == 0:
                            continue
                        used_lr = opt.param_groups[0]["lr"]
                        opt.step()
                        lr.step()
                        step += 1
                        gathered_scores = [None] * world
                        dist.all_gather_object(
                            gathered_scores, scores[start : start + 2] if known else []
                        )
                        if rank == 0:
                            valid_scores = [
                                s for batch in gathered_scores for s in batch
                            ]
                            metrics = (
                                {
                                    key: float(np.mean([s[key] for s in valid_scores]))
                                    for key in valid_scores[0]
                                }
                                if valid_scores
                                else {}
                            )
                            line = json.dumps(
                                dict(
                                    update=step,
                                    lr=used_lr,
                                    grad_norm=norm,
                                    scored_candidates=len(valid_scores),
                                    active_groups=int(cov["group_active"].sum()),
                                    **metrics,
                                ),
                                allow_nan=False,
                            )
                            with (a.output / "metrics.jsonl").open("a") as stream:
                                print(line, file=stream)
                            print(line, flush=True)
                            if step % 10 == 0 or step == a.updates:
                                export(backend, a.output / f"step-{step:04d}")
                        if evaluation and step % 20 == 0:
                            with torch.random.fork_rng(devices=[local]):
                                for i in range(rank, len(evaluation), world):
                                    sample = evaluation[i]
                                    ev = evidence(sample)
                                    er = condition(ev["rgb"], ev["masks"])
                                    vr = backend.rollout(
                                        ev["rgb"], er, 20261104, steps=50
                                    )
                                    frames = decode(vr["videos"])
                                    result = score(
                                        sample, ev, frames, a.teachers, gpu, work
                                    )
                                    target = a.output / f"eval-{step:04d}" / f"{i:06d}"
                                    target.mkdir(parents=True)
                                    save_video(target / "result.mp4", frames)
                                    (target / "scores.json").write_text(
                                        json.dumps(result) + "\n"
                                    )
                            dist.barrier()
                        if step == a.updates:
                            break
                stalls = stalls + 1 if step == before else 0
                if stalls >= 20:
                    raise RuntimeError(
                        "No usable gradient for 20 consecutive rollout groups"
                    )
                round_id += 1
                del rolls, batches, entries, cov, roll
                backend.cache = []
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

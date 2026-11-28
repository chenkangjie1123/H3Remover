"""Prepare source-only evidence from videos and reviewed per-instance masks."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pipeline.video import load_aligned

from .teachers import call


def condition(rgb, masks):
    out = rgb.copy()
    kernel = np.ones((3, 3), np.uint8)
    for i in range(len(rgb)):
        union = masks[:, i].any(axis=0)
        out[i, union] = np.rint(
            0.65 * rgb[i, union] + 0.35 * np.array([255, 48, 48])
        ).astype(np.uint8)
        for mask in masks[:, i]:
            core = cv2.erode(
                mask.astype(np.uint8),
                kernel,
                iterations=2,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out[i, mask & ~core.astype(bool)] = 255
    return out


def prepare(row, base, out, config, gpu):
    instances = []
    if not row.get("masks"):
        raise ValueError("Supply one mask video per selected instance")
    for path in row["masks"]:
        rgb, mask, fit = load_aligned(base / row["video"], base / path)
        instances.append(mask)
    masks = np.stack(instances)
    x, y, w, h = row.get("roi", fit)
    if not (0 <= x < x + w <= 960 and 0 <= y < y + h <= 544):
        raise ValueError("ROI must lie on the training canvas")
    valid = np.zeros(rgb.shape[:-1], bool)
    valid[:, y : y + h, x : x + w] = True
    masks &= valid[None]
    if "void" in row:
        void = np.load(base / row["void"])
        if void.dtype != bool or void.shape != valid.shape:
            raise ValueError("Expected a boolean void map on the aligned video grid")
        valid &= ~void
    eligible = np.asarray(row.get("eligible", [True] * 124))
    if eligible.shape != (124,) or eligible.dtype != bool:
        raise ValueError("Expected 124 source eligibility flags")
    areas = masks.sum(axis=(-1, -2))
    eligible &= (areas > 0).any(axis=0) & ((areas == 0) | (areas >= 32)).all(axis=0)
    if not eligible.any():
        raise ValueError("No reliable target frames in this clip")
    for name, value in dict(
        rgb=rgb, masks=masks, valid=valid, eligible=eligible, roi=(x, y, w, h)
    ).items():
        np.save(out / name, value)
    for kind in ("osor", "depth", "hps"):
        call(kind, config, out, out / "rgb.npy", out / (kind + ".npy"), gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--teachers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="0", help="Physical CUDA device or UUID")
    a = p.parse_args()
    rows = [json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    if not rows:
        p.error("Empty manifest")
    a.output.mkdir(parents=True, exist_ok=False)
    for i, row in enumerate(rows):
        with tempfile.TemporaryDirectory(dir=a.output) as work:
            out = Path(work) / "sample"
            out.mkdir()
            prepare(row, a.manifest.parent, out, a.teachers, a.gpu)
            out.rename(a.output / f"{i:06d}")
        print(f"Prepared {i + 1}/{len(rows)}", flush=True)
    (a.output / "index.json").write_text(
        json.dumps(
            {
                "count": len(rows),
                "teachers": hashlib.sha256(a.teachers.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()

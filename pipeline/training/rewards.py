"""Source-fixed geometry, effect-aware appearance and preference rewards."""

import numpy as np

CONFIG = {
    "min_instance_pixels": 32,
    "anchor_stride": 8,
    "min_anchor_samples_per_fold": 1024,
    "scale_range": (0.25, 4.0),
    "holdout_median_max_SOURCE_span_fraction": 0.03,
    "holdout_p90_max_SOURCE_span_fraction": 0.08,
    "deletion_noise_floor_SOURCE_span_fraction": 0.01,
    "order_deadband_own_frame_span_fraction": 0.01,
    "min_source_pairs": 32,
    "pair_offsets_yx": [
        (0, n) if horizontal else (n, 0)
        for n in (1, 8, 32, 128)
        for horizontal in (True, False)
    ],
}


def outside_tight_boxes(targets, shape):
    outside = np.ones(shape, dtype=bool)
    for mask in targets:
        if mask is None:
            return None
        if mask.shape != shape or mask.dtype != bool:
            raise ValueError("SOURCE boolean instance mask/canvas mismatch")
        y, x = np.nonzero(mask)
        if len(y) < CONFIG["min_instance_pixels"]:
            return None
        outside[y.min() : y.max() + 1, x.min() : x.max() + 1] = False
    return outside if targets else None


def calibrate_video(source_depth, edited_depth, targets_by_frame):
    """Fit one clip-wide gauge, then check disjoint fixed held-out pixels.

    No target pixel is used in either fold. No frame-specific fitting, pixel
    warping, retiming, or candidate-dependent mask generation is performed.
    """
    if source_depth.shape != edited_depth.shape or source_depth.ndim != 3:
        raise ValueError("Native time and canvas must match")
    if len(targets_by_frame) != len(source_depth):
        raise ValueError("SOURCE masks required at every original timestamp")
    fit_s, fit_e, test_s, test_e = [], [], [], []
    total_anchors = 0
    stride = CONFIG["anchor_stride"]
    for frame, targets in enumerate(targets_by_frame):
        outside = outside_tight_boxes(targets, source_depth.shape[1:])
        if outside is None:
            continue
        y, x = np.nonzero(outside[::stride, ::stride])
        y, x = y * stride, x * stride
        s = np.asarray(source_depth[frame, y, x], np.float64)
        e = np.asarray(edited_depth[frame, y, x], np.float64)
        good = np.isfinite(s) & np.isfinite(e)
        total_anchors += len(s)
        fold = ((y // stride + x // stride + frame) % 2) == 0
        fit_s.append(s[good & fold])
        fit_e.append(e[good & fold])
        test_s.append(s[good & ~fold])
        test_e.append(e[good & ~fold])
    if not fit_s:
        return {"known": False, "reason": "NO_SOURCE_BACKGROUND_ANCHORS"}
    s, e, hs, he = [np.concatenate(parts) for parts in (fit_s, fit_e, test_s, test_e)]
    if min(len(s), len(hs)) < CONFIG["min_anchor_samples_per_fold"]:
        return {"known": False, "reason": "INSUFFICIENT_SOURCE_BACKGROUND_ANCHORS"}
    span = float(np.quantile(s, 0.95) - np.quantile(s, 0.05))
    if not np.isfinite(span) or span <= 1e-6:
        return {"known": False, "reason": "DEGENERATE_SOURCE_DEPTH_SCALE"}
    # Robust scalar regression; every retained coordinate remains unchanged.
    selected = np.ones(len(s), bool)
    for _ in range(5):
        ev, sv = e[selected], s[selected]
        variance = float(np.mean((ev - ev.mean()) ** 2))
        if variance <= (span * 1e-6) ** 2:
            return {"known": False, "reason": "DEGENERATE_EDITED_DEPTH_SCALE"}
        scale = float(np.mean((ev - ev.mean()) * (sv - sv.mean())) / variance)
        shift = float(np.median(s - scale * e))
        residual = np.abs(s - (scale * e + shift))
        selected = residual <= max(float(np.quantile(residual, 0.7)), span * 1e-6)
    residual = np.abs(hs - (scale * he + shift))
    median, p90, p95 = map(float, np.quantile(residual, [0.5, 0.9, 0.95]))
    known = (
        CONFIG["scale_range"][0] <= scale <= CONFIG["scale_range"][1]
        and median / span <= CONFIG["holdout_median_max_SOURCE_span_fraction"]
        and p90 / span <= CONFIG["holdout_p90_max_SOURCE_span_fraction"]
    )
    return {
        "known": bool(known),
        "reason": None if known else "CROSS_VIDEO_DEPTH_GAUGE_NOT_COMPARABLE",
        "scale": scale,
        "shift": shift,
        "source_span": span,
        "heldout_abs_residual_median": median,
        "heldout_abs_residual_p90": p90,
        "heldout_abs_residual_p95": p95,
        "deletion_threshold": max(
            CONFIG["deletion_noise_floor_SOURCE_span_fraction"] * span, p95
        ),
        "fit_samples": len(s),
        "heldout_samples": len(hs),
        "SOURCE_fixed_anchor_pool": total_anchors,
        "reward_regions_changed_by_anchor_selection": False,
        "one_clip_global_numeric_unit_conversion": True,
        "spatial_or_temporal_alignment": False,
        "semantic_deletion_success_certified": False,
    }


def _epsilon(depth, allowed):
    values = np.asarray(depth, np.float64)[allowed & np.isfinite(depth)]
    if not values.size:
        return None
    span = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    # Do not impose a metric floor: positive scale/shift must preserve orders.
    return span * CONFIG["order_deadband_own_frame_span_fraction"]


def background_evidence(source, edited, allowed):
    if (
        source.ndim != 2
        or source.shape != edited.shape
        or allowed.shape != source.shape
        or allowed.dtype != bool
    ):
        raise ValueError("Same native frame and SOURCE-only background required")
    eps_s, eps_e = _epsilon(source, allowed), _epsilon(edited, allowed)
    if eps_s is None:
        return dict(
            known=False, reason="NO_FINITE_SOURCE_BACKGROUND", error=None, pairs=0
        )
    count = mismatch = finite_edited = source_ordered = 0
    touched = np.zeros(source.shape, bool)
    source = np.asarray(source, np.float64)
    edited = np.asarray(edited, np.float64)
    for dy, dx in CONFIG["pair_offsets_yx"]:
        h, w = source.shape
        if dy >= h or dx >= w:
            continue
        a = (slice(0, h - dy), slice(0, w - dx))
        b = (slice(dy, h), slice(dx, w))
        ds, de = source[a] - source[b], edited[a] - edited[b]
        support = allowed[a] & allowed[b] & np.isfinite(ds)
        good = np.isfinite(de)
        ss = np.where(np.abs(ds) <= eps_s, 0, np.sign(ds))
        ee = np.where(np.abs(de) <= (eps_e if eps_e is not None else 0), 0, np.sign(de))
        # Never improve by dropping candidate-dependent invalid values.
        mismatch += int((support & (~good | (ss != ee))).sum())
        source_ordered += int((support & (ss != 0)).sum())
        finite_edited += int((support & good).sum())
        count += int(support.sum())
        touched[a] |= support
        touched[b] |= support
    result = dict(
        pairs=count,
        mismatched_pairs=mismatch,
        finite_edited_pairs=finite_edited,
        source_ordered_pairs=source_ordered,
        source_tied_pairs=count - source_ordered,
        background_pixels=int(allowed.sum()),
        covered_background_pixels=int(touched.sum()),
        same_native_coordinates=True,
        independent_frame_scales=True,
        source_fixed_pairs=True,
        texture_or_physical_correctness_certified=False,
    )
    if count < CONFIG["min_source_pairs"]:
        return dict(
            **result,
            known=False,
            reason="INSUFFICIENT_SOURCE_BACKGROUND_PAIRS",
            error=None,
        )
    return dict(**result, known=True, reason=None, error=mismatch / count)


def hps_indices(eligible):
    indices = np.flatnonzero(eligible)
    if len(indices) <= 8:
        return indices
    return indices[[round(j * (len(indices) - 1) / 7) for j in range(8)]]


def full_reward(
    rgb,
    edit,
    masks,
    valid,
    eligible,
    depth,
    edited_depth,
    protected,
    source_mu,
    edited_mu,
):
    """All arrays share the source clock. masks is [instances,time,height,width]."""
    n = len(rgb)
    if (
        n != 124
        or rgb.dtype != np.uint8
        or rgb.shape != edit.shape
        or edit.dtype != np.uint8
    ):
        raise ValueError("Expected 124 aligned uint8 RGB frames")
    if (
        masks.ndim != 4
        or masks.shape[1:] != rgb.shape[:-1]
        or valid.shape != masks.shape[1:]
        or protected.shape != valid.shape
        or depth.shape != valid.shape
        or edited_depth.shape != valid.shape
        or eligible.shape != (n,)
        or any(a.dtype != bool for a in (eligible, masks, valid, protected))
    ):
        raise ValueError("Reward evidence and masks must share the source grid")
    if not np.isfinite(depth).all() or not np.isfinite(edited_depth).all():
        raise ValueError("Nonfinite native inverse depth")
    expected = (len(hps_indices(eligible)),)
    if (
        source_mu.shape != expected
        or edited_mu.shape != expected
        or not np.isfinite([source_mu, edited_mu]).all()
    ):
        raise ValueError("Missing or nonfinite HPS frame scores")
    targets = []
    for i in range(n):
        visible = [m for m in masks[:, i] if m.any()]
        if eligible[i] and (not visible or any(m.sum() < 32 for m in visible)):
            raise ValueError("Invalid source eligibility")
        targets.append(visible if eligible[i] else [None])
    gauge = calibrate_video(
        np.where(valid, depth, np.nan), np.where(valid, edited_depth, np.nan), targets
    )
    if not gauge["known"]:
        return None
    T = E = C = 0.0
    for i in np.flatnonzero(eligible):
        background = valid[i] & ~np.logical_or.reduce(targets[i])
        region = background & protected[i]
        order = background_evidence(depth[i], edited_depth[i], background)
        if not order["known"] or region.sum() < 32:
            return None
        delta = depth[i].astype(np.float64) - (
            gauge["scale"] * edited_depth[i] + gauge["shift"]
        )
        T += (
            np.mean(
                [(delta[m] > gauge["deletion_threshold"]).mean() for m in targets[i]]
            )
            / n
        )
        E += order["error"] / n
        difference = (
            np.abs(rgb[i][region].astype(np.float32) - edit[i][region]).mean(axis=-1)
            / 255.0
        )
        C += difference.mean(dtype=np.float64) / n
    H = float(np.mean(edited_mu.astype(np.float64) - source_mu))
    return dict(
        T=float(T),
        E=float(E),
        C=float(C),
        delta_HPS=H,
        reward=float(T - E - 5 * C + 0.05 * H),
    )

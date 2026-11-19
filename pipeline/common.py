import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def validate_motion(probability, frames):
    if probability.ndim != 3 or probability.shape[0] != frames or not probability.size:
        raise ValueError('Pi4 motion must contain one probability map per prepared frame.')
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError('Invalid Pi4 motion probabilities.')


def motion_fraction(mask, probability, threshold):
    if not mask.any():
        return 0.0
    if probability.shape != mask.shape:
        probability = cv2.resize(probability.astype(np.float32), mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return float(np.mean(probability[mask] >= threshold))


def collect_instances(predictor, session_id, probability, height, width, *,
                      threshold, fraction, min_frames, min_pixels, max_objects, direct=False):
    validate_motion(probability, len(probability))
    count = len(probability)
    records, packed, seen = {}, {}, set()
    stream = predictor.handle_stream_request(dict(type='propagate_in_video', session_id=session_id,
                 propagation_direction='forward', start_frame_index=0, output_prob_thresh=0.5))
    for result in stream:
        frame = int(result['frame_index'])
        if frame in seen or not 0 <= frame < count:
            raise ValueError('SAM returned duplicate or out-of-range frame indices.')
        seen.add(frame)
        output = result['outputs']
        ids = array(output['out_obj_ids']).reshape(-1)
        if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
            raise ValueError('SAM object IDs must be finite integers.')
        masks = array(output['out_binary_masks'])
        if masks.size == 0:
            masks = np.zeros((0, height, width), bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.shape != (len(ids), height, width) or len(set(ids.tolist())) != len(ids):
            raise ValueError('SAM mask dimensions or object IDs do not match the prepared video.')
        if not np.isin(masks, [0, 1]).all():
            raise ValueError('Expected binary SAM masks, not logits.')
        if len(ids) >= max_objects:
            raise RuntimeError('SAM reached the configured object capacity. Increase --max-objects and rerun in a new output directory.')
        masks = masks.astype(bool)
        packed[frame] = (ids.astype(int), np.packbits(masks, axis=-1))
        for oid, mask in zip(ids, masks):
            row = records.setdefault(int(oid), {'positive_frames': 0, 'visible_frames': 0})
            row['visible_frames'] += int(mask.any())
            if not direct:
                ratio = motion_fraction(mask, probability[frame], threshold)
                row['positive_frames'] += int(mask.sum() >= min_pixels and ratio >= fraction)
    if seen != set(range(count)):
        raise RuntimeError(f'SAM processed {len(seen)}/{count} frames. No partial mask is accepted.')
    selected = {oid for oid, row in records.items()
                if (row['visible_frames'] > 0 if direct else row['positive_frames'] >= min_frames)}
    union = np.zeros((count, height, width), bool)
    for frame, (ids, bits) in packed.items():
        for oid, mask in zip(ids, np.unpackbits(bits, count=width, axis=-1).astype(bool)):
            if int(oid) in selected:
                union[frame] |= mask
    return union, {str(oid): dict(row, selected=oid in selected) for oid, row in records.items()}

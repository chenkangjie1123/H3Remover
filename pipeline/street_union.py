"""Two independent paths: semantic tracks and category-free motion proposals."""
import cv2
import numpy as np

from .common import array

DIRECT_PROMPTS = ['person', 'vehicle', 'bicycle', 'motorcycle']


def proposals(prob, direct, threshold=.5, min_area=32):
    h, w = direct.shape
    active = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if area < min_area:
            continue
        component = labels == i
        uncovered = component & ~direct
        if uncovered.sum() < min_area or uncovered.sum() < .25 * area:
            continue
        # Seed from the unexplained pixels. A moving rider and another object
        # may touch an already-selected vehicle in the original heatmap.
        n, sub, boxes, _ = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            x, y, width, height, a = boxes[j]
            if a >= min_area:
                yield sub == j, [(x+width/2)/w, (y+height/2)/h, width/w, height/h]


def choose_masks(masks, component, direct):
    result = np.zeros_like(direct)
    for mask in masks:
        area = int(mask.sum())
        if not area:
            continue
        overlap = int((mask & component).sum())
        if overlap/area >= .5 and overlap/component.sum() >= .15 and (mask & ~direct).sum()/area >= .5:
            result |= mask
    return result


def motion_masks(rgb, probabilities, direct, detector, threshold, min_area, on_frame=None):
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(detector, device='cuda', confidence_threshold=.5)
    extra = np.zeros_like(direct)
    counts = []
    for t, frame in enumerate(rgb):
        state = None
        attempted = accepted = 0
        for component, box in proposals(probabilities[t], direct[t], threshold, min_area):
            if state is None:
                state = processor.set_image(Image.fromarray(frame))
            processor.reset_all_prompts(state)
            state = processor.add_geometric_prompt(box, True, state)
            masks = array(state['masks']).reshape(-1, *direct.shape[1:]).astype(bool)
            chosen = choose_masks(masks, component, direct[t])
            attempted += 1
            accepted += int(chosen.any())
            extra[t] |= chosen & ~direct[t]
        counts.append([attempted, accepted])
        if on_frame:
            on_frame(t, attempted, accepted)
    return extra, counts

from pathlib import Path
import cv2
import numpy as np


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Cannot read video: {path}')
        limit = int(np.ceil(124 / 24 * fps)) + 1
        for _ in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError('Empty video.')
    return np.stack(frames), fps, count


def annotate(source, masks):
    output = source.copy()
    kernel = np.ones((3, 3), np.uint8)
    for i, mask in enumerate(masks):
        edge = mask & ~cv2.erode(mask.astype(np.uint8), kernel, iterations=2,
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        output[i, mask] = np.rint(.65 * source[i, mask] + .35 * np.array([255, 48, 48])).astype(np.uint8)
        output[i, edge] = 255
    return output


def load_inputs(video, mask):
    rgb, fps, count = read_video(video)
    raw_mask, mask_fps, mask_count = read_video(mask)
    if rgb.shape != raw_mask.shape or count != mask_count or abs(fps - mask_fps) > 1e-3:
        raise ValueError('Source and mask must have identical dimensions, frame count and FPS.')
    if np.max(np.ptp(raw_mask.astype(np.int16), axis=-1)) > 8:
        raise ValueError('Use a grayscale binary mask, not a colored mask overlay.')
    indices = np.floor(np.arange(124) * fps / 24 + .5).astype(int)
    if indices[-1] >= len(rgb):
        raise ValueError('Both inputs must cover 124 output frames at 24 FPS (about 5.17 seconds).')
    height, width = rgb.shape[1:3]
    scale = min(960 / width, 544 / height)
    w, h = round(width * scale), round(height * scale)
    x, y = (960 - w) // 2, (544 - h) // 2
    source = np.zeros((124, 544, 960, 3), np.uint8)
    masks = np.zeros((124, 544, 960), bool)
    for i, native in enumerate(indices):
        source[i, y:y+h, x:x+w] = cv2.resize(rgb[native], (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        masks[i, y:y+h, x:x+w] = cv2.resize(raw_mask[native, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST_EXACT) >= 128
    if not masks.any():
        raise ValueError('The mask contains no selected target.')
    return source, annotate(source, masks)


def save_video(path, frames):
    import subprocess
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s', '960x544', '-r', '24', '-i', 'pipe:0',
                    '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', '-map_metadata', '-1', str(path)],
                   input=frames.tobytes(), check=True)

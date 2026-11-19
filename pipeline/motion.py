import argparse
from pathlib import Path

import numpy as np

from .common import sha256, validate_motion, write_json

MODEL_REVISION = '59f307317fb630c9c5a5aa635038d98ed106aaf5'

def main():
    parser = argparse.ArgumentParser(description='Estimate motion on the prepared H3Remover clip with Pi4.')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--weights', default='yyfz233/Pi4')
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--max-size', type=int, default=512)
    args = parser.parse_args()
    import torch
    from pi4 import Pi4
    from pi4.io import load_images
    if not torch.cuda.is_available():
        raise RuntimeError('Pi4 requires a CUDA GPU.')
    source = args.work_dir / 'source.mp4'
    frames, metadata = load_images(source, stride=1, max_frames=124, max_size=args.max_size)
    if len(frames) != 124 or metadata['source_frame_indices'] != list(range(124)):
        raise ValueError('Pi4 must process all 124 prepared frames without temporal subsampling.')
    local = Path(args.weights).exists()
    model = Pi4.from_pretrained(args.weights, **({} if local else {'revision': args.revision})).cuda().eval()
    images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
    with torch.inference_mode():
        prediction = model(images, query_frame=0)
    probability = prediction['dynamic_mask'].detach().float().cpu().numpy()[0]
    validate_motion(probability, 124)
    np.save(args.work_dir / 'motion.npy', probability.astype(np.float16), allow_pickle=False)
    write_json(args.work_dir / 'motion.json', {'source_sha256': sha256(source),
        'frames': 124, 'size': list(probability.shape[1:]), 'weights': args.weights,
        'revision': None if local else args.revision,
        'semantics': 'Pi4 sequence-level moving-object probability',
        'thresholding': 'Raw probabilities retained; selection is performed by the SAM stage.'})


if __name__ == '__main__':
    main()

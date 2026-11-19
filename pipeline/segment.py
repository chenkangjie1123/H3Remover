import argparse
import json
from pathlib import Path

import numpy as np

from .common import collect_instances, sha256, validate_motion, write_json
from video import annotate, read_video, save_video


def main():
    parser = argparse.ArgumentParser(description='Select SAM 3.1 instances using Pi4 motion evidence.')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--prompts', nargs='+', default=['person', 'vehicle', 'bicycle', 'motorcycle', 'animal'])
    parser.add_argument('--motion-threshold', type=float, default=0.5)
    parser.add_argument('--motion-fraction', type=float, default=0.5)
    parser.add_argument('--min-motion-frames', type=int, default=2)
    parser.add_argument('--min-pixels', type=int, default=32)
    parser.add_argument('--max-objects', type=int, default=64)
    args = parser.parse_args()
    if not 0 < args.motion_threshold < 1 or not 0 < args.motion_fraction <= 1:
        parser.error('Motion threshold and fraction must be within (0, 1].')
    if not 1 <= args.min_motion_frames <= 124 or args.min_pixels < 1 or args.max_objects < 2:
        parser.error('Invalid frame, area or capacity setting.')
    source = args.work_dir / 'source.mp4'
    rgb, fps, count = read_video(source)
    if rgb.shape != (124, 544, 960, 3) or count != 124 or abs(fps-24) > 1e-3:
        raise ValueError('Expected the prepared 124-frame source at 24 FPS and 960x544.')
    motion_info = json.loads((args.work_dir / 'motion.json').read_text())
    if motion_info['source_sha256'] != sha256(source):
        raise ValueError('Motion data belongs to a different source video.')
    probability = np.load(args.work_dir / 'motion.npy', allow_pickle=False)
    validate_motion(probability, 124)
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    if not torch.cuda.is_available():
        raise RuntimeError('SAM 3.1 requires a CUDA GPU.')
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=args.checkpoint,
        max_num_objects=args.max_objects, use_fa3=False, compile=False, async_loading_frames=False)
    union = np.zeros((124, 544, 960), bool)
    evidence = {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for prompt in args.prompts:
                print(f'Segmenting: {prompt}', flush=True)
                session = predictor.handle_request(dict(type='start_session', resource_path=str(source),
                                      offload_video_to_cpu=True))['session_id']
                try:
                    predictor.handle_request(dict(type='add_prompt', session_id=session, frame_index=0, text=prompt))
                    masks, records = collect_instances(predictor, session, probability, 544, 960,
                        threshold=args.motion_threshold, fraction=args.motion_fraction,
                        min_frames=args.min_motion_frames, min_pixels=args.min_pixels,
                        max_objects=args.max_objects)
                    union |= masks
                    evidence[prompt] = records
                finally:
                    predictor.handle_request(dict(type='close_session', session_id=session))
    finally:
        predictor.shutdown()
    write_json(args.work_dir / 'selection.json', {'source_sha256': sha256(source),
        'prompts': args.prompts, 'motion_threshold': args.motion_threshold,
        'motion_fraction': args.motion_fraction, 'min_motion_frames': args.min_motion_frames,
        'min_pixels': args.min_pixels, 'max_objects': args.max_objects,
        'instances': evidence, 'mask_pixels_per_frame': union.sum(axis=(1, 2)).tolist(),
        'policy': 'Once selected, an instance is removed in all frames where SAM returns its mask.'})
    if not union.any():
        raise RuntimeError('No moving instance was selected. See selection.json; H3 inference was not started.')
    save_video(args.work_dir / 'mask.mp4', np.repeat(union[..., None], 3, axis=-1).astype(np.uint8) * 255, crf=0)
    save_video(args.work_dir / 'mask_preview.mp4', annotate(rgb, union))
    print('Mask and mask_preview.mp4 are ready.', flush=True)


if __name__ == '__main__':
    main()

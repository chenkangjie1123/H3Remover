import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from pipeline.common import sha256, write_json
from video import read_video, save_video


def prepare(source, folder):
    frames, fps, count = read_video(source)
    indices = np.floor(np.arange(124) * fps / 24 + 0.5).astype(int)
    if indices[-1] >= len(frames):
        raise ValueError('The input must cover about 5.17 seconds (124 output frames at 24 FPS).')
    h, w = frames.shape[1:3]
    scale = min(960/w, 544/h)
    width, height = round(w*scale), round(h*scale)
    x, y = (960-width)//2, (544-height)//2
    prepared = np.zeros((124, 544, 960, 3), np.uint8)
    for i, index in enumerate(indices):
        prepared[i, y:y+height, x:x+width] = cv2.resize(frames[index], (width, height),
                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    save_video(folder / 'source.mp4', prepared, crf=0)
    write_json(folder / 'source.json', {'input_sha256': sha256(source), 'input_fps': fps,
        'input_frame_count': count, 'source_frame_indices': indices.tolist(),
        'output_frames': 124, 'output_fps': 24, 'content_rect_xywh': [x, y, width, height]})


def run_stage(name, command, root, folder, env):
    print(f'[{name}] Starting; log: {folder / (name + ".log")}', flush=True)
    start = time.monotonic()
    with (folder / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'{name} failed (exit {code}). See {folder / (name + ".log")}. Completed artifacts are preserved.')
    return round(time.monotonic()-start, 3)


def main():
    parser = argparse.ArgumentParser(description='Pi4 motion → SAM 3.1 masks → H3Remover. Each model runs in its own process.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pi4-python', required=True, help='Python executable in the Pi4 environment.')
    parser.add_argument('--sam-python', required=True, help='Python executable in the SAM 3.1 environment.')
    parser.add_argument('--h3-python', default=sys.executable)
    parser.add_argument('--pi4-weights', default='yyfz233/Pi4')
    parser.add_argument('--sam-checkpoint')
    parser.add_argument('--prompts', nargs='+', default=['person', 'vehicle', 'bicycle', 'motorcycle', 'animal'])
    parser.add_argument('--base', type=Path, default=Path('weights/MiniMax-H3'))
    parser.add_argument('--adapter', type=Path, default=Path('weights/h3remover'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-objects', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--prepare-only', action='store_true', help='Stop after saving the mask preview, before loading H3.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    folder = args.output_dir.resolve()
    if not args.video.is_file():
        parser.error('Input video does not exist.')
    if args.gpu < 0 or args.max_objects < 2 or args.steps < 2:
        parser.error('Invalid GPU index, object capacity or sampling steps.')
    for executable in [args.pi4_python, args.sam_python, args.h3_python, 'ffmpeg']:
        if not shutil.which(executable):
            parser.error(f'Executable not found: {executable}')
    args.pi4_python = os.path.abspath(shutil.which(args.pi4_python))
    args.sam_python = os.path.abspath(shutil.which(args.sam_python))
    args.h3_python = os.path.abspath(shutil.which(args.h3_python))
    if Path(args.pi4_weights).exists():
        args.pi4_weights = str(Path(args.pi4_weights).resolve())
    if not args.prepare_only:
        for path in [args.base / 'modular_model_index.json', args.adapter / 'h3remover_config.json', args.adapter / 'h3remover.safetensors']:
            if not path.is_file():
                parser.error(f'Missing model file: {path}. Run python download_weights.py first.')
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    stages = [
        ('motion', [args.pi4_python, '-m', 'pipeline.motion', '--work-dir', str(folder), '--weights', args.pi4_weights]),
        ('segment', [args.sam_python, '-m', 'pipeline.segment', '--work-dir', str(folder),
                     '--max-objects', str(args.max_objects), '--prompts', *args.prompts]),
    ]
    if args.sam_checkpoint:
        stages[1][1].extend(['--checkpoint', str(Path(args.sam_checkpoint).resolve())])
    if not args.prepare_only:
        stages.append(('remove', [args.h3_python, str(root / 'infer.py'), '--base', str(args.base.resolve()),
            '--adapter', str(args.adapter.resolve()), '--video', str(folder / 'source.mp4'),
            '--mask', str(folder / 'mask.mp4'), '--output', str(folder / 'edited.mp4'),
            '--seed', str(args.seed), '--steps', str(args.steps)]))
    report = {'status': 'running', 'stages_seconds': {}}
    try:
        write_json(folder / 'run.json', report)
        start = time.monotonic(); prepare(args.video, folder)
        report['stages_seconds']['prepare'] = round(time.monotonic()-start, 3)
        for name, command in stages:
            report['stages_seconds'][name] = run_stage(name, command, root, folder, env)
            write_json(folder / 'run.json', report)
        report['status'] = 'masks_ready' if args.prepare_only else 'complete'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        write_json(folder / 'run.json', report)
    print(f'Results saved in {folder}')


if __name__ == '__main__':
    main()

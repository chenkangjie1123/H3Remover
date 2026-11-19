import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from video import load_inputs, save_video

PROMPT = (
    'Edit <Video 1>, the original RGB video. <Video 2> shows the same video with '
    'the selected target objects highlighted by translucent red masks with white outlines. '
    'Remove every highlighted target object throughout the video, including its associated '
    'cast shadows, reflections, headlights, emitted light and light spill. Reconstruct the '
    'revealed background coherently across frames. Preserve all unrelated objects, scene '
    'details, colors, textures, lighting, camera motion and timing. Do not remove unrelated '
    'shadows, reflections or light sources. Do not reproduce the masks or white outlines. '
    'Output only the edited original video, without blur, flicker or added objects.'
)


def main():
    parser = argparse.ArgumentParser(description='Remove mask-selected objects from a video with H3Remover.')
    parser.add_argument('--base', default='weights/MiniMax-H3')
    parser.add_argument('--adapter', default=Path('weights/h3remover'), type=Path)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--mask', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--preview-only', action='store_true', help='Validate inputs and save the mask reference without loading models.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 2:
        raise ValueError('Use at least two sampling points.')
    if not shutil.which('ffmpeg'):
        parser.error('FFmpeg is not installed. Install it before running inference.')
    for path in [args.video, args.mask]:
        if not path.is_file():
            parser.error(f'Input file not found: {path}')
    if not args.preview_only:
        for path in [Path(args.base) / 'modular_model_index.json', args.adapter / 'h3remover_config.json', args.adapter / 'h3remover.safetensors']:
            if not path.is_file():
                parser.error(f'Missing weights: {path}. Run python download_weights.py, or set --base and --adapter.')
    print('Preparing synchronized video and target masks...', flush=True)
    source, reference = load_inputs(args.video, args.mask)
    if args.preview_only:
        save_video(args.output, reference)
        print(f'Mask preview saved to {args.output}')
        return

    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from peft import LoraConfig
    from safetensors.torch import load_file
    from offload import ComponentOffload

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required.')
    print('Loading H3Remover...', flush=True)
    config = json.loads((args.adapter / 'h3remover_config.json').read_text())
    if config['format_version'] != 1 or config['rank'] != 16 or config['alpha'] != 16:
        raise ValueError('Unsupported adapter configuration.')
    pipe = ModularPipeline.from_pretrained(
        args.base, workflow='ref2va', components_manager=ComponentsManager(), local_files_only=True)
    pipe.register_to_config(canvas_short_edge=544, canvas_max_pixels=960 * 544,
                            reference_image_short_edge=544)
    pipe.load_components(
        names=['text_encoder', 'tokenizer', 'processor', 'vae', 'audio_vae',
               'transformer_ref', 'scheduler', 'audio_scheduler'],
        pretrained_model_name_or_path=args.base,
        torch_dtype={'text_encoder': torch.bfloat16, 'transformer_ref': torch.bfloat16,
                     'default': torch.float32},
        device_map={'text_encoder': 'cpu', 'transformer_ref': 'cpu', 'vae': 'cpu',
                    'audio_vae': 'cpu', 'default': None},
        low_cpu_mem_usage=True, local_files_only=True)
    model = pipe.transformer_ref
    targets = [f'transformer_blocks.{i}.attn.{suffix}' for i in range(50)
               for suffix in ['to_q', 'to_k', 'to_v', 'to_out.0']]
    model.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets,
                                lora_dropout=0., bias='none', use_dora=False,
                                use_rslora=False), adapter_name='default')
    weights = load_file(args.adapter / 'h3remover.safetensors', device='cpu')
    parameters = {name: param for name, param in model.named_parameters() if 'lora_' in name}
    if len(parameters) != 400 or set(weights) != set(parameters):
        raise ValueError('The adapter does not match the full Ref2VA transformer.')
    with torch.no_grad():
        for name, param in parameters.items():
            weight = weights[name]
            if weight.shape != param.shape or not torch.isfinite(weight).all():
                raise ValueError(f'Invalid adapter tensor: {name}')
            param.data = weight.float().clone()
    for module in pipe.components.values():
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
    offload = ComponentOffload(pipe.components, 'cuda:0')
    pipe.vae.enable_tiling()
    references = [MiniMaxH3VideoReference(frames=frames, fps=24.)
                  for frames in [source, reference]]
    try:
        print('Removing selected objects...', flush=True)
        with torch.inference_mode():
            result = pipe(prompt=PROMPT, references=references, height=544, width=960,
                          num_frames=124, num_inference_steps=args.steps,
                          generator=torch.Generator(device='cpu').manual_seed(args.seed),
                          output=['videos'])
        videos = result['videos']
        if isinstance(videos, torch.Tensor):
            videos = videos.detach().float().cpu().numpy()
        if isinstance(videos, np.ndarray):
            if videos.ndim == 5:
                videos = videos[0]
            if videos.shape == (3, 124, 544, 960):
                videos = videos.transpose(1, 2, 3, 0)
            frames = videos if videos.dtype == np.uint8 else np.rint(np.clip(videos, 0, 1) * 255).astype(np.uint8)
        else:
            if isinstance(videos[0], list):
                videos = videos[0]
            frames = np.stack([np.asarray(frame.convert('RGB')) for frame in videos])
        if frames.shape != source.shape:
            raise ValueError('Unexpected output dimensions.')
        save_video(args.output, frames)
        print(f'Edited video saved to {args.output}', flush=True)
    finally:
        offload.close()


if __name__ == '__main__':
    main()

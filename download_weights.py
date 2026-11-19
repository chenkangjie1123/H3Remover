import argparse
import hashlib
from pathlib import Path

BASE_ID = 'MiniMaxAI/MiniMax-H3'
BASE_REVISION = '0919fb684102ef115126612f3fbb84d1027e7317'
ADAPTER_ID = 'chenkangjie1123/H3Remover'
ADAPTER_REVISION = '02eb73f64261b07505ed0c935462bbadfbd0a6cb'
ADAPTER_SHA256 = '6fe5b881ec1a9804af005a3ed512e54a6c3341b0e5a2204323a2ed6482262c84'
BASE_FILES = ['modular_model_index.json', 'transformer_ref/*', 'text_encoder/*',
              'tokenizer/*', 'processor/*', 'vae/*', 'audio_vae/*', 'scheduler/*', 'audio_scheduler/*']


def download(root, adapter_only=False):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        print('Downloading the H3Remover adapter...', flush=True)
        folder = Path(snapshot_download(ADAPTER_ID, revision=ADAPTER_REVISION,
                      allow_patterns=['h3remover.safetensors', 'h3remover_config.json'],
                      local_dir=Path(root) / 'h3remover'))
        with (folder / 'h3remover.safetensors').open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != ADAPTER_SHA256:
            raise RuntimeError('Adapter checksum mismatch. Do not use this download.')
        if not adapter_only:
            print('Downloading the full Ref2VA base and shared components...', flush=True)
            snapshot_download(BASE_ID, revision=BASE_REVISION, allow_patterns=BASE_FILES,
                              local_dir=Path(root) / 'MiniMax-H3')
    except GatedRepoError as error:
        raise SystemExit('Accept the license on the requested Hugging Face model page, run hf auth login, then rerun this command.') from error
    print(f'Weights ready in {Path(root)}. Completed files are reused when rerunning.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the pinned H3Remover inference weights.')
    parser.add_argument('--output-dir', type=Path, default=Path('weights'))
    parser.add_argument('--adapter-only', action='store_true', help='Use this if you already have the full Ref2VA base.')
    args = parser.parse_args()
    download(args.output_dir, args.adapter_only)

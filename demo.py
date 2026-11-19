"""Local Gradio interface for the same mask preparation and editing commands as the CLI."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import gradio as gr

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
MASK, STREET = "Upload a binary mask", "Find street targets automatically"


def build(work, h3_python=sys.executable):
    work = Path(work)
    def run(args):
        result = subprocess.run([h3_python, *map(str, args)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            lines = result.stdout.strip().splitlines()
            raise gr.Error(lines[-1] if lines else "Command failed. See the terminal for details.")

    def prepare(video, mask, mode):
        if not video:
            raise gr.Error("Choose a source video first.")
        folder = Path(tempfile.mkdtemp(dir=work))
        if mode == STREET:
            pi4, sam = os.getenv("PI4_PYTHON"), os.getenv("SAM31_PYTHON")
            if not (pi4 and sam):
                raise gr.Error("Automatic masks need PI4_PYTHON and SAM31_PYTHON. Use an example or upload a mask instead.")
            out = folder / "street"
            visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            if not visible.isdigit():
                raise gr.Error("Set CUDA_VISIBLE_DEVICES to a numeric GPU index for automatic masks.")
            run(["auto_remove.py", "--prepare-only", "--video", video, "--output-dir", out,
                 "--gpu", visible, "--pi4-python", pi4, "--sam-python", sam])
            src, binary, overlay = out/"source.mp4", out/"mask.mp4", out/"mask_preview.mp4"
        else:
            if not mask:
                raise gr.Error("Choose a synchronized black-and-white mask video.")
            src, binary, overlay = video, mask, folder/"overlay.mp4"
            run(["infer.py", "--preview-only", "--video", src, "--mask", binary, "--output", overlay])
        return str(overlay), (video, mask if mode == MASK else None, mode, str(src), str(binary)), None

    def edit(video, mask, mode, reviewed):
        key = (video, mask if mode == MASK else None, mode)
        if not reviewed or tuple(reviewed[:3]) != key:
            raise gr.Error("Preview these targets before running removal.")
        out = Path(tempfile.mkdtemp(dir=work)) / "removed.mp4"
        run(["infer.py", "--video", reviewed[3], "--mask", reviewed[4], "--output", out])
        return str(out)

    with gr.Blocks(title="H3Remover", analytics_enabled=False, delete_cache=(3600, 86400)) as app:
        gr.Markdown("# H3Remover\nRemove objects and their visual effects from video.")
        with gr.Tab("Edit a video"):
            mode = gr.Radio([MASK, STREET], value=MASK, label="How would you like to select targets?")
            with gr.Row():
                video = gr.Video(label="Source video", sources=["upload"])
                mask = gr.Video(label="Binary mask · white removes", sources=["upload"])
            gr.Examples([[str(EXAMPLES/n/"source.mp4"), str(EXAMPLES/n/"mask.mp4")] for n in ["street", "bicycle"]],
                        inputs=[video, mask], example_labels=["City traffic", "Lifting a bicycle"], label="Try an input pair")
            gr.Markdown("Use a clip at least 5.17 seconds long. An uploaded mask must match its dimensions, frame count and frame rate.")
            with gr.Row():
                preview = gr.Button("1 · Preview targets")
                remove = gr.Button("2 · Remove selected objects", variant="primary")
            state = gr.State(None)
            with gr.Row():
                overlay = gr.Video(label="Selected targets · inspect the red overlay", interactive=False)
                result = gr.Video(label="Edited video", interactive=False)
            mode.change(lambda m: gr.Video(visible=m == MASK), mode, mask, queue=False)
            preview.click(prepare, [video, mask, mode], [overlay, state, result],
                          concurrency_id="model", concurrency_limit=1, api_name="preview_targets")
            remove.click(edit, [video, mask, mode, state], result,
                         concurrency_id="model", concurrency_limit=1, api_name="remove_objects")
        with gr.Tab("Watch a finished example"):
            gr.Markdown("### City traffic\nRecorded model output, not a new inference. Left: selected targets. Right: H3Remover.")
            gr.Video(value=str(ROOT/"assets"/"street-comparison.mp4"), interactive=False, label="Before and after")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--h3-python", default=sys.executable, help="Python in your H3Remover environment")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="h3remover-demo-") as work:
        build(work, args.h3_python).queue(max_size=8).launch(
            server_name="127.0.0.1", server_port=args.port, share=False, max_file_size="250mb",
            allowed_paths=[work, str(EXAMPLES), str(ROOT/"assets"/"street-comparison.mp4")])

"""Local Gradio interface for the same mask preparation and editing commands as the CLI."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MASK, STREET = "Upload a binary mask", "Find street targets automatically"
STUDIO = ROOT / "assets" / "studio"
CSS = """
.gradio-container{width:100%!important;max-width:1184px!important;margin:auto!important;padding:0!important}
.gradio-container>.main{padding-top:12px!important}
footer{display:none!important}
.studio-head{display:flex;align-items:center;justify-content:space-between;color:#192029;padding:12px 0 22px;border-bottom:1px solid #d4d9df}
.studio-head h1{font:680 29px/1.15 system-ui;margin:0;letter-spacing:-.045em}
.studio-head small,.studio-tag{font:600 12px/1.4 system-ui;letter-spacing:.12em;text-transform:uppercase;color:#516171}
.studio-logo{background:#223d50;color:white;padding:6px 9px;border-radius:6px;margin-right:10px;font:700 17px/1 system-ui;vertical-align:middle}
.sheet{padding:18px 20px 14px;border-left:3px solid #31759a;background:#f0f4f7;color:#1d2a34}
.sheet h2{font:650 22px/1.3 system-ui;letter-spacing:-.03em;margin:4px 0}.sheet p{margin:3px 0;font:15px/1.5 system-ui}
.deck-title{font:650 14px/1.5 system-ui;color:#344857;letter-spacing:.015em;padding:2px 0}
.studio-deck video{display:block;width:100%!important;height:auto!important;aspect-ratio:16/9;object-fit:contain!important;background:#131a20}
#studio-tools{align-self:stretch;justify-content:center;padding:14px 16px;border:1px solid #d7dee4}
@media(max-width:640px){
 .gradio-container>.main{padding:12px 12px 24px!important}
 .studio-head{align-items:baseline;gap:8px;padding:4px 0 14px}
 .studio-head h1{font-size:24px}.studio-head small{font-size:10px;text-align:right}
}
"""


def studio_theme():
    theme = gr.themes.Base(primary_hue="slate", font=["system-ui", "sans-serif"])
    # Preserve the walkthrough's light palette on either system color scheme.
    return theme.set(**{key: getattr(theme, key[:-5]) for key in vars(theme)
                        if key.endswith("_dark") and hasattr(theme, key[:-5])})


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
            run(["-m", "pipeline.auto_remove", "--prepare-only", "--video", video, "--output-dir", out,
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
        gr.HTML('<header class="studio-head"><h1><span class="studio-logo">H3</span>Removal Studio</h1><small>H3Remover &nbsp;/&nbsp; Video workspace</small></header>')
        with gr.Tab("Example workspace"):
            with gr.Row():
                gr.HTML('<div class="sheet"><small class="studio-tag">Sample footage &nbsp;/&nbsp; MOSE</small><h2>City traffic</h2><p>Review the vehicle masks and the finished road scene.</p></div>', scale=3)
                choices = gr.Radio(["Target overlay", "Removal result"], value="Target overlay", label="Review layer", elem_id="studio-tools", scale=2)
            with gr.Row():
                with gr.Column(min_width=250):
                    gr.HTML('<div class="deck-title">Source monitor</div>')
                    gr.Video(value=str(EXAMPLES/"street"/"source.mp4"), interactive=False, show_label=False, buttons=[], elem_id="studio-source", elem_classes="studio-deck", label="Original street video", loop=True)
                with gr.Column(min_width=250):
                    monitor_title = gr.HTML('<div class="deck-title">Review monitor &nbsp;·&nbsp; Target overlay</div>')
                    monitor = gr.Video(value=str(STUDIO/"selection.mp4"), interactive=False, show_label=False, buttons=[], elem_id="studio-review", elem_classes="studio-deck", label="Review monitor", loop=True)
            def review(layer):
                result = layer == "Removal result"
                return (str(STUDIO/("result.mp4" if result else "selection.mp4")),
                        '<div class="deck-title">Review monitor &nbsp;·&nbsp; '+("Removal result" if result else "Target overlay")+'</div>')
            choices.change(review, choices, [monitor, monitor_title], queue=False, show_progress="hidden")
            gr.Markdown("Recorded example · Open **Edit a video** to run removal on your own clip.")
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
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--h3-python", default=sys.executable, help="Python in your H3Remover environment")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="h3remover-demo-") as work:
        build(work, args.h3_python).queue(max_size=8).launch(
            server_name="127.0.0.1", server_port=args.port, share=False, max_file_size="250mb",
            theme=studio_theme(), css=CSS, js=Path(__file__).with_name("playback.js").read_text(),
            allowed_paths=[work, str(EXAMPLES), str(STUDIO)])

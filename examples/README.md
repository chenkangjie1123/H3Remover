# Examples

To try these clips interactively, launch the [Gradio Studio](studio/) from the repository root with `python examples/studio`. See the [demo setup](../README.md#gradio-demo) for installation.

| Example | Source |
| --- | --- |
| `street` | MOSE, `b0240233`: multiple vehicles crossing an urban road |
| `bicycle` | DAVIS, `bike-packing`: a person lifting a bicycle in a detailed indoor scene |

Each directory contains a source video and a synchronized binary target mask. White pixels identify the selected object. The README previews use an animated cursor to switch from the masked input to the AI-generated H3Remover video on the same canvas.

Both examples are real-world videos from public research datasets.

The Gradio studio's target overlay and saved result (`assets/studio/`) and the recorded comparison (`assets/street-comparison.mp4`) use the same MOSE clip and license below. These previews are resized for playback.

- **MOSE:** Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip H. S. Torr, and Song Bai. [MOSE: A New Dataset for Video Object Segmentation in Complex Scenes](https://mose.video/MOSEv1/), ICCV 2023. The street example and its adapted preview are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Its target mask combines the dataset annotation with additional vehicle segmentation.
- **DAVIS:** Federico Perazzi and collaborators, *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*, CVPR 2016; Jordi Pont-Tuset and collaborators, *The 2017 DAVIS Challenge on Video Object Segmentation*. [Dataset](https://davischallenge.org/) · [Terms](https://github.com/davisvideochallenge/davis-2017#terms-of-use) · [License](DAVIS-LICENSE).

Clips have been temporally resampled and resized to 124 frames for the inference interface. Masks have been converted to binary videos, and previews have been resized and frame-sampled. These are selected demonstrations. Example media retain their dataset licenses, separately from the model license.

The bicycle input repeats its last frame after frame 68 to fill the 124-frame canvas. Its preview shows the 69 content frames.

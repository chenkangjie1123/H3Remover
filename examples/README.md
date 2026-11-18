# Examples

| Example | Source |
| --- | --- |
| `street` | MOSE, `b0240233`: multiple vehicles crossing an urban road |
| `bus` | DAVIS, `bus`: a moving bus partially occluded by trees |

Each directory contains a source video and a synchronized binary target mask. White pixels identify the selected object. The README previews show the masked input on the left and the AI-generated H3Remover output on the right.

Both examples are real-world videos from public research datasets.

- **MOSE:** Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip H. S. Torr, and Song Bai. [MOSE: A New Dataset for Video Object Segmentation in Complex Scenes](https://mose.video/MOSEv1/), ICCV 2023. The street example and its adapted previews are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The target mask combines the dataset annotation with additional vehicle segmentation.
- **DAVIS:** Federico Perazzi and collaborators, *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*, CVPR 2016; Jordi Pont-Tuset and collaborators, *The 2017 DAVIS Challenge on Video Object Segmentation*. [Dataset](https://davischallenge.org/) · [Terms](https://github.com/davisvideochallenge/davis-2017#terms-of-use) · [License](DAVIS-LICENSE).

Clips have been temporally resampled and resized to 124 frames for the inference interface. Masks have been converted to binary videos, and previews have been resized and frame-sampled. These are selected demonstrations. Example media retain their dataset licenses, separately from the model license.

# Image Captioning with ViT and GPT-2

This repository contains an exploratory image-captioning project implemented in **`Image_Captioning.ipynb`**. The notebook investigates the Flickr8k dataset, trains a vision-to-language model that conditions GPT-2 on visual features from a frozen Vision Transformer (ViT), and reports a preliminary caption-generation evaluation.

> **Important:** The reported BLEU values are not valid standard benchmark results. The notebook's evaluation procedure pairs captions and images incorrectly and applies nonstandard candidate filtering. See [Evaluation caveat](#evaluation-caveat) before interpreting them.

## Project contents

Only the following project artifacts are identified:

- 'Notebook/Image_Captioning.ipynb' contains the project’s analysis, ViT–GPT-2 implementation, training, inference, and evaluation.
- `model.pt` — optional checkpoint produced by the notebook; it may not be present in every copy of the project.
- 'Dataset/' folder contains instructions for downloading and preparing Flickr8k. The dataset itself is not included in this repository.
- 'Requirements.txt' lists the Python packages needed to run the notebook.


## Dataset and exploratory analysis

The notebook uses **Flickr8k**, with **8,092 images** and **40,460 captions**. Its sampled split counts include **30,000 training** and **5,000 validation** examples; a test split also exists. These are sample/example counts as reported in the notebook, not a claim that each image is assigned to exactly one split.

The exploratory data analysis includes caption-length statistics, frequent bigrams, vocabulary and part-of-speech exploration, and image aspect-ratio categories.

| Caption length statistic | Value |
| --- | ---: |
| Mean | 11.78 tokens |
| Median | 11 tokens |
| Minimum | 1 token |
| Maximum | 38 tokens |

Aspect-ratio counts in the analyzed sample:

| Category | Images |
| --- | ---: |
| Landscape | 679 |
| Portrait | 285 |
| Square | 36 |

The aspect-ratio figures describe the notebook's sample, not necessarily every image in Flickr8k.

## Model and training approach

The principal model combines:

- **Vision encoder:** `google/vit-base-patch16-224-in21k`, kept frozen.
- **Language model:** pretrained GPT-2.
- **Visual-to-language conditioning:** a learned linear projection maps all ViT visual tokens into GPT-2's embedding space. The projected visual tokens are supplied as a causal prefix to GPT-2.

Images are resized to **224 × 224**. GPT-2 tokenization uses a maximum sequence length of **20**. The reported optimization configuration is:

| Setting | Value |
| --- | ---: |
| Optimizer | SGD |
| Learning rate | 0.125 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Gradient clipping | 0.25 |
| Epochs | 5 |

The notebook identifies **epoch 2** as the best epoch and refers to the best checkpoint as `model.pt`.

### Reported training and validation losses

| Epoch | Training loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 2.84 | 2.676 |
| 2 | 2.48 | 2.655 |
| 3 | 2.41 | 2.724 |
| 4 | 2.44 | 2.808 |
| 5 | 2.54 | 2.982 |

These are the values reported in the notebook; they should not be interpreted as an independently reproduced run.

## Reported caption example

The notebook shows this target and generated text:

- **Target:** “A little tan dog with large ears running through the grass .”
- **Generated:** “Virtle is walking through the grass . It is a small dog . It is carrying a red”

This is an illustrative notebook example, not a claim of representative performance.

## Evaluation caveat

The notebook reports the following BLEU values:

| Metric | Reported value |
| --- | ---: |
| BLEU-1 | 0.4047 |
| BLEU-2 | 0.2450 |
| BLEU-3 | 0.1466 |
| BLEU-4 | 0.0880 |

**Do not present these numbers as standard or benchmark-comparable BLEU scores.** The evaluation has a known pairing flaw: it uses the first image in a batch while using captions from different images. It also excludes candidate captions that are too short for a given BLEU order, making the procedure nonstandard. The values are therefore **preliminary notebook outputs only** and require a corrected evaluation before they can support a performance claim.

## Running and reproducibility notes

1. Open `Image_Captioning.ipynb` in a Jupyter environment.
2. Make the Flickr8k image and caption data available at the paths expected by the notebook, or update the notebook's data-path settings to match your local copy. The exact local dataset paths are environment-dependent; no universal path is specified here.
3. Install the libraries imported by the notebook in your environment. A verified dependency manifest is not included, so this README does not prescribe an unverified package list or versions.
4. Run the notebook cells in order, checking data paths and runtime/device settings for your environment.

Reproduction may be affected by environment-specific file paths and by a multiprocessing installation failure encountered in the notebook. The notebook also includes an exploratory **1,000-example subset** and a hyperparameter experiment that is incomplete; neither should be mistaken for a complete, independently validated result.

### Checkpoint compatibility

`model.pt` is a **full-object serialization**, rather than a weights-only checkpoint. Loading it may require the original model class definitions and a compatible Python/PyTorch environment. Treat full-object checkpoints as trusted files only, and do not assume they will load across arbitrary code or library versions. The notebook's identification of a best checkpoint does not mean the optional file is included in every project copy.

## Limitations and next steps

- Correct image-to-caption pairing in evaluation; use each image's own references and a clearly specified, standard BLEU implementation and policy.
- Evaluate on the held-out test split and report the evaluation protocol, rather than treating the current preliminary scores as benchmark results.
- Review generation stopping and sequence-length behavior, and inspect outputs on a broader, correctly sampled set of images.
- Complete and document the exploratory hyperparameter experiment before drawing conclusions from it.
- Record the actual runtime environment and dependencies, and consider saving model weights plus configuration in a portable, documented format.
- Make dataset paths configurable and document the exact split construction for reproducible runs.

# XAI Surface Defect Detection

A small CNN that classifies steel surface defects (NEU-DET style, 200×200 grayscale
images) plus a set of **GALORE-style explanation methods** that answer not only
*"what is it?"* but also *"why this class and not the other one?"*.

The entry point for everyday use is [`src/analyze_image.py`](src/analyze_image.py):
it takes any image (or a whole folder), predicts the defect class and renders a
10-panel explanation figure.

## Defect classes

The model is trained on six classes (folder names under `data/sorted/`):

`Crazing`, `Inclusion`, `Patches`, `Pitted_surface`, `Rolled_in_scale`, `Scratches`

The shipped checkpoint `checkpoints/best_model.pth` was trained for 30 epochs and
reached **98.52 % validation accuracy** (`val_loss` 0.0383).

## Explanation methods

All methods build on Grad-CAM on the last convolutional block, combined into
higher-level maps. Every map is normalized to `[0, 1]`, and a region is extracted
by thresholding at the 90 % quantile; `strength` is the mean map value inside that
region.

| Method | Question it answers | Formula (conceptually) |
| --- | --- | --- |
| **Class attribution** (`xai/gradcam.py`) | Where does the evidence for class *y* sit? | Grad-CAM on `softmax(logits)[y]` |
| **Difficulty / confidence** (`xai/gradcam.py`) | Which regions make the model unsure / sure? | Grad-CAM on `1 − max p` resp. `max p` |
| **Deliberative** (`xai/deliberative.py`) | Which regions make the model hesitate between class *a* and *b*? | `attr(a) · attr(b) · difficulty` |
| **Discriminant** (`xai/discriminant.py`) | Why class *y\** and **not** *y_c*? | `attr(y*) · (1 − attr(y_c)) · confidence` |
| **Counterfactual** (`xai/counterfactual.py`) | What would a real example of *y_c* look like, and what makes it *y_c*? | discriminant on the query image **and** on a reference image of *y_c* |

## Project structure

```
XAI_SurfaceDetection/
├── checkpoints/best_model.pth      # trained weights (state dict + class names + metrics)
├── data/
│   ├── Input/                      # raw dataset, all images flat in one folder
│   └── sorted/                     # class subfolders, created by the notebook
├── scripts/prepare_dataset.ipynb   # sorts data/Input into data/sorted by filename prefix
├── src/
│   ├── analyze_image.py            # ← main CLI: predict + explain an image or folder
│   ├── dataset.py                  # transforms, 70/15/15 split, DataLoaders
│   ├── model.py                    # DefectCNN (4 conv blocks, global avg pool, linear head)
│   ├── train.py                    # training loop, saves the best checkpoint
│   ├── evaluate.py                 # test accuracy, classification report, confusion matrix
│   ├── test_*.py                   # demo/debug scripts for the individual XAI methods
│   └── xai/                        # gradcam, region, deliberative, discriminant, counterfactual
├── outputs/                        # generated figures (analysis/, confusion_matrix.png)
├── main.py                         # smoke test: forward pass with a dummy batch
└── requirements.txt
```

## Setup

Python 3.14 was used for development; anything ≥ 3.10 should work.

```bash
cd XAI_SurfaceDetection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Quick check that the model builds:

```bash
python main.py
```

### Prepare the dataset

`data/sorted/` is what `dataset.py` reads (via `torchvision.datasets.ImageFolder`).
It is generated from the flat `data/Input/` folder by
`scripts/prepare_dataset.ipynb`, which copies each image into a class folder based
on its filename prefix (`crazing_*` → `Crazing/`, `pitted_*` → `Pitted_surface/`, …).

Run that notebook once (from `scripts/`, so its relative paths resolve) before
training, evaluating or analyzing images. The repository already ships a populated
`data/sorted/` with 300 images per class, so you can skip this step if it is
present.

## Running `analyze_image.py`

> Run it from the **project root**. The script puts its own directory (`src/`) on
> `sys.path`, which is how `from dataset import …` and `from xai… import …` resolve.

### Single image

```bash
python src/analyze_image.py --image data/sorted/Scratches/scratches_1.jpg
```

The image does not have to come from the dataset — any `.jpg/.png/.bmp/.tif` works.
It is converted to grayscale and normalized exactly like the validation data
(no resizing, so a 200×200 input is what the model expects).

### Whole folder

```bash
python src/analyze_image.py --folder data/Input
```

Every supported image in the folder is analyzed (`.jpg`, `.jpeg`, `.png`, `.bmp`,
`.tif`, `.tiff`), figures are saved without opening a window, errors on single
images are reported and skipped, and a summary of all predictions is printed at the
end.

### Custom output directory

```bash
python src/analyze_image.py --image path/to/defect.jpg --output-dir outputs/my_run
```

`--image` and `--folder` are mutually exclusive and one of them is required.
`--output-dir` defaults to `outputs/analysis/`; figures are saved as
`<image_stem>_analysis.png` at 200 dpi. With `--image` the figure is also shown
interactively (`plt.show()`); with `--folder` it is only written to disk.

If you run it on a headless machine, disable the interactive window with
`MPLBACKEND=Agg python src/analyze_image.py --image …`.

### Console output

```
Device: cpu

============================================================
Analyzing: scratches_1.jpg
============================================================

Prediction: Scratches 93.77%
Alternative: Inclusion 6.00%

All classes:
  Crazing: 0.00%
  Inclusion: 6.00%
  Patches: 0.00%
  Pitted_surface: 0.23%
  Rolled_in_scale: 0.00%
  Scratches: 93.77%

Insecurity strength: 0.975
Discriminant strength: 0.238
Counterfactual reference: Inclusion

Saved: outputs/analysis/scratches_1_analysis.png
```

*Prediction* is the top-1 class, *Alternative* the runner-up — all explanations are
computed for exactly this pair. `analyze_image()` also returns these numbers as a
dict, so it can be imported and used programmatically.

### Reading the figure

The saved PNG is a 2×5 grid:

| | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |
| --- | --- | --- | --- | --- | --- |
| **Row 1** | input image + prediction | attribution for the predicted class | attribution for the alternative class | difficulty map | ambiguity map (predicted vs. alternative) |
| **Row 2** | insecurity region + strength | discriminant map "why A, not B?" | region specific to the predicted class | counter example: a real image of the alternative class | region specific to the alternative class |

The counter example in row 2 is picked from the validation split — the first image
belonging to the alternative class (`find_reference_image`), so it is deterministic
for a given split seed.

## Training and evaluation

```bash
python src/train.py        # 30 epochs, Adam (lr 1e-3, weight decay 1e-4), CrossEntropyLoss
python src/evaluate.py     # test accuracy + classification report + outputs/confusion_matrix.png
```

`train.py` saves a checkpoint to `checkpoints/best_model.pth` **whenever the
validation loss improves**, including `epoch`, `model_state_dict`,
`optimizer_state_dict`, `val_loss`, `val_accuracy` and `class_names`. Running it
overwrites the shipped checkpoint, so back it up first if you want to keep it.

The split in `dataset.py` is 70 % train / 15 % validation / 15 % test with
`RANDOM_SEED = 42`, so training, evaluation and the counterfactual reference pool
all see the same partitioning.

## XAI demo scripts

Each explanation method has a standalone script that runs it on images from the
dataset split and opens the resulting figures with `plt.show()` (nothing is written
to disk). They take no arguments and are also run from the project root:

```bash
python src/test_gradcam.py         # plain Grad-CAM for all classes
python src/test_deliberative.py    # insecurity / ambiguity maps, also for top-k class pairs
python src/test_discriminant.py    # "why A, not B?" maps
python src/test_counterfactual.py  # query image vs. counter example
```

## Notes

* **Device:** CUDA is used automatically when available, otherwise CPU. The model
  is small enough that CPU analysis of one image takes a few seconds.
* **Checkpoint compatibility:** `DefectCNN` is instantiated with
  `num_classes=len(class_names)` from the dataset folder, so the number of class
  folders in `data/sorted/` must match the checkpoint (6).
* **`.gitignore`:** generated and environment files (`.venv/`, `__pycache__/`,
  `.idea/`, `.DS_Store`, `outputs/`, `data/sorted/`) are ignored. Files that were
  already added to the index stay tracked until you run
  `git rm -r --cached <path>` for them.

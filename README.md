# Lung Disease Prediction from Chest X-ray Images

**ML4HD project C3** — Machine Learning for Human Data, University of Padova.
Dataset: **ChestMNIST** (the MedMNIST v2 packaging of NIH ChestX-ray14).

Everything lives in one notebook: **`ChestMNIST_C3.ipynb`**. It downloads the
data, explores it, defines every model from scratch in TensorFlow, runs the
full experiment grid, produces the result tables and figures, and ends with a
live demo. Nothing else is required to run it.

No pretrained weights and no `keras.applications` models are used anywhere,
per the course rules.

---

## Running it

On a fresh GPU machine:

```bash
pip install "tensorflow[and-cuda]>=2.16" numpy scikit-learn matplotlib scikit-image ipywidgets
```

The plain `tensorflow` package has no CUDA — install the `[and-cuda]` build or
TensorFlow silently falls back to CPU and training takes days. Part 1 of the
notebook prints whether a GPU was found; check it before starting a long run.

Open the notebook and run all cells, or execute it headlessly — recommended
over SSH, since a dropped connection cannot kill the kernel:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 ChestMNIST_C3.ipynb
```

### QUICK_MODE

The config cell opens with `QUICK_MODE = True`: six small models on a
4,000-image subset at 28px, about ten minutes, enough to prove the pipeline
runs before committing hours to it. Its outputs go to `results_quick/` rather
than `results/`, so a smoke run can never be mistaken for a finished
experiment.

Set `QUICK_MODE = False` for the real grid — roughly **3 hours on an H200**, or
2 hours if you drop the 224px run (`RESOLUTIONS = [64, 128]`).

The grid cell is **resumable**: completed runs are detected on disk and
skipped, so re-running after an interruption continues rather than restarts.

---

## What the experiment does

Four axes, each varied while the others are held fixed, so every difference is
attributable to exactly one change.

| Axis | What varies | Cells |
|---|---|---|
| **A** | Architecture ladder | 6 |
| **B** | Loss function (imbalance handling) | 4 |
| **C** | Input resolution (64 / 128 / 224) | 3 |
| **D** | Preprocessing (augmentation, flips, CLAHE, normalisation) | 5 |

18 cells, but the reference cell of B, C and D is the *same configuration* as
the winner of A, so identical configurations are detected and reused.
**15 models actually get trained.**

### The architecture ladder

| Run | Change |
|---|---|
| `cnn_gap` | Plain residual CNN + global average pooling — the control |
| `cnn_se_gap` | + Squeeze-and-Excitation channel attention |
| `cnn_cbam_gap` | + CBAM channel & spatial attention (reproduces a prior submission) |
| `cnn_lq` | Label queries, no label interaction — isolates per-disease pooling |
| `cnn_lq_self` | + learned label self-attention |
| `cnn_lq_graph` | + fixed co-occurrence graph prior |

The contribution is the **label-query head**. Instead of pooling the feature
map to one vector and applying 14 independent classifiers, 14 learned label
embeddings cross-attend to the spatial feature map, so each disease pools its
own view of the image and the label representations then exchange information.

Two measured facts motivate it:

- Cardiomegaly is a global measurement (the cardiothoracic ratio) while a
  nodule is a few pixels wide. One pooled vector cannot serve both.
- The labels are strongly dependent — P(Infiltration | Edema) = 0.43,
  P(Infiltration | Pneumonia) = 0.42, P(Effusion | Cardiomegaly) = 0.38,
  P(Pneumothorax | Emphysema) = 0.30. Plain BCE assumes independence.

It also yields **per-disease attention maps in one forward pass**, where a
pooling model needs one Grad-CAM backward pass per class.

---

## Why accuracy is not the headline number

Predicting "no finding" for all 14 labels on every test image scores
**~94.7% binary accuracy**. That is why every method in MedMNIST v2's Table 3
reports ACC ≈ 0.947 while their AUCs range from 0.649 to 0.778.

We report macro/micro **AUC-ROC**, **average precision** (far more honest at
0.18% prevalence — Hernia has 144 positives in 78,468 training images), and F1
at thresholds **tuned on validation, never on test**.

### Reference numbers

MedMNIST v2 Table 3, ChestMNIST test macro AUC:

| Method | AUC |
|---|---|
| ResNet-18 (28) | 0.768 |
| ResNet-18 (224) | 0.773 |
| ResNet-50 (224) | 0.773 |
| AutoKeras | 0.742 |
| Google AutoML Vision | 0.778 |

Results are also reported on the 8-pathology subset for direct comparison with
ChestX-ray8 Table 3 (ResNet-50 + W-CEL, mean AUC ≈ 0.696).

Realistically, **~0.78–0.82 macro AUC is the ceiling without pretraining.**
This project is framed around the architectural and imbalance findings, not a
headline accuracy number.

---

## Output

The notebook writes:

```
data/              the .npz archives (~5.7 GB, downloaded automatically)
results/           one JSON per run: config, history, per-class metrics, cost
checkpoints/       one trained model per run
figures/           AUC-vs-cost frontier, per-class AUC, training curves
report/            LaTeX tables, ready to \input
```

Everything except `data/` and `checkpoints/` is small enough to commit back.

# Applied Machine Learning — 2-Hour Hands-On Workshop Plan

A compressed, **notebook-driven** session that walks through the full supervised ML workflow end to end. The primary material is [`workshop.ipynb`](workshop.ipynb) (pneumonia detection with a CNN). Essential concepts from [`1_overview.ipynb`](1_overview.ipynb) are introduced in Block 1 of the notebook — participants do **not** run notebook 1 during the session.

**Duration:** 2 hours (120 min), no break assumed. Add 10 min buffer if Q&A is expected.

---

## What participants will do

Run (or follow live) the pneumonia notebook from `DATA_ROOT` setup through test-set evaluation. By the end they will have seen every stage of a real PyTorch project:

```
Problem → Data → Explore → Preprocess → Model → Train → Validate → Test → Report
```

---

## Primary notebook cell map

Use this as the run order during the session. Cell numbers refer to [`workshop.ipynb`](workshop.ipynb).

| Cells | Section | Workshop block |
|-------|---------|----------------|
| 0–1 | Intro & ML workflow concepts | Block 1 |
| 2–8 | Config, imports, load data, class counts, preview | Block 2 |
| 9–12 | Normalize, `PneumoniaDataset`, `DataLoader` | Block 3 |
| 13–17 | `PneumoniaCNN`, training loop, test accuracy | Block 4 |
| 18–23 | Learning curves, metrics, confusion matrix, error images | Block 5 |

---

## Session overview

| Block | Time | Focus |
|-------|------|-------|
| 1. ML workflow & problem framing | 10 min | Essential concepts (no coding) |
| 2. Load & explore data | 20 min | Cells 2–8 |
| 3. Preprocess & augment | 20 min | Cells 9–12 |
| 4. Build, train, test | 45 min | Cells 13–17 |
| 5. Evaluate & wrap-up | 25 min | Cells 18–23 |

---

## Learning outcomes (2-hour scope)

Participants will be able to:

- Name the stages of a supervised ML workflow and what happens at each stage.
- Load image data with train / validation / test splits and inspect class balance.
- Explain why we normalize pixels and augment training images only.
- Read a simple CNN in PyTorch (`Conv2d`, `BatchNorm`, `MaxPool`, `Dropout`, `Linear`).
- Describe model, loss function, optimizer, and epoch in the training loop.
- Interpret accuracy, a classification report, and a confusion matrix on held-out test data.

*Out of scope for 2 hours:* linear regression from scratch, hyperparameter sweeps, transfer learning (`bonus-finetuning.ipynb`), and extended practice exercises. Point to [`1_overview.ipynb`](1_overview.ipynb) for deeper theory.

---

## Prerequisites

**Before the session (required)**

1. Python 3.10+ with Jupyter or VS Code notebooks.
2. Install: `numpy`, `pandas`, `matplotlib`, `torch`, `torchvision`, `scikit-learn`, `seaborn`, `Pillow`, `tqdm`.
3. Download [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and set `DATA_ROOT` in cell 2.
4. **Smoke-test** cells 2–9 locally so data paths work before the workshop.

**Facilitator prep (strongly recommended)**

- Pre-run cells 21–24 once so training time is known on your hardware.
- If 12 epochs exceed ~25 min on CPU, set `NUM_EPOCHS = 4` for the live demo and share the full-run notebook output separately.
- Have learning-curve and confusion-matrix plots ready as fallback if training overruns.

---

## Block 1 — ML workflow & problem framing (10 min)

**Material:** Discussion only. One slide or whiteboard optional. Concepts align with the opening of [`1_overview.ipynb`](1_overview.ipynb).

### The workflow (draw or show once)

| Step | Pneumonia notebook example |
|------|----------------------------|
| Define problem | Binary classification: pneumonia vs normal on chest X-ray |
| Choose metric | Accuracy, precision/recall per class (clinical cost of false negatives) |
| Prepare data | Load JPEGs, normalize, augment, batch with `DataLoader` |
| Choose model | CNN (`PneumoniaCNN`) — images need spatial feature learning |
| Train | Minimize `BCEWithLogitsLoss` with `RMSprop` over epochs |
| Validate | Monitor val loss/accuracy each epoch; scheduler adjusts LR |
| Test | Final metrics on held-out test set — **once**, at the end |
| Report | Plots, confusion matrix, inspect misclassified images |

### Essential concepts (60-second each)

- **Supervised learning** — learn from labeled examples (`PNEUMONIA` / `NORMAL` folders).
- **Model** — parameterized function (here: CNN weights).
- **Loss** — number to minimize; wrong predictions increase loss.
- **Optimizer** — updates weights using gradients from the loss.
- **Epoch** — one pass through the training set.
- **Train / val / test** — train to learn; val to tune and monitor; test to estimate real-world performance.

### Checkpoint

> *What is the real-world goal, and what proxy metric does this notebook use?*

---

## Block 2 — Load & explore data (20 min)

**Notebook:** [`workshop.ipynb`](workshop.ipynb) cells **2–8**

### Run (facilitator leads, participants follow)

1. **Cell 3** — Set `DATA_ROOT`, `MAX_SAMPLES_PER_CLASS`, `NUM_EPOCHS`.
2. **Cell 4** — Imports (`torch`, `torchvision`, `sklearn`, etc.).
3. **Cell 6** — `get_training_data()`, image size 150×150; load `train`, `val`, `test`.
4. **Cell 7** — Bar chart of class counts → **imbalance** (more pneumonia than normal in training).
5. **Cell 8** — Display one pneumonia and one normal image.

### Concepts learned by running

| Code | Idea |
|------|------|
| Folder layout `train/val/test` | Standard split; test untouched until Block 5 |
| `get_training_data()` | Files → `(image_array, label)` pairs |
| Class count plot | Imbalanced data affects metrics; motivates augmentation |
| `cmap='gray'` | Single-channel X-rays |

### Facilitator script (while plots render)

- Data was QC'd by physicians (see cell 5 markdown) — **cleaning** is part of the workflow.
- Labels come from folder names — **encoding** classes as 0/1 happens in the loading code.

### If short on time

- Keep `MAX_SAMPLES_PER_CLASS = 50` in the configuration cell for faster loads.

---

## Block 3 — Preprocess & augment (20 min)

**Notebook:** cells **9–12**

### Run

1. **Cell 11** — Split loaded lists into `x_train`, `y_train`, etc.; divide by 255 → pixels in **[0, 1]**.
2. **Cell 12** — `PneumoniaDataset` with `transforms`: rotation, zoom, shift, flip; build `DataLoader`s.

### Concepts

| Technique | Why |
|-----------|-----|
| Normalize / 255 | Stable gradients; faster CNN convergence |
| Augmentation (train only) | Artificially expand data; fight overfitting |
| `Dataset` + `DataLoader` | Batched, shuffled input to the training loop |
| `shuffle=True` (train) | Stochastic training; different batches each epoch |

### Discussion (2 min)

- Augment **training** data only — val/test must reflect real deployment images.
- For medical imaging: aggressive rotation may be unsafe; note the trade-off.

### Checkpoint

> *Why do we divide by 255? Why not augment the test set?*

---

## Block 4 — Build, train & test (45 min)

| Architecture | Strength | Limitation |
|--------------|----------|------------|
| MLP | Simple, fast on tabular data | Ignores spatial/temporal structure |
| CNN | Strong on images and grids | Needs more data than linear models |
| RNN/LSTM | Captures temporal dependencies | Harder to train on long sequences |
| GNN | Models irregular relationships | Requires careful graph construction |
| U-Net | Precise pixel-level outputs | Needs detailed mask annotations |

### 6.3 RNN / LSTM — Recurrent networks

```{figure} https://upload.wikimedia.org/wikipedia/commons/b/b5/Recurrent_neural_network_unfold.svg
:alt: RNN unrolled over time
:width: 560px
:align: center

RNN folded (left) and unrolled across time steps (right). Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg).
```

```{figure} https://upload.wikimedia.org/wikipedia/commons/5/56/LSTM_cell.svg
:alt: LSTM cell
:width: 380px
:align: center

LSTM cell with gates controlling information flow. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_cell.svg).
```

**Data flow (RNN):** at each time step $t$: $(x_t, h_{t-1}) \rightarrow h_t \rightarrow y_t$


| Component                  | Data in → Data out                                             |
| -------------------------- | -------------------------------------------------------------- |
| **Input $x_t$**            | Measurement at time $t$ (e.g. sensor reading, word token)      |
| **Hidden state $h_{t-1}$** | Summary of all prior steps — passed forward in time            |
| **Recurrent cell**         | Combines $x_t$ and $h_{t-1}$ with *shared* weights → new $h_t$ |
| **Output $y_t$**           | Prediction at step $t$ (next value, class label, etc.)         |


The diagram shows the network **unrolled**: the same cell is reused at every step (left: folded; right: unfolded across time).

**LSTM** adds a **cell state** and **gates** (forget, input, output) that control what information is kept or discarded — better for long sequences where plain RNNs lose context.

**Data input examples**


|                | Shape / format         | Example                                                                                           |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------- |
| **Usual**      | `(seq_len, 1)`         | Univariate time series: daily website traffic → predict next day                                  |
| **Usual**      | `(seq_len, embed_dim)` | Text: token sequence embedded with word vectors                                                   |
| **Scientific** | `(seq_len, n_vars)`    | Weather station: hourly `[temp, humidity, wind_speed, pressure]` over a week → forecast next 24 h |
| **Scientific** | `(seq_len, 4)`         | DNA / protein: one-hot or learned embeddings per base or amino acid along a sequence              |


### 6.4 GNN — Graph Neural Network

```{figure} https://upload.wikimedia.org/wikipedia/commons/e/ec/Message_Passing_Neural_Network.png
:alt: GNN message passing
:width: 500px
:align: center

Message passing between graph nodes. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Message_Passing_Neural_Network.png).
```

**Data flow:** node features + edge connections → **message passing** → updated node embeddings → graph-level or node-level prediction


| Step            | Data in → Data out                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------- |
| **Graph input** | Nodes (atoms, mesh points) with feature vectors; edges (bonds, adjacency) define who talks to whom |
| **Message**     | Each node gathers information from its neighbors along edges                                       |
| **Aggregate**   | Neighbor messages are combined (sum, mean, max)                                                    |
| **Update**      | Node embedding is refreshed using its old state + aggregated messages                              |
| **Readout**     | Pool all node embeddings for a graph-level prediction, or predict per-node labels                  |


Unlike CNNs (regular grids) or MLPs (fixed vectors), GNNs handle **irregular structure** — molecules, citation networks, finite-element meshes.

**Data input examples**


|                | Structure     | Example                                                                                                           |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Usual**      | Nodes + edges | Recommendation graph: users and items linked by clicks                                                            |
| **Scientific** | Nodes + edges | **Molecule:** C, O, N atoms as nodes; single/double bonds as edges → predict solubility or reactivity             |
| **Scientific** | Nodes + edges | **Simulation mesh:** mesh vertices as nodes, adjacency from elements → predict stress or temperature at each node |


### 6.5 U-Net — Encoder–decoder for segmentation

```{figure} https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png
:alt: U-Net architecture
:width: 520px
:align: center

U-Net encoder–decoder with skip connections. Source: [U-Net paper (Ronneberger et al., 2015)](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
```

**Data flow:** image → **encoder** (contract) → **bottleneck** → **decoder** (expand) → per-pixel mask


| Path                 | Data in → Data out                                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Encoder (left)**   | Repeated Conv + MaxPool: spatial size shrinks, channel depth grows (captures context)                                            |
| **Bottleneck**       | Smallest spatial map with richest semantic features                                                                              |
| **Decoder (right)**  | Upsampling (transposed conv) restores spatial resolution                                                                         |
| **Skip connections** | High-resolution encoder features are **concatenated** into the decoder — preserves fine boundaries (cell edges, lesion outlines) |
| **Output**           | One label per pixel (segmentation mask)                                                                                          |


Same conv layers as a CNN, but the U-shape and skip connections target **pixel-wise** output rather than a single class label.

**Data input examples**


|                | Input → output                            | Example                                                                               |
| -------------- | ----------------------------------------- | ------------------------------------------------------------------------------------- |
| **Usual**      | `(3, H, W)` image → `(H, W)` class mask   | Autonomous driving: label each pixel as road, car, pedestrian                         |
| **Scientific** | `(1, H, W)` grayscale → `(H, W)` mask     | Microscopy: segment individual cells in a brightfield image                           |
| **Scientific** | `(C, H, W)` multispectral → `(H, W)` mask | Remote sensing: land-cover class per pixel from many spectral bands                   |
| **Scientific** | `(C, H, W)` field stack → `(H, W)` mask   | Climate / weather: delineate storm regions or ice extent from stacked variable fields |



**Notebook:** cells **13–17** — **core of the workshop**

### Run

1. **Cell 15** — Read `PneumoniaCNN` architecture (print `model`); point out:
   - `Conv2d` + `ReLU` — learn local patterns
   - `BatchNorm2d` — stabilize training
   - `MaxPool2d` — downsample
   - `Dropout` — regularization / reduce overfitting
   - `Linear` — final binary output
2. **Cell 15** — `BCEWithLogitsLoss`, `RMSprop`, `ReduceLROnPlateau`.
3. **Cell 16** — `run_epoch()`:
   - `model.train()` vs `model.eval()`
   - `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
4. **Cell 16** — Training loop: `NUM_EPOCHS` (default 6; set in configuration cell).
5. **Cell 17** — Test-set loss and accuracy.

### Training loop anatomy (talk through `run_epoch`)

```text
for each batch:
    forward pass  → outputs
    compute loss  → compare outputs to labels
    backward pass → gradients
    optimizer.step() → update weights
```

Link to [`1_overview.ipynb`](1_overview.ipynb): same pattern as SGD on linear regression — loss, gradient, update — but automated by PyTorch autograd.

### While epochs run (~15–30 min depending on hardware)

Use the wait time for **concept Q&A**, not new code:

- Batch size and learning rate (mention; do not sweep).
- Overfitting: train acc ↑ but val acc flat → need more data, augmentation, or regularization.
- Why `scheduler.step(val_acc)` uses validation, not training, accuracy.

### If training is too slow

| Option | Trade-off |
|--------|-----------|
| `NUM_EPOCHS = 4` | Noisier curves; still demonstrates workflow |
| Smaller subset via `train[:200]` | Faster; mention this is demo-only |
| Facilitator pre-trained checkpoint | Load weights; run cells 24–36 only |

### Checkpoint

> *Name the four steps inside one training batch.*

---

## Block 5 — Evaluate & wrap-up (25 min)

**Notebook:** cells **18–23**

### Run

1. **Cell 20** — Plot train vs val accuracy and loss over epochs.
2. **Cell 21** — Test predictions; `classification_report` (precision, recall, F1 per class).
3. **Cell 22** — Confusion matrix heatmap.
4. **Cell 23** — Six correct and six incorrect predictions — **qualitative error analysis**.

### Evaluation essentials

- **Test set** used once — no tuning on test labels.
- **Accuracy alone** can mislead when classes are imbalanced — read per-class recall.
- **False negatives** (missed pneumonia) may be costlier than false positives in clinical use — connect metric to problem context.

### Model selection checklist (5 min — verbal)

1. Define problem and metric first.
2. Start with a clear baseline (even a simple CNN counts).
3. Respect train / val / test splits.
4. Monitor train vs val gap for overfitting.
5. Report test metrics and show failure cases.

### Closing (3 min)

Each participant names:

- One workflow step they will reuse in their project.
- One follow-up from [`1_overview.ipynb`](1_overview.ipynb) (SGD, learning rate) if they want theory depth.

---

## Facilitator quick reference

### 2-hour pacing

| Behind schedule | Ahead of schedule |
|-----------------|-------------------|
| Skip error-image cells (34, 36) | Ask: what hyperparameter would you change first? |
| Show pre-computed training plots | Reduce `NUM_EPOCHS` discussion; preview `bonus-finetuning.ipynb` |
| Verbal summary instead of live confusion matrix | Brief demo of one misclassified image only |

### Common issues

| Issue | Fix |
|-------|-----|
| `DATA_ROOT` not found | Must contain `train/`, `val/`, `test/` subfolders |
| CUDA OOM | Lower batch size in `DataLoader` |
| Val accuracy ~50% | Check labels, class mapping, sufficient epochs |
| Loss NaN | Reduce learning rate or check normalization |

### Optional take-home (not in session)

- [`1_overview.ipynb`](1_overview.ipynb) — linear regression + SGD + batch-size / learning-rate exercises.
- [`bonus-finetuning.ipynb`](bonus-finetuning.ipynb) — ResNet-18 transfer learning.
- Re-run notebook 2 with a changed augmentation or one extra Conv block.

---

## Pre-workshop email (template)

> **2-hour Applied ML workshop — please complete before we meet:**
>
> 1. Clone the repo and install dependencies.
> 2. Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
> 3. Open `docs/source/tutorial/workshop.ipynb`, set `DATA_ROOT` in the configuration cell, and run through the data-loading cells successfully.
>
> We will run the **full ML workflow** in this single notebook. Theory from `1_overview.ipynb` will be summarized at the start; work through that notebook afterward for deeper background.

---

## Appendix — Concept source map

| Concept | Covered in session | Deeper dive |
|---------|-------------------|-------------|
| Why use ML in science | Block 1 (verbal) | `1_overview` cells 0 |
| Problem definition & metrics | Blocks 1, 5 | `1_overview` cells 0 |
| SGD, loss, gradients | Block 4 (link) | `1_overview` SGD section |
| Batch size, learning rate | Block 4 (mention) | `1_overview` practice section |
| Feature engineering, scaling, augmentation | Block 3 | `workshop.ipynb` cells 11–12 |
| CNN architecture | Block 4 | `workshop.ipynb` cell 15 |
| Train / val / test, metrics | Blocks 2, 5 | `workshop.ipynb` throughout |
| Transfer learning | Take-home | `bonus-finetuning.ipynb` |

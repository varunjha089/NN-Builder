# Future plan

What is still open, in the suggested order. Done work is in git and CLAUDE.md; this file lists only what is left.

## 1. Needs the owner (cannot be done from here)

- **Stable signing certificate.** Create a self-signed code-signing certificate in Keychain Access; the build then uses it through an
  environment variable (never in the repository). Ends the Keychain "Always Allow" question after every update.
- **Release 0.4.0.** Phases 2–4, smoother running and the text language are installed on the owner's Mac but not released; the
  Windows / Linux / Intel downloads are still 0.3.0. It is also the first real test of the automatic updater (Check for Updates →
  Install and restart).
- **First real runs of the new training features** (only CI has run them, on a CPU): ImageNet weights, Grad-CAM and the mistakes
  gallery, TorchScript / ONNX export, bfloat16, channels-last, the automatic batch split on a real out-of-memory, save memory, the
  measured memory and data-wait figures. A short Kaggle run of the ResNet-18 demo with ImageNet weights and export would cover most.

## 1b. Handling more kinds of architecture (asked 2026-11-05; in this order, each building on the one before)

1. **Sentinel-1/2 data** (section 5): images with 2-32 bands from GeoTIFF / .npy, band choice, per-band normalisation, radar in dB,
   composites for display; then multi-label (BigEarthNet). *Done (2026-11-05): N-band input from Kaggle .tif / .npy folders, band choice, per-band statistics, dB, composites (CI-tested).
   Also done (2026-11-05): own files ("My own files" source, e.g. Google Earth Engine exports): folder on the Mac copied to the machine
   (rsync, signature) or uploaded as a private Kaggle dataset, or a folder already on the machine; GEE band names (B4 = B04), NaN pixels,
   S1 already in dB recognised; multispectral segmentation (images + masks, and a split scene + label image cut into tiles, no-data
   labels ignored); a generated GEE export script (Code Editor for scene + WorldCover labels, Python for class patches; not run against a
   real GEE account). Next in this step: multi-label (BigEarthNet), Hugging Face band arrays, first-conv expansion for ImageNet weights,
   a spatially separate test region for scenes.*
   **Plan for the next two parts (2026-11-06, being built):**
   - *Multi-label* (several classes per image, BigEarthNet): task "multilabel". Labels from a CSV in the dataset folder (own files or
     Kaggle): a file column plus either one column of class names separated by `;` `,` `|` or spaces, or one 0/1 column per class;
     class names read from the CSV on the Mac. The Output block stays (one logit per class, no softmax); loss = BCE with logits
     (label smoothing pulls targets towards 0.5), prediction = sigmoid ≥ 0.5. Scores: micro F1 (the run's main number, "val_f1"),
     macro F1, mAP (average precision per class), exact match; per-class precision / recall / F1 / AP in Results; early stopping
     monitors val_f1. Works with ordinary pictures and with bands.
   - *S1 + S2 fusion*: a second band group (bands2, band_order2, sensor2, s1_db2) read from a second folder with the same relative file
     names (any extension), e.g. `s2/forest/x.tif` + `s1/forest/x.npy`; scenes: a second image folder. Early fusion: one Input
     receives both groups stacked (S2 bands, then S1 bands). Mid fusion: two Input blocks, `group` 1 and 2 (canvas: Input inspector;
     text: `input 1`, `input 2`), each a branch, joined by Concat / Add; the generated forward slices the stacked tensor per group, so
     the data path stays one tensor. Sentinel-1 dB per group; statistics per band; pictures from the first group.
     Real multi-input (different sizes / sources per Input) stays step 2.
2. **Several inputs and outputs:** more than one Input block (each bound to a data source or a band group: S1 + S2 fusion, Siamese
   pairs) and more than one Output (multi-task heads, each with its loss and weight); shapes, codegen, text language and harness.
3. **Weight sharing:** a block that reuses another block's weights (Siamese / twin networks).
4. **Losses and custom blocks:** BCE, focal, Dice, MSE / L1 (regression), contrastive / triplet per output; a "custom PyTorch" block
   (code plus a declared output shape, checked on the GPU machine by the one-batch check).
5. **Transformer additions:** cross-attention (two inputs), attention masks (causal / padding), sinusoidal and rotary positions, RMSNorm.
6. **Other data:** Conv1d / Conv3d and their pools (signals, volumes, video), LSTM / GRU, token embeddings for text.
7. **Object detection:** an anchor-free head (FCOS / CenterNet style) with box losses and mAP, and box datasets.
8. **Dynamic structures:** mixture-of-experts layers (top-k routing), early-exit heads.
9. **Multi-GPU training:** DistributedDataParallel / FSDP on machines with several GPUs (college server, multi-GPU EC2).

## 2. The text language (details in [PROGRAMMING_LANGUAGE.md](PROGRAMMING_LANGUAGE.md), section 9)

- A changed block that a function or loop made (on the canvas) still prints the text again; settings changes, moves, and blocks added,
  removed or rewired on plain lines already keep the hand-written text.
- Completion of block words and settings while typing (colours and ⌘-click on numbers are done).
- The text form on the clipboard next to copied blocks.
- Import a torchvision model as text; variables for sweeps; a diff of two designs by block number.

## 3. Models and training

- **MobileNet v2 / EfficientNet-B0 demos with ImageNet weights** (the plan named MobileNet; the current MobileNet demo is v1, which
  torchvision has no weights for).
- Mixup / CutMix and RandAugment; a class-weighted loss for imbalanced data (the Dataset tab already reads the class balance).
- A hyperparameter sweep (grid over learning rate and batch size) on the seed-group machinery.
- `torch.compile` as an option.

## 4. Workflow and window

- Very large designs (thousands of blocks): the canvas costs ~0.3 MB per block. Ideas: a lighter block drawing (no glow, no edge labels)
  above ~1,000 blocks; editing inside the ×N view (it is read-only today).

- A run queue (several runs one after another); a macOS notification when a run ends; notes on each run.
- Click a confusion-matrix cell to see those images.
- Kaggle's 20 GB output limit: warn before a run would exceed it.
- The layer table's automatic groups (from shapes) suit plain CNNs and ViTs; hybrid designs drawn on the canvas can get odd group names
  (designs written as text are grouped by their lines).

## 5. Sentinel-1 and Sentinel-2 datasets

### Why the current data path is not enough

Today a dataset has 1 or 3 channels (`dataset.rs` refuses anything else). Every image goes through PIL as grey or RGB, is normalised with one
fixed mean/std, and has one label. Sentinel data needs:

- **Many bands.** Sentinel-2 has 13 bands (usually 10–12 are used). Sentinel-1 has 2 (VV and VH radar).
- **Float values that PIL cannot read.** Sentinel-2 is 16-bit reflectance (0–10000). Sentinel-1 is radar backscatter, linear or in dB.
  Files come as GeoTIFF, `.npy`, HDF5 or array columns.
- **Normalisation per band**, not one ImageNet-style mean.
- **Several labels per image** for the big benchmarks (BigEarthNet: 19 classes per patch).

### Datasets to target (confirm the exact Hugging Face / Kaggle ids when starting)

| Dataset | Sensor | Task | Size | Why |
|---|---|---|---|---|
| EuroSAT-MS | S2, 13 bands, 64×64 | 10 classes | 27k | Small and fast: the first test |
| So2Sat LCZ42 | S1 (8 ch) + S2 (10 ch), 32×32 | 17 local climate zones | ~400k | Paired S1 + S2: the clean fusion test |
| BigEarthNet v2 (reBEN) | S1 (2) + S2 (12), 120×120 | 19 labels per image | ~550k | The standard paper benchmark; needs multi-label |
| Sen1Floods11 | S1 (+ S2) | flood segmentation | ~4.8k chips | S1 segmentation through the existing U-Net path |

### Phases

**S1: N-band input (core)**
- `Dataset.channels` becomes any number from 1 to 32, plus a `bands` list (e.g. `["B02","B03","B04","B08",…]`) and a `sensor`
  (`s1` | `s2` | `other`).
- Harness: a loader for `.tif` (rasterio / tifffile), `.npy`, HDF5 and Hugging Face array columns that reads float tensors with no PIL step.
- Band selection with presets ("S2 10 m + 20 m", "S1 VV + VH").
- Sentinel-1 option: convert linear values to dB and clip.
- Per-band mean/std, computed on the GPU machine from a sample of the training images, cached next to the data and saved in the
  checkpoint so a resumed run normalises the same way.
- Augmentation on tensors: flips and 90° rotations (safe for satellite images). Colour jitter is off above 3 channels.

**S2: Multi-label**
- Task "multi-label": sigmoid + BCE loss, a threshold of 0.5.
- Metrics: micro/macro F1 and mAP instead of accuracy; per-class F1 in Results; early stopping on `val_f1`.

**S3: S1 + S2 fusion**
- Early fusion first: stack the S1 and S2 bands into one input (works with the current single Input block).
- Mid fusion later: two Input blocks, one branch each, joined by Concat. This needs multi-input support in `shapes.rs` and codegen
  (the bigger change).

**S4: Pretrained weights with more than 3 channels**
- ImageNet weights expect 3 channels. The first convolution gets the RGB weights for B04/B03/B02, and the mean of those weights for the
  other bands (the usual trick).
- Option: Sentinel-pretrained ResNet/ViT weights (SSL4EO-S12, SatMAE style), if their licences allow.

**S5: Display and teaching**
- Samples, Grad-CAM and mistakes show a true-colour (B04/B03/B02) or false-colour (B08/B04/B03) composite, made on the GPU machine.
  Sentinel-1 is shown as a VV / VH / ratio composite.
- Dataset tab: a band table (wavelength, resolution) and a size warning (BigEarthNet is about 60–120 GB: Kaggle's disk and its 12-hour
  limit matter).
- A demo: "EuroSAT-MS ResNet-18 (13 bands)".

### Tests

- Rust: shapes and codegen with 13 and 2 channels; `clean()` rules for bands and sensor.
- CI (never on the Mac): synthetic 13-band GeoTIFFs and `.npy` files through the real harness (loading, per-band statistics, multi-label
  BCE, composite images, first-convolution expansion for pretrained weights). Nothing is downloaded or trained on the Mac.

### Suggested order

S1 with EuroSAT-MS, then S2 (BigEarthNet labels), then S4, S3 and S5. S1 + S2 together already give a usable 13-band classifier.

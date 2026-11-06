# Future plan

What is still open, in the suggested order. Done work is in git and CLAUDE.md; this file lists only what is left.

## 0. Next six tasks (asked 2026-11-06, in this order)

1. **Check data** (done 2026-11-06): a run that only loads the data and runs the one-batch check on the training machine, no training;
   it reports files per class, band names and value ranges per band, empty pixels, CSV rows without a file, already-dB radar,
   constant or all-empty bands, class balance, and shows a few composites; minutes instead of a wasted GPU hour.
2. **Unbalanced classes** (done 2026-11-06): class counts from Read folder / Read labels in the Dataset tab, an option to weight the loss by class
   (inverse frequency; BCE pos_weight for multi-label), and per-class thresholds for multi-label chosen on the validation data.
3. **Finish the Sentinel step** (done 2026-11-06): a spatially separate test region for scenes (honest scores), BigEarthNet's own layout (a folder of
   single-band GeoTIFFs per patch), ImageNet weights for more than 3 bands (first convolution expanded), Hugging Face band arrays.
4. **Google Drive / Cloud Storage source** (done 2026-11-06; real buckets / links not tried): Earth Engine exports reach the training machine directly (rclone / gsutil on the machine),
   without passing through the Mac.
5. **Real-data run** (done 2026-11-06 for Hugging Face giswqs/EuroSAT_MS, 13 bands, on a Kaggle T4): Check data passed (no warning, 10
   classes, realistic band values); 5 epochs of a 143,594-parameter CNN: test accuracy 92.1 % (top-5 99.8 %), 17 s per epoch, 0.15 GB
   GPU memory. Found and fixed: Kaggle refused notebooks whose title was over 50 characters (400 Bad Request). Seen: data_wait 0.65
   (the GPU waits for the data loader about two thirds of the time: Hugging Face band arrays decoded per image). Still not tried: an
   Earth Engine export of the owner's own (the GEE script, own files upload to Kaggle).
6. **Release 0.4.0** (released 2026-11-06) with housekeeping: upload-artifact action update (Node.js 20 warning), delete the local backup branch, the first
   real updater test (see section 1).

## 0b. Why the architecture steps (section 1b) are in this order

- First the real-data run and release 0.4.0 (section 0, tasks 5 and 6): nothing below is worth building on untested data paths.
- Losses (2) are the smallest change and help the owner's segmentation now; they also define the per-output loss that steps 4 and 8 use.
- The custom block (3) comes next because it unblocks everything the app lacks while the rest is built, and its one-batch shape check
  is reused by every later block.
- Several inputs / outputs (4) is the big structural change (a batch becomes a dict); weight sharing (5), cross-attention (7) and
  detection heads (8) all need it, so it goes before them.
- Time series (6) after 4–5 because it reuses the multi-input data path and pairs; it is the step most useful for remote sensing.
- Transformer additions (7) after 6, because masks are needed there for missing dates and cross-attention needs 4.
- Detection (8) needs losses (2), several heads (4) and a new dataset kind, so it is late.
- Multi-GPU (9) is independent of the model; it pays off once large ViTs and time series exist.
- Fine-tuning and MoE (10) are refinements on top of everything else.

## 1. Needs the owner (cannot be done from here)

- **Stable signing certificate.** Create a self-signed code-signing certificate in Keychain Access; the build then uses it through an
  environment variable (never in the repository). Ends the Keychain "Always Allow" question after every update.
- **First real test of the automatic updater.** 0.4.0 is released (2026-11-06); the app installed on the owner's Mac is the same code
  but still says 0.3.0, so Help → Check for Updates → Install and restart there is the test.
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
   **Multi-label and S1 + S2 fusion (2026-11-06): built as planned below** (CI trains a mid-fusion S2 + S1 set whose classes differ only in
   the radar, and a BigEarthNet-like multi-label set with early fusion). Left in this step: Hugging Face band arrays, first-conv
   expansion for ImageNet weights, a separate test region for scenes, BigEarthNet's own layout (one folder of per-band GeoTIFFs per
   patch). The plan as written before building:
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
Reordered 2026-11-06 (the owner asked "can it handle all modern architectures?"). Today it covers 2-D image classification,
multi-label and segmentation: CNNs (VGG / ResNet / MobileNet-style), U-Net, ViTs from Tiny to g/14, hybrids, ImageNet weights, S1 + S2
fusion. What limits it is less the block list than the fixed training setup: one input, one output, cross-entropy / BCE only, no
escape hatch, 2-D only, supervised only, one GPU. The steps below, in this order, each building on the ones before
(why the order: section 0b):

2. **Losses:** Dice, focal, Dice + CE, MSE / L1 / Huber (regression, a "regress" task), chosen in the Train tab; Dice / focal first,
   for unbalanced Sentinel segmentation. Harness `loss_of` gains the cases; class weights keep working. *Dice, Dice + CE and focal done
   (2026-11-06); MSE / L1 / Huber with a "regress" task (numbers per image from table columns) done the same day; both CI-tested, not
   yet on a real run. Left: per-pixel regression (canopy height / biomass maps: a number per pixel from a raster).*
3. **Custom PyTorch block (escape hatch):** *Done 2026-11-06 (CI-tested; not yet on a real run): the code is one per design
   (Graph.python, a python """ … """ section in the text), the output shape is a "# out:" line in the class; custom loss in the Train
   tab. Left: Custom blocks with several inputs, copy/paste of blocks carrying their class into another design.* The plan was: a block holding a few lines of PyTorch (`__init__` and `forward` bodies) plus a declared
   output shape (formula in the input shape); shapes trusts the declaration, codegen pastes the code into the model class, the
   one-batch check on the GPU machine compares the real shape with the declared one. Also a custom loss (a function of logits and
   targets). Saved in the .nnb and the text language; no Python ever runs on the Mac.
4. **Several inputs and outputs:** *Several outputs done 2026-11-06 (multi-task heads from the labels table / HF columns, one
   Output per head, per-head loss and weight, per-head scores) and inputs at their own sizes (the second band group at its own image
   size, one Input each) the same day; both CI-tested, not yet on real data. Left: more than two inputs, inputs from different sources
   (not twin files), segmentation + extra outputs, per-head class weights.* The plan: more than one Input (each bound to a data source or band group; inputs of different sizes, e.g.
   10 m S2 and 20 m bands, or S1 and S2 at their own sizes) and more than one Output (multi-task heads, each with its task, loss
   and weight, and its own scores in Results); shapes, codegen, text language, harness (a batch becomes a dict of tensors).
5. **Weight sharing:** *Done 2026-11-06 (CI-tested, not yet on real data): Shared-weights block, Twin branch button, Subtract,
   before/after pairs as a second band group with the same bands. Left: a pair layout of its own (two files listed in a table, or two
   dates in one file), change-map datasets like OSCD tried for real, contrastive / triplet losses.* The plan: a block that reuses another block's weights (or a whole branch: Siamese / twin networks); with step 4 this
   gives change detection from before / after image pairs (OSCD-style), with a pair dataset layout.
6. **Time dimension:** *Done 2026-11-06 (CI-tested, not yet on real data): time-series datasets (dates per sample, .npy or GeoTIFF
   date by date), Conv3D, Temporal pool (mean / max / attention, U-TAE-like), To sequence, Conv1D, LSTM / GRU, Token pool last; the
   transformer blocks work along the dates. Left: real dates (date encodings, uneven spacing), missing dates (masks: step 7), the Earth
   Engine script for a monthly stack, Hugging Face time series, a second band group with a time series, PASTIS tried for real.* The plan: time-series data (a stack of dated Sentinel-2 / Sentinel-1 images per place: a folder per patch, or a band
   axis of T×C in one GeoTIFF; Earth Engine script for monthly composites), Conv3d / Conv1d and their pools, temporal attention over
   the date axis per pixel (U-TAE-style) and over tokens (TSViT-style), LSTM / GRU, date encodings; per-pixel crop-type
   segmentation (PASTIS-style) and per-patch classification.
7. **Transformer additions:** *Done 2026-11-06 (CI-tested): causal masks, rotary and sinusoidal positions, RMSNorm, SwiGLU,
   cross-attention, token concat; empty dates of a time series filled from their neighbours. Left: padding masks (sequences of different
   lengths), token embeddings for text (no text datasets in the app), KV-cache / generation.* The plan: cross-attention (two inputs, needs step 4), attention masks (causal / padding; needed by step 6 for
   missing dates), sinusoidal and rotary positions, RMSNorm, gated MLP (SwiGLU); token embeddings for text.
8. **Object detection:** *Done 2026-11-06 (CI-tested, not yet on real data): YOLO-format datasets, CenterNet-style Detection output,
   focal + L1 loss, mAP@0.5 and @0.5:0.95, boxes drawn in Results. Left: oriented boxes (DOTA), COCO JSON, tiling large scenes with
   their boxes, cross-class NMS, a detection demo design, multi-scale heads (FPN).* The plan: an anchor-free head (FCOS / CenterNet style) with box losses and box mAP, box datasets (COCO / YOLO txt),
   boxes drawn in the mistakes gallery; needs steps 2 and 4.
9. **Multi-GPU training:** *Done 2026-11-06 (CI-tested with two CPU processes; never on real GPUs): DistributedDataParallel started
   by train.py itself, signals forwarded, flags synced, rank 0 evaluates and checkpoints. Left: FSDP (weights split over the GPUs for
   models too big for one), automatic batch split with several GPUs, several machines, choosing Kaggle's T4×2 (not possible through the
   API).* The plan: DistributedDataParallel / FSDP on machines with several GPUs (college server, multi-GPU EC2), for the large
   ViTs and long time series.
10. **Fine-tuning and dynamic structures:** *Done 2026-11-06 (CI-tested): frozen layers, LoRA on Dense and Self-attention, mixture of
    experts, early exits with confidence stopping in the results. Left: LoRA on convolutions, adapters (bottleneck) blocks, expert
    parallelism across GPUs, exits that really stop at inference (exported model).* The plan: LoRA / adapters on pretrained backbones, freezing by layer group; mixture-of-experts layers
    (top-k routing), early-exit heads.

**Left out on purpose:** diffusion models, GANs, VAEs, LLM pretraining and self-supervised pretraining (MAE, DINO, CLIP). Each needs its
own training loop (noise schedules and sampling, two optimisers, KL terms, augmented views); that is research code, which suits a code
editor better than a canvas. With step 3 their parts (a U-Net block with timestep input, a projection head) can still be prototyped.

## 2. The text language (details in [PROGRAMMING_LANGUAGE.md](PROGRAMMING_LANGUAGE.md), section 9)

- A changed block that a function or loop made (on the canvas) still prints the text again; settings changes, moves, and blocks added,
  removed or rewired on plain lines already keep the hand-written text.
- Completion of block words and settings while typing (colours and ⌘-click on numbers are done).
- The text form on the clipboard next to copied blocks.
- Import a torchvision model as text; variables for sweeps; a diff of two designs by block number.

## 3. Models and training

- **MobileNet v2 / EfficientNet-B0 demos with ImageNet weights** (the plan named MobileNet; the current MobileNet demo is v1, which
  torchvision has no weights for).
- Mixup / CutMix and RandAugment.
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

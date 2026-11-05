# Future plan

## A text language for designs

Numbered blocks with their connections written as text, editable next to the canvas: the detailed plan is in
[PROGRAMMING_LANGUAGE.md](PROGRAMMING_LANGUAGE.md).

## Sentinel-1 and Sentinel-2 datasets

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

# TEST_APP.md: what to test by hand

The automatic tests (`cargo test` on every change, GitHub Actions on every push) check the shape engine against PyTorch on ~290 designs and
train small networks on **generated data on a CPU**. They cannot click through the window, use real datasets, a real GPU, real cloud
accounts or several GPUs. This file lists what still needs a person: open the app, do the steps, and compare with **Expect**.

How to use it:

- Work in a **new project** (Projects → New project, e.g. "Test 2026-11"), not in a project you care about. Delete it at the end (it goes
  to the Trash).
- Kaggle gives about 30 GPU hours a week; the tests below marked **GPU** use a few minutes each (5 epochs or fewer). AWS / spot tests cost
  money: do them only when you want to.
- Mark each line: ✅ works · ❌ wrong (write what you saw) · ⏭ skipped. Problems go to Claude with the run id (Train tab → Runs) or a
  screenshot.

Status column: **CI** = covered by the automatic tests on generated data (the logic is right; the window and real data are not checked) ·
**Real** = already tried for real once · **Never** = never tried in any form outside the code · **✅ Window** = seen in the app ·
**✅ GPU** = run on a real Kaggle GPU (T4) with generated data · **✅ Real data** = a real dataset on a real GPU. The ✅ lines of
6 November 2026 were done by Claude through the app's own buttons (macOS accessibility) on 12 TEST projects; the owner approved the
Keychain question.

---

## 0. Install and update

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 0.1 | Updater | Your installed app says 0.3.0: Help → Check for Updates… → Install and restart | It downloads 0.4.0, restarts, About says 0.4.0; projects and settings unchanged | ✅ Done (owner, 6 Nov): 0.3.0 → 0.4.0 by the updater |
| 0.2 | Keychain after update | Open Connection after the update | At most one Keychain question ("Always Allow"); tokens still there | ✅ Done: one "Always Allow" per new build |
| 0.3 | Manual | Help → Documentation | The PDF opens (69 pages, dated 6 November 2026) | ✅ Window |

## 1. Projects, settings, menus

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 1.1 | New / open / close | New project, close (File → Close Project), open it again | Same design and dataset come back | ✅ Window: 12 TEST projects opened and closed (File → Close Project) |
| 1.2 | Demo | Projects → Start from a demo → ResNet-18 → ⌘S | Saved as a project; an unsaved demo vanishes when another opens | ✅ Window |
| 1.3 | Delete | Delete the test project at the end | Its folder is in the Trash, off the list | ✅ Window |
| 1.4 | Settings | ⚙ → night mode, panel colours, Design size | Applied at once, kept after restart | ✅ Window |
| 1.5 | Undo / redo | Change a few blocks, ⌘Z / ⇧⌘Z | Steps undone and redone in order | ✅ Window |

## 2. Design canvas (new blocks and tools)

For each, add the block from the palette, select it, change its settings in the inspector, and look at the block's shape and the error /
warning text.

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 2.1 | Palette groups | Look at the palette | Groups: Image, Up-sampling, Dense and outputs (incl. Detection output, Early exit), Normalise, Time series, Transformer (incl. Cross-attention, RMSNorm, SwiGLU, Mixture of experts), Branches (incl. Subtract, Shared weights), Your own (Custom) | ✅ Window: every group and button, incl. Twin branch |
| 2.2 | Custom block | Add Custom (PyTorch) → Insert an example → Class: SE | The block shows the input's shape; the code box edits; the Text tab shows a `python """ … """` section | ✅ Window + GPU: class SE shown, trained on a T4 (TEST 1); "Insert an example" not pressed |
| 2.3 | Custom shape error | In the code, change `# out: c, h, w` to `# out: c * 2, h, w` | The next block's shape follows (channels doubled); a wrong formula shows a red message | ✅ Window |
| 2.4 | Twin branch | Fused dataset (2.x below or any), Input → conv → conv; select the last conv → Twin branch (shared weights) | A second Input (band group 2) and Shared-weights copies appear; parameters do not grow | ✅ Window |
| 2.5 | Subtract | Subtract the two branches, tick absolute difference | Shape stays; the summary says \|a − b\| | ✅ Window |
| 2.6 | Frozen / LoRA | Tick "frozen" on a conv; set LoRA rank 4 on a Dense | The design summary shows "N trained, M frozen" | ✅ Window: "146,328 trained, 24,576 frozen" (TEST 1) |
| 2.7 | MoE / exits | After To tokens add Mixture of experts; after a Global pool add Early exit | Shapes kept; an Early exit before Global pool says it needs a vector | ✅ Window + GPU: MoE and exit shown and trained (TEST 1) |
| 2.8 | Transformer options | Self-attention: causal, rotary; Class token + position: sinusoidal | The block summary shows causal · rotary; an odd head width with rotary is refused | ✅ Window: "4 heads · rotary"; summaries now also show sinusoidal and LoRA (fixed) |
| 2.9 | Output for | With extra outputs (4.6), select an Output → "Output for" | The list shows the main task and each extra output | ✅ Window |
| 2.10 | Large designs | Demo ViT-g/14 → Collapse repeated blocks | Fast; one ×40 node | ✅ Window |
| 2.11 | Custom blocks everywhere | TEST 22: Custom blocks on an image (AddCoords `c + 2`, CBAM inside a function, SpaceToDepth `c * 4, h // 2, w // 2`), on tokens (TokenMix) and on a vector (GEGLU `n // 2`), custom loss, ONNX export | One-batch check passes; trains; model.onnx / model.ts written | ✅ GPU (real CIFAR-10): one-batch check passed, 65.3 % in 5 epochs, exit 53.9 %, ONNX + TorchScript written |
| 2.12 | MoE GShard options + demo | Demo MoE Transformer (GShard); Mixture of experts: capacity 1.25, random second expert (`moe 8 top 2 hidden 256 cap 1.25 random`) | 2,198,306 parameters; text folds into `repeat 3`; trains | ✅ GPU (real CIFAR-10): PyTorch 2,198,306 = the app's, 75.95 % in 10 epochs; CI: no expert over its capacity, random only while training |

## 3. Text tab

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 3.1 | Round trip | Paste `doc/examples/everything.nnb` (CIFAR-10 3×32×32) | 97 blocks, no warning, 217,396 parameters (16,832 frozen) | ✅ Window |
| 3.2 | Time series text | Paste `doc/examples/timeseries.nnb` with a time-series dataset (4 bands, 6 dates, 32×32) | No error; Conv3D, Temporal pool, LSTM … on the canvas | ✅ Window |
| 3.3 | New words | Type `5 shared 2`, `subtract abs`, `lstm 32 bi`, `moe 4 top 2 hidden 64`, `exit 0.3`, `dense 64 lora 4`, `conv 8 3x3 frozen`, `detect` | Each builds the right block; ⌘-click selects it | ✅ Window: TEST 1's text shows python section, frozen, custom, exit, rotary, lora, moe |
| 3.4 | Canvas → text | Change a setting on the canvas | Only that line changes; hand-written functions stay | ✅ Window |
| 3.5 | Auto numbering | Text tab: Enter after `2 conv …` in a design with lines 1–7; Enter after `a …` inside a repeat; Enter on the new label-only line; delete a line, Renumber | `3 ` typed and 3–7 become 4–8 (references too) and the blocks stay in place on the canvas; `b ` inside the group; the label goes and the numbers close up; Renumber fills the gap | ✅ Window (MoE demo, scratch home) |

## 4. Dataset tab, one kind of data at a time

Each: set the Dataset tab as described, then **Check data on Kaggle** (a few minutes, no training) and look at the data report (Results).

| # | Test | Dataset | Expect | Status |
|---|---|---|---|---|
| 4.1 | Sentinel-2, Hugging Face | `giswqs/EuroSAT_MS`, 13 bands | 0 warnings, 10 classes, band means ~1000–2500 | ✅ Real data: 0 warnings; 92.1 % (5 epochs) |
| 4.2 | Own files → Kaggle upload | A folder on this Mac with class subfolders of GeoTIFFs (e.g. a small Earth Engine export) | First run uploads a private Kaggle dataset, later runs reuse it; data check finds all classes | ✅ GPU: 9 folders uploaded as private Kaggle datasets and trained (TEST 2–10) |
| 4.3 | Earth Engine script | Dataset → "Get Sentinel-1/2 data from Google Earth Engine" → run the script in your GEE account | The exports land in Drive with the tab's band order; "Set this tab to match" fills it in | ✅ Real data: the app's script run in the Code Editor (exports queued) and in the Python API (exported, trained); also 40 existing exports |
| 4.4 | Cloud folder | Own files → in the cloud: a Drive link or gs:// bucket of the export | The machine fetches it; data check OK | ✅ Real: Google Drive link (fetched on Kaggle); gs:// / rclone not tried |
| 4.5 | Multi-label | A labels CSV (or BigEarthNet preset with a BigEarthNet subset) | Classes read from the CSV; micro F1 / mAP after training | ✅ GPU: TEST 4, micro F1 / mAP 1.0 |
| 4.6 | Extra outputs | Labels CSV with a class column + a number column; Dataset → Extra outputs → add "numbers" with that column | Design needs two Outputs; results show both scores | ✅ GPU: TEST 3, class 100 %, biomass R² 0.987, wet 97.5 % |
| 4.7 | Regression | Task Regression, labels CSV with a numeric column (e.g. a biomass or yield table you have) | Read labels lists the numeric columns; R² / scatter after training | ✅ GPU: TEST 2, R² 0.998 (data check: both targets, 0 warnings) |
| 4.8 | Segmentation, scene | Earth Engine scene + WorldCover label (layout "large images + label image"), test tiles "strip" | Tiles counted; mIoU after training | ✅ GPU |
| 4.9 | Fusion S2 + S1 | Second band group Sentinel-1 from a twin folder | Data check shows 2 groups; mid fusion trains | ✅ GPU: TEST 4 (S1 stacked with S2) |
| 4.10 | Own sizes | Same, tick "its own image size" (e.g. S1 at half) | Input 2 has the smaller size; an Input for all bands is refused | ✅ GPU: TEST 5, S1 16×16 next to S2 32×32: 100 % |
| 4.11 | Before / after pairs | Two folders of the same bands at two dates, same file names | Accepted (names repeat); Siamese design trains | ✅ GPU: TEST 6, Siamese 100 % (constant lr) |
| 4.12 | Time series | Tick "a time series", dates per sample; .npy (dates, bands, H, W) or GeoTIFF date by date | Input (bands, dates, H, W); data check pictures = median date | ✅ GPU: TEST 7 (100 %, data check 0 warnings), TEST 8 (crop map 95.0 %, mIoU 0.869) |
| 4.13 | Detection | A YOLO-format set (images/ + labels/ .txt; e.g. a ship / car / building set from Kaggle or Roboflow, exported as YOLO) | Classes typed in label order; data check counts boxes per class | ✅ GPU: TEST 9, data check 191 / 200 boxes, mAP@0.5 0.951 |
| 4.14 | Class balance | Any unbalanced set → Read folder / Read labels | Counts per class and a 3:1 warning | ✅ Window |

## 5. Train tab settings (each needs one short **GPU** run, 3–5 epochs)

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 5.1 | Recipe | Use recommended → train | Cosine lr curve, warm-up visible in the lr of each epoch | ✅ GPU |
| 5.2 | Losses | Focal (classification), Dice + CE (segmentation), Huber (regression) | Run starts; Results say "Trained with … loss" | ✅ GPU: focal (TEST 1, 10), Dice + CE (TEST 8), Huber (TEST 2) |
| 5.3 | Custom loss | Loss: your own (Python) → keep the example | Run trains; a broken function gives a clear error | ✅ GPU: TEST 6 |
| 5.4 | Class weights | Unbalanced classes: inverse frequency | Log lists the weights; rare-class recall rises | ✅ GPU: sqrt (TEST 1), inverse 0.55 / 5.5 (TEST 10) |
| 5.5 | Thresholds | Multi-label + "a threshold per class" | Results list the thresholds | ✅ GPU: TEST 4, threshold column in Results |
| 5.6 | ImageNet weights + LoRA | ResNet-18 demo, weights resnet18, "train only the new layers"; or ViT-B/16 + LoRA on attention | Start event: far fewer trainable parameters; accuracy close to full fine-tuning | ✅ Real data: ViT-B/16 LoRA 93.6 % with 0.27 % of the weights trained vs full 96.8 % (after a fix) |
| 5.7 | Save memory | ViT demo, "save memory" on | Run card memory drops, ~30 % slower | ✅ Real data |
| 5.8 | Seeds | 3 seeds → Compare selected → Export | Overlaid curves; CSV + LaTeX table with mean ± s.d. | ✅ GPU |
| 5.9 | GPU memory estimate | Change batch size | Estimate updates; warns above the target's memory | ✅ Window |

## 6. Where training runs

| # | Test | Steps | Expect | Status |
|---|---|---|---|---|
| 6.1 | Kaggle | Any design, Train on Kaggle | Live curves every 30 s; results downloaded; long project names work | ✅ Real: 15 runs on Kaggle (2 at a time: Kaggle's limit) |
| 6.2 | College server (SSH) | Connection → add the server → Prepare → Train | SLURM / systemd / tmux chosen; run survives closing the app; results come back | Never |
| 6.3 | Several GPUs | On a machine with ≥ 2 GPUs: GPUs = all → Train | Start event `gpus: N`; epochs ~N× faster; one set of curves | ✅ Real: Kaggle T4 x2, TEST 2: 2 workers, R² 0.998 in 20 s (after a fix: it did not exit) |
| 6.4 | Several GPUs, stop | Same, press Stop mid-epoch | All workers stop; results written | Not possible yet: needs an SSH machine with ≥ 2 GPUs (Kaggle cannot be stopped by the API); CI: spot notice |
| 6.5 | AWS on demand | Connection → AWS instance → Train | The app starts the instance, the run stops it at the end | ✅ Real: EC2 g4dn.xlarge, Mumbai (2026-11-07) |
| 6.6 | AWS spot | Persistent spot request, interruption behaviour stop | After an interruption the run resumes by itself at the same batch | ✅ Real: EC2 spot, Mumbai (2026-11-08); interruption -> checkpoint + exit 75 kept resumable (resume at boot not seen: no spot capacity to restart) |
| 6.7 | Continue elsewhere | Spot instance terminated, Mac copy on → Continue on another machine | Resumes on the other machine from the checkpoint | ✅ Real: spot instance terminated -> Continue on gap-experiment from the Mac copy, finished 99.11 % (2026-11-08) |

## 7. Results tab, per task

| # | Test | After a run of | Expect | Status |
|---|---|---|---|---|
| 7.1 | Classification | 4.1 | Accuracy, top-5, per-class, confusion matrix, Grad-CAM, mistakes | ✅ Window: TEST 1 (confusion matrix, per class, mistakes, class weights, focal) |
| 7.2 | Multi-label | 4.5 | Micro / macro F1, mAP, exact match, per-class table, ✓ ✗ ○ samples | ✅ Window: TEST 4 (micro / macro F1, mAP, exact match, thresholds) |
| 7.3 | Regression | 4.7 | R², MAE, RMSE cards, per-target table, scatter plot, samples with both numbers | ✅ Window: TEST 2 (R², MAE, RMSE, scatter plot, Huber) |
| 7.4 | Segmentation | 4.8 | Pixel accuracy, mIoU, IoU per class, image / true / predicted masks | ✅ Window: TEST 8 (mIoU, masks, Dice + CE) |
| 7.5 | Detection | 4.13 | mAP@0.5, mAP@0.5:0.95, AP per class, images with green (found) and dashed (true) boxes | ✅ Window: TEST 9 (mAP cards, boxes "2 found · 2 true"; a stray classification section removed) |
| 7.6 | Extra outputs | 4.6 | "Extra outputs" table under the main cards | ✅ Window: TEST 3 (Extra outputs table) |
| 7.7 | Early exits | A classifier with an Early exit | "Early exits" table: accuracy per exit, share leaving early | ✅ Window: TEST 1 (exit 79.3 %; 23.8 % could stop there at 83.8 %) |
| 7.8 | Data check | Any "Check data" | Sizes, counts, bands, warnings, sample images | ✅ Window: data checks of TEST 2, 7, 9 |
| 7.9 | Export | "Also save … TorchScript and ONNX" | model.ts and model.onnx in Show files | ✅ GPU |

## 8. Real-data experiments worth running (each a real test of a new feature)

1. **Crop types from a time series:** a monthly Sentinel-2 stack (6–12 dates) over fields with a crop map as labels; Conv3D + Temporal
   pool (attention) + a small decoder + Segmentation output. Compare with a single date (time steps off).
2. **Change detection:** before / after Sentinel-2 pairs (e.g. new buildings or a flood) with a Twin branch and Subtract |a − b|.
3. **Flood mapping with S1 + S2:** Sentinel-1 at its own size next to Sentinel-2, mid fusion; Dice + CE loss for the rare flood class.
4. **Biomass / yield regression:** patches with a number each (labels CSV), Huber loss; read R² and the scatter plot.
5. **Ships or cars:** a YOLO-format detection set; Detection output on a ¼-size map; read mAP@0.5.
6. **LoRA vs full fine-tuning:** ViT-B/16 with ImageNet weights on EuroSAT RGB: full fine-tuning against LoRA rank 4 on attention
   (trainable parameters, accuracy, time). ✅ Done 6 Nov (1 epoch, 4,000 images, T4): LoRA 93.6 % training 228,874 weights (0.27 %),
   full 96.8 % training 85.8 M; about 100 s per epoch both.

## 9. Known limits (expected, not bugs)

- The Mac never trains; tests on the Mac only check designs and code.
- Kaggle may give a notebook two T4s ("GPU T4 x2"): with GPUs = all, a Kaggle run then trains on both.
- Not built yet: rotated boxes, COCO JSON, real dates / missing-date masks in time series, FSDP, LoRA on convolutions, per-pixel
  regression (see FUTURE_PLAN.md).

## Results log

| Date | Test # | Result | Notes / run id |
|---|---|---|---|
| 2026-11-06 | 4.1, 6.1, 7.1, 7.8 | ✅ | EuroSAT-MS on Kaggle T4: data check 0 warnings (20261106-114810-check); 92.1 % (20261106-115114) |
| 2026-11-06 | 5.6, 8.6 | ✅ | TEST 11 ViT-B/16 + LoRA 93.6 % (20261106-204440); TEST 12 full fine-tuning 96.8 % (20261106-204235); real EuroSAT RGB |
| 2026-11-06 | 2.x, 7.1, 7.7 | ✅ | TEST 1 hybrid with every new block, real EuroSAT-MS 13 bands: 83.7 %, exit 79.3 % (20261106-205440) |
| 2026-11-06 | 4.2, 4.7, 6.3 | ✅ | TEST 2: data check (20261106-205650-check); 2 GPUs: trained in 28 s but never exited (205947, cancelled) → fixed; again 214127: R² 0.998, ended by itself |
| 2026-11-06 | 4.6, 7.6 | ✅ | TEST 3 extra outputs (20261106-210609) |
| 2026-11-06 | 4.5, 4.9, 5.5, 7.2 | ✅ | TEST 4 multi-label + S1 (20261106-211340) |
| 2026-11-06 | 4.10 | ✅ | TEST 5 own sizes 100 % (20261106-211652) |
| 2026-11-06 | 4.11, 5.3 | ✅ | TEST 6 Siamese + custom loss: 62.5 % with cosine / batch 32 (212212, learning slowly); 100 % with constant lr / batch 16 (214630) |
| 2026-11-06 | 4.12, 7.8 | ✅ | TEST 7 time series: data check (212428-check); 50 % with cosine / batch 32 (214303); 100 % with constant lr / batch 16 (215322) |
| 2026-11-06 | 4.12, 5.2, 7.4 | ✅ | TEST 8 crop map 95.0 %, mIoU 0.869 (20261106-213013) |
| 2026-11-06 | 4.13, 7.5 | ✅ | TEST 9 detection: data check (213229-check), mAP@0.5 0.951, @0.5:0.95 0.628 (213439) |
| 2026-11-06 | 5.2, 5.4 | ✅ | TEST 10 unbalanced: weights 0.55 / 5.5, focal, 100 % (20261106-213656) |
| 2026-11-06 | — | fixed | Found by these tests: LoRA + ImageNet weights refused (fixed, CI); block summaries missing positions / LoRA; a poll error stayed on finished runs; several GPUs did not exit; a stray section on detection results |
| 2026-11-07 | 1.4, 1.5, 2.3, 2.4, 2.5, 2.9 | ✅ | In the window: night mode kept after a restart; ⌘Z / ⇧⌘Z; a custom block's `# out:` 32 → 64 channels moved the next shapes, `h / 2` gave "use // (whole-number division)"; Twin branch made 3 shared blocks; Subtract shows \|a − b\|; "Output for" lists the extra outputs |
| 2026-11-07 | 3.1, 3.2, 3.4 | ✅ | everything.nnb pasted: 97 blocks, 217,396 params; timeseries.nnb: 23,204 params; LSTM 16 → 24 on the canvas changed only that line of the text |
| 2026-11-07 | 4.8 | ✅ | TEST 15 scene segmentation (generated scene + label, strip test tiles, Dice + CE): 95.9 % pixels, mIoU 0.919 (20261107-052912) |
| 2026-11-07 | — | fixed | A moment without DNS failed a Kaggle upload (20261107-052609): Kaggle calls are now tried 3 times. A third seed failed on Kaggle's limit of 2 GPU notebooks: now "queued" and pushed when one is free, also while another seed is the run shown |
| 2026-11-07 | 5.1, 5.8, 7.9 | ✅ | TEST 13 everything.nnb on CIFAR-10, 2 epochs, recommended recipe, 3 seeds: the third waited ("queued") and was pushed when seed 2 ended; 49.30 / 52.53 / 51.01 % → 50.95 ± 1.62 % (top-5 94.70 ± 0.43 %, checked by hand); warm-up cut to 1 epoch, then cosine to 0; Compare (curves + table) and Export CSV + LaTeX; model.ts + model.onnx (group g20261107-063547) |
| 2026-11-07 | 5.7 | ❌ → fixed | "save memory" was ticked but not used (the design has early exits); the Train tab now says so |
| 2026-11-07 | 4.14 | ✅ | TEST 10 Read folder: "common 200 · rare 20, unbalanced (10.0×): consider class weights"; fixed: "other bands" b1 / b2 showed Sentinel-2 names as tooltips |
| 2026-11-07 | 5.7 | ✅ | TEST 12 ViT-B/16 on real EuroSAT RGB, batch 64, save memory on: peak 2.18 GB instead of 7.22 GB, 114 s instead of 99 s per epoch (+15 %), same accuracy 96.8 % (20261107-065029 vs 20261106-204235) |
| 2026-11-07 | 6.5 | ✅ | EC2 g4dn.xlarge (Mumbai, Deep Learning AMI, the owner's instance): Check instance (stopped, shutdown Stop); Prepare 0.7 min (CUDA, T4, the AMI's PyTorch 2.12); run 20261107-085601 40.5 %; results fetched from the stopped instance (092313, 79.5 %, 0.5 min); the app quit mid-run, the run went on (095956); Stop mid-run ends "stopped" with results (134745, 75.7 %); the instance stops itself after every run. 29.3 billed minutes ≈ $0.28 in all (10 of them a slow CIFAR-10 download) |
| 2026-11-07 | 4.3 | ✅ | TEST 17: 40 real Earth Engine exports (S2 L1C 13 bands, S1 VV/VH dB, LZW, 512×512) fused, Read folder takes each group's band names from its own folder; Check data on Kaggle passed (one batch 2×15×64×64; S1 seen as dB; no empty pixels) (20261107-133659-check) |
| 2026-11-07 | 0.3, 1.2, 1.3, 2.10, 5.9 | ✅ | Manual opens; LeNet demo saved as a project and deleted (to the Trash); ViT-g/14 (1,011,216,522 params, 287 layers) collapses to 8 boxes at once; memory estimate 0.61 GB at batch 8, 1.5 GB at 512 |
| 2026-11-07 | — | fixed | Found by the EC2 and Earth Engine tests (all with tests): an instance started for work that failed was left running; ssh wait too short for an AMI's first boot; "does not answer" while metrics.jsonl did not exist; a "starting" run never polled again; Fetch hidden when needed and not updating the state; a poll overwriting a fetch (race); Stop killed the data-loading workers first; LZW GeoTIFFs on Kaggle (imagecodecs); 13-band TIFF header ("432-bit"); band names taken from the other group's folder; a full path as the second group's folder |
| 2026-11-07 | 4.3, 4.4 | ✅ | TEST 19: the app's segmentation script (S2 SR B2 B3 B4 B8 2024 + ESA WorldCover, EPSG:3857, 10 m) run for a 5.5 km tile in Earth Engine (2 tasks, 35 s); the Drive folders shared by link as the cloud source; Kaggle fetched them in 5 s (nothing through the Mac), cut 64 tiles of 64×64, strip test split 48 / 8, one batch passed (20261107-162826-check); trained 10 epochs: pixel accuracy 79.8 %, built-up IoU 0.71, tree cover 0.67, mIoU 0.148 (20261107-164424) |
| 2026-11-07 | 6.6, 6.7 | ⏭ | Spot g4dn.xlarge in Mumbai: no capacity in any zone (ap-south-1a/b/c, spot ~$0.25/h); the 4 requests cancelled at once, nothing launched or charged. To retry when AWS has spot capacity (AWS_TEST_PLAN section 8) |
| 2026-11-07 | 4.3 | ✅ | The Dataset tab's script pasted unchanged into the Code Editor (project of the owner; only the area added instead of drawing): ran without errors, the two tasks nnb_image / nnb_label showed the app's settings (EPSG:3857, 10 m, Drive folders, GeoTIFF) and were submitted. They wait behind about 700 thesis tasks of the same project. Earth Engine ignored priority 9999 (task priority needs a paid plan); default 100. Both SUCCEEDED at 21:08 (about 26 EECU-seconds each), into the existing test folders of the same names; no second scene.tif appeared there (the Python run's files, of the same computation, stayed unchanged) |
| 2026-11-08 | 6.6 | ✅ (partly) | Spot g4dn.xlarge in Mumbai (persistent request, interruption behaviour stop), TEST 20 (LeNet-5, MNIST, 60 epochs). Interrupted at epoch 8 by stopping the instance: 1st try ended the run as "stopped" (fixed: on spot a shutdown TERM is an interruption); 2nd try exited 75 but run.sh removed resume_wanted (fixed: exit 75 keeps it); 3rd try (083354) interrupted cleanly, but AWS then had no spot capacity to start the instance again, so the resume at boot by itself was not seen on the real instance (it is covered by CI and the stand-in flow test). Also fixed: the ssh wait after a start kept a paid instance waiting 18 min when this Mac's address was not allowed |
| 2026-11-08 | 6.7 | ✅ | The interrupted run's spot request cancelled and the instance terminated; the app showed "terminated" and offered Continue on another machine; on gap-experiment (on demand) the run resumed from the Mac's checkpoint copy (epoch 8 -> resume at epoch 9), finished 60 epochs, 99.11 % (090658, resumed_from 083354); the old run marked "continued" |
| 2026-11-08 | 6.3 (again), 6.4 | ✅ / ⏭ | TEST 21 (Small CNN, CIFAR-10, 30 epochs, GPUs = all) on Kaggle: 2 GPUs (T4 x2), 32 s per epoch, 85.42 % (101430). 6.4 cannot be done on Kaggle (its API cannot stop a notebook: the app's Stop is for SSH/AWS machines); no machine with 2+ GPUs is reachable over SSH (college: none; AWS: only g4dn.xlarge allowed). Added: a "GPU use" log line halfway through every epoch |
| 2026-11-10 | 2.11, 2.12 | ✅ | Kaggle T4, real CIFAR-10. TEST 22 custom PyTorch blocks (5 Custom classes, custom loss, export): one-batch check passed, 65.25 % (top-5 96.9 %) in 5 epochs (20261110-074047). TEST 23 GShard MoE Transformer (8 experts, top 2, cap 1.25, random 2nd): 75.95 % (top-5 98.6 %) in 10 epochs, 31 s per epoch, 0.7 GB GPU memory (20261110-073911) |
| 2026-11-10 | 3.5 | ✅ | MoE demo in a scratch home, typed by keyboard: Enter after `2 conv` → `3 `, 3–7 became 4–8, all 52 old blocks kept id and canvas place; Enter inside the repeat → `b `, b–o → c–p with `<-` refs; Enter on a label-only line removed it and closed the gap; deleted a line, Renumber closed the gap (ids kept). Found and fixed: a demo's first text edit rearranged the canvas (design on disk had no labels), and opening another design with the Text tab open applied the old text to it |
| | | | |

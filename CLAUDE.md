# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Status

Version 0.4.0 (2026-11-06). Public repository github.com/varunjha089/NN-Builder (MIT); releases there feed the app's automatic update.
The automatic tests use stand-ins for Kaggle, ssh and EC2 (never real accounts). By hand (TEST_APP.md, 57 of 61 ✅ on 2026-11-07): Kaggle
(many runs, 2 GPU notebooks at a time), EC2 on demand on the owner's g4dn.xlarge (AWS_TEST_PLAN.md: every phase, $0.28), real Earth Engine
exports (the app's script in the Code Editor and the Python API; existing S1+S2 exports fused) fetched by Kaggle from a Drive link. Open:
spot 6.6 / 6.7 (no Mumbai capacity on 2026-11-07; retry planned), college server 6.2 / 6.4. The earlier Python/Blockly app
(0.1.1, 2026-11-03) is in the Trash.
Open work (owner tasks, the design language, models, workflow, Sentinel-1/2) is in FUTURE_PLAN.md; what the owner still has to try by hand
(window, real data, real GPUs / clouds) is TEST_APP.md: add a line there for every new feature, and its status (CI / Real / Never); the design language in
PROGRAMMING_LANGUAGE.md; the first real EC2 tests and their cost guards in AWS_TEST_PLAN.md. CHANGELOG.md lists every commit by time (dd_mm_yy_hr_min_sec, newest first); after committing, run
`python3 scripts/make_changelog.py` and commit the result as "CHANGELOG: …" (such commits are left out of the list).

## Layout

```
Cargo.toml                 workspace: crates/core, app/src-tauri (version 0.1.0 in [workspace.package] and app/src-tauri/tauri.conf.json)
crates/core/src/
  model.rs                 Graph {nodes, edges}, KINDS (incl. multiply, seg_output, and for transformers to_tokens, cls_pos, attention,
                           token_pool: shapes of rank 2 = (tokens, dimension)), defaults(kind), arity, starter()
  templates.rs             famous architectures (LeNet-5, small CIFAR CNN, AlexNet, VGG-16, GoogLeNet, ResNet-18, MobileNet v1, small DenseNet,
                           Transformer encoder on image rows, ViT-Base/16, ViT-Tiny, U-Net, Attention U-Net): TEMPLATES (key, title, year, task,
                           input, about), build(key) with a left-to-right layout (a column per depth); plus one original design,
                           MSG-Former (msg_former: multi-scale dilated convolutions + squeeze-and-excitation gate + residual, then
                           Transformer encoders; 466,522 params, counted by hand in the test) and ViT-g/14 (vit_g14, 1,011,216,522 params,
                           counted by hand; too big to train on a 16 GB T4, said in its description; CI builds it in PyTorch, ~4 GB RAM)
                           Demos with a built-in dataset switch it to the one that fits their input (CIFAR-10 for 3×32×32, MNIST for
                           1×28×28) instead of resizing another (apply_template in main.rs).
                           Dataset → Read info (Hugging Face) keeps the chosen image/label columns when the dataset has them; else the
                           class column named "label", else the one with most classes (the columns arrive sorted by name, so CIFAR-100's
                           coarse_label would otherwise replace fine_label).
                           Train tab: Random seed field (TrainSettings.seed: weights, derived test split, batch order).
                           Validation and early stopping: val_fraction (held out of the training data, evaluation transforms; 0 =
                           validate on test as before), early_stop + patience + monitor (val_loss | val_acc | val_miou); clean() forces
                           val_fraction 0.1 when early stopping is on. harness: best_model.pt (atomic), "best" / "early_stop" events,
                           es state in the checkpoint (survives spot resume), final test on the best weights; CI: ci_early_stop_check.py.
                           Training recipe: optimizer adam | adamw | sgd, weight_decay, schedule constant | cosine | plateau (halve after 3
                           epochs without lower val loss; state "plat" in the checkpoint), warmup_epochs (linear), label_smoothing,
                           grad_clip; the learning rate is a function of the global step (lr_at), so it resumes exactly; "lr" in each epoch
                           event, "lr_drop" events. Train tab "Use recommended" (AdamW 0.05, cosine, 5 warm-up, smoothing 0.1, clip 1).
                           CI: ci_recipe_check.py. Train tab also: what is missing for the target, and a GPU memory estimate (16 B/param
                           + 2× layer outputs × batch, vs Kaggle/g4dn 16 GB or the SSH machine's nvidia-smi memory).
                           Kaggle live progress: while running, `kaggle kernels logs` (cloud::kaggle_logs) gives metrics.jsonl and log.txt
                           every 30 s (kaggle_live). runs::preflight refuses a run (before creating it) without tools / token / key.
                           Phase 2: runs::start_seeds (one group "g<time>", ids "<time>-s<seed>"; RunInfo.seed / .group), summary
                           (rows + mean / sample s.d. per group) and export (<project>/exports/runs-<time>.csv + .tex booktabs);
                           Train tab seeds 1/3/5, Runs checkboxes, Compare panel (overlaid validation curves) + Export; Results: top-5,
                           per-class precision/recall/F1 (from the confusion matrix); shapes.rs macs per node + total_macs (tests: MSG
                           93,499,136 by hand, ResNet-18 1.80–1.83 G). Tests: summary.rs (incl. pdflatex of the table); CI recipe
                           check uses 6 classes and checks val_top5 / test_top5.
                           Phase 3: TrainSettings.pretrained (resnet18 | vgg16 | vit_b_16; harness load_pretrained: each design tensor in
                           order gets the next unused torchvision tensor of the same shape, a BatchNorm the one after the matching
                           convolution (producers(): forward hooks); stops when < 80 % of weight tensors match;
                           ImageNet normalisation) + freeze ("backbone" = train only unloaded tensors); Grad-CAM (last 4-D layer) on the
                           samples, "mistakes" (16 misclassified of up to 2000 test images); export (model.ts + model.onnx, fetched
                           with the results). CI: ci_pretrained_check.py compares the loaded demos with torchvision's own models.
                           Phase 4: GroupNorm block (groups must divide channels); Design: copy / paste (document copy/paste events,
                           JSON {nnb_blocks}), Residual block after selected, collapseRepeats (×N read-only view of identical consecutive
                           groups); remote: upload_dir / download_file use rsync or tar / cat over ssh (NNB_NO_RSYNC=1 in tests; flows
                           test runs a whole job that way); Windows: no ControlMaster options, password login refused with advice;
                           Build workflow: macOS-Intel (x86_64-apple-darwin cross-build, NN-Builder_X_x64.app.zip); Dataset: hf_preview
                           (class balance from datasets-server /statistics, 8 example images from /first-rows as data URLs on request),
                           no-image-column error.
                           GPU memory / smoother running: TrainSettings.auto_split (default on: a batch that runs out of memory is
                           processed in 2, 4, 8 ... parts, gradients weighted by share, "memory_split" event, parts kept in the
                           checkpoint, version 5; evaluation batches halve the same way; one_batch_check uses 2 images) and save_memory
                           (codegen checkpoint_cuts: forward cut where only one tensor crosses, ~sqrt(n) parts, _partN methods +
                           torch.utils.checkpoint; generated only when on; not with several outputs / early exits: the Train tab
                           says so, canSaveMemory); bf16 on compute capability 8+, channels-last for conv nets,
                           workers "auto" = min(8, CPUs) with prefetch 4, "memory" event (measured peak after the first epoch), init_process_group(device_id=) on CUDA (no NCCL "guessing device ID" warning; TypeError fallback for PyTorch < 2.3), torchvision download bar silenced, gpu_use() log line halfway through each epoch (rank 0, CUDA: GPUs
                           training, per GPU memory used / total and % busy from nvidia-smi, so no CUDA context on other GPUs),
                           data_wait per epoch; loader workers ignore INT/TERM/HUP/USR1/USR2 (quiet_worker: Stop signals the whole process group, and a dead
                           worker made the run fail instead of stop; CI: ci_resume_check part 4 with NNB_TEST_WORKERS=2); Train tab "GPU memory" box and run-card notes; status polls slow to 15 s when the
                           window is hidden. Mac side: memo'd blocks (not onlyRenderVisibleElements: it empties the minimap), undo history capped
                           by block count, charts thinned to ~1500 points. CI: ci_memory_check.py (parts = plain forward and gradients
                           on 7 demos; a run with NNB_TEST_OOM_ABOVE=8 ends with the same weights as without). The design text
                           language (crates/core/src/dsl.rs, tests/text.rs, web/src/TextTab.tsx; Graph.text + Node.label; save_graph
                           = dsl::sync, design_from_text = dsl::apply, design_format; Text tab after Design; block labels in the block
                           headers; manual sec:text; dsl::patch_settings rewrites only the lines of
                           blocks whose settings changed on the canvas; dsl::patch_structure edits only the lines of blocks added,
                           removed or rewired on plain lines (a group's outside input may change); TextTab colours words (highlight())
                           and ⌘-click selects a block; the text applies 0.25 s after typing; canvas and text stay separate tabs; Design.tsx groupByText groups the layer table by text line;
                           wrap_rows lays long text designs out in rows of 24; File → Import/Export Design… reads/writes .nnb too;
                           doc/language.tex = the manual's language chapter (\input by the manual;
                           its examples are doc/examples/*.nnb via \VerbatimInput; tests/text.rs builds every example and every
                           verbatim snippet, so update both together); `net "…"` (dsl::design_name) names the design: kept by sync/reprint (Format)
                           and written into train.py's first line; its block reference (sec:blocks-text) lists every block's
                           values (short form, name=, default, range); tests/text.rs block_reference_forms checks each form. The parser
                           accepts only settings a block has (and shapes::float_limits for rates), dsl::effective drops unknown keys; doc/examples/everything.nnb = every language feature and block kind, checked in tests/text.rs
                           and run in PyTorch by CI via dump_cases) is described in PROGRAMMING_LANGUAGE.md (section 9: as built, what is left) (FUTURE_PLAN.md: Sentinel-1/2).
                           Large designs (measured 2026-11-05, examples/big_designs.rs builds 1B-100B ViTs from text and times them):
                           cost follows blocks, not parameters; the window needs ~0.3 MB per block drawn (3,507 blocks ≈ 1 GB),
                           the canvas suggests the ×N view above 500 blocks. Most blocks per design: Settings → Design size
                           (settings.json max_blocks, null = recommended = dsl::recommended_max_blocks(RAM): a quarter of memory at
                           0.3 MB/block, ≥ 5,000; RAM from sysctl / /proc/meminfo / CIM; dsl::set_max_blocks at start and on save). shapes::analyze and dsl/codegen use id maps (no
                           per-block search: 3,507 blocks analysed in ~21 ms); Design.tsx reuses unchanged React Flow nodes. Limits:
                           shapes::limits() per kind (filters 65,536, heads 512); a value outside is an error, never clamped silently.
                           Help menu: Documentation opens the bundled manual PDF (tauri.conf.json bundle.resources).
                           Check data: runs::start_check (id "<time>-check", RunInfo.check, epochs 0) sets TrainSettings.check_only
                           for that run only (never saved); harness check_only(): data_report (sizes, class_counts via labels_of or a
                           sample, band_report of raw values, warnings), one_batch_check, 12 samples, results.json check_only, "done"
                           with warnings; Train tab "Check data on …", Runs "data check", ResultsTab DataReport. CI: ci_multiband_check
                           check_data (S1 left linear must be flagged, no epoch events).
                           Unbalanced classes: TrainSettings.class_weights ("" | inverse | sqrt) -> harness class_weights() sets
                           LOSS_WEIGHT (CE weight, multi-label BCE pos_weight; seg counts pixels), "class_weights" event; tune_thresholds
                           (clean() forces val_fraction ≥ 0.1) -> tune_thresholds() per-class F1-best threshold on validation, THRESH used
                           by multilabel_scores and pictures, thresholds.json; cloud::folder_counts -> "counts" in kaggle_files / local
                           describe, DatasetTab FolderCounts. CI: imbalanced case (10:1) and multilabel thresholds + sqrt pos_weight.
                           Losses: TrainSettings.loss ("" CE/BCE | focal | dice | dice_ce; codegen::LOSSES) and focal_gamma (default 2,
                           clean() 0-5); train_script refuses dice for task classify; harness LOSS / GAMMA, loss_of + dice_loss (-100
                           pixels masked, LOSS_WEIGHT weights every loss), recipe.loss in "start", results.json "loss"; Train tab Loss
                           select, ResultsTab line. CI: ci_multiband_check loss_math (formulas on small tensors) and focal on
                           imbalanced + multilabel, dice_ce on segfiles, dice on scene.
                           Custom blocks: kind "custom" (param "class"), Graph.python = the design's PyTorch code (serde default; saved
                           with the design); custom.rs reads it as text (classes(): name, __init__ args = input dims, "# out:" shape
                           expressions; out_shape(), calc() with + - * // %; split_section / set_section for the text's python """ … """
                           section). shapes "custom" uses the declaration (params 0: counted on the machine); codegen pastes the code
                           above class Net and builds Name(c, h, w); harness one-batch check names a wrong # out:. DSL: `custom NAME`,
                           expand blanks the section (line numbers kept), sync = sync_blocks + set_section(g.python). Custom loss:
                           TrainSettings.loss "custom" + custom_loss (must define def custom_loss(out, y); pasted before the harness;
                           regression gets standardised targets). UI: palette "Your own", CustomFields (class select, code textarea,
                           example SE), Design save keeps graph.python; Train tab custom loss box (CUSTOM_LOSS examples). Tests:
                           design.rs custom_pytorch_blocks, text.rs custom_blocks_and_python_section, everything.nnb (Unshuffle, run by
                           CI), CI ci_multiband_check custom_blocks (trains SE + Down with a custom loss; a wrong # out: must stop at
                           the check).
                           Several outputs (multi-task): Dataset.heads (dataset::Head: name, task classify|multilabel|regress, columns,
                           classes, loss, weight; head_problems) + label_column (classify main task from the table; empty = folder
                           name); Output param "head" (0 main; shapes per-head sizes via ds.head_size, one Output per head, missing
                           heads reported; Analysis.head_names); codegen forward returns a tuple in head order (save memory off then);
                           DSL `output N`. Harness HEADS/MULTITASK: labels are tuples (main_y/main_out/y_to/y_split), y_head per row
                           (table or HF row, with_heads), head_stats (standardised regression heads), loss_of = main + Σ weight·head_loss,
                           evaluate total loss + head_scores -> HEAD_SCORES ("heads" in epoch / done / results.json). UI: DatasetTab
                           HeadsBox, Output inspector "Output for", ResultsTab table, Train chart lines. CI: ci_multiband_check
                           multitask (class from folders + 2 numbers + a class from labels.csv).
                           Inputs at their own sizes: Dataset.image_size2 (group 2's [h, w], empty = resized to image_size;
                           separate_inputs(), group_shape(); not larger than group 1, not for scenes); shapes gives Input group 2 its own
                           shape and refuses a group-0 Input; codegen slices x[:, c1:, :h2, :w2]. Harness SIZE2 / C1: read_groups resizes
                           group 2 to SIZE2, read_bands pads it with NaN into the top-left corner (stats NaN-aware, normalise -> 0),
                           flips per group block, band_report on each group's own pixels. DatasetTab SecondGroup "its own image size".
                           CI: ci_multiband_check fusion_sizes (S2 32×32 + S1 16×16).
                           Shared weights: kind "shared" (param "of" = the shared block's id; shapes::shared_step runs that block's step on
                           this input, which must equal the source's input; params 0; Kahn order waits for the source; SHAREABLE kinds)
                           and "subtract" (a - b, abs). codegen calls the source's module (module_of). DSL `shared LABEL` (label -> id in
                           expand, whole or relative to the block's own group; printed as "shared @id" then resolve_shared in print /
                           sync / patch_*), `subtract abs`. Data: a pair in time = bands2 == bands (duplicate names allowed; harness
                           BANDS names the second copy B04_2). UI: Design "Twin branch (shared weights)" (addTwin), inspectors. Tests:
                           design.rs siamese_shared_weights, text.rs shared_weights_and_subtract, everything.nnb (tw / df), CI siamese
                           (before/after pairs, PyTorch's parameter count = the app's).
                           Time series: Dataset.time_steps (dates per sample; input_shape (C, T, H, W); not fused / HF / scenes);
                           kinds conv3d (tkernel, same | valid), temporal_pool (mean | max | attention: TemporalPool), to_sequence
                           ((T, C): ToSequence), lstm / gru (Recurrent, units, bidirectional), conv1d (Conv1dSteps; kernel / stride
                           are numbers: dsl::effective keeps them so, shapes::int reads a pair's first value), token_pool "last";
                           shapes guards rank 4 (only these and the elementwise / join / flatten / global_pool / batchnorm / custom
                           blocks), seg_output checks the last two sizes; codegen TIME_MODULES, BatchNorm3d, AdaptivePool3d. Harness
                           TIME: load_raw + time_orient ((T,C,H,W), (T,H,W,C), (C,T,H,W), GeoTIFF T·C bands date by date) -> (C, T, H, W),
                           per-date resize, normalise any rank, composite = median of the dates, no channels-last with Conv3d. DatasetTab
                           "a time series" + dates; Design palette "Time series". Tests: text.rs timeseries_design (doc/examples/
                           timeseries.nnb, also in dump_cases = CI PyTorch compare), time_series_blocks_in_text; CI ts_seq + ts_seg.
                           Transformer additions: attention causal / rotary (SelfAttention writes q/k/v out for rotary: same parameters,
                           rotate()), cls_pos pos learned | sinusoidal | none, kinds cross_attention (two inputs, CrossAttention with kdim),
                           rmsnorm (RMSNorm), swiglu (SwiGLU, units); concat joins token sequences (N1 + N2, D). DSL causal, rotary/rope,
                           sinusoidal, nopos, xattn, rms, swiglu N. Harness fill_dates (empty dates of a time series interpolated). CI:
                           ci_multiband_check transformer_math (imports the generated tokens/train.py: causal, rotary distance-only scores,
                           sinusoidal values, RMSNorm, forward + backward); everything.nnb has ca / rn / sw and rotary attention.
                           Object detection: task "detect" (kaggle_train_dir = images, kaggle_mask_dir = YOLO .txt labels; classes ≥ 1),
                           kind detect_output (DetectHead: K heatmaps + w, h, dx, dy per cell; stride must divide the image; EXPECTED
                           "detect"); train_script allows only CenterNet's loss or custom. Harness DETECT / DET_OUT: yolo_boxes, det_pairs,
                           det_target (Gaussian peaks, size, offset, mask: a fixed-size label), DetFiles (flips move boxes), det_loss
                           (CenterNet focal + L1), det_decode (3×3 peaks, top 100), det_truth (boxes back from the target), det_scores
                           (AP every-point, mAP@0.5 and @0.5:0.95); evaluate -> map50 in acc slot, map in mIoU slot; samples with boxes.
                           UI: Dataset task radio + DetectFolders, Results cards / per-class AP / BoxImage, Train chart. CI:
                           ci_multiband_check detection (targets <-> boxes exact, perfect mAP 1, detector on squares and discs).
                           Several GPUs: TrainSettings.gpus (1 default, N, 0 = all; clean() ≤ 64; "gpus" in TRAIN). Harness: RANK / WORLD
                           from NNB_RANK / NNB_WORLD; __main__ -> wanted_workers() (NNB_TEST_PROCESSES = CI's CPU processes) -> run_parallel():
                           Popen of train.py per rank (MASTER_ADDR/PORT), the parent writes py_pid.txt and forwards SIGTERM/INT/HUP/USR1/USR2,
                           exits with rank 0's code, ending the other workers a minute after rank 0 ends (a worker hanging on exit
                           kept a Kaggle notebook running: rank 0 now also destroys the process group; the other ranks wait at a barrier until rank 0 has
                           written the results, then all leave the group together: one leaving early sometimes aborted on exit in CI; a
                           worker's code no longer fails a run whose rank 0 ended 0). main(): init_process_group (nccl /
                           gloo, 1 h timeout), rank 0 loads data and weights
                           first (barrier), ShardBatches (each rank every WORLD-th image of the one-GPU batch), SyncBatchNorm + DDP after the
                           one-batch check (rank 0), sync_flags (STOP / INTERRUPT / checkpoint due, MAX all-reduce) every step and at the
                           epoch end, sums all-reduced, rank 0 evaluates / checkpoints and broadcasts early stop + plateau lr; emit() and
                           checkpoint() silent on other ranks, which return after training. OOM with DDP: a message (no auto split).
                           Train tab "GPUs". CI: ci_multiband_check parallel (2 gloo processes: one set of events; SIGUSR1 -> 75 -> resume).
                           Step 9: param "frozen" on any SHAREABLE block (effective keeps it only when true; DSL word frozen; shapes
                           NodeResult.frozen + Analysis.frozen_params; codegen FROZEN list, harness requires_grad_(False)); "lora" rank on
                           dense / attention (LoRALinear: base frozen, B·A; SelfAttention explicit projections; params count base + rank·(in+out));
                           kind moe (MoE: router + experts, top_k, nnb_aux balancing loss × AUX_WEIGHT); kind exit_output (classify only,
                           from a vector; forward collects exits -> self.nnb_exits, EXIT_WEIGHTS; extra_losses; exit_scores: per-exit accuracy
                           + stopping at 90 % confidence) -> "exits" in epoch / results; ResultsTab table. DSL: frozen, lora N, moe E top K
                           hidden U, exit W (kind_word exit / detect). CI: ci_multiband_check adapters (frozen and LoRA base weights unchanged,
                           adapters learnt, trainable count = app's).
                           Sentinel step finished: scene_split ("" random | strip = eastern tenth, one tile column gap | files =
                           scene_test_image/_label) in harness scene_split(); BigEarthNet layout: read_patch_folder (one GeoTIFF per band,
                           <patch>_<band>.tif, bands resized to the largest), label tables via read_table (CSV/TSV/Parquet, pyarrow on
                           demand), folder_index for nested patch folders, split column used as given (given_val), s1_name pairs the
                           fusion group (FUSE_BY_PATH), class names with commas matched whole (split_labels); DatasetTab BigEarthNet 19
                           preset; ImageNet weights on bands: load_pretrained widens the first conv (expand_rgb: B04/B03/B02 get R/G/B,
                           others the mean, ×3/C); HF band arrays: HFBands + hf_array (arrays, GeoTIFF / .npy bytes), problems() allows
                           huggingface (not fused). CI cases: scene strip, ben, pretrained_ms, hfbands (local parquet as the HF repo).
                           Cloud source: Dataset.cloud_url (gs:// | https://drive.google.com/… | rclone:remote:path | file:// for CI;
                           dataset::cloud_url_problem) with source local: no Mac folder, no upload; harness cloud_fetch() into
                           <data>/cloud/<key> (google-cloud-storage with ~/.config/nnb/gcs-key.json or anonymous, gdown, rclone copy);
                           Creds.gcs_key (Keychain "gcs", set_gcs, Connection tab) copied with copy_tokens to SSH/AWS; GeeBox
                           "Export to: Cloud Storage" (toCloudStorage, nnb_export/ prefix). Real gs:// / Drive / rclone not tested
                           (CI uses file://).
                           Kaggle runs: data in /tmp/nnb_data (not /kaggle/working, the notebook output), checkpoint removed at the
                           end, kaggle_output deleted after metrics/results/model/log are copied; the run card shows the last check.
                           Kaggle allows 2 GPU notebooks at a time: a push refused for that (runs::is_session_limit) makes the run
                           "queued" (RunInfo.kaggle_attach keeps the dataset slug) and status() pushes it again when due, so 3 or 5
                           seeds no longer fail; the Train tab follows the shown run every 3 s and the project's other active runs every 30 s
                           (so a queued seed is pushed while another seed is shown). Kaggle CLI calls are tried 3 times on
                           connection errors (cloud::kcap / transient).
  dataset.rs               Multispectral / radar: bands, band_order, sensor (s2 | s1 | other), s1_db; SENTINEL2_BANDS (EuroSAT order),
                           SENTINEL2_GEE, SENTINEL1_BANDS, band_key (B4 = B04, used in problems() and harness band_key); input channels =
                           bands.len() (≤ 32); from Kaggle or own files. Source "local" (own files, e.g. Google Earth Engine exports):
                           local_dir (Mac path, or local_on_machine = a path on the SSH machine), layout "" | "scene" (scene_image +
                           scene_label, segmentation tiles); kaggle_train/test/mask_dir are reused as the subfolders; local_key() =
                           nnb-<name>-<fnv hash> = the folder's name on the machine (<workdir>/data/local/<key>, runs::upload_local with
                           .nnb_signature) and on Kaggle (cloud::kaggle_upload: private dataset <user>/<key>, staged as links in
                           tools/kaggle-upload, signature in <settings>/uploads.json; create / version --dir-mode zip). codegen adds
                           local_key to DATASET. localdata.rs: files, signature, describe (class folders, first GeoTIFF header via
                           tiff_info: size, bands, GDAL band names from tag 42112; TIFF + BigTIFF). main.rs: local_scan, pick_folder.
                           harness: data_root/local_root, tiff_read (installs imagecodecs + reloads tifffile when a compressed GeoTIFF needs it:
                           Earth Engine exports are LZW; Kaggle has tifffile without it), read_array + pick_bands (S1 already in dB = negative values: not converted),
                           normalise (NaN -> 0), read_mask/mask_classes (non-class values -> -100, ignored in loss, evaluate, units),
                           BandFolder, BandPairs (multiband images + masks), SceneTiles (scene -> cached .npy, memory-mapped tiles;
                           tiles need ≥ 50 % data and one labelled pixel), multiband_stats (NaN-aware), composite() (2 bands -> a, b, a-b).
                           DatasetTab LocalBox + GeeBox (generated GEE script: Code Editor scene + WorldCover, Python class patches).
                           Multi-label: task "multilabel", labels_csv (localdata::csv_classes + main.rs local_csv read the class names;
                           HF: a ClassLabelList column, hf_info reports Sequence/List of ClassLabel so); harness label_rows/LabelFiles,
                           loss_of (BCE with logits), multilabel_scores (micro/macro F1, mAP, exact, per_class); evaluate returns micro F1
                           as accuracy, the scores dict in the confusion slot, mAP in the mIoU slot; val_map per epoch; Results/Train tabs.
                           Regression: task "regress" (numbers per image), Dataset.classes = the target columns (shapes needs ≥ 1),
                           labels_csv required for kaggle/local; localdata::csv_classes "numeric" (columns numeric in every row, min/
                           max/mean); losses "" = mse | l1 | huber (train_script refuses class losses for regress and vice versa).
                           Harness REGRESS: label_rows reads float targets (non-numeric rows left out), hf_targets for HF, target_stats()
                           -> TARGET_MEAN/_STD ("targets" event), loss_of standardises, evaluate -> regression_scores (r2 in acc slot,
                           scores in confusion slot, MAE in mIoU slot; val_mae, no train_acc), monitor val_miou falls back to val_loss;
                           results r2/mae/rmse/per_target/scatter/target_stats + target_stats.json; data_report "targets". runs listing
                           "task". UI: Dataset task radio, LabelsCsv target checkboxes, HF target columns; Train tab losses + R² chart;
                           ResultsTab cards, per-target table, Scatter, samples. CI: ci_multiband_check regress (Huber, early stop R²).
                           Fusion: bands2, band_order2, sensor2, s1_db2, fuse_dir (second group: a folder inside the dataset folder, never a full path (problems()), same relative file names, stacked after
                           group 1); group_channels(); Input param "group" (0 all, 1, 2): shapes allows two Inputs only when fused (one
                           per group), codegen slices x[:, :c1] / x[:, c1:], DSL `input 1` / `input 2`; harness GROUPS, read_groups,
                           fuse_path, set_fuse; Design inspector "Bands" + "Add an Input for band group 2"; DatasetTab SecondGroup.
                           CI ci_multiband_check.py (examples/multiband_case.rs): Kaggle S2/S1 + gee, s1gee, scene, segfiles, fusion
                           (mid), multilabel (early fusion) cases.
  dataset.rs               Dataset (builtin | huggingface | kaggle; task classify | segment; resize (built-in); hf_mask_col, kaggle_mask_dir;
                           image_size, channels, classes), BUILTINS, input_shape, problems()
  shapes.rs                analyze(graph, dataset) -> Analysis: computation order (Kahn), per-node in/out shapes, params, rf, padding/stride
                           used, plain-words messages (status ok/warn/error/blocked/unconnected), edge shapes; parameter readers (int, pair, text)
  codegen.rs               TrainSettings (incl. augmentation aug_*); model_code (Net with forward following the graph, LAYERS, EXPECTED); train_script = header (DATASET,
                           TRAIN, CLASSES as Python literals) + model + python/harness.py (include_str!)
  store.rs                 home() (NN_BUILDER_HOME), settings.json (projects, last_project, projects_dir, machines, prepared), Project (files);
                           the draft: a demo opens as the one unsaved project in <settings>/draft (not in the projects list); save_as moves
                           it into the projects folder; discard_draft throws an unsaved one away: on the next demo (create_draft), on
                           create/open of a project, and when the app quits (RunEvent::Exit) or starts
  remote.rs                Machine (ssh | aws) + clean() validation, ssh/rsync commands (ControlMaster ~/.ssh/nnb-%C, BatchMode), connect_command
                           (one-time login), prepare_script (+ parse_prepare: GPUs, SLURM, systemd+linger, tmux, timeout, PyTorch; no Python set + /opt/pytorch/bin/python
                           = the AWS Deep Learning AMI's; the Python's venv activate is sourced there and in run.sh (that AMI keeps CUDA
                           inside the venv); pip falls back to sudo -n), choose_launch,
                           put_secret_script (file via stdin, chmod 600)
  cloud.rs                 tools venv (settings/tools: kaggle + boto3), kaggle_test/files/push/status/output (CLI with env credentials and a
                           private KAGGLE_CONFIG_DIR), class_folders, aws() (python/helpers/aws_ec2.py: describe/start/stop)
  runs.rs                  spot: run_script (run.sh with flock, time budget over resumes, IMDS watcher), RESUME_SH + crontab @reboot,
                           Poll, aws_explain, mirror_checkpoint, cancel_remote, resume_on (see "Spot instances" below)
                           start (Kaggle push | ssh/aws: aws_up, prepare if needed, optional token copy, run.sh with timeout + exit code +
                           shutdown on AWS, rsync, launch via slurm/systemd/tmux/nohup), status (rate-limited polls: Kaggle 30 s, ssh 3 s;
                           metrics appended by byte offset; fetch when ended), stop, fetch (AWS: start/copy/stop), listing, results
  jobs.rs                  one background job with a log (Jobs), stream/capture for external commands
crates/core/tests/         design.rs (shapes, codegen compiles with python3 -m py_compile, machine validation), drafts.rs (demo drafts),
                           flows.rs (whole ssh run and Kaggle run through stand-in ssh/rsync/python/kaggle scripts)
crates/core/examples/dump_cases.rs   300 random designs -> JSON for python/verify_shapes.py (GitHub Actions only)
python/harness.py          the training part of every train.py (spot: SIGUSR1 = checkpoint + exit 75, SIGUSR2 = checkpoint now, timed and atomic
                           checkpoints, mid-epoch resume; see "Spot instances") (data: torchvision / datasets / ImageFolder from Kaggle; segmentation: SegPairs of
                           image + mask (HF mask column, or Kaggle images/masks folders paired by name), per-pixel loss, pixel accuracy + mean IoU;
                           augmentation on training data only (seg: image and mask together); one-batch check against
                           EXPECTED; AMP; checkpoint.pt + resume; SIGTERM/SIGHUP = stop + save; metrics.jsonl + "NNB_EVENT" lines; results.json)
python/helpers/aws_ec2.py  boto3 start/stop/describe (+ lifecycle, state reason, spot request; refuses to start an interrupted spot instance) (run with the tools venv; credentials in the environment for that call only)
python/verify_shapes.py    PyTorch comparison (CI)
app/src-tauri/src/main.rs  Tauri commands (thin: they call nnb-core), keychain (service "NN Builder": kaggle, huggingface, aws), Hugging Face
                           datasets-server/whoami calls (ureq), the menu bar (build_menu / on_menu): App (About, Settings… ⌘,), File (New
                           Project ⌘N, Open Project… ⌘O with a folder dialog, Save Project… ⌘S (the draft), Close Project (no project open: the Projects page;
                           an unsaved draft is discarded); Projects page: Delete per project (asks in the row, then project_delete moves
                           the folder to the Trash with the trash crate, never a hard delete, and store::forget takes it off the list;
                           a missing folder is only removed from the list), New from Demo ▸, Import/Export Design… (model.json, dialogs from
                           tauri-plugin-dialog), Export train.py ⌘E, Show Project in Finder ⇧⌘R), Edit (Undo ⌘Z / Redo ⇧⌘Z of the design,
                           Cut/Copy/Paste/Select All, Arrange ⌘L), View (Night Mode ⇧⌘N), Connection, Window, Help (Documentation ⌘? opens the manual PDF,
                           bundled as a resource from doc/ via tauri.conf.json; Check for Updates…). Dialog actions run in Rust and emit
                           "project-changed" / "toast"; the rest is emitted as "menu" (id) to the window. Undo/Redo: App.tsx keeps snapshots
                           (Design calls remember(prev) before each change; changes within 0.6 s are one step; in a text field the menu
                           items undo the typing instead)
web/src/                   Design.tsx: Save as PNG (PngExporter inside <ReactFlow>: getNodesBounds + html-to-image toPng of the viewport at 2×,
                           ≤ 16,000 px, glow off via .png-export; main.rs design_png writes export/<name> architecture.png; menu
                           File → Export Design as PNG = event nnb-export-png). The canvas reads left to right (handles left/right; arrange() = a column per depth; designs saved in the
                           old top-to-bottom view are arranged once when opened); it also has the "Famous architectures" picker (template_apply keeps the old design as model.previous.json and
                           fits the dataset) and the skip-connection tool (inserts Add/Concat after the selected block, with a 1×1 strided
                           convolution / pooling / Dense projection when shapes differ); the layer table (nothing selected) is grouped
                           by groupLayers(): Input, image stages by resolution (pool + a following Dropout stay in their stage), Embedding
                           and Encoder block k for transformers, Classifier; collapsible, with parameter subtotals; every block name in
                           its kind's colour (COLORS, neon at night), status colour only for warnings/errors;
                           App.tsx (Projects page: New project and Open side by side (Open: a path field (store::project_path cleans quotes, ~/, a file
                           inside the project) + Open folder + Choose… = pick_folder), under them "Running Task" (heading neon pink; runs_active =
                           runs::active_everywhere: starting/running/queued runs of every listed project from run.json, no network, re-read
                           every 30 s; a tile opens the project at that run on Train), then "Opened projects" as tiles, most recent first, with a
                           description (project_describe = store::set_description, "description" in project.json, empty removes it), folding
                           (localStorage "nnb-recent-open"), then the demos as clickable tiles in a section that
                           folds (remembered in localStorage "nnb-demos-open"); .page scrolls at the
                           window edge with the content centred, at most 1180 px; demo cards on the Projects page and File → New from Demo open "<Architecture> demo" as the draft: nothing
                           in the projects folder until Save… / File → Save Project… ⌘S (project_save_as); unsaved, it is discarded
                           without asking (the owner's choice); runs refuse a draft);
                           App.tsx (top bar: Project ▾, Design, Dataset, Code, Train, Results, and Connection set apart on the right; job panel;
                           Code and Project tabs), Design.tsx (canvas, palette, inspector), Preview.tsx (sliding window, charts, images),
                           DatasetTab, TrainTab, ResultsTab, ConnectionTab, api.ts (typed invoke wrappers)
                           theme.ts: day / night (neon) theme: `data-theme` on <html> switches the CSS variables in styles.css; night
                           draws boxes and text as neon glows (each layer glows in its kind's colour made bright by neon(), passed as
                           --c); remembered in localStorage; the ☾/☀ button in the top bar and View → Night Mode; also sets the window's
                           dark title bar (permission core:window:allow-set-theme). Canvas drawings read the variables (cssVar) and redraw
                           on a theme change (useTheme). New colours go in styles.css as variables, not hard-coded in components.
                           Night panel colours: in night mode the Design screen is three separate panels (palette, canvas, inspector)
                           whose backgrounds are --night-palette/-canvas/-inspector, chosen in the Settings tab (⚙ in the top bar,
                           NN Builder → Settings… ⌘,): presets + colour pickers, kept in localStorage "nnb-night-panels" (theme.ts
                           setPanels/usePanels); the Settings tab also has the theme and the automatic-update switch.
.github/workflows/ci.yml   core tests + PyTorch comparison on GitHub's machines
doc/nn_builder_manual.tex  the user manual (LaTeX, plain thebibliography, no biber): app tour, step-by-step design, arithmetic, block
                           reference, dataset/code/connection/train/results, spot handling, the 15 demos (numbers from the app),
                           troubleshooting; nn_builder_manual.pdf is committed next to it. Build: cd doc && latexmk -pdf
                           nn_builder_manual.tex. Keep it in step with features (new blocks, demos, settings, menus) and recompile.
                           doc/figures/*.jpg: screenshots (1440×900 window, JPEG). Take them from a copy started with a neutral
                           NN_BUILDER_HOME (e.g. /Users/Shared/nnb-screens with projects_dir "~/NN_Builder_Projects", auto_update false) so
                           no user name, path or real project shows; never the Connection tab (it shows Keychain account names).
                           Kinds of data (sec:data-kinds): doc/figures/kind_*.png = GENERATED example data drawn by
                           doc/screens/data_pictures.py (numpy + matplotlib; no download), data_*.png = the Dataset tab drawn by the real UI
                           (web/dist) in headless Chrome with doc/screens/stub.js answering app_state / project_load from
                           examples/doc_screens.rs (no app window needed; PYTHON=<python with Pillow> trims the empty page):
                           (cd web && npm run build) && doc/screens/render.sh. Re-run both after Dataset-tab changes.
                           doc/test_architectures.tex (sec:tests, \input after the demos): the 18 TEST projects of 2026-11-07 (TEST_APP.md),
                           each with doc/figures/tests/testNN.jpg (Design screen, night, taken from copies in a neutral NN_BUILDER_HOME,
                           1500 px JPEG) and testNN_arch.png (Save as PNG, split into rows of ~3,000 px at empty gaps), what it tests, the parameter arithmetic of its main blocks and a per-block table (per kind for
                           > 40 blocks). The tables were generated once from nnb_core::shapes::analyze of those projects (the projects
                           are the owner's and stay out of the repository); edit the text by hand.
                           The MSG-Former section (sec:msg) has the layer table, parameter and compute breakdowns and ablations: if the
                           template changes, update those numbers (templates test pins the total).
```

## Build, test, install (macOS)

- Quit NN Builder (Cmd-Q). `cd web && npm ci && npm run build`; `cargo test -p nnb-core`; `cargo clippy -p nnb-core -p nn-builder-app` (no warnings);
  `cd app && npm ci && npx tauri build`; replace `/Applications/NN Builder.app` with `target/release/bundle/macos/NN Builder.app`, then delete
  the bundle output (one copy only).
- Look at the window with a scratch settings folder: `open -n --env NN_BUILDER_HOME=<scratch> "target/release/bundle/macos/NN Builder.app"`
  (starting the binary directly may give a process without a window).
- The keychain items belong to the app; an unsigned rebuild may make macOS ask once to allow access to them again ("Always Allow").
  main.rs caches each item in memory after the first read (SECRETS), so a running app asks the Keychain at most once per item.
  Starting the app reads no Keychain item: settings.json "secrets_meta" (Kaggle user name, whether a Hugging Face token exists, the
  first characters of the AWS key id; nothing secret) is shown instead, updated by set/clear; the Hugging Face dataset calls read the
  token only when secrets_meta says one is stored (hf_token()).

## Automatic updates

- `app/src-tauri/src/update.rs` (adapted from the LaTeX Editor): `check` reads `releases/latest` of `REPO` without signing in (override with
  `NN_BUILDER_UPDATE_URL`); `pick_asset` by OS/arch from the names the Build workflow gives (`NN-Builder_X_aarch64.app.zip`,
  `NN-Builder_X_x64-setup.exe`, `nn-builder_X_amd64.AppImage` / `.deb`); `download` checks size and the release's `digest` SHA-256; `install`:
  macOS unzips (ditto) next to the running bundle and swaps by rename; Windows runs the installer at exit; AppImage replaced; .deb via xdg-open.
- Commands `update_check(manual)` (automatic at start only when `auto_update` is on and `last_update_check` is over a day old; both in
  settings.json), `update_install` (a background job ending in "UPDATE_READY"), `update_restart` (Tauri restart, or installer + exit).
  Window: the banner under the top bar; Help menu → Check for Updates…; the switch on the Projects page.
- If the packaged file names change, change `pick_asset`, its test and `.github/workflows/build.yml` together.

## Releasing a version

1. Raise the version in `Cargo.toml` ([workspace.package]), `app/src-tauri/tauri.conf.json` and `web/package.json`; `cargo build` (Cargo.lock);
   test; commit; push `main`.
2. Create a draft release `vX.Y.Z` with the notes (API, token header file; see ../HOW_THINGS_WERE_DONE.txt), then push the tag `vX.Y.Z`.
   The Build workflow builds macOS / Windows / Linux and its `release` job attaches the four files (it refuses a tag that differs from the version).
3. When the files are attached, publish the draft (`draft: false`): only then does `releases/latest` show it and the apps offer the update.
   Check: `curl https://api.github.com/repos/varunjha089/NN-Builder/releases/latest` lists the assets with a `digest`.

## Rules for changes

- A new block kind: model.rs (KINDS, defaults, arity), shapes.rs (step), codegen.rs (modules), Design.tsx (COLORS, GROUPS, summary, inspector
  fields), dump_cases.rs (so CI compares it with PyTorch). A new template: templates.rs (TEMPLATES + build); the template test checks it is valid
  at its input and compiles.
- Git: every change is committed (one commit per logical change, message says what and why; the owner asked for this on 2026-11-04).
- Never run PyTorch, train or download datasets on the Mac (tests included). No real Kaggle/AWS/SSH calls in tests: stand-ins only.
- Remote commands only through remote.rs (validated fields, sh_quote, scripts on ssh stdin). Secrets only through the keychain and env/stdin.

## What NN_Builder is

A desktop app (Rust) for **designing** neural networks on a canvas. The Mac only designs: it never trains, never downloads datasets and needs
no PyTorch. A finished design becomes a complete Python (PyTorch) training script, including loading a dataset from Kaggle or Hugging Face, and
the app runs it automatically on a GPU somewhere else:

- **Kaggle** (free GPU notebooks, through the Kaggle API),
- **a college GPU server** (SSH, password or key),
- **an AWS EC2 g4dn.xlarge** (NVIDIA T4 16 GB, 4 vCPU; SSH with a key file).

One button ("Train on ..."): the app generates the code, uploads it, sets up the machine (once), starts the run, follows it, brings the results
back and, on AWS, stops the instance. On the college server and AWS the run must not depend on the SSH connection: it keeps going when the
Mac sleeps, the network drops or the app closes, and the app reconnects to it later.

## Stack

| Part | Choice | Why |
|---|---|---|
| Desktop app | Tauri 2 (Rust backend) | Same as CourseShelf (proven builds for macOS/Windows/Linux, own updater); a real installer, no browser-window launcher |
| Canvas | Web UI in the Tauri window: React + TypeScript + React Flow (node graph) | A graph, not a block stack, so branching designs work: ResNet skips, U-Net, Inception, concatenation |
| Shape check, code generation | Rust (no Python on the Mac) | Instant, offline; one engine for the shape check and the generated code |
| Generated training code | Python 3.9+ with PyTorch + torchvision (+ `datasets` / `kaggle` packages as needed) | Runs on all three targets; fastest training (CUDA) |
| Remote control | System `ssh`/`rsync` (OpenSSH) for college/AWS; Kaggle REST API; Hugging Face Hub API; AWS EC2 API (`aws-sdk-ec2` crate) | `~/.ssh/config`, agents, ProxyJump and host-key checks work as in the Terminal |
| Secrets | macOS Keychain (Windows Credential Manager / Secret Service on other systems), via the `keyring` crate | Never in settings files, project files, generated code, logs or command lines |

A from-scratch C++ engine is NOT part of the app: training speed comes from GPU kernels (CUDA/cuDNN), and the Mac does not
train. It could come back later as an optional "export to C++" for teaching.

## 1. The canvas

- **Nodes** = layers: Input, Conv2D, Depthwise Conv2D, Pool (max/avg), Global pool, BatchNorm, LayerNorm, Dropout, Flatten, Dense, activations,
  Add (residual), Concat, Upsample / ConvTranspose (U-Net), Output (classification head; later segmentation head).
- **Edges** = tensors. Each edge shows its shape, e.g. `(64, 14, 14)`. A node with problems turns red (blocks Train) or yellow (warning), with the
  reason in plain words on the node.
- **Setting filters, kernel, stride, padding, pooling** (from the earlier plan, still valid): fields on the node; kernel and stride have a
  "square" link (unlink for height × width); padding `same` / `valid` / a number; pool stride "same as pool size" unless unticked; dilation and
  bias under "Advanced"; presets (3×3 keep size, 3×3 stride 2, 1×1, 5×5, 7×7 stride 2).
- **Shape rule** (shown on click, with the node's own numbers): `out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1`. `same`
  with stride > 1: the app computes explicit padding (PyTorch refuses `padding="same"` there).
- **Checks:** output size < 1, Dense without Flatten/Global pool, Add of different shapes, Concat with different heights/widths, class count
  ≠ dataset classes, Output not last or missing (red); stride drops rows, stride skips pixels, a very large Dense after Flatten, even kernel with
  `same` (yellow).
- **Teaching aids:** the sliding-window animation for Conv/Pool, receptive field per node, a strip showing the image getting smaller and deeper.
- **Templates:** LeNet-5, a small VGG, ResNet-18, U-Net, as starting points.
- **Saved as** `model.json` (nodes, edges, positions) in the project folder; the Rust engine is the only reader/writer.

## 2. From design to Python

- The Rust engine turns the graph into a `train.py` with: the `nn.Module` (a `forward` that follows the graph), the dataset code (section 3), the
  training loop (Adam/SGD, learning rate, batch size, epochs, mixed precision on CUDA), **a checkpoint every epoch and automatic resume**, and
  `metrics.jsonl` (one JSON event per line: start, batch, epoch, done, stopped, error), `results.json` (test accuracy, confusion matrix, sample
  predictions, first-layer filters, feature maps) and `model.pt`.
- **First a one-batch check** on the GPU machine: load one batch, run forward + backward, compare the real shapes with the app's; on a mismatch
  stop at once with a clear message (the Mac cannot test this itself).
- **Correctness of the Rust shape engine:** GitHub Actions (not the Mac) compares it with a real PyTorch forward pass on many random graphs
  (shapes and parameter counts) on every push. A red check there blocks a release.
- The generated code is readable and also downloadable ("Export train.py" / a Jupyter notebook).

## 3. Datasets (chosen on the Mac, downloaded only on the GPU machine)

- **Hugging Face:** search and pick a dataset; the app reads its splits, features and first rows through the public datasets-server API (no
  download) and lets you choose the image column and the label column (class names from the `ClassLabel` feature). Generated code:
  `datasets.load_dataset(name, split=...)` + transforms. Gated/private datasets need a Hugging Face token on the target.
- **Kaggle:** search and pick a dataset; the app reads its file list through the Kaggle API and recognises the common layouts (folder per class,
  CSV of file + label). On Kaggle it is attached to the notebook directly; on college/AWS it is downloaded with the Kaggle API on the machine.
- **Built-in:** MNIST, Fashion-MNIST, CIFAR-10 (torchvision downloads them on the machine).
- The app shows the size before a run; the data is cached on the machine (`~/.nn_builder/data`) and not downloaded again.

## 4. Where training runs

### Kaggle GPU (no SSH)

- The app pushes the generated script as a Kaggle notebook through the API (`kernels push` with metadata: GPU on, internet on, the dataset
  attached), checks `kernels status` every minute, then downloads the outputs (`kernels output`: metrics, results, model).
- Runs entirely on Kaggle's servers, so it is independent of the Mac by design.
- Limits: about 30 GPU hours a week, about 12 hours per run, runs may queue; the account must be phone-verified for GPU and internet. Whether the
  API can choose the GPU type (T4 ×2 or P100) must be checked. A gated Hugging Face dataset needs its token stored once by hand in the
  notebook's Kaggle "Secrets" (the API cannot set secrets).

### College GPU server (SSH)

- **Machine settings** store only: host (or `~/.ssh/config` alias), user, port, key file path or "password", jump host, remote folder
  (default `~/.nn_builder`), optional existing Python. Never a password or key contents.
- **Password login:** the app opens a Terminal window once (`ssh -M -f -N`, ControlMaster); the user types the password there; all later
  commands reuse that connection (`BatchMode`, never prompting). A key (`ssh-copy-id`) is recommended where allowed.
- **Prepare (once):** check Python, GPUs (`nvidia-smi`), disk; make `~/.nn_builder/venv` and install PyTorch matching the driver (no sudo).
- **A run:** `rsync` the run folder to `~/.nn_builder/runs/<id>/`, then start it detached. **Independence from SSH**, in order of preference
  (the app detects what the server allows):
  1. **SLURM** (`sbatch`) if the cluster has a job queue: the proper way on shared clusters;
  2. `systemd-run --user` (survives logout only if "lingering" is enabled, which may need the administrator);
  3. `tmux` / `nohup setsid` (some servers kill all of a user's processes at logout: the app tests this once in Prepare and warns).
- GPU on the machine (Train tab): empty = all; "0,1" = `CUDA_VISIBLE_DEVICES`; "free" = run.sh waits (runs::free_gpu_wait) until
  GPUs = N GPUs are idle (under FREE_GPU_MIB 1024 MiB used and FREE_GPU_BUSY 10 % busy in two `nvidia-smi` looks 15 s apart, every minute;
  GPUs = all: as soon as one is, taking every idle one), then sets `CUDA_VISIBLE_DEVICES` to them. Info events in metrics.jsonl, a log line
  every 10 min; not counted in the time budget; Stop while waiting -> "stopped" event + exit 143; no wait under SLURM (`SLURM_JOB_ID`).
  NNB_TEST_LOOK / NNB_TEST_EVERY shorten the pauses (test run_sh_waits_for_free_gpus). SLURM asks `--gres=gpu:<GPUs, at least 1>`.
  gpu_use() reports the GPUs named in `CUDA_VISIBLE_DEVICES` (physical nvidia-smi numbers).
- The app copies new `metrics.jsonl` lines every few seconds while open, re-attaches to running jobs by run id after a restart, and copies the
  results back when the run ends. Checkpoints + resume cover a killed job.

### AWS EC2 g4dn.xlarge (SSH with a key file)

- **Automatic start/stop** with the EC2 API: the app starts the instance, waits for SSH, runs Prepare (first time) and the run, exactly as for the
  college server (detached, independent of SSH).
- **Cost safety:** the run ends with `sudo shutdown -h now` (the instance's shutdown behaviour must be "stop", not "terminate"), plus a maximum
  run time set in the app. The app also shows the instance state and an estimated cost.
- **Results while the instance is stopped:** next time the app opens it starts the instance, copies the results and stops it again; optionally
  the run uploads `model.pt` and the results to a private Hugging Face repository so they are available without restarting it.
- **Spot instances** (built 2026-11-04; detected from the instance, nothing to switch on). Recommended setup: a **persistent** spot request with
  interruption behaviour **stop** (AWS restarts the instance when capacity returns; its disk, data cache and checkpoints survive), and the
  machine setting "Keep a copy of checkpoints on this Mac". How each situation is handled:

  | Situation | What happens |
  |---|---|
  | 2-minute interruption notice (instance metadata `spot/instance-action`) | run.sh's watcher sends SIGUSR1; train.py saves a mid-epoch checkpoint (atomic) and exits 75; `resume_wanted` stays |
  | Rebalance recommendation (earlier warning) | SIGUSR2: train.py saves a checkpoint now and keeps training |
  | Hard interruption without notice / crash of the instance | at most `checkpoint_minutes` of training lost (spot default 10 min when the setting is 0) |
  | Interrupted during a checkpoint write, or a full disk | `torch.save` to `.tmp`, fsync, `os.replace`: the previous checkpoint stays whole; a failed write is a warning |
  | Interrupted during evaluation / final results | checkpoint says "epoch trained, not evaluated" (or "all epochs done"); the resume only evaluates |
  | Instance restarted by AWS while the Mac is off | crontab `@reboot` runs `resume.sh` (after 45 s for the GPU driver): the run continues by itself |
  | Instance back while the app is open | the app follows the new public address (describe), sees the run not running with `resume_wanted`, runs `resume.sh now` |
  | Two starters at once (cron + app) | `flock` on run.lock, and resume.sh skips a run whose train.py (py_pid.txt) is alive: one copy only |
  | Resumed mid-epoch | `EpochOrder` sampler fixed by (seed, epoch): the remaining batches of that epoch in the same order; running sums, GradScaler and RNG restored; a different batch size restarts the epoch |
  | Interrupted again and again / price too high | at most `max_resumes` automatic resumes (default 20; then exit 76 "gave up"); `max_hours` counts running time summed over resumes (used_seconds.txt) |
  | Instance terminated (one-time request / behaviour terminate) | describe says terminated: state "terminated"; with the Mac copy, **Continue on another machine** (resume_on: new run, same train.py + checkpoint + metrics; the old run is marked "continued") |
  | Old instance comes back after the run moved elsewhere | when that run is opened and its instance answers, the app stops it there and removes resume_wanted; the note says to cancel the spot request in the console (the app cannot act while closed) |
  | Stop pressed while the instance is down | `cancel` is recorded: stopped (not resumed) as soon as it answers |
  | Starting an interrupted spot instance by hand / no capacity | explained, not retried in a loop: AWS restarts it itself; "insufficient capacity" says try later or use another machine |
  | User Stop / time limit | TERM: the run ends normally with results; resume_wanted removed; never resumed |
  | Shutdown without the notice (stop from the console, maintenance, a missed notice) | spot only: run.sh exports NNB_SPOT=1; TERM while `systemctl is-system-running` says stopping makes train.py checkpoint and exit 75 and run.sh keep resume_wanted (it was a Stop before: found by the first real spot test, 2026-11-08; CI: ci_resume_check with NNB_TEST_SHUTDOWN) |

  Mac side: status asks AWS (describe, at most every 30 s) instead of ssh while the instance is down; checkpoints are copied in a background
  thread with rsync (temporary file + rename, so the copy is always whole; mirror.json says which one). The window shows the last checkpoint
  and the Mac copy. Tests: flows.rs runs the whole spot path with stand-ins (fake EC2 metadata via curl, fake crontab; the stand-in ssh drops
  `bash -l` so the real crontab is never used, and the test asserts that; never `stop_instance` in tests, the stand-in instance is this Mac);
  the PyTorch side (mid-epoch checkpoint, exit 75, resume at the right batch, SIGUSR2) runs in CI: examples/resume_case.rs +
  python/ci_resume_check.py.
- AWS key: an IAM access key allowed only `ec2:StartInstances`, `ec2:StopInstances` for that one instance, `ec2:DescribeInstances`,
  `ec2:DescribeInstanceAttribute` (the shutdown behaviour), plus `ec2:DescribeSpotInstanceRequests` for spot details (optional). A new AWS
  account must first get a G-instance vCPU quota (often 0 at the start).
- Cost guards (added 2026-11-07, before the first real EC2 run; AWS_TEST_PLAN.md): runs::aws_up describes first and refuses an on-demand
  instance whose shutdown behaviour is Terminate (a note when it cannot be read); it returns "was" (the state before). When this call
  started the instance (started_here) and anything fails before train.py is launched (aws_up's own ssh wait, start_remote ->
  launch_remote, resume_on, Prepare, fetch), aws_stop_again stops it and the error says so (or "STOP IT IN THE EC2 CONSOLE"); an instance
  that was already running is left running. Tested in flows.rs with FAKE_EC2 (calls.txt).

## 5. Secrets

Kaggle API token, Hugging Face token, AWS access key: in the Keychain only. When a run needs one on a remote machine (Kaggle download on
college/AWS, gated Hugging Face dataset, upload of results), it is written there as a file readable only by the user (`chmod 600`, e.g.
`~/.kaggle/kaggle.json`, `~/.cache/huggingface/token`) through ssh stdin, never on a command line, and the app says so before the first time.
Nothing secret is ever written into `train.py`, `model.json`, `metrics.jsonl`, logs or the repository.

## Build order (each step usable on its own)

1. Rust app: canvas, shape engine, checks, code generation, Export train.py; GitHub Actions comparison with PyTorch.
2. Datasets: Hugging Face and Kaggle pickers and the generated loading code.
3. **Kaggle runs** (no SSH, nothing to install: the quickest complete loop).
4. College GPU over SSH: Prepare, detached runs (SLURM / systemd-run / tmux), reconnect, results.
5. AWS g4dn: start/stop through the API, self-stop, cost display, results from a stopped instance.

## Rules (from the workspace CLAUDE.md, apply once code exists)

- Program and data are separate: projects live in folders the owner chooses; tests use scratch folders and never the owner's projects.
- **The Mac never trains and never downloads datasets.** Tests on the Mac use the Rust engine only; anything needing PyTorch runs in GitHub
  Actions. No real Kaggle/AWS/college runs in tests: stand-ins for `ssh`, `rsync`, the Kaggle API and the EC2 API.
- No real cloud action (starting an EC2 instance, pushing a Kaggle notebook) without the owner asking.
- Every feature gets a test; the shape engine keeps its PyTorch comparison green.
- After each change: update this file, commit and push (once the repository exists), build and install, never lose user data.

## Open questions for the owner (still open; the app handles every case: it detects SLURM etc.)

1. Does the college server use SLURM (a job queue), or do you just ssh in and run?
2. Does your AWS account already have a quota for g4dn (G and VT instances)? On-demand only, or also spot?
3. Is your Kaggle account phone-verified (needed for GPU and internet in notebooks)?
4. Public GitHub repository like CourseShelf and The Data Application?

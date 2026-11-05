# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Status

Version 0.3.0 (2026-11-05). Public repository github.com/varunjha089/NN-Builder (MIT); releases there feed the app's automatic update.
Kaggle, a college server and EC2 are tested with stand-ins only (never against real accounts in tests). The earlier Python/Blockly app
(0.1.1, 2026-11-03) is in the Trash.
Open work (owner tasks, the design language, models, workflow, Sentinel-1/2) is in FUTURE_PLAN.md; the design language in
PROGRAMMING_LANGUAGE.md. CHANGELOG.md lists every commit by time (dd_mm_yy_hr_min_sec, newest first); after committing, run
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
                           torch.utils.checkpoint; generated only when on); bf16 on compute capability 8+, channels-last for conv nets,
                           workers "auto" = min(8, CPUs) with prefetch 4, "memory" event (measured peak after the first epoch),
                           data_wait per epoch; Train tab "GPU memory" box and run-card notes; status polls slow to 15 s when the
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
                           verbatim snippet, so update both together); doc/examples/everything.nnb = every language feature and block kind, checked in tests/text.rs
                           and run in PyTorch by CI via dump_cases) is described in PROGRAMMING_LANGUAGE.md (section 9: as built, what is left) (FUTURE_PLAN.md: Sentinel-1/2).
                           Large designs (measured 2026-11-05, examples/big_designs.rs builds 1B-100B ViTs from text and times them):
                           cost follows blocks, not parameters; the window needs ~0.3 MB per block drawn (3,507 blocks ≈ 1 GB),
                           the canvas suggests the ×N view above 500 blocks. Most blocks per design: Settings → Design size
                           (settings.json max_blocks, null = recommended = dsl::recommended_max_blocks(RAM): a quarter of memory at
                           0.3 MB/block, ≥ 5,000; RAM from sysctl / /proc/meminfo / CIM; dsl::set_max_blocks at start and on save). shapes::analyze and dsl/codegen use id maps (no
                           per-block search: 3,507 blocks analysed in ~21 ms); Design.tsx reuses unchanged React Flow nodes. Limits:
                           shapes::limits() per kind (filters 65,536, heads 512); a value outside is an error, never clamped silently.
                           Help menu: Documentation opens the bundled manual PDF (tauri.conf.json bundle.resources).
                           Kaggle runs: data in /tmp/nnb_data (not /kaggle/working, the notebook output), checkpoint removed at the
                           end, kaggle_output deleted after metrics/results/model/log are copied; the run card shows the last check.
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
                           (one-time login), prepare_script (+ parse_prepare: GPUs, SLURM, systemd+linger, tmux, timeout, PyTorch), choose_launch,
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
web/src/                   Design.tsx: the canvas reads left to right (handles left/right; arrange() = a column per depth; designs saved in the
                           old top-to-bottom view are arranged once when opened); it also has the "Famous architectures" picker (template_apply keeps the old design as model.previous.json and
                           fits the dataset) and the skip-connection tool (inserts Add/Concat after the selected block, with a 1×1 strided
                           convolution / pooling / Dense projection when shapes differ); the layer table (nothing selected) is grouped
                           by groupLayers(): Input, image stages by resolution (pool + a following Dropout stay in their stage), Embedding
                           and Encoder block k for transformers, Classifier; collapsible, with parameter subtotals; every block name in
                           its kind's colour (COLORS, neon at night), status colour only for warnings/errors;
                           App.tsx (Projects page: New project and Open side by side, then the demos as clickable tiles in a section that
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
- Free GPU chosen from `nvidia-smi` (shared GPUs); `CUDA_VISIBLE_DEVICES` set for the run.
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

  Mac side: status asks AWS (describe, at most every 30 s) instead of ssh while the instance is down; checkpoints are copied in a background
  thread with rsync (temporary file + rename, so the copy is always whole; mirror.json says which one). The window shows the last checkpoint
  and the Mac copy. Tests: flows.rs runs the whole spot path with stand-ins (fake EC2 metadata via curl, fake crontab; the stand-in ssh drops
  `bash -l` so the real crontab is never used, and the test asserts that; never `stop_instance` in tests, the stand-in instance is this Mac);
  the PyTorch side (mid-epoch checkpoint, exit 75, resume at the right batch, SIGUSR2) runs in CI: examples/resume_case.rs +
  python/ci_resume_check.py.
- AWS key: an IAM access key allowed only `ec2:StartInstances`, `ec2:StopInstances`, `ec2:DescribeInstances` for that one instance, plus
  `ec2:DescribeSpotInstanceRequests` for spot details (optional). A new AWS account must first get a G-instance vCPU quota (often 0 at the start).

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

# Changelog

Every change to NN Builder, newest first. Each entry is headed by its time as `dd_mm_yy_hr_min_sec` (day, month, two-digit year,
hour, minute, second; the commit time) and the commit. Regenerate with `python3 scripts/make_changelog.py`.

## 05_11_26_17_58_10

**Test: read the manual chapter with Windows line endings too** (`5a2f05d`)

The Windows build checks files out with CRLF, so manual_snippets_build found no snippets there.

## 05_11_26_17_50_07

**dsl: clippy (redundant slice)** (`43fffbb`)

## 05_11_26_17_49_50

**Manual: a full chapter on writing designs in the NNB language** (`2dd973d`)

doc/language.tex (included by the manual) teaches the language from the first line to functions and
loops: settings, connections, labels, let and arithmetic, repeat, for, def, several inputs and paths
into groups, a table of every block, the rules in brief, the error messages, and how text and canvas
stay in step. Its examples are files in doc/examples/ (LeNet-5, a small ResNet with a function and a
loop, a small ViT, a small U-Net) shown with \VerbatimInput; tests/text.rs builds every example (no
error or warning, text kept, code compiles) and every snippet in the chapter. A misspelled setting now
suggests the nearest word ('stide' -> 'stride').

## 05_11_26_17_44_03

**Docs: README, manual and CLAUDE.md for the changes since 0.3.0; scripts/make_changelog.py** (`3469f15`)

README: Text tab and the design language, menus (Close Project, Help, .nnb), Projects page Delete and
Settings (Design size), training recipe and validation, GPU memory features, how large a design can be,
and what CI checks. Manual date. CLAUDE.md: CHANGELOG.md and how to regenerate it.

## 05_11_26_17_25_15

**Settings: Design size, the most blocks per design, recommended from the computer's memory** (`2e3df2f`)

The 20,000-block limit was a fixed guard. It is now a setting: recommended as a quarter of this
computer's memory at the measured 0.3 MB per block (16 GB: 13,000; 124 GB: 105,000; at least 5,000),
and editable (100 to 10,000,000), with a warning when a value could need more than half the memory.
RAM from sysctl (macOS), /proc/meminfo (Linux) or CIM (Windows). The repeat error names the setting.

## 05_11_26_17_19_23

**Text: a repeat that makes too many blocks is reported on its own line, with the size** (`b6276fd`)

repeat 5000 with 7 blocks inside reported 'more than 20000 blocks' on a block inside the repeat. The
repeat line now says how many blocks it would make, the memory the canvas would need (~0.3 MB per
block) and how many passes fit.

## 05_11_26_17_11_27

**Help menu with the manual; large designs: no silent limits, faster engine, lighter redraws** (`2d7787b`)

- Help menu: Documentation (Cmd-?) opens the user manual PDF, now bundled in the app as a resource;
  Check for Updates moved there (from the app and Connection menus).
- Measured 1B-100B ViT designs (examples/big_designs.rs): the cost follows the number of blocks, not
  parameters; the window needs ~0.3 MB per block drawn (3,507 blocks about 1 GB).
- Bug: Conv2D filters were silently capped at 4096 and heads at 64, so wide designs (8192+) became a
  different network that failed with a confusing Add error. Limits are now one table (filters 65,536,
  heads 512) and any value outside its range is an error on its block; the inspector allows the same.
- Engine: analyze, codegen, same_structure and patch_settings use id maps instead of a search per block
  (3,507 blocks: analysis 136 -> 21 ms, sync 44 -> 5 ms, codegen 63 -> 10 ms).
- Canvas: unchanged blocks keep their node object (no redraw of every block per edit), edges memoised,
  the x N collapse computed once; above 500 blocks a note suggests the x N view.
- Manual (size table), CLAUDE.md, README, FUTURE_PLAN.md updated.

## 05_11_26_16_45_44

**Projects page: delete a project (to the Trash) or remove a missing one from the list** (`9f89701`)

Each project row gets Delete (Remove for a missing folder), confirmed in the row. project_delete moves
the project's folder to the Trash with the trash crate (NSFileManager on macOS, so it can be put back;
never a hard delete) and only for folders with a project.json in the projects list; store::forget takes
it off the list and off last_project; an open project is closed. Manual and CLAUDE.md updated.

## 05_11_26_12_41_50

**Text and canvas in step on every change; File > Close Project** (`4ab2d53`)

- dsl::patch_structure: blocks added, removed or rewired on plain lines on the canvas change only those
  lines (new block: next free number after its input; implicit inputs become <- when the line above
  changed; a block inserted in front of a call or loop becomes its input). Functions, loops, names and
  comments stay; every result is checked against the canvas. Tested on the everything design.
- Every canvas action keeps the design text (save() carries it), so the Rust side edits lines instead
  of printing a new text. Blocks keep their id from the old design only for the same label and kind.
- Text tab: coloured words, Cmd-click on a number selects its block, applies 0.25 s after typing.
  Canvas and text stay separate tabs (owner's choice).
- Computed numbers print short (0.1 + 0.2 -> 0.3), so the text is not reprinted for rounding.
- File > Close Project: no project open, the Projects page; an unsaved demo is discarded.
- Manual, CLAUDE.md, PROGRAMMING_LANGUAGE.md and FUTURE_PLAN.md updated.

## 05_11_26_11_43_38

**Text language: settings changed on the canvas keep functions; table by line; rows; .nnb files** (`8b2eaaf`)

- dsl::sync: when the canvas changed only settings of blocks that have their own top-level line, only
  those lines are rewritten (indentation, label, inputs and comment kept) and the result is checked
  against the canvas; functions, loops, names and comments stay. Anything else still prints the text.
- Layer table: for a design written as text, ordered by the text and grouped by line (the blocks of one
  repeat / for / call line are one group named after the line; plain lines by their line range).
- A long design typed into an empty canvas is laid out in rows of 24 columns, not one wide row.
- File > Export Design... writes .nnb design text or .json; Import Design... reads both.
- FUTURE_PLAN.md now lists everything still open (owner tasks, language, models, workflow, Sentinel);
  PROGRAMMING_LANGUAGE.md section 9, the manual and CLAUDE.md updated.

## 05_11_26_11_33_49

**Text language: refuse settings a block does not have; explain canvas loops; keep unapplied drafts** (`9535756`)

Found by testing in the app:
- "1 input stride 2" was accepted and kept as a meaningless setting: stride / pad applied to every
  block kind and name=value took any name. Now stride / pad only for convolutions and pools, and
  name=value only for the block's own settings (the error lists them).
- A loop on the canvas printed "<- ?id" and the text said only "unexpected character '?'"; it now
  says the arrows go in a circle and to remove one.
- A text with a problem was lost when switching tabs; it is kept as a draft and shown again (with its
  problem) while the design has not changed.

## 05_11_26_11_26_16

**Text language: a design that uses every feature, and two parser fixes it found** (`aafa9c0`)

doc/examples/everything.nnb is a hybrid CNN + transformer written only as text: functions calling
functions with defaults and named values, several inputs (in2), for with tuples, repeat with i, let,
every operator, paths into loop passes, an omitted label, comments, mixed case, and every block kind
but seg_output. tests/text.rs builds it, requires no error or warning at 3x32x32, checks values and
wiring, compiles the generated code and round-trips it; dump_cases adds it to CI's PyTorch comparison.

Fixed: a label spelled like a block word ("cat concat <- ...") was read as the block; a name equal to a
setting word ("let heads = 2" then "attention heads heads") replaced the setting word itself.

## 05_11_26_11_09_25

**Text tab: the design as numbered lines, in step with the canvas** (`46c917a`)

A Text tab after Design shows the design text with line numbers. Typing applies it 0.5 s after it is
valid (one undo step; blocks keep their place); problems in the text are listed by line and leave the
canvas as it was; the design's own warnings and errors are shown on the lines of their blocks, and a
click selects the block on the canvas. Help lists the language (blocks, <-, repeat, for, def, let,
paths into groups); Format prints the text again; Copy. Each block shows its label in its header.
App saves through one persist() that takes the synced design (text and labels) back only when nothing
changed meanwhile. Manual section "The Text tab", PROGRAMMING_LANGUAGE.md section 9 (as built, what is
left), CLAUDE.md.

## 05_11_26_10_56_54

**NNB text engine: numbered-line design language kept in step with the canvas** (`2d1ec0f`)

dsl.rs: lexer, parser and expansion with labels, <- inputs, repeat, for (with tuples), functions
(def, defaults, several inputs in/in2), let and arithmetic; a printer that folds identical groups
into repeat (ViT-g/14: 16 lines); sync keeps hand-written text while the blocks only move, else
prints it again; apply keeps block ids and positions by label. Graph gains text, nodes a label;
save_graph syncs and returns the graph; design_from_text and design_format commands. The Text tab
in the window is still to come. Tests: tests/text.rs (all demos round-trip, every kind, functions,
loops, paths into groups, errors with lines, ids kept).

## 05_11_26_10_26_44

**Canvas: draw every block again (visible-only drawing left the minimap empty)** (`8858e24`)

onlyRenderVisibleElements never measures off-screen blocks, so React Flow's minimap had nothing to
show (seen on ViT-g/14). The memoised blocks keep large designs smooth without it.

## 05_11_26_10_24_44

**Smoother running: automatic batch split on out-of-memory, "save memory", bf16, faster loading** (`5ee361c`)

GPU side (generated train.py):
- A batch that runs out of GPU memory is processed in 2, 4, 8 ... parts whose gradients add up to the
  whole batch's, so the run continues instead of failing (same batch, schedule and results); the split
  is kept in the checkpoint. Evaluation batches halve the same way; the one-batch shape check uses two
  images so it cannot run out of memory first.
- "Save memory": activation checkpointing. Codegen cuts forward where only one tensor crosses, into
  about sqrt(n) parts, and recomputes each part in the backward pass (generated only when switched on).
- bfloat16 on GPUs that have it, channels-last for convolutional networks, data workers = CPU cores
  (up to 8) with prefetching.
- Measured peak memory after the first epoch and the share of time spent waiting for data.

Window: a GPU memory box in the Train tab (the estimate accounts for save memory), run-card notes
for the measured memory, splits and slow data loading; status polls slow down while the window is
hidden. Large designs: blocks redrawn only when they change, only visible blocks drawn above 150,
undo history capped by size, charts thinned to ~1500 points.

CI: ci_memory_check.py compares the checkpointed forward with the plain one (outputs and gradients) on
seven demos, and a run with simulated out-of-memory against a normal run (same weights). Manual and
CLAUDE.md updated.

## 05_11_26_10_17_20

**PROGRAMMING_LANGUAGE.md: detailed plan for NNB text, a numbered-block language for designs** (`07e37b2`)

Syntax (numbered lines, <- inputs, repeat blocks), the parameter table for every block kind, errors,
the printer and round-trip guarantee, window/editor/file integration, grammar, tests and steps.
FUTURE_PLAN.md points to it.

## 05_11_26_09_52_09

**FUTURE_PLAN.md: plan for Sentinel-1 and Sentinel-2 datasets** (`620e079`)

N-band input with per-band normalisation, multi-label, S1+S2 fusion, pretrained weights beyond 3
channels, composites for display; target datasets EuroSAT-MS, So2Sat LCZ42, BigEarthNet, Sen1Floods11.

## 05_11_26_09_43_03

**ImageNet weights: match BatchNorms by the convolution that feeds them** (`482c4d4`)

All BatchNorm tensors of a width have the same shape, and the ResNet-18 demo registers a downsampling
block as conv1, projection, BN, BN while torchvision has conv1, bn1, conv2, bn2, downsample. Matching by
order and shape gave the projection's BatchNorm the weights of bn2 (CI: outputs differed by 8.9).
A forward pass with hooks now finds each BatchNorm's producing convolution in both networks.

## 05_11_26_09_36_18

**ONNX export: install the onnx package on the GPU machine when missing** (`630210f`)

torch.onnx.export needs the onnx package, which CI (CPU PyTorch only), Kaggle images or a fresh venv
may lack, so model.onnx was silently skipped with a warning. The recipe check now shows the export
warnings when the file is missing.

## 05_11_26_09_31_13

**Phase 3 and 4: ImageNet weights, Grad-CAM, export, GroupNorm, x N view, Windows SSH, Intel build** (`c4380a1`)

Phase 3 (training side, checked in CI since the Mac never runs PyTorch):
- Start from ImageNet weights (torchvision resnet18 / vgg16 / vit_b_16), loaded tensor by tensor in
  order by shape, with ImageNet normalisation and an optional frozen backbone; ci_pretrained_check.py
  compares the loaded demos with torchvision's own models.
- Grad-CAM on the sample images and a gallery of misclassified test images in Results.
- Optional TorchScript (model.ts) and ONNX (model.onnx) export, fetched with the results.

Phase 4 (design and connection):
- GroupNorm block (shape rule, codegen, PyTorch comparison case).
- Copy / paste of blocks, "Residual block after selected", and a read-only x N view that collapses
  identical consecutive blocks (ViT-g/14: 40 encoder blocks shown once).
- Remote copies without rsync (tar / cat over ssh) and no ControlMaster on Windows; the flows test
  runs a whole job through the no-rsync path.
- Build workflow cross-builds macOS Intel (x86_64) and the updater picks it.
- Dataset tab: class balance and 8 example images from the datasets-server, and a clear error for
  datasets without an image column.

App start-up no longer reads the Keychain (secrets_meta in settings.json, nothing secret), and the
Hugging Face calls read the token only when one is stored, so an unsigned rebuild cannot hold up the
window with a Keychain prompt. Train tab selects stay inside their boxes. Manual, README and
CLAUDE.md updated.

## 05_11_26_09_00_16

**Phase 2: seeds, comparing runs, CSV/LaTeX export, top-5, per-class metrics, MACs** (`3f7a881`)

- Train with 3 or 5 seeds in one go (a group of runs, seeds s, s+1, ...).
- Runs table: seed column, checkboxes, select a group; Compare selected:
  validation curves overlaid, per-run table (recipe, epochs, best epoch,
  test accuracy, top-5, time) and mean ± sample s.d. per group.
- Export CSV + LaTeX (booktabs, with a mean ± s.d. row) into the project's
  exports folder.
- Top-5 accuracy (more than five classes) per epoch and at the end;
  per-class precision / recall / F1 with macro averages in Results.
- MACs per image for every design (convolutions, dense, attention).
- Tests: summary/export (mean and s.d., CSV, LaTeX that compiles), MACs
  against a hand count (MSG-Former 93,499,136) and ResNet-18's 1.8 G; CI
  recipe check with six classes checks top-5. Manual, README updated.

## 05_11_26_08_45_29

**Version 0.3.0** (`d558ee2`)

## 05_11_26_08_42_24

**Phase 1: Kaggle live progress, training recipe, checks before training** (`8104fdb`)

- Kaggle: while the notebook runs, its log (kaggle kernels logs) gives
  the curves and the log every 30 s instead of 'running' until the end.
- Training recipe: AdamW, weight decay, learning-rate schedule (cosine
  decay to 0, or halve when the validation loss stalls), linear warm-up,
  label smoothing, gradient clipping; 'Use recommended' in the Train tab.
  The rate is computed from the global step, so spot resumes continue it
  exactly; the plateau state is in the checkpoint. Defaults unchanged.
- Before training: a run is refused before it is created when the tools,
  the Kaggle token or the AWS key are missing; the Train tab says what is
  missing and estimates the GPU memory against the target's GPU.
- Tests: recipe settings and compatibility, live Kaggle progress with the
  stand-in CLI; CI trains with the recipe and checks the learning rate
  per epoch (warm-up then cosine).
- Manual, README and CLAUDE.md updated.

## 05_11_26_08_35_02

**README: install on macOS, Windows and Linux; tools find 'python' on Windows** (`ff27ac4`)

README: a table of the release files and requirements per system; step
by step for macOS (Open Anyway in Privacy & Security on macOS 15, the
quarantine command), Windows (SmartScreen), Linux (.deb with apt, the
AppImage and FUSE 2); first steps; updating and uninstalling (where
projects, settings and keys live); building from source per system.
Says plainly that SSH/AWS training is not supported on Windows yet.

Install tools: also try 'python' (the usual name on Windows).

## 04_11_26_22_23_58

**Read each Keychain item once per launch** (`140eb13`)

The window refresh and the run status (every 3 s while a run is open)
read the Kaggle, Hugging Face and AWS items from the Keychain each time.
For an unsigned app every read can make macOS ask for the login password
again, so the dialog kept coming back. The items are now cached in memory
after the first read, and the cache is updated when one is saved or
removed.

## 04_11_26_22_20_45

**Validation split and early stopping (Train tab)** (`758476b`)

- Validation part of the training data (val_fraction): held out with the
  evaluation transforms (no augmentation); the per-epoch curves use it,
  and the test data is used once, at the end. 0 keeps the old behaviour.
- Early stopping: stop when validation loss / accuracy / mIoU has not
  improved for 'patience' epochs; the best epoch's weights (best_model.pt,
  written atomically) become model.pt and the test accuracy is measured
  on them. Turning it on always gets a validation part (never the test
  data). The state is in the checkpoint, so it survives spot resumes.
- Train tab section, run card message, manual and CLAUDE.md updated.
- Tests: settings/compatibility/compile locally; CI runs real training
  that must stop early with a 432/108/60 split and restore the best epoch.

## 04_11_26_22_16_38

**Kaggle runs: small outputs, no duplicate download, show the last check** (`f5ee442`)

The first real Kaggle run (LeNet-5, MNIST, 3 epochs on a T4: 98.44%)
worked, but the app downloaded 66 MB to get 360 KB of results: MNIST and
the checkpoint were written to /kaggle/working, the notebook's output. Now
the data goes to /tmp/nnb_data, the checkpoint is removed when a Kaggle
run ends (Kaggle runs are never resumed), and the downloaded output folder
is deleted once metrics, results, model and log are copied. While a Kaggle
run is going, the run card says when Kaggle was last checked.

## 04_11_26_21_59_34

**No 'large Dense' warning on transformer MLPs** (`172ad73`)

The warning (and its advice to use Global pool instead of Flatten) was
also shown on the Dense layers of every Transformer encoder block, which
are large by design (ViT-g/14: 8.7 M parameters each). It now applies only
to Dense layers on a flat vector. Test: none on ViT-B/16 and ViT-g/14,
still on VGG-16.

## 04_11_26_19_40_30

**Ignore the whole MSG-Former paper folder (source, figures, PDF) while it is written** (`78d422c`)

## 04_11_26_19_39_39

**Ignore the MSG-Former paper draft until its training runs are done** (`a5817fa`)

## 04_11_26_19_36_38

**Train tab: Random seed field** (`5221bde`)

The seed was in the training settings (default 1) but not in the window,
so repeated runs with different seeds (mean and spread of the accuracy)
were not possible from the app.

## 04_11_26_19_33_44

**Read info keeps the chosen Hugging Face columns** (`1df4c90`)

The column list arrives sorted by name, so for uoft-cs/cifar100 Read info
picked coarse_label (20 classes) over fine_label (100) and silently
replaced the user's choice. Now it keeps the chosen image and label
columns when the dataset has them, else prefers a column named 'label',
else the class column with the most classes.

## 04_11_26_19_17_05

**Manual: screenshots of the application and a full MSG-Former section** (`48618d8`)

doc/figures: nine screenshots (Projects, Design with LeNet-5 and
MSG-Former, Dataset, Code, Train, Settings, day theme), taken from a copy
with a neutral settings folder so no personal path or account shows.

MSG-Former section rewritten: the idea, the block as equations with the
reason for each part, an overview diagram, a layer-by-layer table (shapes,
parameters, receptive field), where the parameters and the computation
are (about 93.5 M MACs per image), training recipe, ablations with their
parameter counts, how to adapt it, and its status (untrained). Four new
references (DeepLabv3, HDC, CoAtNet, MobileViT), all checked on arXiv.

App fixes found while taking the screenshots: a demo opens with the
built-in dataset that fits its input (CIFAR-10 / MNIST) instead of
resizing another; the layer table keeps shapes on one line and fits the
inspector; demo tiles skip a citation line like 'Zhai et al., Title.'.

## 04_11_26_19_04_40

**User manual (doc/, LaTeX + PDF); IAM policy shows the spot permission** (`5a85585`)

doc/nn_builder_manual.tex: installing and the window, projects and unsaved
demos, the Design screen, a step-by-step CNN, the arithmetic (output size,
parameters, receptive field, training memory), every block, transformers
and segmentation, Dataset / Code / Connection / Train / Results, EC2 spot
handling, the fifteen demo architectures (as built, numbers from the app,
differences from the papers, how to train them), troubleshooting, glossary;
references to the original papers on arXiv (or their official page when
not on arXiv). Compiled PDF committed next to it.

Connection tab: the IAM policy text now includes
ec2:DescribeSpotInstanceRequests (used for spot details).

## 04_11_26_18_45_12

**CI resume check: absolute data path; clear message when the kaggle command is missing** (`f288e09`)

## 04_11_26_18_42_04

**Remove Python bytecode committed by mistake; ignore __pycache__** (`7fae751`)

## 04_11_26_18_41_46

**EC2 spot instances: checkpoint on the interruption notice, resume automatically** (`7bf5ddc`)

train.py (harness): SIGUSR1 = spot interruption notice -> mid-epoch
checkpoint and exit 75; SIGUSR2 = rebalance recommendation -> checkpoint
now; timed checkpoints (checkpoint_minutes, spot default 10); atomic
writes (tmp + fsync + replace); resume in the middle of an epoch with the
same batch order (EpochOrder), sums, GradScaler and RNG restored.

run.sh: one copy at a time (flock), running-time budget summed over
resumes, IMDSv2 watcher; resume.sh + crontab @reboot restarts a cut-off
run when AWS brings the instance back, without the Mac; max_resumes.

App: spot detected from the instance; follows the new public address,
asks AWS instead of ssh while the instance is down (interrupted /
terminated), restarts a cut-off run, copies new checkpoints to the Mac
(machine setting), 'Continue on another machine' from that copy, stops a
continued run if its old instance comes back, Stop while unreachable.

Tests: the whole spot path with stand-ins (fake metadata service and
crontab; the real crontab is never used); CI interrupts real PyTorch
training mid-epoch and checks the resume.

## 04_11_26_18_12_47

**Settings page; night Design screen in three panels with chosen colours** (`7a54f20`)

Night mode shows the Design screen as three separate panels (blocks,
canvas, inspector). Their background colours are chosen in the new
Settings tab (⚙ in the top bar, NN Builder → Settings… ⌘,): six presets,
a colour picker per panel, a live preview, a warning for light colours,
reset. Settings also holds the theme and the automatic-update switch.

## 04_11_26_18_03_08

**ViT-g/14 demo (1 billion parameters); demo section folds** (`ea1c9d1`)

ViT-g/14 from Scaling Vision Transformers: patch 14, width 1408, 40
blocks, 16 heads, MLP 6144; 1,011,216,522 parameters with a 10-class
head, counted by hand in the test. Its description says it does not fit
a 16 GB T4 for training.

The Projects page's demo section can be folded; the choice is remembered.

## 04_11_26_17_39_35

**MSG-Former: an original demo architecture** (`708fba2`)

Multi-scale gated residual stages (3×3 convolutions with dilation 1, 2, 3
side by side, Concat, 1×1 fuse, squeeze-and-excitation gate via Multiply,
1×1 shortcut) on 32×32 images, then two Transformer encoder blocks on the
8×8 map as tokens. 466,522 parameters, counted by hand in the test; CI
compares it with PyTorch like the other demos.

Layer table: vector layers inside a stage (the gate) stay in that stage;
only a single patch convolution joins the Embedding group.

## 04_11_26_17_24_11

**Projects page redesign; pages use the whole window** (`f51ecc1`)

The page stopped at 1100 px, leaving the scrollbar in the middle and an
empty strip on the right. Now pages scroll at the window edge with the
content centred. Projects page: New project and Open side by side as
cards (Enter works in both fields), demos as clickable tiles with
'Open →' instead of a row of identical buttons.

## 04_11_26_17_21_10

**Unsaved demos discarded automatically; table names in block colours** (`19c2dfd`)

- An unsaved demo is thrown away without asking when another demo or
  project is opened or created, and when the app quits or starts (the
  owner's choice); the Discard/Cancel dialog is gone.
- Layer table: every block name in its kind's colour, as on the canvas
  (neon at night), also inside the stage summaries; rows are coloured by
  status only for warnings and errors.

## 04_11_26_17_12_28

**Demos open as one unsaved draft until saved** (`b8e8dfa`)

Clicking a demo again and again created AlexNet demo, AlexNet demo 2, ...
in the projects folder. Now a demo opens as the single draft in the
settings folder (replaced by the next demo, after asking when it was
changed); Save… / File → Save Project… ⌘S moves it into the projects
folder. Runs need a saved project.

## 04_11_26_17_06_46

**Layer table grouped into stages** (`ece3421`)

Input, image stages by resolution (a pool stays with the stage it pools),
Embedding and Encoder block k for transformers, Classifier. Each group is
collapsible and shows its contents, output shape and parameter subtotal.

## 04_11_26_15_31_20

**Night mode: neon theme (View → Night Mode ⇧⌘N, ☾/☀ in the top bar)** (`ac1ed78`)

Colours moved into CSS variables; night draws boxes and text as neon glows,
each layer in a bright version of its kind's colour. Canvas drawings follow
the theme. Remembered per window; also switches the title bar to dark.

## 04_11_26_12_12_41

**Menus: Undo/Redo of the design in Edit, a fuller File menu** (`155ad76`)

Edit: Undo (Cmd-Z) / Redo (Shift-Cmd-Z) of every design change (blocks, arrows, settings, moves, templates; quick successive
changes are one step; in a text field they undo the typing), Cut/Copy/Paste/Select All, Arrange Left to Right (Cmd-L).
File: New Project (Cmd-N), Open Project… with a folder dialog (Cmd-O), New from Demo (all architectures), Import / Export
Design (model.json), Export train.py (Cmd-E), Show Project in Finder (Shift-Cmd-R), Close Window. App menu: About, Check for
Updates…. Dialogs come from tauri-plugin-dialog.

## 04_11_26_12_05_53

**Projects page: demo architectures below Open** (`48b382c`)

## 04_11_26_12_02_45

**Automatic updates from GitHub releases; release workflow; version 0.2.0** (`839a99a`)

The app checks the public GitHub releases at start (at most once a day; switch on the Projects page) and from Connection menu →
Check for Updates…; a bar offers "Install and restart": the installer for this system is downloaded, checked against the release's
SHA-256, put in place (macOS: unzip next to the running .app and swap by rename) and the app restarts. Settings, projects and
keychain entries are untouched. The Build workflow builds macOS, Windows and Linux and attaches the four installers to a tag's
release. Python compile checks in the tests also find `python` (Windows runners).

## 04_11_26_11_58_58

**Audit fixes: tmux launch quoting, Kaggle push check, settings race, ~/.ssh left alone** (`0e3e41b`)

- tmux: the run path was quoted inside double quotes, so a tmux-launched run could not start; now started from its folder
  (tested: a whole run through tmux in the flow test when tmux is installed)
- Kaggle push counts as done only on "successfully pushed"; the file list retries without the paging option; train.py passes
  the dataset to `kaggle datasets download` as an argument (the -d option is gone in newer CLIs)
- settings.json changes are serialised (a background Prepare and a machine save could overwrite each other)
- an existing ~/.ssh folder keeps its permissions (only a folder the app creates is set to 700)

## 04_11_26_11_55_17

**Projects page: start a new project from a demo architecture** (`0223f63`)

Cards for every famous architecture on the Projects page (title, year, task, input, one line about it); "Create project"
makes "<Architecture> demo" in the projects folder with the network loaded and the dataset fitted, and opens it.

## 04_11_26_11_52_13

**Clippy: is_multiple_of in the attention heads check** (`eebdc7b`)

## 04_11_26_11_51_55

**App: left-to-right canvas, Arrange button, Transformer blocks in the palette and inspector** (`4eeb976`)

Blocks connect from their right side to the next block's left side; new blocks, skip-connection joins and projections are
placed to the right; "Arrange left to right" lays out a design in columns by depth, and designs saved top to bottom are arranged
once when opened. The Transformer palette group (To tokens, Class token + position, Self-attention, Token pool) with its settings;
Conv2D kernel and stride up to 32 (ViT patches).

## 04_11_26_11_50_46

**Core: Transformer and ViT, token blocks, left-to-right layout** (`cd63010`)

New blocks for transformers: To tokens (patches or rows), Class token + position, Self-attention (multi-head) and Token pool;
LayerNorm and Dense work on token sequences, with plain-words errors elsewhere (BatchNorm on tokens, Output without Token pool,
heads not dividing the dimension). New templates: ViT-Base/16 (parameter count equal to torchvision's vit_b_16), ViT-Tiny for
CIFAR-10, and a Transformer encoder that reads image rows as tokens. Kernel and stride limits raised to 32 (ViT's 16x16 patches
were being clamped). Templates and new projects are laid out left to right.

## 04_11_26_11_47_23

**App: famous architectures picker, skip-connection tool, segmentation and augmentation settings** (`b8db63d`)

Design: a "Famous architectures" picker (the current design is kept as model.previous.json; the dataset's image size, resize and
task follow the architecture), Multiply and Segmentation output blocks, and a skip-connection tool on the selected block that
inserts Add or Concat and, when shapes differ, a 1x1 strided convolution, pooling or Dense projection on the skip path.
Dataset: task (classification / segmentation), mask column (Hugging Face) or masks folder (Kaggle), resizing of built-in sets.
Train: data augmentation options; mean-IoU curve for segmentation. Results: mean IoU, IoU per class, image / true / predicted masks.

## 04_11_26_11_44_18

**Core: famous architectures, segmentation, Multiply block, data augmentation** (`3b389eb`)

Templates (templates.rs): LeNet-5, a small CIFAR CNN, AlexNet, VGG-16, GoogLeNet (9 Inception modules), ResNet-18, MobileNet v1,
a small DenseNet, U-Net and Attention U-Net, laid out by depth; tests check each is valid at its own input and that VGG-16 and
ResNet-18 have exactly torchvision's parameter counts. CI also compares the templates with PyTorch.

Segmentation: dataset task "segment" (Hugging Face mask column, or Kaggle images + masks folders paired by name), a
Segmentation output block (1x1 convolution, must return to the input size), per-pixel loss, pixel accuracy and mean IoU.
Multiply block for attention gates (equal shapes, a one-channel map, or one weight per channel). Built-in datasets can be
resized (e.g. CIFAR-10 at 224x224). Augmentation (flips, padded random crop, rotation, colour jitter) on training images only;
segmentation flips/rotates image and mask together.

## 04_11_26_11_36_05

**NN Builder 0.1.0: design networks on a canvas, train them on Kaggle, a college GPU or AWS** (`6bdc448`)

Rust (Tauri 2) app with a React Flow canvas: blocks for convolution (incl. depthwise and transposed), pooling, upsampling,
normalisation, dropout, dense, Add and Concat, with filters/kernel/stride/padding/pooling set in an inspector. A Rust shape engine
shows every output shape, parameter count and receptive field and explains problems in plain words. The design becomes a standalone
PyTorch train.py (dataset from torchvision, Hugging Face or Kaggle; AMP, checkpoints with resume, one-batch shape check).

Training never runs on the Mac: Kaggle notebooks through the Kaggle API, or SSH machines (college server, AWS EC2 started and
stopped by the app) where runs are detached with SLURM / systemd-run / tmux / nohup so they do not depend on the connection.
A Connection tab and menu hold the tools, the tokens (system keychain only) and the machines.

Tests: shape engine, generated Python compiles, whole SSH and Kaggle runs through stand-ins; CI compares 300 random designs
with real PyTorch.

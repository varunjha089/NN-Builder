# NN Builder

Design neural networks on a canvas, check every shape while you build, and train them on a GPU somewhere else: **Kaggle**, a **college GPU
server** (SSH) or an **AWS EC2** instance (g4dn.xlarge). Your computer only designs: it never trains, never downloads datasets and needs no PyTorch.

**Manual:** [doc/nn_builder_manual.pdf](doc/nn_builder_manual.pdf) (source: [doc/nn_builder_manual.tex](doc/nn_builder_manual.tex)): using the
application, designing a network step by step, the arithmetic behind every number, all blocks, training on Kaggle / SSH / EC2 (spot), and
the fifteen demo architectures explained, with references to the original papers (arXiv).

Version 0.4.0 (newer changes on main are listed in [CHANGELOG.md](CHANGELOG.md)) · MIT licence · Rust (Tauri 2) + React; the generated
training code is Python (PyTorch). Planned work: [FUTURE_PLAN.md](FUTURE_PLAN.md); the design language: [PROGRAMMING_LANGUAGE.md](PROGRAMMING_LANGUAGE.md).

## What it does

- **Design tab:** a node graph (React Flow). Blocks: Input, Conv2D, Depthwise Conv2D, ConvTranspose2D, Pool, Global pool, Upsample, BatchNorm,
  LayerNorm, GroupNorm, Dropout, Flatten, Dense, Activation, Add, Concat, Multiply, Subtract, Shared weights; transformer blocks (To tokens,
  Class token + position with learnt / sinusoidal / no positions, Self-attention with causal masks, rotary positions and LoRA,
  Cross-attention, RMSNorm, SwiGLU, Mixture of experts, Token pool); time-series blocks (Conv3D, Temporal pool, To sequence, Conv1D, LSTM,
  GRU); outputs (Output, Segmentation output, Detection output, Early exit); and **Custom (PyTorch)**: your own `nn.Module` class with a
  declared output shape. Any block with weights can be **frozen**; Dense and Self-attention take a **LoRA** rank. Branches work (ResNet
  skips with Add, U-Net with Concat, attention gates with Multiply, Siamese twins with Shared weights: the **Twin branch** button).
  The network reads left to right; "Arrange left to right" lays out any design.
  - **Famous architectures** to start from (Projects page: "Start from a demo architecture" opens one unsaved, ⌘S keeps it; or Design → Famous architectures): LeNet-5, a small CIFAR-10 CNN, AlexNet, VGG-16, GoogLeNet (Inception v1), ResNet-18, MobileNet v1,
    a small DenseNet, a Transformer encoder (image rows as tokens), ViT-Base/16, ViT-Tiny (CIFAR-10), ViT-g/14 (1 billion parameters), U-Net and Attention U-Net (VGG-16,
    ResNet-18 and ViT-Base/16 have exactly torchvision's parameter counts), plus one original design, MSG-Former: multi-scale dilated
    convolutions with a squeeze-and-excitation gate and residual shortcuts, then Transformer encoder blocks.
  - **Skip connections:** select a block and choose an earlier one: an Add (residual) or Concat is inserted, with a 1×1 convolution (or pooling)
    on the skip path when channels or sizes differ.
  - Filters, kernel, stride, padding (`same` / `valid` / a number), dilation, pooling size and stride (= pool size unless unticked) are set in
    the inspector; kernel and stride have a "square" link; Conv2D has presets (3×3 keep size, 3×3 stride 2, 1×1, 5×5, 7×7 stride 2).
  - Every block shows its output shape and parameters, every arrow the shape it carries. Problems are explained in plain words (red: image too
    small for the window, Dense without Flatten, Add of different shapes, Concat of different sizes, a loop, no Output; yellow: a stride that
    drops rows or skips pixels, a huge Dense layer). Click a block for the formula with its numbers, its receptive field and an animation of
    the window sliding over the padded input.
- **Text tab:** the same design as numbered lines, kept in step with the canvas both ways (typing changes the canvas a moment later; a
  change on the canvas edits only the lines it concerns). One block per line, inputs with `<-`, and repeated parts written once:
  `repeat 12 { … }`, `for c, s in (64, 1), (128, 2) { … }`, functions `def block(c, s = 1) { … }`, names and arithmetic (`let w = 64`,
  `conv w * 2 3x3`). Coloured text, errors on their line, ⌘-click a number to select its block; File → Import / Export Design reads and
  writes `.nnb` text. **How to write it:** the manual's chapter "Writing designs as text: the NNB language" (a tutorial from the first line
  to functions and loops, every block, the rules, the error messages; source [doc/language.tex](doc/language.tex), examples in
  [doc/examples/](doc/examples/)). A design using every feature: [doc/examples/everything.nnb](doc/examples/everything.nnb).
  ```
  1 input
  2 conv 64 3x3
  3 repeat 2 {
      a conv 64 3x3 none
      b add <- a, in
    }
  4 global_pool
  5 output
  ```
- **Menus:** File (New Project, Open Project…, Save Project, Close Project, New from Demo, Import / Export Design (`.json` or `.nnb`),
  Export train.py, Show Project in Finder), Edit (Undo / Redo of every design change, Arrange Left to Right), View (Night Mode),
  Connection, and Help (Documentation opens this manual inside the app; Check for Updates…).
- **Projects page:** new project, open a recent one, **Delete** a project (its folder goes to the Trash, from where it can be put back),
  and the demo architectures. **Settings** (⚙): day / neon night theme, the night panel colours, automatic updates, and **Design size**
  (the most blocks a design may have: recommended from the computer's memory, editable).
- **Dataset tab:** tasks **classification**, **multi-label** (several classes per image, e.g. BigEarthNet), **regression** (numbers per
  image), **segmentation** (images + masks, or a large scene and its label image cut into tiles) and **object detection** (YOLO-format
  boxes; a CenterNet-style head, mAP), plus **extra outputs** (several targets at once). Sources: built-in (MNIST, Fashion-MNIST, CIFAR-10),
  **Hugging Face** (splits, columns and class names read through the public datasets-server API, nothing downloaded), **Kaggle** (the file
  list read through the Kaggle API), **your own files** (copied to the training machine or uploaded once as a private Kaggle dataset, e.g.
  Google Earth Engine exports, with a generated export script) or a **cloud folder** (Cloud Storage, Google Drive, rclone) that the machine
  fetches itself. **Satellite images** of 2–32 bands (Sentinel-2, Sentinel-1 in dB) from GeoTIFF / .npy, two sensors together (also each at
  its own size), before/after pairs and **time series** of dated images. **Check data** runs the data loading on the training machine and
  reports what it found before any GPU hour is spent; unbalanced classes get class weights, focal or Dice losses.
- **Data augmentation** (Train tab): flips, rotation, padded random crop, brightness/contrast, on training images only; for segmentation the
  image and its mask are flipped and rotated together.
- **Code tab:** the standalone `train.py` the GPU machine runs: the model, the data loading, a training loop with mixed precision, a
  checkpoint every epoch and automatic resume, a one-batch shape check before training, and the results. Export it to run it anywhere.
- **Training recipe and validation:** Adam / AdamW / SGD, weight decay, cosine or plateau schedule with warm-up, label smoothing, gradient
  clipping ("Use recommended"); a validation split with early stopping on validation loss, accuracy or mIoU; checks before a run (tools,
  tokens) and a GPU-memory estimate for the target.
- **GPU memory:** a batch that runs out of GPU memory is processed in 2, 4, 8 … parts with the same result instead of failing; "save
  memory" recomputes layer outputs instead of keeping them (much less memory, ~30% slower); bfloat16 on newer GPUs, channels-last
  convolutions, data loading on all CPU cores; the run card shows the real peak memory and whether the GPU waited for data.
- **Losses and your own code:** cross-entropy, focal, Dice, Dice + CE, MSE, L1, Huber, or your own `custom_loss(out, y)` in Python.
- **Several GPUs:** on a machine with several GPUs train.py starts a DistributedDataParallel worker per GPU (stop, spot interruptions and
  resume work as with one).
- **Better models and insight:** start from torchvision's ImageNet weights (ResNet-18, VGG-16, ViT-B/16) and fine-tune or train only the
  head; Grad-CAM heat maps and a gallery of misclassified images; save the trained model as TorchScript and ONNX.
- **Designing:** copy and paste blocks, a residual-block shortcut, GroupNorm, and a ×N view that draws repeated blocks once (ViT-g/14's
  40 encoder blocks as one node); the Dataset tab warns about datasets without images and shows the class balance and example images.
- **Compare and report:** run the same training with 3 or 5 seeds in one go, compare runs (overlaid curves, a table with mean ± s.d.),
  export the table as CSV and as a LaTeX table; top-5 accuracy and per-class precision / recall / F1; the design's MACs per image.
- **Train tab:** one button. On **Kaggle** the script is pushed as a private GPU notebook (with the dataset attached); the curves follow
  the notebook's log every 30 s and the results are downloaded when it ends. On an **SSH machine** or **AWS** it is copied there and started detached (SLURM, systemd-run, tmux or
  nohup+setsid, whichever the machine allows), so it keeps going when your Mac sleeps or the network drops; the curves update while the app is
  open, and the results come back when the run ends. AWS: the app starts the instance, and the run stops it at the end.
- **EC2 spot instances**: when AWS gives its 2-minute interruption notice, the run saves a checkpoint in the middle of the epoch and stops;
  when the instance comes back (persistent spot request, interruption behaviour "stop") it continues at that batch by itself, even while your
  computer is off. Checkpoints are also written every few minutes and safely (never half-written); the app can keep a copy of each one on
  your computer, so a run whose instance was terminated can **continue on another machine** (e.g. a college GPU). Limits on automatic
  resumes and on total running time keep the cost in check.
- **Results tab:** test accuracy, confusion matrix, sample predictions, first-layer filters, feature maps; for the other tasks F1 / mAP,
  R² / MAE / RMSE with a scatter plot, mAP with the boxes drawn, every extra output's score and the early exits; the files (model.pt,
  train.py, results.json) are in the run folder.
- **What to test by hand:** [TEST_APP.md](TEST_APP.md), the checklist of everything to try in the real window, on real data and on real
  GPUs (the automatic tests run on GitHub's CPUs with generated data).
- **Connection tab** (also the **Connection** menu): the tools (the official `kaggle` CLI and `boto3`, in a small Python environment of the
  app), your Kaggle token, Hugging Face token and AWS access key (kept in the system keychain only), and your machines (college server with a
  password or key, jump host; AWS instance with region, instance id and key file). Password machines: you log in once in a Terminal window the
  app opens; the app never sees or stores the password.

## Updates

The app checks the GitHub releases once a day (switch on the Projects page; or Help menu → Check for Updates…). When a newer version
exists, a bar offers "Install and restart": the download is checked against the release's SHA-256, then the app replaces itself and restarts.
Your projects, settings and keychain entries are not touched.

## Install

Download the file for your system from the **[latest release](https://github.com/varunjha089/NN-Builder/releases/latest)**. The app is
small (5 MB on macOS); it needs no PyTorch and no GPU on your computer, because training runs on Kaggle, a server or AWS.

| System | File | Requirements |
|---|---|---|
| macOS (Apple silicon: M1 or newer) | `NN-Builder_X_aarch64.app.zip` | macOS 11 or newer |
| macOS (Intel) | `NN-Builder_X_x64.app.zip` | macOS 10.13 or newer (from version 0.4.0) |
| Windows 10 / 11 (64-bit) | `NN-Builder_X_x64-setup.exe` | Microsoft Edge WebView2 (built into Windows 11; the installer adds it on Windows 10 if missing) |
| Linux (64-bit), Debian / Ubuntu | `nn-builder_X_amd64.deb` | Ubuntu 22.04 or newer, Debian 12 or newer |
| Linux (64-bit), any distribution | `nn-builder_X_amd64.AppImage` | FUSE 2 (`libfuse2`; on Ubuntu 24.04 `libfuse2t64`) |

The builds are not signed with a paid Apple or Microsoft certificate, so each system warns the first time you open the app.

### macOS

1. Unzip the `.app.zip` for your Mac (`aarch64` = Apple silicon, `x64` = Intel; double-click it) and drag **NN Builder.app** into **Applications**.
2. Open it. macOS says it cannot verify the developer: click **Done**, then open **System Settings → Privacy & Security**, scroll to
   *"NN Builder" was blocked…* and click **Open Anyway** (enter your password). This is needed once.
   (On macOS 14 and older, right-click the app → **Open** → **Open** does the same.)
3. When you save a Kaggle, Hugging Face or AWS key, macOS asks whether NN Builder may use it in your keychain: enter your login password and
   click **Always Allow**.

Or in Terminal, after moving the app to Applications: `xattr -dr com.apple.quarantine "/Applications/NN Builder.app"`, then open it normally.

### Windows

1. Run `NN-Builder_X_x64-setup.exe`. If Windows SmartScreen shows *"Windows protected your PC"*, click **More info → Run anyway**.
2. Follow the installer; NN Builder is then in the Start menu.
3. Keys are stored in the Windows Credential Manager.

**Training on Windows:** Kaggle works, and so do a college server or AWS over SSH **with a key** (files go up and down through `tar`
and `ssh`, which Windows 10/11 include). Logging in once with a *password* is not possible on Windows (it needs SSH connection sharing,
which Windows' OpenSSH lacks): set up a key for the machine.

### Linux

Debian / Ubuntu (`.deb`, installs the dependencies too):

```
sudo apt install ./nn-builder_X_amd64.deb
nn-builder-app        # or start "NN Builder" from the applications menu
```

Any distribution (`.AppImage`, nothing to install):

```
chmod +x nn-builder_X_amd64.AppImage
./nn-builder_X_amd64.AppImage
```

If the AppImage does not start, install FUSE 2: `sudo apt install libfuse2` (Ubuntu 22.04) or `sudo apt install libfuse2t64` (24.04).
Keys are stored through the Secret Service (GNOME Keyring or KWallet), which normal desktops have.

### First steps (all systems)

1. **Connection → Install tools** (once): makes a small private Python environment with the `kaggle` and `boto3` helpers (about 60 MB). It
   needs **Python 3.9–3.14** on your computer: macOS and Linux usually have it; on Windows install it from
   [python.org](https://www.python.org/downloads/) (tick *Add python.exe to PATH*).
2. **Connection → Kaggle:** paste your API token (kaggle.com → Settings → API → Create New Token). The Kaggle account must be
   phone-verified for GPU notebooks. Add a college server or an AWS instance here too if you have one.
3. **Projects:** create a project, or open a demo architecture and save it (⌘S / Ctrl+S).
4. **Design → Dataset → Train → Results.** The full guide is the [manual](doc/nn_builder_manual.pdf).

### Updating and uninstalling

- **Updates:** the app checks the releases once a day (Settings) and offers *Install and restart*; or download the new file and install it
  over the old one. Projects, settings and keys are kept.
- **Uninstall:** macOS: move NN Builder.app to the Trash. Windows: Settings → Apps → NN Builder → Uninstall. Linux: `sudo apt remove
  nn-builder`, or delete the AppImage.
  Your **projects** stay where you saved them (by default `~/NN_Builder_Projects`). The app's **settings** folder is
  `~/Library/Application Support/NN Builder` (macOS), `%APPDATA%\NN Builder` (Windows) or `~/.local/share/nn-builder` (Linux); the **keys** are
  the "NN Builder" entries in the keychain / Credential Manager / Secret Service. Delete those too to remove everything.

## Build from source

Needs [Rust](https://rustup.rs) (stable), [Node.js](https://nodejs.org) 20 or newer, and the Tauri system packages:

- **macOS:** Xcode Command Line Tools (`xcode-select --install`).
- **Windows:** Microsoft C++ Build Tools ("Desktop development with C++") and WebView2.
- **Linux (Debian/Ubuntu):** `sudo apt install build-essential curl wget file libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf`

```
git clone https://github.com/varunjha089/NN-Builder.git && cd NN-Builder
cd web && npm ci && npm run build && cd ..          # the window
cargo test -p nnb-core                               # core tests (no network, no PyTorch)
cd app && npm ci && npx tauri build && cd ..         # the installer
```

The result is in `target/release/bundle/`: `macos/NN Builder.app` (copy it to Applications), `nsis/…-setup.exe` (Windows), or `deb/` and
`appimage/` (Linux).

## Where things are

- Settings and the tools environment: `~/Library/Application Support/NN Builder` (macOS; `NN_BUILDER_HOME` overrides it). Secrets: the keychain.
- Projects: folders you choose (default `~/NN_Builder_Projects/<name>`): `model.json` (the graph), `dataset.json`, `train.json`, `runs/<date-time>/`
  (`train.py`, `run.sh`, `metrics.jsonl`, `log.txt`, `results.json`, `model.pt`).
- On a remote machine: `~/.nn_builder/` (`venv/`, `data/`, `runs/`).

## How large a design can be

The app never holds weights, so the parameter count does not matter to it; the number of blocks does (the canvas needs about 0.3 MB per
block drawn). Measured: a 100-billion-parameter ViT with 392 blocks opens in about 190 MB; a 25-billion one with 3,507 blocks in about 1 GB.
Settings → Design size sets the limit (recommended: a quarter of the computer's memory); the ×N view and the Text tab handle very large
designs without drawing every block. Details in the manual.

## How correctness is checked

`cargo test -p nnb-core` checks the shape engine on hand-made designs (CNN, residual Add, a small U-Net, loops, unconnected blocks), compiles the
generated Python, and runs a whole SSH run and a whole Kaggle run through stand-in `ssh`, `rsync`, `python` and `kaggle` programs.
GitHub Actions (`.github/workflows/ci.yml`) also builds 300 random designs (and every demo, and `doc/examples/everything.nnb`) in real PyTorch
and compares every layer's shape and the parameter count with the app's. It also runs real (CPU) trainings of generated scripts: spot
interruption and resume, early stopping, the training recipe with Grad-CAM and TorchScript / ONNX export, the GPU-memory features (the
recomputed forward equals the plain one; a split batch ends with the same weights), and the ImageNet weights against torchvision's own
models. `cargo test -p nnb-core --test text` checks the design language: every demo round-trips through the text, every block kind, functions,
loops, errors on the right line, and canvas edits that keep the hand-written text.

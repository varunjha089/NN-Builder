# NN Builder

Design neural networks on a canvas, check every shape while you build, and train them on a GPU somewhere else: **Kaggle**, a **college GPU
server** (SSH) or an **AWS EC2** instance (g4dn.xlarge). Your computer only designs: it never trains, never downloads datasets and needs no PyTorch.

Version 0.1.0 · MIT licence · Rust (Tauri 2) + React; the generated training code is Python (PyTorch).

## What it does

- **Design tab:** a node graph (React Flow). Blocks: Input, Conv2D, Depthwise Conv2D, ConvTranspose2D, Pool, Global pool, Upsample, BatchNorm,
  LayerNorm, Dropout, Flatten, Dense, Activation, Add, Concat, Multiply, Output, Segmentation output. Branches work (ResNet skips with Add,
  U-Net with Concat, attention gates with Multiply).
  - **Famous architectures** to start from: LeNet-5, a small CIFAR-10 CNN, AlexNet, VGG-16, GoogLeNet (Inception v1), ResNet-18, MobileNet v1,
    a small DenseNet, U-Net and Attention U-Net (VGG-16 and ResNet-18 have exactly torchvision's parameter counts).
  - **Skip connections:** select a block and choose an earlier one: an Add (residual) or Concat is inserted, with a 1×1 convolution (or pooling)
    on the skip path when channels or sizes differ.
  - Filters, kernel, stride, padding (`same` / `valid` / a number), dilation, pooling size and stride (= pool size unless unticked) are set in
    the inspector; kernel and stride have a "square" link; Conv2D has presets (3×3 keep size, 3×3 stride 2, 1×1, 5×5, 7×7 stride 2).
  - Every block shows its output shape and parameters, every arrow the shape it carries. Problems are explained in plain words (red: image too
    small for the window, Dense without Flatten, Add of different shapes, Concat of different sizes, a loop, no Output; yellow: a stride that
    drops rows or skips pixels, a huge Dense layer). Click a block for the formula with its numbers, its receptive field and an animation of
    the window sliding over the padded input.
- **Dataset tab:** classification or **segmentation** (images + masks; pixel accuracy and mean IoU). Built-in (MNIST, Fashion-MNIST, CIFAR-10), **Hugging Face** (reads the dataset's splits, columns and class names through
  the public datasets-server API, nothing is downloaded) or **Kaggle** (reads the file list through the Kaggle API and finds the folder with one
  subfolder per class; for segmentation an images folder and a masks folder). Images are resized and converted to grey or colour as you choose
  (built-in sets too, e.g. CIFAR-10 at 224×224 for AlexNet).
- **Data augmentation** (Train tab): flips, rotation, padded random crop, brightness/contrast, on training images only; for segmentation the
  image and its mask are flipped and rotated together.
- **Code tab:** the standalone `train.py` the GPU machine runs: the model, the data loading, a training loop with mixed precision, a
  checkpoint every epoch and automatic resume, a one-batch shape check before training, and the results. Export it to run it anywhere.
- **Train tab:** one button. On **Kaggle** the script is pushed as a private GPU notebook (with the dataset attached) and the results are
  downloaded when it ends. On an **SSH machine** or **AWS** it is copied there and started detached (SLURM, systemd-run, tmux or
  nohup+setsid, whichever the machine allows), so it keeps going when your Mac sleeps or the network drops; the curves update while the app is
  open, and the results come back when the run ends. AWS: the app starts the instance, and the run stops it at the end.
- **Results tab:** test accuracy, confusion matrix, sample predictions, first-layer filters, feature maps; the files (model.pt, train.py,
  results.json) are in the run folder.
- **Connection tab** (also the **Connection** menu): the tools (the official `kaggle` CLI and `boto3`, in a small Python environment of the
  app), your Kaggle token, Hugging Face token and AWS access key (kept in the system keychain only), and your machines (college server with a
  password or key, jump host; AWS instance with region, instance id and key file). Password machines: you log in once in a Terminal window the
  app opens; the app never sees or stores the password.

## Install and run

macOS: build with the steps below and copy `NN Builder.app` to /Applications. First steps in the app: Connection → Install tools → add your
Kaggle token (and/or a machine) → Project → New → Design → Dataset → Train.

## Build

```
cd web && npm ci && npm run build && cd ..          # the window
cargo test -p nnb-core                               # core tests (no network, no PyTorch)
cd app && npm ci && npx tauri build && cd ..         # target/release/bundle/macos/NN Builder.app
```

## Where things are

- Settings and the tools environment: `~/Library/Application Support/NN Builder` (macOS; `NN_BUILDER_HOME` overrides it). Secrets: the keychain.
- Projects: folders you choose (default `~/NN_Builder_Projects/<name>`): `model.json` (the graph), `dataset.json`, `train.json`, `runs/<date-time>/`
  (`train.py`, `run.sh`, `metrics.jsonl`, `log.txt`, `results.json`, `model.pt`).
- On a remote machine: `~/.nn_builder/` (`venv/`, `data/`, `runs/`).

## How correctness is checked

`cargo test -p nnb-core` checks the shape engine on hand-made designs (CNN, residual Add, a small U-Net, loops, unconnected blocks), compiles the
generated Python, and runs a whole SSH run and a whole Kaggle run through stand-in `ssh`, `rsync`, `python` and `kaggle` programs.
GitHub Actions (`.github/workflows/ci.yml`) also builds 300 random designs in real PyTorch and compares every layer's shape and the parameter
count with the app's.

# ---------------------------------------------------------------- NN Builder training harness (the same on Kaggle, a college GPU and AWS)
# Above this line: DATASET, TRAIN, CLASSES, EXPECTED, LAYERS and the Net class, written by the app from the design.
# Writes (next to this script, or in /kaggle/working on Kaggle): metrics.jsonl (one JSON event per line; also printed as "NNB_EVENT ..."),
# checkpoint.pt (every epoch, and every TRAIN["checkpoint_minutes"] inside an epoch; written atomically; a restarted run resumes from it,
# also in the middle of an epoch with the same batch order), model.pt and results.json at the end.
# Signals: TERM/INT/HUP = stop (the run ends, with results). USR1 = EC2 spot interruption notice: save a checkpoint at once and exit with
# RESUME_EXIT (75), so the run is started again later and continues. USR2 = rebalance recommendation: save a checkpoint now, keep training.
# Tasks: "classify" (one label per image) and "segment" (a class for every pixel; images + masks).
import glob
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
OUT = os.getcwd() if ON_KAGGLE else os.path.dirname(os.path.abspath(__file__))
# on Kaggle the data goes outside /kaggle/working, which is the notebook's output (downloaded by the app when the run ends)
DATA = os.environ.get("NNB_DATA") or ("/tmp/nnb_data" if ON_KAGGLE else os.path.join(os.path.expanduser("~"), ".nn_builder", "data"))
TASK = DATASET.get("task", "classify")
SEG = TASK == "segment"
AUG = TRAIN.get("aug", {})
STOP = False
INTERRUPT = False
CKPT_NOW = False
RESUME_EXIT = 75
STATS = {"mnist": ((0.1307,), (0.3081,)), "fashion_mnist": ((0.2860,), (0.3530,)),
         "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def emit(event, **data):
    data = {"event": event, "time": round(time.time(), 3), **data}
    line = json.dumps(data)
    with open(os.path.join(OUT, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("NNB_EVENT " + line, flush=True)


def on_stop(signum, frame):
    global STOP
    STOP = True


def on_interrupt(signum, frame):
    global INTERRUPT
    INTERRUPT = True


def on_checkpoint(signum, frame):
    global CKPT_NOW
    CKPT_NOW = True


class Interrupted(Exception):
    """a spot interruption notice arrived during evaluation: leave at once (the checkpoint is saved by the caller)"""


def save_checkpoint(path, state):
    """atomic: an interruption or a full disk during the write leaves the previous checkpoint intact. False when it could not be written."""
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except OSError as e:
        emit("warning", message=f"could not write the checkpoint ({e}); the previous one is kept")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


class EpochOrder(torch.utils.data.Sampler):
    """the training order of one epoch, fixed by (seed, epoch): a run resumed in the middle of an epoch continues with the batches it had not
    seen yet, in the same order"""

    def __init__(self, n, seed, epoch, start=0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return max(0, self.n - self.start)


def resized():
    return DATASET["source"] != "builtin" or DATASET.get("resize")


def stats():
    if DATASET["source"] == "builtin" and not DATASET.get("resize"):
        return STATS[DATASET["builtin"]]
    c = DATASET["channels"]
    if DATASET["source"] == "builtin":
        m, s = STATS[DATASET["builtin"]]
        return (tuple(m) * 3)[:c] if len(m) < c else m[:c], (tuple(s) * 3)[:c] if len(s) < c else s[:c]
    return (0.5,) * c, (0.5,) * c


# ---------------------------------------------------------------- data
def image_transform(train):
    """resize / convert, then (training only) augmentation, then tensor + normalise"""
    from torchvision import transforms
    mean, std = stats()
    steps = []
    if resized():
        h, w = DATASET["image_size"]
        steps.append(transforms.Lambda(lambda im: im.convert("L" if DATASET["channels"] == 1 else "RGB")))
        steps.append(transforms.Resize((h, w)))
    if train and not SEG:               # segmentation: flips/rotation are applied to image and mask together in SegPairs
        if AUG.get("crop"):
            size = tuple(DATASET["image_size"]) if resized() else tuple(EXPECTED["input"][1:])
            steps.append(transforms.RandomCrop(size, padding=int(AUG["crop"])))
        if AUG.get("hflip"):
            steps.append(transforms.RandomHorizontalFlip())
        if AUG.get("vflip"):
            steps.append(transforms.RandomVerticalFlip())
        if AUG.get("rotate"):
            steps.append(transforms.RandomRotation(float(AUG["rotate"])))
    if train and AUG.get("color"):
        steps.append(transforms.ColorJitter(0.2, 0.2, 0.2 if DATASET["channels"] == 3 or not resized() else 0.0))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def to_pil(img):
    from PIL import Image
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        return Image.open(img)
    return Image.fromarray(__import__("numpy").asarray(img))


class HFImages(torch.utils.data.Dataset):
    def __init__(self, ds, tfm):
        self.ds, self.tfm = ds, tfm
        self.names = {n: i for i, n in enumerate(CLASSES)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        label = row[DATASET["hf_label_col"]]
        if isinstance(label, str):
            label = self.names[label]
        return self.tfm(to_pil(row[DATASET["hf_image_col"]])), int(label)


class HFPairs:
    """(image, mask) of a Hugging Face split, read one row at a time"""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return row[DATASET["hf_image_col"]], row[DATASET["hf_mask_col"]]


class SegPairs(torch.utils.data.Dataset):
    """image + mask pairs: anything indexable giving (image, mask), each a PIL image, a file path or an array"""
    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train
        self.tfm = image_transform(train)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image
        from torchvision.transforms import functional as TF
        img, mask = (to_pil(x) for x in self.pairs[i])
        h, w = DATASET["image_size"]
        mask = mask.convert("L").resize((w, h), Image.NEAREST)
        img = img.convert("L" if DATASET["channels"] == 1 else "RGB").resize((w, h), Image.BILINEAR)
        if self.train:
            if AUG.get("hflip") and random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if AUG.get("vflip") and random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if AUG.get("rotate"):
                a = random.uniform(-float(AUG["rotate"]), float(AUG["rotate"]))
                img, mask = TF.rotate(img, a), TF.rotate(mask, a, interpolation=TF.InterpolationMode.NEAREST)
        m = np.array(mask, dtype=np.int64)
        if len(CLASSES) == 2 and m.max() > 1:
            m = (m > 0).astype(np.int64)                    # binary masks stored as 0/255
        return self.tfm(img), torch.from_numpy(m)


def kaggle_root():
    """the downloaded / attached Kaggle dataset folder"""
    owner, name = DATASET["kaggle_slug"].split("/")
    if ON_KAGGLE:
        for cand in [f"/kaggle/input/{name}", f"/kaggle/input/datasets/{owner}/{name}"] + glob.glob(f"/kaggle/input/**/{name}", recursive=True):
            if os.path.isdir(cand):
                return cand
        raise SystemExit(f"DATA: the Kaggle dataset {DATASET['kaggle_slug']} is not attached to this notebook (looked in /kaggle/input).")
    root = os.path.join(DATA, "kaggle", f"{owner}__{name}")
    if not os.path.isdir(root) or not os.listdir(root):
        os.makedirs(root, exist_ok=True)
        exe = os.path.join(os.path.dirname(sys.executable), "kaggle")
        emit("info", message=f"downloading {DATASET['kaggle_slug']} from Kaggle")
        try:
            r = subprocess.run([exe if os.path.exists(exe) else "kaggle", "datasets", "download", DATASET["kaggle_slug"], "-p", root, "--unzip"],
                               capture_output=True, text=True)
        except FileNotFoundError:
            raise SystemExit("DATA: the kaggle command is not installed on this machine (Prepare installs it with the training packages; "
                             "or: pip install kaggle)")
        if r.returncode != 0:
            raise SystemExit("DATA: Kaggle download failed (is the Kaggle token on this machine? Connection tab): " + (r.stderr or r.stdout)[-500:])
    return root


def kaggle_folder(root, sub):
    p = os.path.join(root, sub) if sub else root
    if os.path.isdir(p):
        return p
    found = [d for d in glob.glob(os.path.join(root, "**", sub), recursive=True) if os.path.isdir(d)] if sub else []
    if not found:
        raise SystemExit(f"DATA: the folder {sub!r} is not in the Kaggle dataset.")
    return found[0]


def kaggle_pairs(root):
    """images and masks with the same name (masks may add _mask / _lab / _label, or use another extension)"""
    idir, mdir = kaggle_folder(root, DATASET["kaggle_train_dir"]), kaggle_folder(root, DATASET["kaggle_mask_dir"])
    masks = {}
    for f in os.listdir(mdir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXT:
            for suffix in ("_mask", "_lab", "_label", "-mask", ""):
                if suffix and stem.endswith(suffix):
                    masks.setdefault(stem[: -len(suffix)], os.path.join(mdir, f))
            masks.setdefault(stem, os.path.join(mdir, f))
    pairs = []
    for f in sorted(os.listdir(idir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXT and stem in masks:
            pairs.append((os.path.join(idir, f), masks[stem]))
    if not pairs:
        raise SystemExit(f"DATA: no image in {DATASET['kaggle_train_dir']!r} has a mask with the same name in {DATASET['kaggle_mask_dir']!r}.")
    return pairs


def split(n, seed, part=0.1):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(n * part))
    return idx[k:], idx[:k]


def load_data(cfg):
    from torch.utils.data import Subset
    src = DATASET["source"]
    train_t, test_t = image_transform(True), image_transform(False)
    if src == "builtin":
        if SEG:
            raise SystemExit("DATA: the built-in datasets are for classification.")
        from torchvision import datasets
        cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[DATASET["builtin"]]
        root = os.path.join(DATA, "torchvision")
        train, test = cls(root, train=True, download=True, transform=train_t), cls(root, train=False, download=True, transform=test_t)
        train_eval = cls(root, train=True, download=False, transform=test_t)
    elif src == "huggingface":
        from datasets import load_dataset
        kw = {"cache_dir": os.path.join(DATA, "huggingface")}
        conf = DATASET["hf_config"] or None
        tr = load_dataset(DATASET["hf_repo"], conf, split=DATASET["hf_train_split"], **kw)
        if DATASET["hf_test_split"]:
            te = load_dataset(DATASET["hf_repo"], conf, split=DATASET["hf_test_split"], **kw)
        else:
            parts = tr.train_test_split(test_size=0.1, seed=cfg["seed"])
            tr, te = parts["train"], parts["test"]
        if SEG:
            train, test, train_eval = SegPairs(HFPairs(tr), True), SegPairs(HFPairs(te), False), SegPairs(HFPairs(tr), False)
        else:
            train, test, train_eval = HFImages(tr, train_t), HFImages(te, test_t), HFImages(tr, test_t)
    else:
        root = kaggle_root()
        if SEG:
            pairs = kaggle_pairs(root)
            tr_i, te_i = split(len(pairs), cfg["seed"])
            train, test = SegPairs([pairs[i] for i in tr_i], True), SegPairs([pairs[i] for i in te_i], False)
            train_eval = SegPairs([pairs[i] for i in tr_i], False)
        else:
            from torchvision.datasets import ImageFolder
            folder = kaggle_folder(root, DATASET["kaggle_train_dir"])
            full_train, full_test = ImageFolder(folder, transform=train_t), ImageFolder(folder, transform=test_t)
            if list(full_train.classes) != list(CLASSES):
                raise SystemExit(f"DATA: the class folders are {full_train.classes}, but the design expects {CLASSES}.")
            if DATASET["kaggle_test_dir"]:
                train, train_eval = full_train, full_test
                test = ImageFolder(kaggle_folder(root, DATASET["kaggle_test_dir"]), transform=test_t)
            else:
                tr_i, te_i = split(len(full_train), cfg["seed"])
                train, test, train_eval = Subset(full_train, tr_i), Subset(full_test, te_i), Subset(full_test, tr_i)
    # a validation part of the training data (the same images without augmentation), so that the test data is used only once, at the end
    val = None
    if cfg.get("val_fraction"):
        tr_i, va_i = split(len(train), cfg["seed"] + 7, cfg["val_fraction"])
        train, val = Subset(train, tr_i), Subset(train_eval, va_i)
    if cfg.get("limit_train"):
        train = Subset(train, range(min(cfg["limit_train"], len(train))))
    return train, val, test


# ---------------------------------------------------------------- checks and results
def one_batch_check(model, loader, device):
    """run one batch forward + backward and compare every layer's shape with what the app computed"""
    seen = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, list(o.shape[1:])))
             for name, _, _ in LAYERS]
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    if list(x.shape[1:]) != EXPECTED["input"]:
        raise SystemExit(f"CHECK: the images are {list(x.shape[1:])}, but the design's Input is {EXPECTED['input']}.")
    out = model(x)
    if SEG and list(y.shape[1:]) != list(out.shape[2:]):
        raise SystemExit(f"CHECK: the masks are {list(y.shape[1:])}, but the network outputs {list(out.shape[2:])}.")
    nn.functional.cross_entropy(out, y).backward()
    model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    bad = [f"{label}: expected {EXPECTED[name]}, got {seen.get(name)}" for name, label, _ in LAYERS if seen.get(name) != EXPECTED[name]]
    if bad:
        raise SystemExit("CHECK: shapes differ from the design: " + "; ".join(bad[:5]))
    if int(y.max()) >= len(CLASSES):
        raise SystemExit(f"CHECK: a label is {int(y.max())}, but there are only {len(CLASSES)} classes"
                         + (" (mask pixel values must be class numbers 0, 1, 2, ...)" if SEG else "") + ".")
    emit("check", message="one batch passed: every layer has the shape the design shows", batch=list(x.shape))


def evaluate(model, loader, device):
    """loss, accuracy (per image, or per pixel for segmentation), confusion matrix, mean IoU (segmentation)"""
    model.eval()
    k = len(CLASSES)
    loss_sum, n_items, cm = 0.0, 0, torch.zeros(k * k, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            if INTERRUPT:
                raise Interrupted()
            x, y = x.to(device), y.to(device)
            out = model(x).float()
            loss_sum += nn.functional.cross_entropy(out, y).item() * len(y)
            n_items += len(y)
            pred = out.argmax(1)
            cm += torch.bincount((y * k + pred).flatten().cpu(), minlength=k * k)
    cm = cm.view(k, k)
    acc = float(cm.diag().sum()) / max(1, int(cm.sum()))
    iou = [float(cm[i, i]) / max(1, int(cm[i].sum() + cm[:, i].sum() - cm[i, i])) for i in range(k)]
    present = [iou[i] for i in range(k) if int(cm[i].sum() + cm[:, i].sum()) > 0]
    return loss_sum / max(n_items, 1), acc, cm.tolist(), (sum(present) / max(1, len(present))), iou


def norm01(t):
    t = t.float()
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / (hi - lo if hi > lo else 1.0)).clamp(0, 1)


def small(t, limit=48):
    """an image tensor (C,H,W) shrunk to at most limit pixels a side, as nested lists with 3 decimals"""
    if t.shape[-1] > limit or t.shape[-2] > limit:
        t = nn.functional.interpolate(t.unsqueeze(0).float(), size=(min(limit, t.shape[-2]), min(limit, t.shape[-1])), mode="nearest")[0]
    return [[[round(float(v), 3) for v in row] for row in ch] for ch in t]


def pictures(model, test, device):
    model.eval()
    res = {"filters": None, "feature_maps": [], "samples": []}
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        w = convs[0].weight.detach().cpu()[:32]
        res["filters"] = {"kernel": list(w.shape[2:]), "images": [small(norm01(f[:3] if f.shape[0] == 3 else f[:1])) for f in w]}
    maps = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: maps.__setitem__(name, o.detach().cpu()))
             for name, _, conv in LAYERS if conv]
    mean, std = stats()
    idx = list(range(min(8 if SEG else 16, len(test))))
    xs = torch.stack([test[i][0] for i in idx]).to(device)
    with torch.no_grad():
        out = model(xs).float()
    for h in hooks:
        h.remove()
    for name, label, conv in LAYERS:
        if conv and name in maps and maps[name].dim() == 4:
            res["feature_maps"].append({"layer": label, "channels": int(maps[name].shape[1]),
                                        "maps": [small(norm01(c.unsqueeze(0)), 32)[0] for c in maps[name][0, :8]]})
    m_t, s_t = torch.tensor(mean).view(-1, 1, 1), torch.tensor(std).view(-1, 1, 1)
    k = max(1, len(CLASSES) - 1)
    for j, i in enumerate(idx):
        img = (test[i][0] * s_t + m_t).clamp(0, 1)
        if SEG:
            pred = out[j].argmax(0).cpu()
            truth = test[i][1]
            res["samples"].append({"image": small(img, 64), "truth": small((truth.float() / k).unsqueeze(0), 64),
                                   "pred": small((pred.float() / k).unsqueeze(0), 64),
                                   "pixel_acc": round(float((pred == truth).float().mean()), 3)})
        else:
            prob = torch.softmax(out[j], 0).cpu()
            p = int(prob.argmax())
            res["samples"].append({"image": small(img), "label": int(test[i][1]), "pred": p, "prob": round(float(prob[p]), 3)})
    return res


# ---------------------------------------------------------------- training
def main():
    from torch.utils.data import DataLoader
    global CKPT_NOW
    for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if s is not None:
            signal.signal(s, on_stop)
    for s, f in ((getattr(signal, "SIGUSR1", None), on_interrupt), (getattr(signal, "SIGUSR2", None), on_checkpoint)):
        if s is not None:
            signal.signal(s, f)
    if not ON_KAGGLE:
        with open(os.path.join(OUT, "py_pid.txt"), "w") as f:
            f.write(str(os.getpid()))
    cfg = TRAIN
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        emit("info", message="no CUDA GPU found: training on the CPU (slow)")
    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train, val, test = load_data(cfg)
    workers = cfg.get("num_workers", 2) if device.type == "cuda" else 0
    bs = cfg["batch_size"]

    def loader(epoch, skip_batches=0):
        return DataLoader(train, batch_size=bs, sampler=EpochOrder(len(train), cfg["seed"], epoch, skip_batches * bs), num_workers=workers,
                          pin_memory=device.type == "cuda")

    eval_bs = 64 if SEG else 256
    # per-epoch validation: the held-out part of the training data, or (without one) the test data as before
    vl = DataLoader(val if val is not None else test, batch_size=eval_bs, shuffle=False, num_workers=workers)
    test_l = DataLoader(test, batch_size=eval_bs, shuffle=False, num_workers=workers) if val is not None else vl
    # early stopping: the monitored value, the best epoch so far and how many epochs since then (kept in the checkpoint)
    monitor = cfg.get("monitor", "val_loss")
    early, patience = bool(cfg.get("early_stop")), max(1, int(cfg.get("patience", 5)))
    es = {"best": None, "best_epoch": 0, "bad": 0}
    best_path = os.path.join(OUT, "best_model.pt")
    opt = (torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9) if cfg["optimizer"] == "sgd"
           else torch.optim.Adam(model.parameters(), lr=cfg["lr"]))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ck_path = os.path.join(OUT, "checkpoint.pt")
    steps = (len(train) + bs - 1) // bs
    zero = {"loss_sum": 0.0, "n": 0, "correct": 0, "units": 0, "seconds": 0.0}
    # where training stands: `epoch` = last finished epoch, `step` = batches already done of the next one (with their running sums)
    pos = {"epoch": 0, "step": 0, "sums": dict(zero)}
    resumes = 0
    dirty = {"v": False}  # trained since the last checkpoint
    if os.path.exists(ck_path):
        try:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch older than 1.13
            ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        pos = {"epoch": ck["epoch"], "step": ck.get("step", 0), "sums": ck.get("sums") or dict(zero)}
        es = ck.get("es") or es
        if ck.get("steps") not in (None, steps):
            pos["step"], pos["sums"] = 0, dict(zero)  # a different batch size or dataset size: restart the epoch
        rng = ck.get("rng") or {}
        try:
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu())
            if "cuda" in rng and device.type == "cuda":
                torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
            if "python" in rng:
                random.setstate(rng["python"])
        except (RuntimeError, ValueError, TypeError):
            pass
        resumes = ck.get("resumes", 0) + 1
        emit("resume", epoch=pos["epoch"] + 1, step=pos["step"], steps=steps, resumes=resumes)

    def state():
        rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": pos["epoch"], "step": pos["step"],
                "steps": steps, "sums": pos["sums"], "rng": rng, "resumes": resumes, "es": es, "version": 3}

    def checkpoint(reason):
        if save_checkpoint(ck_path, state()):
            dirty["v"] = False
            emit("checkpoint", epoch=pos["epoch"] + (1 if pos["step"] else 0), step=pos["step"], steps=steps, reason=reason)

    def interrupted():
        if dirty["v"]:
            checkpoint("spot interruption")
        emit("interrupted", epoch=pos["epoch"] + 1, step=pos["step"], steps=steps)
        sys.exit(RESUME_EXIT)

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    emit("start", device=str(device), gpu=gpu, torch=torch.__version__, python=sys.version.split()[0], params=n_params, task=TASK,
         train_size=len(train), val_size=len(val) if val is not None else 0, test_size=len(test), epochs=cfg["epochs"],
         early_stop=early, patience=patience if early else None, monitor=monitor if early else None, amp=amp, augmentation={k: v for k, v in AUG.items() if v})
    if pos["epoch"] == 0 and pos["step"] == 0:
        one_batch_check(model, loader(1), device)
    every = max(1, steps // 20)
    # the run's own setting, else the default for a spot instance (set by run.sh)
    ck_every = float(cfg.get("checkpoint_minutes") or os.environ.get("NNB_CHECKPOINT_MINUTES") or 0) * 60
    last_ck = time.time()
    stopped = False
    try:
        for epoch in range(pos["epoch"] + 1, cfg["epochs"] + 1):
            model.train()
            t0 = time.time()
            sums = pos["sums"] if pos["step"] else dict(zero)
            for step, (x, y) in enumerate(loader(epoch, pos["step"]), pos["step"] + 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sums["loss_sum"] += loss.item() * len(y)
                sums["n"] += len(y)
                sums["correct"] += int((out.argmax(1) == y).sum())
                sums["units"] += y.numel()
                pos.update(step=step, sums=sums)
                dirty["v"] = True
                if step % every == 0 or step == steps:
                    emit("batch", epoch=epoch, step=step, steps=steps, loss=round(loss.item(), 5))
                if INTERRUPT:
                    sums["seconds"] += time.time() - t0
                    interrupted()
                if step < steps and (CKPT_NOW or (ck_every and time.time() - last_ck >= ck_every)):
                    sums["seconds"] += time.time() - t0
                    t0 = time.time()
                    checkpoint("rebalance recommendation" if CKPT_NOW else "timer")
                    CKPT_NOW, last_ck = False, time.time()
                if STOP:
                    stopped = True
                    break
            val_loss, val_acc, _, miou, _ = evaluate(model, vl, device)
            ev = dict(epoch=epoch, train_loss=round(sums["loss_sum"] / max(sums["n"], 1), 5), train_acc=round(sums["correct"] / max(sums["units"], 1), 5),
                      val_loss=round(val_loss, 5), val_acc=round(val_acc, 5), seconds=round(sums["seconds"] + time.time() - t0, 2), partial=stopped)
            if SEG:
                ev["val_miou"] = round(miou, 5)
            emit("epoch", **ev)
            # best epoch so far (lower validation loss, or higher validation accuracy / mIoU); its weights are kept in best_model.pt
            value = {"val_loss": val_loss, "val_acc": val_acc, "val_miou": miou}.get(monitor, val_loss)
            better = es["best"] is None or (value < es["best"] - 1e-4 if monitor == "val_loss" else value > es["best"] + 1e-4)
            if better and not stopped:
                es.update(best=value, best_epoch=epoch, bad=0)
                if early:
                    save_checkpoint(best_path, model.state_dict())
                    emit("best", epoch=epoch, monitor=monitor, value=round(value, 5))
            elif not stopped:
                es["bad"] += 1
            pos.update(epoch=epoch, step=0, sums=dict(zero))
            checkpoint("epoch")
            last_ck = time.time()
            if early and es["bad"] >= patience and epoch < cfg["epochs"]:
                emit("early_stop", epoch=epoch, best_epoch=es["best_epoch"], monitor=monitor, best=round(es["best"], 5), patience=patience)
                break
            if stopped or INTERRUPT:
                break
        if INTERRUPT:
            interrupted()
        if early and os.path.exists(best_path):
            # the final model is the best epoch's, not the last one's
            model.load_state_dict(torch.load(best_path, map_location=device))
        val_loss, val_acc, confusion, miou, iou = evaluate(model, test_l, device)
    except Interrupted:
        interrupted()
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    res = {"task": TASK, "test_acc": val_acc, "test_loss": val_loss, "classes": CLASSES, "confusion": confusion}
    if SEG:
        res.update(miou=miou, iou=iou)
    res.update(pictures(model, test, device))
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    emit("stopped" if stopped else "done", test_acc=round(val_acc, 5), **({"miou": round(miou, 5)} if SEG else {}),
         **({"best_epoch": es["best_epoch"]} if early else {}))
    if ON_KAGGLE:
        # a Kaggle run is never resumed: the checkpoint would only make the output the app downloads bigger
        for f in (ck_path, best_path):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0) and not isinstance(e.code, int):
            emit("error", message=str(e.code))
            sys.exit(3)
        raise
    except Exception as e:
        emit("error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        raise

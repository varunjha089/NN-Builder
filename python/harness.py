# ---------------------------------------------------------------- NN Builder training harness (the same on Kaggle, a college GPU and AWS)
# Above this line: DATASET, TRAIN, CLASSES, EXPECTED, LAYERS and the Net class, written by the app from the design.
# Writes (next to this script, or in /kaggle/working on Kaggle): metrics.jsonl (one JSON event per line; also printed as "NNB_EVENT ..."),
# checkpoint.pt (every epoch; a restarted run resumes from it), model.pt and results.json at the end.
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
DATA = os.environ.get("NNB_DATA") or ("/kaggle/working/data" if ON_KAGGLE else os.path.join(os.path.expanduser("~"), ".nn_builder", "data"))
STOP = False
STATS = {"mnist": ((0.1307,), (0.3081,)), "fashion_mnist": ((0.2860,), (0.3530,)),
         "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}


def emit(event, **data):
    data = {"event": event, "time": round(time.time(), 3), **data}
    line = json.dumps(data)
    with open(os.path.join(OUT, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("NNB_EVENT " + line, flush=True)


def on_stop(signum, frame):
    global STOP
    STOP = True


def stats():
    if DATASET["source"] == "builtin":
        return STATS[DATASET["builtin"]]
    c = DATASET["channels"]
    return (0.5,) * c, (0.5,) * c


# ---------------------------------------------------------------- data
def image_transform():
    from torchvision import transforms
    mean, std = stats()
    steps = []
    if DATASET["source"] != "builtin":
        h, w = DATASET["image_size"]
        steps.append(transforms.Lambda(lambda im: im.convert("L" if DATASET["channels"] == 1 else "RGB")))
        steps.append(transforms.Resize((h, w)))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class HFImages(torch.utils.data.Dataset):
    def __init__(self, ds, tfm):
        self.ds, self.tfm = ds, tfm
        self.names = {n: i for i, n in enumerate(CLASSES)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        from PIL import Image
        row = self.ds[i]
        img = row[DATASET["hf_image_col"]]
        if not isinstance(img, Image.Image):
            img = Image.open(img) if isinstance(img, str) else Image.fromarray(__import__("numpy").asarray(img))
        label = row[DATASET["hf_label_col"]]
        if isinstance(label, str):
            label = self.names[label]
        return self.tfm(img), int(label)


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
        r = subprocess.run([exe if os.path.exists(exe) else "kaggle", "datasets", "download", "-d", DATASET["kaggle_slug"], "-p", root, "--unzip"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("DATA: Kaggle download failed (is the Kaggle token on this machine? Connection tab): " + (r.stderr or r.stdout)[-500:])
    return root


def load_data(cfg):
    from torch.utils.data import Subset, random_split
    tfm = image_transform()
    src = DATASET["source"]
    g = torch.Generator().manual_seed(cfg["seed"])
    if src == "builtin":
        from torchvision import datasets
        cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[DATASET["builtin"]]
        root = os.path.join(DATA, "torchvision")
        train, test = cls(root, train=True, download=True, transform=tfm), cls(root, train=False, download=True, transform=tfm)
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
        train, test = HFImages(tr, tfm), HFImages(te, tfm)
    else:
        from torchvision.datasets import ImageFolder
        root = kaggle_root()

        def folder(sub):
            p = os.path.join(root, sub) if sub else root
            if not os.path.isdir(p):
                found = [d for d in glob.glob(os.path.join(root, "**", sub), recursive=True) if os.path.isdir(d)] if sub else []
                if not found:
                    raise SystemExit(f"DATA: the folder {sub!r} is not in the Kaggle dataset.")
                p = found[0]
            return ImageFolder(p, transform=tfm)
        full = folder(DATASET["kaggle_train_dir"])
        if [c for c in full.classes] != list(CLASSES):
            raise SystemExit(f"DATA: the class folders are {full.classes}, but the design expects {CLASSES}.")
        if DATASET["kaggle_test_dir"]:
            train, test = full, folder(DATASET["kaggle_test_dir"])
        else:
            n_test = max(1, len(full) // 10)
            train, test = random_split(full, [len(full) - n_test, n_test], generator=g)
    if cfg.get("limit_train"):
        train = Subset(train, range(min(cfg["limit_train"], len(train))))
    return train, test


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
    nn.functional.cross_entropy(out, y).backward()
    model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    bad = [f"{label}: expected {EXPECTED[name]}, got {seen.get(name)}" for name, label, _ in LAYERS if seen.get(name) != EXPECTED[name]]
    if bad:
        raise SystemExit("CHECK: shapes differ from the design: " + "; ".join(bad[:5]))
    if int(y.max()) >= len(CLASSES):
        raise SystemExit(f"CHECK: a label is {int(y.max())}, but there are only {len(CLASSES)} classes.")
    emit("check", message="one batch passed: every layer has the shape the design shows", batch=list(x.shape))


def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    k = len(CLASSES)
    confusion = [[0] * k for _ in range(k)]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).float()
            loss_sum += nn.functional.cross_entropy(out, y, reduction="sum").item()
            pred = out.argmax(1)
            correct += int((pred == y).sum())
            n += len(y)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1
    return loss_sum / max(n, 1), correct / max(n, 1), confusion


def norm01(t):
    t = t.float()
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / (hi - lo if hi > lo else 1.0)).clamp(0, 1)


def small(t, limit=48):
    """an image tensor (C,H,W) shrunk to at most limit pixels a side, as nested lists with 3 decimals"""
    if t.shape[-1] > limit or t.shape[-2] > limit:
        t = nn.functional.adaptive_avg_pool2d(t.unsqueeze(0), (min(limit, t.shape[-2]), min(limit, t.shape[-1])))[0]
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
    idx = list(range(min(16, len(test))))
    xs = torch.stack([test[i][0] for i in idx]).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(xs).float(), 1).cpu()
    for h in hooks:
        h.remove()
    for name, label, conv in LAYERS:
        if conv and name in maps and maps[name].dim() == 4:
            res["feature_maps"].append({"layer": label, "channels": int(maps[name].shape[1]),
                                        "maps": [small(norm01(c.unsqueeze(0)), 32)[0] for c in maps[name][0, :8]]})
    m_t, s_t = torch.tensor(mean).view(-1, 1, 1), torch.tensor(std).view(-1, 1, 1)
    for j, i in enumerate(idx):
        img = (test[i][0] * s_t + m_t).clamp(0, 1)
        p = int(prob[j].argmax())
        res["samples"].append({"image": small(img), "label": int(test[i][1]), "pred": p, "prob": round(float(prob[j][p]), 3)})
    return res


# ---------------------------------------------------------------- training
def main():
    from torch.utils.data import DataLoader
    for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if s is not None:
            signal.signal(s, on_stop)
    cfg = TRAIN
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        emit("info", message="no CUDA GPU found: training on the CPU (slow)")
    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train, test = load_data(cfg)
    workers = cfg.get("num_workers", 2) if device.type == "cuda" else 0
    tl = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    vl = DataLoader(test, batch_size=256, shuffle=False, num_workers=workers)
    opt = (torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9) if cfg["optimizer"] == "sgd"
           else torch.optim.Adam(model.parameters(), lr=cfg["lr"]))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = 1
    ck_path = os.path.join(OUT, "checkpoint.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        emit("resume", epoch=start_epoch)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    emit("start", device=str(device), gpu=gpu, torch=torch.__version__, python=sys.version.split()[0], params=n_params,
         train_size=len(train), test_size=len(test), epochs=cfg["epochs"], amp=amp)
    if start_epoch == 1:
        one_batch_check(model, tl, device)
    steps = len(tl)
    every = max(1, steps // 20)
    stopped = False
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for step, (x, y) in enumerate(tl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += int((out.argmax(1) == y).sum())
            n += len(y)
            if step % every == 0 or step == steps:
                emit("batch", epoch=epoch, step=step, steps=steps, loss=round(loss.item(), 5))
            if STOP:
                stopped = True
                break
        val_loss, val_acc, _ = evaluate(model, vl, device)
        emit("epoch", epoch=epoch, train_loss=round(loss_sum / max(n, 1), 5), train_acc=round(correct / max(n, 1), 5),
             val_loss=round(val_loss, 5), val_acc=round(val_acc, 5), seconds=round(time.time() - t0, 2), partial=stopped)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, ck_path)
        if stopped:
            break
    val_loss, val_acc, confusion = evaluate(model, vl, device)
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    res = {"test_acc": val_acc, "test_loss": val_loss, "classes": CLASSES, "confusion": confusion}
    res.update(pictures(model, test, device))
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    emit("stopped" if stopped else "done", test_acc=round(val_acc, 5))


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

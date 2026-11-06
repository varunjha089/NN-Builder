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
import math
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
MULTILABEL = TASK == "multilabel"   # any number of classes per image: one sigmoid per class, binary cross-entropy, F1 and mAP
# regression: numbers per image (biomass, canopy height, yield …); CLASSES are then the names of the target columns, the Output gives
# one number per target. The network learns the targets standardised with the training data's mean and spread (TARGET_MEAN / _STD);
# scores (R², MAE, RMSE) are in the targets' own units
REGRESS = TASK == "regress"
# several outputs (multi-task): extra targets next to the main task, each with its own Output block (head 1, 2, ...). A label is then a
# tuple (main, head 1, ...) and the network returns a tuple in the same order; everything that is not about the extra heads works on
# the main part (main_y / main_out)
HEADS = DATASET.get("heads") or []
MULTITASK = bool(HEADS)
HEAD_STATS = {}      # regression heads: k -> (mean, std) of their targets on the training data
HEAD_SCORES = None   # the extra heads' scores of the last evaluation


def main_y(y):
    return y[0] if MULTITASK and isinstance(y, (tuple, list)) else y


def main_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def y_to(y, device, **kw):
    return tuple(t.to(device, **kw) for t in y) if isinstance(y, (tuple, list)) else y.to(device, **kw)


def y_split(y, parts):
    return list(zip(*[t.tensor_split(parts) for t in y])) if isinstance(y, (tuple, list)) else y.tensor_split(parts)


def y_head(row, k):
    """head k's target from a row (a dict of the table's or the Hugging Face dataset's values)"""
    h = HEADS[k - 1]
    try:
        if h["task"] == "regress":
            return torch.tensor([float(row[c]) for c in h["columns"]], dtype=torch.float32)
        names = {n: i for i, n in enumerate(h["classes"])}
        if h["task"] == "classify":
            v = row[h["columns"][0]]
            return int(names[str(v).strip()]) if str(v).strip() in names else int(float(v))
        t = torch.zeros(len(h["classes"]))
        if h["columns"]:
            v = row[h["columns"][0]]
            ks = [names[c] if isinstance(c, str) and c in names else int(c) for c in v] if isinstance(v, (list, tuple)) else split_labels(str(v), names)
        else:
            ks = [i for i, c in enumerate(h["classes"]) if str(row[c]).strip() not in ("", "0", "0.0", "false", "False")]
        for i in ks:
            t[i] = 1.0
        return t
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def with_heads(y, row):
    """the main label, and the extra heads' targets when the network has several outputs (a Hugging Face row)"""
    if not MULTITASK:
        return y
    extra = [y_head(row, k) for k in range(1, len(HEADS) + 1)]
    if any(e is None for e in extra):
        bad = HEADS[[e is None for e in extra].index(True)]
        raise SystemExit(f"DATA: a row has no usable value for the output {bad['name']!r} (columns {bad['columns'] or bad['classes']}).")
    return (y, *extra)
# multispectral / radar images (Sentinel-2, Sentinel-1, …): arrays of any number of bands from GeoTIFF or .npy files, not PIL pictures.
# Fusion: a second band group (e.g. Sentinel-1 next to Sentinel-2) from files with the same names in another folder, stacked after the
# first group's bands; the network's Input blocks may take one group each (the generated forward slices the channels).
GROUPS = [{"bands": list(DATASET.get("bands") or []), "order": list(DATASET.get("band_order") or DATASET.get("bands") or []),
           "sensor": DATASET.get("sensor", ""), "db": bool(DATASET.get("s1_db"))}]
if DATASET.get("bands2"):
    GROUPS.append({"bands": list(DATASET["bands2"]), "order": list(DATASET.get("band_order2") or DATASET["bands2"]),
                   "sensor": DATASET.get("sensor2", ""), "db": bool(DATASET.get("s1_db2"))})
# the band names of both groups; a pair in time (the same bands before / after) names the second copy B04_2, B03_2, ...
BANDS = GROUPS[0]["bands"] + [b + "_2" if b in GROUPS[0]["bands"] else b for g in GROUPS[1:] for b in g["bands"]]
MULTI = bool(BANDS)
FUSE = []          # (folder of the first group's files, folder of the second group's), set by load_data
BAND_STATS = None  # per-band (mean, std), measured on the training images (multiband_stats)
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


OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
TEST_OOM_ABOVE = int(os.environ.get("NNB_TEST_OOM_ABOVE") or 0)  # CI only: pretend that more than this many images do not fit


def is_oom(e):
    """the GPU ran out of memory (PyTorch's own error, or the older RuntimeError some libraries still raise)"""
    return (OOM is not RuntimeError and isinstance(e, OOM)) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


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


IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def stats():
    if MULTI:
        return BAND_STATS
    if TRAIN.get("pretrained"):
        return IMAGENET  # pretrained weights expect the images normalised as in their own training
    if DATASET["source"] == "builtin" and not DATASET.get("resize"):
        return STATS[DATASET["builtin"]]
    c = DATASET["channels"]
    if DATASET["source"] == "builtin":
        m, s = STATS[DATASET["builtin"]]
        return (tuple(m) * 3)[:c] if len(m) < c else m[:c], (tuple(s) * 3)[:c] if len(s) < c else s[:c]
    return (0.5,) * c, (0.5,) * c


# ---------------------------------------------------------------- pretrained weights
def producers(model, x):
    """for every module: the module whose output it received (one forward pass with hooks; in eval mode, so no statistics change)"""
    made, kept, seen = {}, [], {}
    def hook(name):
        def f(mod, inp, out):
            if inp and torch.is_tensor(inp[0]) and id(inp[0]) in made:
                seen.setdefault(name, made[id(inp[0])])
            if torch.is_tensor(out):
                kept.append(out)  # keeps the object alive, so its id is not reused
                made[id(out)] = name
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hs:
            h.remove()
        model.train(was)
    return seen


def load_pretrained(model, name):
    """copy torchvision's ImageNet weights `name` (resnet18 | vgg16 | vit_b_16) into the design: each tensor of the design, in order, gets
    the next unused torchvision tensor of the same shape (the demos are built in torchvision's layer order). BatchNorm tensors all have the
    same shape (C,), so a BatchNorm instead gets the torchvision BatchNorm fed by the convolution that feeds it (found by a forward pass;
    a ResNet block with a shortcut projection is laid out in a different order). Tensors without a match (a new output head for another
    number of classes) keep their random start. Returns the names of the loaded tensors."""
    import torchvision
    os.environ.setdefault("TORCH_HOME", os.path.join(DATA, "torch"))
    ref = getattr(torchvision.models, name)(weights="DEFAULT")
    norm = torch.nn.modules.batchnorm._BatchNorm
    def bn_names(m):
        return {n for n, mod in m.named_modules() if isinstance(mod, norm)}
    owner = lambda key: key.rsplit(".", 1)[0] if "." in key else ""
    dev = next(model.parameters()).device
    mine_from = producers(model, torch.zeros(1, *EXPECTED["input"], device=dev)) if bn_names(model) else {}
    ref_from = producers(ref, torch.zeros(1, 3, 224, 224)) if bn_names(ref) else {}
    ref_bn, my_bn = bn_names(ref), bn_names(model)
    src = list(ref.state_dict().items())
    used = [False] * len(src)
    dst = model.state_dict()
    loaded, missing, pairs = [], [], {}
    for key, t in dst.items():  # everything except BatchNorm: by order and shape
        if owner(key) in my_bn:
            continue
        j = next((j for j, (k, st) in enumerate(src) if not used[j] and owner(k) not in ref_bn and st.shape == t.shape), None)
        widen = False
        if j is None and MULTI and t.dim() == 4 and t.shape[1] != 3:
            # the first convolution of a multiband design: ImageNet's 3-channel filters, widened to the bands (expand_rgb)
            j = next((j for j, (k, st) in enumerate(src) if not used[j] and owner(k) not in ref_bn and st.dim() == 4 and st.shape[1] == 3
                      and st.shape[0] == t.shape[0] and st.shape[2:] == t.shape[2:]), None)
            widen = j is not None
        if j is None:
            missing.append(key)
            continue
        used[j] = True
        pairs.setdefault(owner(key), owner(src[j][0]))
        dst[key] = (expand_rgb(src[j][1], t.shape[1]) if widen else src[j][1]).to(t.dtype)
        loaded.append(key)
    index = {k: j for j, (k, _) in enumerate(src)}
    for bn in [n for n, _ in model.named_modules() if n in my_bn]:  # BatchNorm: the one after the matching convolution
        r = pairs.get(mine_from.get(bn))
        rbn = next((n for n in ref_bn if ref_from.get(n) == r), None) if r else None
        for key in [k for k in dst if owner(k) == bn]:
            j = index.get(f"{rbn}.{key.rsplit('.', 1)[1]}") if rbn else None
            if j is None or used[j] or src[j][1].shape != dst[key].shape:
                missing.append(key)
                continue
            used[j] = True
            dst[key] = src[j][1].to(dst[key].dtype)
            loaded.append(key)
    model.load_state_dict(dst)
    weights = [k for k, v in model.state_dict().items() if v.dim() > 1]
    loaded_weights = [k for k in loaded if k in weights]
    if len(loaded_weights) < 0.8 * len(weights):
        raise SystemExit(f"PRETRAINED: the design does not match {name}: only {len(loaded_weights)} of its {len(weights)} weight tensors have "
                         f"a counterpart. Use ImageNet weights with the unchanged demo (its layers may be widened only at the output).")
    emit("pretrained", weights=name, loaded=len(loaded), total=len(dst), new=[k for k in missing if not k.endswith("num_batches_tracked")][:20])
    return loaded


def expand_rgb(w, c):
    """ImageNet's first-layer filters (out, 3, k, k) for c bands: the red, green and blue filters go to B04, B03 and B02 when the bands
    include them (by name), every other band gets the mean of the three; all scaled by 3 / c so the layer's output keeps its size"""
    names = BANDS if c == len(BANDS) else next((g["bands"] for g in GROUPS if len(g["bands"]) == c), [""] * c)
    rgb = {"B04": 0, "B03": 1, "B02": 2}
    out = torch.stack([w[:, rgb[band_key(b)]] if band_key(b) in rgb else w.mean(1) for b in names], 1)
    emit("info", message=f"ImageNet's first-layer filters widened from 3 to {c} channels "
                         f"({sum(band_key(b) in rgb for b in names)} from red / green / blue, the others from their mean)")
    return out * (3.0 / c)


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
        if REGRESS:
            return self.tfm(to_pil(row[DATASET["hf_image_col"]])), with_heads(hf_targets(row), row)
        label = row[DATASET["hf_label_col"]]
        if MULTILABEL:
            y = torch.zeros(len(CLASSES))
            for v in (label if isinstance(label, (list, tuple)) else [label]):
                y[self.names[v] if isinstance(v, str) else int(v)] = 1.0
            return self.tfm(to_pil(row[DATASET["hf_image_col"]])), with_heads(y, row)
        if isinstance(label, str):
            label = self.names[label]
        return self.tfm(to_pil(row[DATASET["hf_image_col"]])), with_heads(int(label), row)


def hf_targets(row):
    """regression: the target columns of a Hugging Face row as a float tensor"""
    try:
        return torch.tensor([float(row[c]) for c in CLASSES], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f"DATA: the target columns {CLASSES} must be numbers in every row ({type(e).__name__}: {e}).")


def hf_array(v):
    """a Hugging Face cell holding a multiband image as a NumPy array: an array / nested list, GeoTIFF or .npy bytes, or a picture"""
    import io
    import numpy as np
    if isinstance(v, dict) and v.get("bytes") is not None:
        data = v["bytes"]
        if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            try:
                import tifffile
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tifffile", "imagecodecs"], check=False)
                import tifffile
            return tifffile.imread(io.BytesIO(data))
        if data[:6] == b"\x93NUMPY":
            return np.load(io.BytesIO(data))
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)))
    if isinstance(v, dict) and v.get("path"):
        return read_array(v["path"])
    return np.asarray(v)


class HFBands(torch.utils.data.Dataset):
    """multiband images from a Hugging Face split (an array column, or GeoTIFF bytes): bands picked by name, normalised per band; labels
    as for pictures (a class, a list of classes for multi-label, or a mask column for segmentation)"""
    def __init__(self, ds, train):
        self.ds, self.train = ds, train
        self.names = {n: i for i, n in enumerate(CLASSES)}
        try:
            # array columns as NumPy arrays (an Array3D read row by row as nested lists is many times slower)
            self.ds = ds.with_format("numpy", columns=[DATASET["hf_image_col"]], output_all_columns=True)
        except Exception:
            pass

    def __len__(self):
        return len(self.ds)

    def raw(self, i):
        import numpy as np
        a = pick_bands(orient(hf_array(self.ds[i][DATASET["hf_image_col"]]), GROUPS[0]["order"], f"row {i}"), GROUPS[0])
        t = torch.from_numpy(np.ascontiguousarray(a))
        h, w = DATASET["image_size"]
        if tuple(t.shape[1:]) != (h, w):
            t = nn.functional.interpolate(t[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
        return t

    def __getitem__(self, i):
        import numpy as np
        row = self.ds[i]
        x = normalise(self.raw(i))
        if SEG:
            y = mask_classes(np.asarray(hf_array(row[DATASET["hf_mask_col"]])).squeeze(), x.shape[1:])
        elif REGRESS:
            y = hf_targets(row)
        elif MULTILABEL:
            y = torch.zeros(len(CLASSES))
            v = row[DATASET["hf_label_col"]]
            for c in (v if isinstance(v, (list, tuple)) else [v]):
                y[self.names[c] if isinstance(c, str) else int(c)] = 1.0
        else:
            v = row[DATASET["hf_label_col"]]
            y = self.names[v] if isinstance(v, str) else int(v)
        if self.train:
            x, y2 = flips(x, y if SEG else None)
            y = y2 if SEG else y
        return x, with_heads(y, row)


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


def local_root():
    """the owner's own files: a folder already on this machine, the private Kaggle dataset the app uploaded (attached to the notebook),
    or the copy the app made in <data>/local/<key>"""
    key = DATASET["local_key"]
    if DATASET.get("cloud_url"):
        root = os.path.join(DATA, "cloud", key)
        cloud_fetch(DATASET["cloud_url"], root)
    elif DATASET.get("local_on_machine"):
        root = os.path.expanduser(DATASET["local_dir"])
    elif ON_KAGGLE:
        found = [c for c in [f"/kaggle/input/{key}"] + glob.glob(f"/kaggle/input/**/{key}", recursive=True) if os.path.isdir(c)]
        if not found:
            raise SystemExit(f"DATA: your files (the Kaggle dataset {key}) are not attached to this notebook (looked in /kaggle/input).")
        root = found[0]
    else:
        root = os.path.join(DATA, "local", key)
    if not os.path.isdir(root):
        raise SystemExit(f"DATA: the folder with your files is not on this machine ({root}).")
    return root


def pip_install(*names):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *names], check=False)


def cloud_fetch(url, root):
    """bring the dataset from the cloud to this machine (once; later runs fetch only what is new or changed). gs://bucket/folder with
    the google-cloud-storage library (the key the app copied to ~/.config/nnb/gcs-key.json, else anonymous: public buckets);
    a shared Google Drive folder or file link with gdown (a zip / tar is unpacked); rclone:remote:path with rclone; file:///folder copies"""
    import shutil
    os.makedirs(root, exist_ok=True)
    done = os.path.join(root, ".nnb_fetched")
    emit("info", message=f"fetching the data from {url}")
    t0, count, size = time.time(), 0, 0
    if url.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            pip_install("google-cloud-storage")
            from google.cloud import storage
        key = os.path.expanduser("~/.config/nnb/gcs-key.json")
        client = storage.Client.from_service_account_json(key) if os.path.exists(key) else storage.Client.create_anonymous_client()
        bucket, _, prefix = url[5:].partition("/")
        prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        try:
            blobs = list(client.list_blobs(bucket, prefix=prefix))
        except Exception as e:
            raise SystemExit(f"DATA: cannot list {url}: {type(e).__name__}: {e} (a private bucket needs the Cloud Storage key: Connection tab, "
                             "and 'copy tokens' when starting the run; Kaggle notebooks can read public buckets only)")
        for b in blobs:
            if b.name.endswith("/"):
                continue
            dest = os.path.join(root, b.name[len(prefix):])
            if os.path.exists(dest) and os.path.getsize(dest) == b.size:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            b.download_to_filename(dest + ".part")
            os.replace(dest + ".part", dest)
            count, size = count + 1, size + (b.size or 0)
        if not blobs:
            raise SystemExit(f"DATA: nothing found under {url}.")
    elif url.startswith("https://drive.google.com/"):
        if os.path.exists(done):
            return
        try:
            import gdown
        except ImportError:
            pip_install("gdown")
            import gdown
        if "/folders/" in url:
            try:
                files = gdown.download_folder(url, output=root, quiet=True) or []
            except Exception as e:
                raise SystemExit(f"DATA: Google Drive folder download failed: {e}. The folder must be shared as 'anyone with the link'; "
                                 "gdown fetches at most 50 files per folder (for more, use Cloud Storage or rclone)")
            count = len(files)
        else:
            f = gdown.download(url, output=root + "/", quiet=True, fuzzy=True)
            if not f:
                raise SystemExit("DATA: Google Drive file download failed (is it shared as 'anyone with the link'?)")
            count = 1
            if f.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                shutil.unpack_archive(f, root)
                os.remove(f)
    elif url.startswith("rclone:"):
        exe = shutil.which("rclone")
        if not exe:
            raise SystemExit("DATA: rclone is not installed on this machine (https://rclone.org/install/; then 'rclone config' once to add the remote)")
        r = subprocess.run([exe, "copy", url[len("rclone:"):], root, "--transfers", "8"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("DATA: rclone copy failed: " + (r.stderr or r.stdout)[-500:])
    elif url.startswith("file://"):
        src = url[len("file://"):]
        if not os.path.isdir(src):
            raise SystemExit(f"DATA: the folder {src} does not exist.")
        shutil.copytree(src, root, dirs_exist_ok=True)
        count = sum(len(fs) for _, _, fs in os.walk(src))
    else:
        raise SystemExit(f"DATA: unknown cloud location {url}")
    with open(done, "w") as f:
        f.write(url)
    emit("info", message=f"data ready in {root}: {count} files fetched ({size / 1e6:.1f} MB known) in {time.time() - t0:.0f} s")


def data_root():
    return local_root() if DATASET["source"] == "local" else kaggle_root()


def kaggle_folder(root, sub):
    p = os.path.join(root, sub) if sub else root
    if os.path.isdir(p):
        return p
    found = [d for d in glob.glob(os.path.join(root, "**", sub), recursive=True) if os.path.isdir(d)] if sub else []
    if not found:
        raise SystemExit(f"DATA: the folder {sub!r} is not in the dataset.")
    return found[0]


def kaggle_pairs(root, ext_ok=IMAGE_EXT):
    """images and masks with the same name (masks may add _mask / _lab / _label, or use another extension)"""
    idir, mdir = kaggle_folder(root, DATASET["kaggle_train_dir"]), kaggle_folder(root, DATASET["kaggle_mask_dir"])
    masks = {}
    for f in os.listdir(mdir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXT + (".npy",):
            for suffix in ("_mask", "_lab", "_label", "-mask", ""):
                if suffix and stem.endswith(suffix):
                    masks.setdefault(stem[: -len(suffix)], os.path.join(mdir, f))
            masks.setdefault(stem, os.path.join(mdir, f))
    pairs = []
    for f in sorted(os.listdir(idir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in ext_ok and stem in masks:
            pairs.append((os.path.join(idir, f), masks[stem]))
    if not pairs:
        raise SystemExit(f"DATA: no image in {DATASET['kaggle_train_dir']!r} has a mask with the same name in {DATASET['kaggle_mask_dir']!r}.")
    return pairs


# ---------------------------------------------------------------- multispectral / radar images
BAND_EXT = (".tif", ".tiff", ".npy")


def band_key(b):
    """a band name for comparing: upper case, Earth Engine's B1 … B9 as B01 … B09 (the same rule as dataset::band_key in the app)"""
    u = str(b).strip().upper()
    return "B0" + u[1:] if len(u) == 2 and u[0] == "B" and u[1].isdigit() else u


DB_NOTE = [False]  # the "already in decibels" note is given once


def load_raw(path):
    """a GeoTIFF / .npy file as it is stored"""
    import numpy as np
    if path.lower().endswith(".npy"):
        return np.load(path)
    try:
        import tifffile
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tifffile", "imagecodecs"], check=False)
        import tifffile
    return tifffile.imread(path)


def read_array(path, order=None):
    """a GeoTIFF / .npy file as a float32 array, bands first when the band axis can be told by the number of bands of the files"""
    return orient(load_raw(path), order, os.path.basename(path))


# time series: every file holds all dates of one sample; the network gets (bands, dates, H, W)
TIME = int(DATASET.get("time_steps") or 0)


def time_orient(a, order, name="a sample"):
    """a time series as float32 (bands of the files, dates, H, W), from (dates, bands, H, W), (dates, H, W, bands), (bands, dates, H, W),
    or a GeoTIFF whose bands go date by date ((dates·bands, H, W) or (H, W, dates·bands): all bands of date 1, then of date 2, ...)"""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    c, t = len(order), TIME
    if a.ndim == 4:
        if a.shape[0] == t and a.shape[1] == c:
            return a.transpose(1, 0, 2, 3)
        if a.shape[0] == t and a.shape[-1] == c:
            return a.transpose(3, 0, 1, 2)
        if a.shape[0] == c and a.shape[1] == t:
            return a
    if a.ndim == 3 and a.shape[0] == t * c:
        return a.reshape(t, c, *a.shape[1:]).transpose(1, 0, 2, 3)
    if a.ndim == 3 and a.shape[-1] == t * c:
        return a.transpose(2, 0, 1).reshape(t, c, *a.shape[:2]).transpose(1, 0, 2, 3)
    raise SystemExit(f"DATA: {name} has shape {tuple(a.shape)}, but a sample should hold {t} dates of {c} bands ({' '.join(order)}): "
                     f"(dates, bands, H, W) in a .npy, or {t * c} bands date by date in a GeoTIFF; check the dates and the band order.")


def orient(a, order=None, name="an image"):
    """an array as float32 (bands, H, W): the band axis is the one that matches the number of bands of the files"""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    order = order or GROUPS[0]["order"]
    # (H, W, bands) -> (bands, H, W): the band axis is the one that matches the number of bands of the files
    if a.ndim == 3 and a.shape[0] != len(order) and a.shape[-1] == len(order):
        a = a.transpose(2, 0, 1)
    if a.ndim != 3 or a.shape[0] != len(order):
        raise SystemExit(f"DATA: {name} has shape {tuple(a.shape)}, but the files should have {len(order)} bands "
                         f"({' '.join(order)}); check the band order in the Dataset tab.")
    return a


def pick_bands(a, g=None):
    """the chosen bands of a (bands, H, W) array of band group g, in the chosen order; Sentinel-1 in decibels when asked (values that
    are already decibels, as Earth Engine's S1_GRD exports them, are recognised by their negative values and left alone)"""
    import numpy as np
    g = g or GROUPS[0]
    if "idx" not in g:
        keys = [band_key(b) for b in g["order"]]
        g["idx"] = [keys.index(band_key(b)) for b in g["bands"]]
    a = a[g["idx"]]
    if g["sensor"] == "s1" and g["db"]:
        if np.nanmin(a) < 0:
            if not DB_NOTE[0]:
                DB_NOTE[0] = True
                emit("warning", message="the Sentinel-1 values are already in decibels (negative values): not converted again")
        else:
            a = 10.0 * np.log10(np.clip(a, 1e-6, None))
    return a


def read_patch_folder(folder, g):
    """BigEarthNet's layout: a folder per patch with one GeoTIFF per band (<patch>_B02.tif, <patch>_VV.tif, …); the chosen bands of group
    g, each brought to the largest band's size (Sentinel-2 has 10, 20 and 60 m bands), in the chosen order"""
    import numpy as np
    files = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in BAND_EXT:
            files[band_key(stem.rsplit("_", 1)[-1])] = os.path.join(folder, f)
    planes = []
    for b in g["bands"]:
        f = files.get(band_key(b))
        if f is None:
            raise SystemExit(f"DATA: {os.path.basename(folder)} has no file for band {b} (found: {', '.join(sorted(files)) or 'none'}).")
        a = read_array(f, [b])
        planes.append(torch.from_numpy(np.ascontiguousarray(a[0], dtype=np.float32)))
    size = max((p.shape for p in planes), key=lambda s: s[0] * s[1])
    planes = [p if tuple(p.shape) == tuple(size) else
              nn.functional.interpolate(p[None, None], size=tuple(size), mode="bilinear", align_corners=False)[0, 0] for p in planes]
    a = torch.stack(planes).numpy()
    g2 = dict(g, idx=list(range(len(g["bands"]))))       # already in the chosen order: only the dB step of pick_bands is left
    return pick_bands(a, g2)


FUSE_BY_PATH = {}  # labels tables that name the second group's file per row (BigEarthNet: s1_name): first path -> second path


def fuse_path(path):
    """the second band group's file for a first-group file: the same path relative to the first group's folder, in the second group's
    folder, with any of the extensions .tif / .tiff / .npy"""
    if path in FUSE_BY_PATH:
        return FUSE_BY_PATH[path]
    base1, base2 = FUSE
    stem = os.path.splitext(os.path.relpath(path, base1))[0]
    for ext in BAND_EXT + tuple(e.upper() for e in BAND_EXT):
        if os.path.exists(os.path.join(base2, stem + ext)):
            return os.path.join(base2, stem + ext)
    raise SystemExit(f"DATA: {stem} has no file in the second band group's folder ({base2}); both groups need the same file names.")


# the second band group at its own size (separate inputs): in the batch tensor it fills the top-left corner of its channels, the rest
# empty (NaN: left out of the statistics, 0 after normalising); the generated forward cuts it out again
SIZE2 = tuple(DATASET["image_size2"]) if len(GROUPS) == 2 and len(DATASET.get("image_size2") or []) == 2 else None
C1 = len(GROUPS[0]["bands"])


def read_groups(path, resize=True):
    """every band group of one image as float32 tensors (first group from `path`, the second from its twin file)"""
    import numpy as np
    out = []
    for k, g in enumerate(GROUPS):
        p = path if k == 0 else fuse_path(path)
        if TIME:
            # (bands, dates, H, W): the chosen bands of every date, each date resized like an image
            t = torch.from_numpy(np.ascontiguousarray(pick_bands(time_orient(load_raw(p), g["order"], os.path.basename(p)), g)))
            h, w = DATASET["image_size"]
            if resize and tuple(t.shape[2:]) != (h, w):
                c, n = t.shape[:2]
                t = nn.functional.interpolate(t.reshape(1, c * n, *t.shape[2:]), size=(h, w), mode="bilinear", align_corners=False,
                                              antialias=True)[0].reshape(c, n, h, w)
            out.append(t)
            continue
        a = read_patch_folder(p, g) if os.path.isdir(p) else pick_bands(read_array(p, g["order"]), g)
        t = torch.from_numpy(np.ascontiguousarray(a))
        h, w = SIZE2 if k == 1 and SIZE2 else DATASET["image_size"]
        if resize and tuple(t.shape[1:]) != (h, w):
            t = nn.functional.interpolate(t[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
        out.append(t)
    if len({tuple(t.shape[1:]) for t in out}) > 1 and not (SIZE2 and resize):
        raise SystemExit(f"DATA: {os.path.basename(path)}: the band groups have different sizes {[tuple(t.shape[1:]) for t in out]}; "
                         "export them on the same grid.")
    return out


def read_bands(path, resize=True):
    """one image (both band groups when fused, stacked) as a float32 tensor (bands, H, W) with the chosen bands, in the chosen order;
    empty pixels (NaN, as Earth Engine writes masked pixels of float images) stay NaN until normalised"""
    groups = read_groups(path, resize)
    if SIZE2 and resize:
        h, w = DATASET["image_size"]
        g2 = torch.full((groups[1].shape[0], h, w), float("nan"))
        g2[:, :SIZE2[0], :SIZE2[1]] = groups[1]
        groups[1] = g2
    return torch.cat(groups, 0)


def normalise(x):
    """per-band (x - mean) / std; empty pixels (NaN / infinite) become 0 = the band's mean"""
    if BAND_STATS is not None:
        m, s = BAND_STATS
        shape = (-1,) + (1,) * (x.dim() - 1)   # (bands, H, W), or (bands, dates, H, W) for a time series
        x = (x - torch.tensor(m).view(shape)) / torch.tensor(s).view(shape)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def read_mask(path, size=None):
    """a label image (PNG / GeoTIFF / .npy) as int64 class numbers; values that are not a class (Earth Engine's no-data, 255 in a
    multi-class mask, negative numbers) become -100, which the loss and the scores leave out. Binary masks stored as 0/255 = 0/1."""
    import numpy as np
    if path.lower().endswith((".npy", ".tif", ".tiff")):
        if path.lower().endswith(".npy"):
            m = np.load(path)
        else:
            import tifffile
            m = tifffile.imread(path)
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[0] if m.shape[0] < m.shape[-1] else m[..., 0]
    else:
        from PIL import Image
        m = np.array(Image.open(path).convert("L"))
    return mask_classes(m, size)


def mask_classes(m, size=None):
    import numpy as np
    m = np.nan_to_num(np.asarray(m, dtype=np.float64), nan=-1)
    if len(CLASSES) == 2 and set(np.unique(m).tolist()) <= {0.0, 255.0}:
        m = (m > 0).astype(np.float64)
    m = np.where((m >= 0) & (m < len(CLASSES)) & (m == np.floor(m)), m, -100).astype(np.int64)
    t = torch.from_numpy(m)
    if size and tuple(t.shape) != tuple(size):
        t = nn.functional.interpolate(t[None, None].float(), size=tuple(size), mode="nearest")[0, 0].long()
    return t


def flips(x, y=None):
    """training augmentation for satellite images: flips and quarter turns (no "up"), on the image and its mask together; with the
    second group at its own size, on each group's block (the same flips and turns, so both still show the same place)"""
    if SIZE2:
        turn = AUG.get("rotate") and x.shape[-1] == x.shape[-2] and SIZE2[0] == SIZE2[1]
    else:
        turn = AUG.get("rotate") and x.shape[-1] == x.shape[-2]
    ops = []
    if AUG.get("hflip") and random.random() < 0.5:
        ops.append(lambda t: t.flip(-1))
    if AUG.get("vflip") and random.random() < 0.5:
        ops.append(lambda t: t.flip(-2))
    if turn:
        k = random.randint(0, 3)
        ops.append(lambda t, k=k: torch.rot90(t, k, (-2, -1)))
    if SIZE2:
        a, b = x[:C1], x[C1:, :SIZE2[0], :SIZE2[1]]
        for op in ops:
            a, b = op(a), op(b)
            y = None if y is None else op(y)
        x = x.clone()
        x[:C1], x[C1:, :SIZE2[0], :SIZE2[1]] = a, b
        return x, y
    for op in ops:
        x = op(x)
        y = None if y is None else op(y)
    return x, y


class BandFolder(torch.utils.data.Dataset):
    """one subfolder per class, each with GeoTIFF / .npy files; normalised per band; flips and quarter turns on the training data"""
    def __init__(self, files, train):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, label = self.files[i]
        x = normalise(read_bands(path))
        if self.train:
            x, _ = flips(x)
        return x, label


class BandPairs(torch.utils.data.Dataset):
    """segmentation of multiband images: (image file, mask file) pairs with the same name; image and mask flipped / turned together"""
    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = self.pairs[i]
        x = normalise(read_bands(img))
        y = read_mask(mask, x.shape[-2:])
        if self.train:
            x, y = flips(x, y)
        return x, y


class SceneTiles(torch.utils.data.Dataset):
    """large images (an Earth Engine export) and their label images cut into tiles of image_size; the arrays are kept as .npy files in
    the data folder and read with memory mapping, so a scene larger than the memory works"""
    def __init__(self, tiles, train):
        self.tiles, self.train, self.open = tiles, train, {}

    def __len__(self):
        return len(self.tiles)

    def arr(self, path):
        import numpy as np
        if path not in self.open:
            self.open[path] = np.load(path, mmap_mode="r")
        return self.open[path]

    def raw(self, i):
        """tile i before normalising (for the band statistics)"""
        import numpy as np
        img, _, y0, x0 = self.tiles[i]
        h, w = DATASET["image_size"]
        return torch.from_numpy(np.array(self.arr(img)[:, y0:y0 + h, x0:x0 + w], dtype=np.float32))

    def __getitem__(self, i):
        import numpy as np
        lab, y0, x0 = self.tiles[i][1:]
        h, w = DATASET["image_size"]
        x = normalise(self.raw(i))
        y = torch.from_numpy(np.array(self.arr(lab)[y0:y0 + h, x0:x0 + w], dtype=np.int64))
        if self.train:
            x, y = flips(x, y)
        return x, y


def set_fuse(root, first):
    """fusion: remember the first group's folder and find the second group's (DATASET["fuse_dir"]) in the dataset"""
    if len(GROUPS) > 1:
        FUSE[:] = [first, kaggle_folder(root, DATASET["fuse_dir"])]


# ---------------------------------------------------------------- multi-label: several classes per image, from a CSV
FILE_COLS = ("file", "filename", "file_name", "image", "image_id", "img", "path", "name", "patch", "patch_id", "id")
LABEL_COLS = ("labels", "label", "tags", "classes", "class", "categories", "landcover")


def split_labels(cell, names):
    """the classes in one CSV cell: names (or class numbers) separated by ; , | or spaces; a name with spaces is kept whole"""
    import re
    cell = cell.strip()
    if not cell:
        return []
    out = []
    # class names that themselves contain , ; | (BigEarthNet: "Beaches, dunes, sands") are taken whole first
    for name in sorted((n for n in names if re.search(r"[;,|]", n)), key=len, reverse=True):
        if name in cell:
            out.append(names[name])
            cell = cell.replace(name, "")
    cell = cell.strip(" ;,|")
    if not cell:
        return out
    parts = [x.strip() for x in re.split(r"[;,|]", cell)] if re.search(r"[;,|]", cell) else ([cell] if cell in names else cell.split())
    for x in parts:
        x = x.strip().strip("'\"[]")
        if x in names:
            out.append(names[x])
        elif x.isdigit() and int(x) < len(CLASSES):
            out.append(int(x))
        elif x:
            raise SystemExit(f"DATA: the label {x!r} is not one of the classes ({', '.join(CLASSES[:8])}{', …' if len(CLASSES) > 8 else ''}).")
    return out


SPLIT_COLS = ("split", "set", "subset", "fold")
FUSE_COLS = ("s1_name", "s1_patch", "s1", "fuse", "second")


def read_table(path):
    """the labels table as (header, rows of strings or lists): CSV / TSV, or Parquet (BigEarthNet's metadata.parquet; needs pyarrow,
    installed when missing)"""
    if path.lower().endswith(".parquet"):
        try:
            import pandas as pd
            import pyarrow  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pandas", "pyarrow"], check=False)
            import pandas as pd
        df = pd.read_parquet(path)
        head = [str(c) for c in df.columns]
        rows = [[v.tolist() if hasattr(v, "tolist") and not isinstance(v, str) else v for v in r] for r in df.itertuples(index=False, name=None)]
        return head, rows
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="\t" if path.lower().endswith(".tsv") else ","))
    return [h.strip() for h in rows[0]], rows[1:]


def folder_index(base, depth=3):
    """name -> path of every folder up to `depth` levels below base (BigEarthNet keeps patches in <tile>/<patch>/)"""
    index, level = {}, [base]
    for _ in range(depth):
        nxt = []
        for d in level:
            try:
                for e in os.scandir(d):
                    if e.is_dir():
                        index.setdefault(e.name, e.path)
                        nxt.append(e.path)
            except OSError:
                pass
        level = nxt
    return index


def label_rows(root):
    """(image file or patch folder, multi-hot tensor, split or None) for every row of the labels table whose image exists; files are
    found relative to the table's folder, then the training folder (also below it, for patch folders), with or without their extension.
    A split column (train / validation / test) is used as given; a column naming the second band group's patch (s1_name) pairs them."""
    path = os.path.join(root, DATASET["labels_csv"])
    if not os.path.isfile(path):
        found = glob.glob(os.path.join(root, "**", DATASET["labels_csv"]), recursive=True)
        if not found:
            raise SystemExit(f"DATA: the labels file {DATASET['labels_csv']!r} is not in the dataset.")
        path = found[0]
    head, rows = read_table(path)
    low = [h.lower() for h in head]
    names = {c: k for k, c in enumerate(CLASSES)}
    fcol = next((i for i, h in enumerate(low) if h in FILE_COLS), 0)
    scol = next((i for i, h in enumerate(low) if h in SPLIT_COLS), None)
    ucol = next((i for i, h in enumerate(low) if h in FUSE_COLS), None) if len(GROUPS) > 1 else None
    onehot = [head.index(c) for c in CLASSES if c in head]
    skip = {fcol, scol, ucol}
    if REGRESS and len(onehot) != len(CLASSES):
        raise SystemExit(f"DATA: the labels table has no column {', '.join(c for c in CLASSES if c not in head)} (columns: {', '.join(head)}).")
    lcol = None if len(onehot) == len(CLASSES) else next((i for i, h in enumerate(low) if h in LABEL_COLS and i not in skip),
                                                          next((i for i in range(len(head)) if i not in skip), None))
    if lcol is None and len(onehot) != len(CLASSES) and not (MULTITASK and TASK == "classify"):
        raise SystemExit("DATA: the labels table needs a file column and a labels column (or one 0/1 column per class).")
    train_dir = kaggle_folder(root, DATASET["kaggle_train_dir"]) if DATASET.get("kaggle_train_dir") else None
    bases = [os.path.dirname(path)] + ([train_dir] if train_dir else []) + [root]
    exts = ("",) + (BAND_EXT if MULTI else IMAGE_EXT)
    index, fuse_index = None, None
    items, missing, no_number = [], 0, 0
    for r in rows:
        if not r or len(r) <= fcol or not str(r[fcol]).strip():
            continue
        y = torch.zeros(len(CLASSES))
        if MULTITASK and TASK == "classify":
            y = None   # the class: from label_column, or the image's folder (found below)
        elif REGRESS:
            try:
                y = torch.tensor([float(r[onehot[k]]) for k in range(len(CLASSES))], dtype=torch.float32)
            except (TypeError, ValueError, IndexError):
                no_number += 1
                continue
            if not bool(torch.isfinite(y).all()):
                no_number += 1
                continue
        elif lcol is None:
            for k, c in enumerate(CLASSES):
                y[k] = float(str(r[onehot[k]]).strip() not in ("", "0", "0.0", "false", "False"))
        else:
            cell = r[lcol] if lcol < len(r) else ""
            ks = [names[v] if v in names else split_labels(str(v), names)[0] for v in cell] if isinstance(cell, (list, tuple)) \
                else split_labels(str(cell), names)
            for k in ks:
                y[k] = 1.0
        f = str(r[fcol]).strip()
        p = next((os.path.join(b, f + e) for b in bases for e in exts if os.path.exists(os.path.join(b, f + e))), None)
        if p is None and MULTI:
            if index is None:   # patch folders deeper down (BigEarthNet: <tile>/<patch>/)
                index = folder_index(train_dir or os.path.dirname(path))
            p = index.get(f)
        if p is None:
            missing += 1
            continue
        if MULTITASK:
            cells = {head[i]: r[i] for i in range(min(len(head), len(r)))}
            if y is None:
                cls = str(cells.get(DATASET["label_column"], "")).strip() if DATASET.get("label_column") else os.path.basename(os.path.dirname(p))
                if cls not in names:
                    no_number += 1
                    continue
                y = names[cls]
            extra = [y_head(cells, k) for k in range(1, len(HEADS) + 1)]
            if any(e is None for e in extra):
                no_number += 1
                continue
            y = (y, *extra)
        if ucol is not None and len(FUSE) == 2:
            if fuse_index is None:
                fuse_index = folder_index(FUSE[1])
            q = fuse_index.get(str(r[ucol]).strip())
            if q is None:
                missing += 1
                continue
            FUSE_BY_PATH[p] = q
        sp = str(r[scol]).strip().lower() if scol is not None else None
        items.append((p, y, {"val": "validation", "valid": "validation"}.get(sp, sp)))
    if missing:
        emit("warning", message=f"{missing} rows of {os.path.basename(path)} name files that are not in the dataset (left out)")
    if no_number:
        emit("warning", message=f"{no_number} rows of {os.path.basename(path)} have an empty, unknown or non-numeric target (left out)")
    if len(items) < 2:
        raise SystemExit(f"DATA: {os.path.basename(path)}: no row names an image that is in the dataset (looked in {', '.join(bases)}).")
    if MULTITASK:
        emit("info", message=f"{len(items)} images with labels for the main task and {len(HEADS)} extra output(s): "
             + ", ".join(h["name"] for h in HEADS))
        return items
    if REGRESS:
        ys = torch.stack([it[1] for it in items])
        emit("info", message=f"{len(items)} images with targets; " + ", ".join(
            f"{c} {float(ys[:, k].min()):.4g} … {float(ys[:, k].max()):.4g}" for k, c in enumerate(CLASSES)))
        return items
    counts = sum(it[1] for it in items)
    emit("info", message=f"{len(items)} images with labels; per class: " + ", ".join(f"{c} {int(n)}" for c, n in zip(CLASSES, counts)))
    return items


class LabelFiles(torch.utils.data.Dataset):
    """multi-label images: (file, multi-hot) pairs; pictures through the transforms, or multiband arrays"""
    def __init__(self, items, train, tfm=None):
        self.items, self.train, self.tfm = items, train, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        if MULTI:
            x = normalise(read_bands(path))
            if self.train:
                x, _ = flips(x)
            return x, y
        from PIL import Image
        return self.tfm(Image.open(path)), y


def scene_split(root, tiles, seed):
    """train and test tiles. "" = 10 % of the tiles at random (neighbours of training tiles: optimistic); "strip" = the eastern tenth of
    every image, with one column of tiles left out between, so no test tile touches a training tile; "files" = a separately exported test
    region (scene_test_image / scene_test_label)"""
    how = DATASET.get("scene_split", "")
    if how == "files":
        te = scene_tiles(root, DATASET["scene_test_image"], DATASET["scene_test_label"])
        emit("info", message=f"test region: {len(te)} tiles from {DATASET['scene_test_image']}")
        return tiles, te
    if how == "strip":
        h, w = DATASET["image_size"]
        widest = {}
        for img, _, _, x0 in tiles:
            widest[img] = max(widest.get(img, 0), x0 + w)
        tr, te, gap = [], [], 0
        for t in tiles:
            cols = widest[t[0]] // w
            cut = max(2, cols - max(1, round(cols * 0.1)))   # the first test column
            col = t[3] // w
            if col >= cut:
                te.append(t)
            elif col == cut - 1:
                gap += 1                                    # touches the test strip: used for neither
            else:
                tr.append(t)
        if not te or not tr:
            raise SystemExit("DATA: the images are too narrow for a test strip (at least 3 columns of tiles): use a smaller image size, "
                             "or a separate test region.")
        emit("info", message=f"test strip: the eastern {len(te)} tiles; {gap} tiles next to it left out; {len(tr)} for training")
        return tr, te
    tr_i, te_i = split(len(tiles), seed)
    return [tiles[i] for i in tr_i], [tiles[i] for i in te_i]


def scene_files(root, what):
    p = os.path.join(root, what)
    if os.path.isfile(p):
        return [p]
    d = kaggle_folder(root, what)
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(BAND_EXT))
    if not files:
        raise SystemExit(f"DATA: no .tif / .tiff / .npy files in {what!r}.")
    return files


def scene_tiles(root, image_what=None, label_what=None):
    """(image .npy, label .npy, row, column) of every tile with enough data: at least half of its pixels have a value in the image and at
    least one has a class in the label image. A large Earth Engine export may be split into several files (name-0000000000-0000000000.tif):
    each image file needs a label file with the same name (or one image file and one label file)."""
    import hashlib
    import numpy as np
    imgs, labs = scene_files(root, image_what or DATASET["scene_image"]), scene_files(root, label_what or DATASET["scene_label"])
    if len(imgs) == 1 and len(labs) == 1:
        pairs = [(imgs[0], labs[0])]
    else:
        by_name = {os.path.splitext(os.path.basename(f))[0]: f for f in labs}
        pairs = [(f, by_name[os.path.splitext(os.path.basename(f))[0]]) for f in imgs if os.path.splitext(os.path.basename(f))[0] in by_name]
        if not pairs:
            raise SystemExit("DATA: no image file has a label file with the same name.")
    cache = os.path.join(DATA, "scenes", DATASET.get("local_key") or DATASET.get("kaggle_slug", "").replace("/", "__"))
    os.makedirs(cache, exist_ok=True)
    h, w = DATASET["image_size"]
    tag = hashlib.sha1(json.dumps([[{k: v for k, v in g.items() if k != "idx"} for g in GROUPS], DATASET.get("fuse_dir"), len(CLASSES)]).encode()).hexdigest()[:8]
    tiles = []
    for img, lab in pairs:
        stem = os.path.splitext(os.path.basename(img))[0]
        ip, lp = os.path.join(cache, f"{stem}-{tag}.npy"), os.path.join(cache, f"{stem}-{tag}-label.npy")
        if not (os.path.exists(ip) and os.path.exists(lp)):
            a = read_bands(img, resize=False).numpy()
            m = read_mask(lab).numpy()
            if m.shape != a.shape[1:]:
                raise SystemExit(f"DATA: {os.path.basename(img)} is {a.shape[2]}×{a.shape[1]} pixels but its label image "
                                 f"{os.path.basename(lab)} is {m.shape[1]}×{m.shape[0]}: export both on the same grid (same region, scale "
                                 "and projection).")
            for path, arr in ((ip, a), (lp, m)):
                np.save(path + ".tmp.npy", arr)
                os.replace(path + ".tmp.npy", path)
            emit("info", message=f"{os.path.basename(img)}: {a.shape[2]}×{a.shape[1]} pixels, {a.shape[0]} bands")
        a, m = np.load(ip, mmap_mode="r"), np.load(lp, mmap_mode="r")
        for y0 in range(0, a.shape[1] - h + 1, h):
            for x0 in range(0, a.shape[2] - w + 1, w):
                lt = m[y0:y0 + h, x0:x0 + w]
                if (lt >= 0).any() and np.isfinite(a[:, y0:y0 + h, x0:x0 + w]).all(0).mean() >= 0.5:
                    tiles.append((ip, lp, y0, x0))
    if len(tiles) < 2:
        raise SystemExit(f"DATA: the images give {len(tiles)} tile(s) of {h}×{w} with data: use a smaller image size or a larger region.")
    emit("info", message=f"{len(tiles)} tiles of {h}×{w} pixels from {len(pairs)} image(s)")
    return tiles


def band_files(folder):
    classes = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    files = [(os.path.join(folder, c, f), k) for k, c in enumerate(classes) for f in sorted(os.listdir(os.path.join(folder, c)))
             if f.lower().endswith(BAND_EXT)]
    if classes != list(CLASSES):
        raise SystemExit(f"DATA: the class folders are {classes}, but the design expects {CLASSES}.")
    if not files:
        raise SystemExit(f"DATA: no .tif / .tiff / .npy files in the class folders of {folder}.")
    return files


def multiband_stats(files, seed, load=None):
    """mean and standard deviation of every band over up to 256 training images; kept in band_stats.json next to train.py, so a resumed
    run normalises exactly as before"""
    global BAND_STATS
    path = os.path.join(OUT, "band_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("bands") == BANDS:
            BAND_STATS = (tuple(d["mean"]), tuple(d["std"]))
            return
    pick = random.Random(seed).sample(files, min(256, len(files)))
    total, sq, n = None, None, None
    for item in pick:
        x = (load(item) if load else read_bands(item[0])).double().flatten(1)
        ok = torch.isfinite(x)                 # empty pixels are left out of the statistics
        x = torch.where(ok, x, torch.zeros_like(x))
        total = x.sum(1) if total is None else total + x.sum(1)
        sq = (x * x).sum(1) if sq is None else sq + (x * x).sum(1)
        n = ok.sum(1) if n is None else n + ok.sum(1)
    n = n.clamp_min(1)
    mean = total / n
    std = (sq / n - mean * mean).clamp_min(1e-12).sqrt()
    BAND_STATS = (tuple(round(float(v), 6) for v in mean), tuple(round(max(float(v), 1e-6), 6) for v in std))
    with open(path, "w") as f:
        json.dump({"bands": BANDS, "mean": BAND_STATS[0], "std": BAND_STATS[1]}, f)
    emit("band_stats", bands=BANDS, mean=list(BAND_STATS[0]), std=list(BAND_STATS[1]))


def composite(x):
    if x.dim() == 4:   # a time series: the median of its dates
        x = x.median(1).values
    return composite_image(x)


def composite_image(x):
    """a picture (3 or 1 channels, 0..1) of a normalised multiband image: true colour B04 B03 B02 for Sentinel-2, VV / VH / VV-VH for
    Sentinel-1, else the first three bands; each channel stretched between its 2nd and 98th percentile"""
    m, s = BAND_STATS
    x = x * torch.tensor(s).view(-1, 1, 1) + torch.tensor(m).view(-1, 1, 1)
    pick = None
    keys = [band_key(b) for b in BANDS]
    if DATASET.get("sensor") == "s2" and all(b in keys for b in ("B04", "B03", "B02")):
        pick = [x[keys.index(b)] for b in ("B04", "B03", "B02")]
    elif DATASET.get("sensor") == "s1" and all(b in keys for b in ("VV", "VH")):
        vv, vh = x[keys.index("VV")], x[keys.index("VH")]
        pick = [vv, vh, vv - vh]
    if pick is None:
        pick = [x[i] for i in range(min(3, x.shape[0]))]
        if len(pick) == 2:
            pick.append(pick[0] - pick[1])  # two bands: a colour picture like Sentinel-1's VV / VH / VV-VH
    out = []
    for c in pick:
        lo, hi = torch.quantile(c.flatten().float(), torch.tensor([0.02, 0.98]))
        out.append(((c - lo) / (hi - lo if hi > lo else 1.0)).clamp(0, 1))
    return torch.stack(out)


def split(n, seed, part=0.1):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(n * part))
    return idx[k:], idx[:k]


def load_data(cfg):
    from torch.utils.data import Subset
    src = DATASET["source"]
    given_val = None   # validation data given by the dataset itself (a split column)
    train_t, test_t = (None, None) if MULTI else (image_transform(True), image_transform(False))
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
        if MULTI:
            train, test, train_eval = HFBands(tr, True), HFBands(te, False), HFBands(tr, False)
            multiband_stats(list(range(len(train))), cfg["seed"], load=train_eval.raw)
        elif SEG:
            train, test, train_eval = SegPairs(HFPairs(tr), True), SegPairs(HFPairs(te), False), SegPairs(HFPairs(tr), False)
        else:
            train, test, train_eval = HFImages(tr, train_t), HFImages(te, test_t), HFImages(tr, test_t)
    elif MULTI and SEG:
        root = data_root()
        if DATASET.get("layout") == "scene":
            first = os.path.join(root, DATASET["scene_image"])
            set_fuse(root, first if os.path.isdir(first) else os.path.dirname(first))
            tiles = scene_tiles(root)
            tr, te = scene_split(root, tiles, cfg["seed"])
            plain = SceneTiles(tr, False)
            multiband_stats(list(range(len(tr))), cfg["seed"], load=plain.raw)
            train, test, train_eval = SceneTiles(tr, True), SceneTiles(te, False), SceneTiles(tr, False)
        else:
            set_fuse(root, kaggle_folder(root, DATASET["kaggle_train_dir"]))
            pairs = kaggle_pairs(root, BAND_EXT)
            tr_i, te_i = split(len(pairs), cfg["seed"])
            tr, te = [pairs[i] for i in tr_i], [pairs[i] for i in te_i]
            multiband_stats(tr, cfg["seed"])
            train, test, train_eval = BandPairs(tr, True), BandPairs(te, False), BandPairs(tr, False)
    elif MULTILABEL or REGRESS or MULTITASK:
        root = data_root()
        if MULTI:
            set_fuse(root, os.path.dirname(os.path.join(root, DATASET["labels_csv"])) if not DATASET.get("kaggle_train_dir")
                     else kaggle_folder(root, DATASET["kaggle_train_dir"]))
        rows = label_rows(root)
        items = [(p, y) for p, y, _ in rows]
        if any(sp in ("train", "test") for _, _, sp in rows):
            # the table's own split (BigEarthNet: train / validation / test)
            tr = [(p, y) for p, y, sp in rows if sp == "train"]
            te = [(p, y) for p, y, sp in rows if sp == "test"]
            va = [(p, y) for p, y, sp in rows if sp == "validation"]
            emit("info", message=f"split from the labels table: {len(tr)} train, {len(va)} validation, {len(te)} test")
            if va:
                given_val = LabelFiles(va, False, test_t)
        else:
            tr_i, te_i = split(len(items), cfg["seed"])
            tr, te = [items[i] for i in tr_i], [items[i] for i in te_i]
        if MULTI:
            multiband_stats(tr, cfg["seed"])
        train, test, train_eval = LabelFiles(tr, True, train_t), LabelFiles(te, False, test_t), LabelFiles(tr, False, test_t)
    elif MULTI:
        root = data_root()
        set_fuse(root, kaggle_folder(root, DATASET["kaggle_train_dir"]))
        files = band_files(kaggle_folder(root, DATASET["kaggle_train_dir"]))
        if DATASET["kaggle_test_dir"]:
            tr_files, te_files = files, band_files(kaggle_folder(root, DATASET["kaggle_test_dir"]))
        else:
            tr_i, te_i = split(len(files), cfg["seed"])
            tr_files, te_files = [files[i] for i in tr_i], [files[i] for i in te_i]
        multiband_stats(tr_files, cfg["seed"])
        train, test, train_eval = BandFolder(tr_files, True), BandFolder(te_files, False), BandFolder(tr_files, False)
    else:
        root = data_root()
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
    val = given_val
    if val is None and cfg.get("val_fraction"):
        tr_i, va_i = split(len(train), cfg["seed"] + 7, cfg["val_fraction"])
        train, val = Subset(train, tr_i), Subset(train_eval, va_i)
    if cfg.get("limit_train"):
        train = Subset(train, range(min(cfg["limit_train"], len(train))))
    return train, val, test


LOSS_WEIGHT = None   # unbalanced classes: a weight per class (multi-label: pos_weight), set by class_weights()
THRESH = None        # multi-label: a decision threshold per class chosen on the validation data (tune_thresholds), else 0.5


LOSS = TRAIN.get("loss") or ""            # "" (cross-entropy / BCE; regression: MSE) | focal | dice | dice_ce | mse | l1 | huber | custom
GAMMA = float(TRAIN.get("focal_gamma") or 2.0)


def loss_of(out, y, smoothing=0.0):
    """the training loss: the main task's, plus each extra output's times its weight (several outputs)"""
    if not MULTITASK:
        return loss_one(out, y, smoothing)
    total = loss_one(out[0], y[0], smoothing)
    for k, h in enumerate(HEADS, 1):
        total = total + float(h.get("weight", 1.0)) * head_loss(k, out[k].float(), y[k])
    return total


def head_loss(k, out, y):
    """an extra output's loss: cross-entropy (or focal), binary cross-entropy, or MSE / L1 / Huber on its standardised targets"""
    h = HEADS[k - 1]
    if h["task"] == "classify":
        ce = nn.functional.cross_entropy(out, y, reduction="none")
        if h.get("loss") == "focal":
            pt = out.log_softmax(1).gather(1, y.unsqueeze(1)).squeeze(1).exp()
            ce = (1 - pt) ** 2 * ce
        return ce.mean()
    if h["task"] == "multilabel":
        return nn.functional.binary_cross_entropy_with_logits(out, y.float())
    mean, std = HEAD_STATS.get(k, (None, None))
    z = y.float() if mean is None else (y.float() - mean.to(y.device)) / std.to(y.device)
    if h.get("loss") == "l1":
        return nn.functional.l1_loss(out, z)
    if h.get("loss") == "huber":
        return nn.functional.huber_loss(out, z, delta=1.0)
    return nn.functional.mse_loss(out, z)


def head_targets(ds, k):
    """head k's targets of a dataset, without reading images where the dataset keeps them"""
    from torch.utils.data import Subset
    if isinstance(ds, Subset):
        inner = head_targets(ds.dataset, k)
        return [inner[i] for i in ds.indices]
    if isinstance(ds, LabelFiles):
        return [y[k] for _, y in ds.items]
    if isinstance(ds, (HFImages, HFBands)):
        cols = {c: list(ds.ds[c]) for c in (HEADS[k - 1]["columns"] or HEADS[k - 1]["classes"])}
        return [y_head({c: v[i] for c, v in cols.items()}, k) for i in range(len(ds))]
    return [ds[i][1][k] for i in random.Random(0).sample(range(len(ds)), min(2000, len(ds)))]


def head_stats(train):
    """regression heads: mean and spread of their targets on the training data (the network learns them standardised)"""
    for k, h in enumerate(HEADS, 1):
        if h["task"] != "regress":
            continue
        ys = torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in head_targets(train, k)]).double()
        mean, std = ys.mean(0), ys.std(0, unbiased=False) if len(ys) > 1 else torch.ones(ys.shape[1], dtype=torch.float64)
        std = torch.where(std > 1e-12, std, torch.ones_like(std))
        HEAD_STATS[k] = (mean.float(), std.float())
        emit("targets", head=h["name"], targets=h["columns"], mean=[round(float(v), 6) for v in mean], std=[round(float(v), 6) for v in std], n=len(ys))


def head_scores(outs, ys):
    """each extra output's scores on a whole evaluation: accuracy, micro F1 + mAP, or R² / MAE / RMSE in the targets' units"""
    res = []
    for k, h in enumerate(HEADS, 1):
        o, y = torch.cat(outs[k]), torch.cat(ys[k])
        d = {"name": h["name"], "task": h["task"]}
        if h["task"] == "classify":
            d.update(metric="accuracy", score=float((o.argmax(1) == y).float().mean()) if len(y) else 0.0)
        elif h["task"] == "multilabel":
            sc = multilabel_scores(torch.sigmoid(o), y, 0.5)
            d.update(metric="micro F1", score=sc["micro_f1"], map=sc["map"], macro_f1=sc["macro_f1"])
        else:
            mean, std = HEAD_STATS.get(k, (torch.zeros(o.shape[1]), torch.ones(o.shape[1])))
            sc = regression_scores(o * std + mean, y.float(), h["columns"])
            d.update(metric="R²", score=sc["r2"], mae=sc["mae"], rmse=sc["rmse"], per_target=sc["per_target"])
        res.append(d)
    return res


def loss_one(out, y, smoothing=0.0):
    """cross-entropy (one class per image / pixel), or binary cross-entropy per class for multi-label (smoothing pulls the targets
    towards 0.5); weighted by class when the Train tab asks for it. Focal: each term times (1 - p_true)^γ, so the easy, already-right
    images / pixels count little. Dice (segmentation, multi-label): 1 - the mean over classes of 2|P∩T| / (|P| + |T|) on the batch's
    probabilities; dice_ce adds the plain cross-entropy (BCE) to it. Unlabelled pixels (-100) are left out of every one."""
    out = out.float()
    if LOSS == "custom":
        # the Train tab's own loss (defined above, in the generated part of train.py)
        return custom_loss(out, standardise(y.float()) if REGRESS else y)
    if REGRESS:
        z = standardise(y.float())
        if LOSS == "l1":
            return nn.functional.l1_loss(out, z)
        if LOSS == "huber":
            return nn.functional.huber_loss(out, z, delta=1.0)
        return nn.functional.mse_loss(out, z)
    w = None if LOSS_WEIGHT is None else LOSS_WEIGHT.to(out.device)
    if MULTILABEL:
        t = y.float()
        if smoothing:
            t = t * (1 - smoothing) + 0.5 * smoothing
        if LOSS == "focal":
            bce = nn.functional.binary_cross_entropy_with_logits(out, t, pos_weight=w, reduction="none")
            p = torch.sigmoid(out)
            pt = p * y.float() + (1 - p) * (1 - y.float())
            return ((1 - pt) ** GAMMA * bce).mean()
        bce = nn.functional.binary_cross_entropy_with_logits(out, t, pos_weight=w)
        if LOSS in ("dice", "dice_ce"):
            d = dice_loss(torch.sigmoid(out), y.float(), 0, w)
            return d if LOSS == "dice" else d + bce
        return bce
    if LOSS == "focal":
        ce = nn.functional.cross_entropy(out, y, weight=w, label_smoothing=smoothing, reduction="none")  # 0 where y = -100
        keep = y >= 0
        pt = out.log_softmax(1).gather(1, y.clamp_min(0).unsqueeze(1)).squeeze(1).exp()
        norm = w[y.clamp_min(0)][keep].sum() if w is not None else keep.sum()
        return ((1 - pt) ** GAMMA * ce)[keep].sum() / norm.clamp_min(1e-6)
    ce = nn.functional.cross_entropy(out, y, weight=w, label_smoothing=smoothing)
    if LOSS in ("dice", "dice_ce") and SEG:
        keep = (y >= 0).unsqueeze(1)
        t = nn.functional.one_hot(y.clamp_min(0), out.shape[1]).movedim(-1, 1).float()
        d = dice_loss(out.softmax(1) * keep, t * keep, 1, w)
        if not bool(keep.any()):
            d = d * 0.0
        return d if LOSS == "dice" else d + torch.nan_to_num(ce)
    return ce


TARGET_MEAN, TARGET_STD = None, None   # regression: per target, from the training data (target_stats)


def standardise(y):
    """regression targets in the units the network learns (mean 0, spread 1 on the training data)"""
    if TARGET_MEAN is None:
        return y
    return (y - TARGET_MEAN.to(y.device)) / TARGET_STD.to(y.device)


def to_units(out):
    """regression: the network's outputs back in the targets' own units"""
    if TARGET_MEAN is None:
        return out
    return out * TARGET_STD.to(out.device) + TARGET_MEAN.to(out.device)


def target_stats(train):
    """regression: mean and spread of every target over the training data (a constant target keeps spread 1)"""
    global TARGET_MEAN, TARGET_STD
    labels = labels_of(train)
    if labels is None:
        labels = [main_y(train[i][1]) for i in random.Random(0).sample(range(len(train)), min(2000, len(train)))]
    ys = torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in labels]).double()
    mean, std = ys.mean(0), ys.std(0, unbiased=False) if len(ys) > 1 else torch.ones(ys.shape[1], dtype=torch.float64)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    TARGET_MEAN, TARGET_STD = mean.float(), std.float()
    emit("targets", targets=CLASSES, mean=[round(float(v), 6) for v in mean], std=[round(float(v), 6) for v in std],
         min=[round(float(v), 6) for v in ys.min(0)[0]], max=[round(float(v), 6) for v in ys.max(0)[0]], n=len(ys))


def regression_scores(pred, true, names=None):
    """per target: MAE, RMSE, R² (1 - residual / total sum of squares; 0 = no better than always predicting the mean), the mean of
    each over the targets, and up to 1000 (truth, prediction) points per target for the scatter plot"""
    names = CLASSES if names is None else names
    k = len(names)
    per = []
    for c in range(k):
        p, t = pred[:, c].double(), true[:, c].double()
        err = p - t
        ss_tot = float(((t - t.mean()) ** 2).sum()) if len(t) else 0.0
        per.append({"target": names[c], "mae": float(err.abs().mean()) if len(t) else 0.0,
                    "rmse": float((err ** 2).mean().sqrt()) if len(t) else 0.0,
                    "r2": 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0,
                    "mean": float(t.mean()) if len(t) else 0.0})
    pick = random.Random(0).sample(range(len(true)), min(1000, len(true)))
    mean = lambda key: sum(d[key] for d in per) / max(1, k)
    return {"r2": mean("r2"), "mae": mean("mae"), "rmse": mean("rmse"), "per_target": per,
            "scatter": {"truth": [[round(float(true[i, c]), 5) for i in pick] for c in range(k)],
                        "pred": [[round(float(pred[i, c]), 5) for i in pick] for c in range(k)]}}


def dice_loss(p, t, axis, w=None, eps=1.0):
    """1 - the (class-weighted) mean soft Dice score; p, t: probabilities and 0/1 targets with the classes on `axis`, summed over the
    batch and the pixels. eps keeps a class that is absent from the batch and not predicted near a score of 1"""
    dims = [d for d in range(p.dim()) if d != axis]
    inter, total = (p * t).sum(dims), (p + t).sum(dims)
    score = (2 * inter + eps) / (total + eps)
    if w is None:
        return 1 - score.mean()
    return 1 - (score * w).sum() / w.sum()


def class_weights(train, how):
    """the weight of each class against imbalance: 'inverse' = N / (K·n_c) (multi-label: pos_weight = negatives / positives per class),
    'sqrt' = its square root; a class without examples gets the largest weight; kept between 0.05 and 50 (pos_weight: 100)"""
    global LOSS_WEIGHT
    counts, _ = class_counts(train)
    n = torch.tensor(counts, dtype=torch.float64)
    if MULTILABEL:
        total = float(len(train)) if labels_of(train) is not None else max(1.0, float(n.sum()))
        w = (total - n).clamp_min(0) / n.clamp_min(1)
        w = w.sqrt() if how == "sqrt" else w
        w = w.clamp(0.05, 100)
    else:
        w = n.sum() / (len(counts) * n.clamp_min(1))
        w = w.sqrt() if how == "sqrt" else w
        w[n == 0] = w[n > 0].max() if bool((n > 0).any()) else 1.0
        w = w.clamp(0.05, 50)
    LOSS_WEIGHT = w.float()
    emit("class_weights", how=how, counts=counts, weights=[round(float(v), 4) for v in LOSS_WEIGHT],
         kind="pos_weight" if MULTILABEL else "weight")


def tune_thresholds(model, data, device, batch=128):
    """multi-label: for each class the threshold (0.05 … 0.95) with the highest F1 on the validation data (ties: the one nearest 0.5)"""
    global THRESH
    from torch.utils.data import DataLoader
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=batch, shuffle=False):
            probs.append(torch.sigmoid(main_out(model(x.to(device))).float()).cpu())
            ys.append(main_y(y) > 0.5)
    p, y = torch.cat(probs), torch.cat(ys)
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    best = []
    for c in range(p.shape[1]):
        def f1(t):
            pred = p[:, c] >= t
            tp, fp, fn = int((pred & y[:, c]).sum()), int((pred & ~y[:, c]).sum()), int((~pred & y[:, c]).sum())
            return 2 * tp / max(1, 2 * tp + fp + fn)
        best.append(max(grid, key=lambda t: (round(f1(t), 6), -abs(t - 0.5))) if bool(y[:, c].any()) else 0.5)
    THRESH = torch.tensor(best)
    emit("thresholds", thresholds=best)
    with open(os.path.join(OUT, "thresholds.json"), "w") as f:
        json.dump({"classes": CLASSES, "thresholds": best}, f)


def multilabel_scores(prob, y, th=None):
    """micro / macro F1 at 0.5 (or the tuned thresholds), mean average precision, exact match, and per class precision / recall / F1 /
    AP / support; th: a fixed threshold instead (an extra output)"""
    pred, y = prob >= (th if th is not None else 0.5 if THRESH is None else THRESH.view(1, -1)), y > 0.5
    tp, fp, fn = (pred & y).sum(0).float(), (pred & ~y).sum(0).float(), (~pred & y).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1))
    used = (tp + fp + fn) > 0
    aps = []
    for k in range(y.shape[1]):
        pos = int(y[:, k].sum())
        if pos == 0:
            aps.append(None)
            continue
        order = prob[:, k].argsort(descending=True)
        hits = y[order, k].float()
        prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
        aps.append(float((prec * hits).sum() / pos))
    valid = [a for a in aps if a is not None]
    per = [{"precision": round(float(tp[k] / (tp[k] + fp[k]).clamp_min(1)), 4), "recall": round(float(tp[k] / (tp[k] + fn[k]).clamp_min(1)), 4),
            "f1": round(float(f1[k]), 4), "ap": None if aps[k] is None else round(aps[k], 4), "support": int(y[:, k].sum())} for k in range(y.shape[1])]
    return {"micro_f1": micro, "macro_f1": float(f1[used].mean()) if bool(used.any()) else 0.0,
            "map": sum(valid) / max(1, len(valid)), "exact": float((pred == y).all(1).float().mean()), "per_class": per}


# ---------------------------------------------------------------- checks and results
def one_batch_check(model, loader, device):
    """run one batch forward + backward and compare every layer's shape with what the app computed"""
    seen = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, list(o.shape[1:])))
             for name, _, _ in LAYERS]
    x, yall = next(iter(loader))
    # two images check every shape; a full batch might not fit before the split is known
    x, yall = x[:2].to(device), y_to(tuple(t[:2] for t in yall) if isinstance(yall, (tuple, list)) else yall[:2], device)
    y = main_y(yall)
    if list(x.shape[1:]) != EXPECTED["input"]:
        raise SystemExit(f"CHECK: the images are {list(x.shape[1:])}, but the design's Input is {EXPECTED['input']}.")
    outs = model(x)
    if MULTITASK and not (isinstance(outs, (tuple, list)) and len(outs) == len(HEADS) + 1):
        raise SystemExit(f"CHECK: the network gives {len(outs) if isinstance(outs, (tuple, list)) else 1} output(s), but the dataset has "
                         f"{len(HEADS) + 1} (the main task and {len(HEADS)} extra).")
    out = main_out(outs)
    if SEG and list(y.shape[1:]) != list(out.shape[2:]):
        raise SystemExit(f"CHECK: the masks are {list(y.shape[1:])}, but the network outputs {list(out.shape[2:])}.")
    if (MULTILABEL or REGRESS) and list(y.shape) != list(out.shape):
        raise SystemExit(f"CHECK: the {'targets' if REGRESS else 'labels'} are {list(y.shape[1:])} per image, but the network outputs {list(out.shape[1:])}.")
    loss_of(outs, yall).backward()
    model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    bad = [f"{label}: expected {EXPECTED[name]}, got {seen.get(name)}" + (" (the class's '# out:' line declares the expected shape)" if "Custom" in label else "")
           for name, label, _ in LAYERS if seen.get(name) != EXPECTED[name]]
    if bad:
        raise SystemExit("CHECK: shapes differ from the design: " + "; ".join(bad[:5]))
    if not REGRESS and int(y.max()) >= len(CLASSES):
        raise SystemExit(f"CHECK: a label is {int(y.max())}, but there are only {len(CLASSES)} classes"
                         + (" (mask pixel values must be class numbers 0, 1, 2, ...)" if SEG else "") + ".")
    emit("check", message="one batch passed: every layer has the shape the design shows", batch=list(x.shape))


def evaluate(model, loader, device):
    """loss, accuracy (per image, or per pixel for segmentation), confusion matrix, mean IoU (segmentation), top-5 accuracy
    (classification with at least 6 classes; else None)"""
    model.eval()
    k = len(CLASSES)
    loss_sum, n_items, cm = 0.0, 0, torch.zeros(k * k, dtype=torch.long)
    top5 = 0 if (not SEG and not MULTILABEL and not REGRESS and k > 5) else None
    probs, targets = [], []
    global HEAD_SCORES
    h_out, h_y = {k: [] for k in range(1, len(HEADS) + 1)}, {k: [] for k in range(1, len(HEADS) + 1)}
    with torch.no_grad():
        for x, yall in loader:
            if INTERRUPT:
                raise Interrupted()
            x, yall = x.to(device), y_to(yall, device)
            outs = model(x)
            out, y = main_out(outs).float(), main_y(yall)
            for k in h_out:
                h_out[k].append(outs[k].float().cpu())
                h_y[k].append(yall[k].cpu())
            if MULTITASK and not SEG:
                # the validation loss is the whole training loss (every output), as in training
                loss_sum += loss_of(outs, yall).item() * len(y)
            if REGRESS:
                loss_sum += 0 if MULTITASK else loss_of(out, y).item() * len(y)
                n_items += len(y)
                probs.append(to_units(out).cpu())
                targets.append(y.float().cpu())
                continue
            if MULTILABEL:
                loss_sum += 0 if MULTITASK else loss_of(out, y).item() * len(y)
                n_items += len(y)
                probs.append(torch.sigmoid(out).cpu())
                targets.append(y.cpu())
                continue
            if not MULTITASK and (not SEG or bool((y >= 0).any())):  # a batch with no labelled pixel has no loss
                loss_sum += nn.functional.cross_entropy(out, y).item() * len(y)
            n_items += len(y)
            pred = out.argmax(1)
            keep = y >= 0                      # segmentation: pixels without a class (-100) are not scored
            cm += torch.bincount((y[keep] * k + pred[keep]).flatten().cpu(), minlength=k * k)
            if top5 is not None:
                top5 += int((out.topk(5, dim=1).indices == y.unsqueeze(1)).any(1).sum())
    HEAD_SCORES = head_scores(h_out, h_y) if MULTITASK else None
    if REGRESS:
        # (loss, R² in the accuracy's place, the scores in the confusion matrix's place, MAE in mean IoU's place)
        sc = regression_scores(torch.cat(probs), torch.cat(targets)) if probs else regression_scores(torch.zeros(0, k), torch.zeros(0, k))
        return loss_sum / max(n_items, 1), sc["r2"], sc, sc["mae"], None, None
    if MULTILABEL:
        # (loss, micro F1 in the accuracy's place, the scores in the confusion matrix's place, mAP in mean IoU's place)
        sc = multilabel_scores(torch.cat(probs), torch.cat(targets)) if probs else multilabel_scores(torch.zeros(0, k), torch.zeros(0, k))
        return loss_sum / max(n_items, 1), sc["micro_f1"], sc, sc["map"], None, None
    cm = cm.view(k, k)
    acc = float(cm.diag().sum()) / max(1, int(cm.sum()))
    iou = [float(cm[i, i]) / max(1, int(cm[i].sum() + cm[:, i].sum() - cm[i, i])) for i in range(k)]
    present = [iou[i] for i in range(k) if int(cm[i].sum() + cm[:, i].sum()) > 0]
    return (loss_sum / max(n_items, 1), acc, cm.tolist(), (sum(present) / max(1, len(present))), iou,
            None if top5 is None else top5 / max(1, n_items))


def norm01(t):
    t = t.float()
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / (hi - lo if hi > lo else 1.0)).clamp(0, 1)


def small(t, limit=48):
    """an image tensor (C,H,W) shrunk to at most limit pixels a side, as nested lists with 3 decimals"""
    if t.shape[-1] > limit or t.shape[-2] > limit:
        t = nn.functional.interpolate(t.unsqueeze(0).float(), size=(min(limit, t.shape[-2]), min(limit, t.shape[-1])), mode="nearest")[0]
    return [[[round(float(v), 3) for v in row] for row in ch] for ch in t]


def grad_cam(model, xs):
    """Grad-CAM (Selvaraju et al., arXiv:1610.02391) on the last layer whose output is an image: for each image, the feature maps of that
    layer weighted by how strongly the predicted class depends on them; (layer label, heat maps of shape (B, H, W) in 0..1) or None"""
    acts, grads = {}, {}
    hooks = []
    for name, label, _ in LAYERS:
        mod = getattr(model, name)
        hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o) if torch.is_tensor(o) and o.dim() == 4 else None))
    with torch.enable_grad():
        xs = xs.clone().requires_grad_(True)
        out = main_out(model(xs)).float()
        last = [n for n, _, _ in LAYERS if n in acts]
        for h in hooks:
            h.remove()
        if not last:
            return None
        a = acts[last[-1]]
        a.retain_grad()
        out.gather(1, out.argmax(1, keepdim=True)).sum().backward()
        if a.grad is None:
            return None
        cam = torch.relu((a.grad.mean((2, 3), keepdim=True) * a).sum(1)).detach()
    cam = nn.functional.interpolate(cam.unsqueeze(1), size=xs.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    flat = cam.flatten(1)
    cam = (cam - flat.min(1)[0].view(-1, 1, 1)) / (flat.max(1)[0] - flat.min(1)[0]).clamp_min(1e-8).view(-1, 1, 1)
    label = next(lb for n, lb, _ in LAYERS if n == last[-1])
    model.zero_grad(set_to_none=True)
    return label, cam.cpu()


def pictures(model, test, device):
    model.eval()
    res = {"filters": None, "feature_maps": [], "samples": []}
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        w = convs[0].weight.detach().cpu()[:32]
        res["filters"] = {"kernel": list(w.shape[2:]), "images": [small(norm01(f[:3] if f.shape[0] == 3 else f[:1])) for f in w]}
        if MULTI:
            res["bands"] = BANDS
    maps = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: maps.__setitem__(name, o.detach().cpu()))
             for name, _, conv in LAYERS if conv]
    mean, std = stats()
    idx = list(range(min(8 if SEG else 16, len(test))))
    xs = torch.stack([test[i][0] for i in idx]).to(device)
    with torch.no_grad():
        out = main_out(model(xs)).float()
    for h in hooks:
        h.remove()
    for name, label, conv in LAYERS:
        if conv and name in maps and maps[name].dim() == 4:
            res["feature_maps"].append({"layer": label, "channels": int(maps[name].shape[1]),
                                        "maps": [small(norm01(c.unsqueeze(0)), 32)[0] for c in maps[name][0, :8]]})
    m_t, s_t = torch.tensor(mean).view(-1, 1, 1), torch.tensor(std).view(-1, 1, 1)
    # what a sample looks like: the picture itself, or a colour composite of a multiband image
    look = composite if MULTI else (lambda x: (x * s_t + m_t).clamp(0, 1))
    k = max(1, len(CLASSES) - 1)
    for j, i in enumerate(idx):
        img = look(test[i][0])
        if SEG:
            pred = out[j].argmax(0).cpu()
            truth = main_y(test[i][1]).clamp_min(0)
            res["samples"].append({"image": small(img, 64), "truth": small((truth.float() / k).unsqueeze(0), 64),
                                   "pred": small((pred.float() / k).unsqueeze(0), 64),
                                   "pixel_acc": round(float((pred == truth).float().mean()), 3)})
        elif REGRESS:
            pred = to_units(out[j]).cpu()
            res["samples"].append({"image": small(img), "target": [round(float(v), 5) for v in main_y(test[i][1])],
                                   "value": [round(float(v), 5) for v in pred]})
        elif MULTILABEL:
            prob = torch.sigmoid(out[j]).cpu()
            res["samples"].append({"image": small(img), "labels": [k for k in range(len(CLASSES)) if main_y(test[i][1])[k] > 0.5],
                                   "preds": [k for k in range(len(CLASSES)) if prob[k] >= (0.5 if THRESH is None else float(THRESH[k]))],
                                   "probs": [round(float(v), 3) for v in prob]})
        else:
            prob = torch.softmax(out[j], 0).cpu()
            p = int(prob.argmax())
            res["samples"].append({"image": small(img), "label": int(main_y(test[i][1])), "pred": p, "prob": round(float(prob[p]), 3)})
    if not SEG and not MULTILABEL and not REGRESS:
        try:
            cam = grad_cam(model, xs)
            if cam:
                res["cam_layer"] = cam[0]
                for j, c in enumerate(cam[1]):
                    res["samples"][j]["cam"] = small(c.unsqueeze(0), 32)[0]
        except Exception as e:  # pictures are a bonus: never lose a run over them
            emit("warning", message=f"no Grad-CAM pictures: {type(e).__name__}: {e}")
        # misclassified test images (looking at up to 2000)
        mistakes = []
        with torch.no_grad():
            for start in range(0, min(len(test), 2000), 128):
                batch = [test[i] for i in range(start, min(start + 128, len(test), 2000))]
                o = main_out(model(torch.stack([b[0] for b in batch]).to(device))).float()
                pr = torch.softmax(o, 1).cpu()
                for b, (x, y) in enumerate(batch):
                    y = main_y(y)
                    p = int(pr[b].argmax())
                    if p != int(y):
                        mistakes.append({"image": small(look(x)), "label": int(y), "pred": p, "prob": round(float(pr[b][p]), 3),
                                         "index": start + b})
                if len(mistakes) >= 16:
                    break
        res["mistakes"] = mistakes[:16]
    return res


def export_model(model, example):
    """TorchScript (model.ts) and ONNX (model.onnx) copies of the trained network, for use without this code; a failure is a warning"""
    import copy
    m = copy.deepcopy(model).cpu().eval()
    x = example[:1].cpu()
    try:
        torch.jit.trace(m, x).save(os.path.join(OUT, "model.ts"))
    except Exception as e:
        emit("warning", message=f"TorchScript export failed: {type(e).__name__}: {e}")
    try:
        import onnx  # noqa: F401  (torch.onnx.export needs it; Kaggle and a fresh venv may not have it)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "onnx"], check=False)
    try:
        args = dict(input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        try:
            torch.onnx.export(m, (x,), os.path.join(OUT, "model.onnx"), dynamo=False, **args)
        except TypeError:  # PyTorch without the dynamo option
            torch.onnx.export(m, (x,), os.path.join(OUT, "model.onnx"), **args)
    except Exception as e:
        emit("warning", message=f"ONNX export failed: {type(e).__name__}: {e}")
    emit("exported", files=[f for f in ("model.ts", "model.onnx") if os.path.exists(os.path.join(OUT, f))])


# ---------------------------------------------------------------- training
# ---------------------------------------------------------------- check data (no training)
def labels_of(ds):
    """the labels of a dataset without reading its images, where the dataset keeps them (else None)"""
    from torch.utils.data import Subset
    if isinstance(ds, Subset):
        inner = labels_of(ds.dataset)
        return None if inner is None else [inner[i] for i in ds.indices]
    if isinstance(ds, BandFolder):
        return [l for _, l in ds.files]
    if isinstance(ds, LabelFiles):
        return [main_y(y) for _, y in ds.items]
    if isinstance(ds, (HFImages, HFBands)) and REGRESS:
        cols = [list(ds.ds[c]) for c in CLASSES]
        return [[float(v) for v in row] for row in zip(*cols)]
    if isinstance(ds, (HFImages, HFBands)) and not SEG:
        return list(ds.ds[DATASET["hf_label_col"]])
    for attr in ("targets", "labels"):
        if isinstance(getattr(ds, attr, None), (list, tuple)) or torch.is_tensor(getattr(ds, attr, None)):
            return [int(v) for v in getattr(ds, attr)]
    return None


def class_counts(ds, sample=64):
    """images per class (multi-label: images having each class; segmentation: pixels per class in up to `sample` items)"""
    k = len(CLASSES)
    if SEG:
        counts = torch.zeros(k, dtype=torch.long)
        for i in random.Random(0).sample(range(len(ds)), min(sample, len(ds))):
            y = ds[i][1]
            counts += torch.bincount(y[y >= 0].flatten(), minlength=k)[:k]
        return [int(c) for c in counts], True
    labels = labels_of(ds)
    exact = labels is not None
    if labels is None:
        labels = [main_y(ds[i][1]) for i in random.Random(0).sample(range(len(ds)), min(500, len(ds)))]
    counts = [0] * k
    for y in labels:
        if MULTILABEL and isinstance(y, (list, tuple)):   # a Hugging Face list of classes
            for c in y:
                counts[CLASSES.index(c) if isinstance(c, str) else int(c)] += 1
        elif MULTILABEL:
            for c in range(k):
                counts[c] += int(float(y[c]) > 0.5)
        elif isinstance(y, str):
            counts[CLASSES.index(y)] += 1
        elif 0 <= int(y) < k:
            counts[int(y)] += 1
    return counts, exact


def band_report(ds, sample=48):
    """per band of the raw files (before normalising): range, mean, spread and empty pixels, from up to `sample` training images"""
    import numpy as np
    files = [p for p, _ in ds.files] if isinstance(ds, BandFolder) else [p for p, _ in ds.items] if isinstance(ds, LabelFiles) else \
        [p for p, _ in ds.pairs] if isinstance(ds, BandPairs) else None
    raws = []
    if files is not None:
        for p in random.Random(1).sample(files, min(sample, len(files))):
            raws.append(read_bands(p).numpy())
    elif isinstance(ds, (SceneTiles, HFBands)):
        for i in random.Random(1).sample(range(len(ds)), min(sample, len(ds))):
            raws.append(ds.raw(i).numpy())
    if not raws:
        return []
    if SIZE2:   # the second group's bands: only their own block (the rest of the tensor is padding)
        rows = [np.concatenate([r[b].reshape(-1) if b < C1 else r[b, :SIZE2[0], :SIZE2[1]].reshape(-1) for r in raws]) for b in range(len(BANDS))]
    else:
        a = np.concatenate([r.reshape(r.shape[0], -1) for r in raws], 1)
        rows = [a[b] for b in range(len(BANDS))]
    out = []
    for b, name in enumerate(BANDS):
        v = rows[b]
        ok = np.isfinite(v)
        f = v[ok]
        out.append({"band": name, "min": float(f.min()) if f.size else None, "max": float(f.max()) if f.size else None,
                    "mean": float(f.mean()) if f.size else None, "std": float(f.std()) if f.size else None,
                    "empty": round(float(1 - ok.mean()), 4), "zero": round(float((f == 0).mean()) if f.size else 0.0, 4)})
    return out


def data_report(train, val, test):
    """what the run would train on, and what looks wrong"""
    from torch.utils.data import Subset
    base = train.dataset if isinstance(train, Subset) else train
    if REGRESS:
        counts, exact = [], True
    else:
        counts, exact = class_counts(train)
    rep = {"sizes": {"train": len(train), "validation": len(val) if val is not None else 0, "test": len(test)},
           "counts": counts, "counts_exact": exact, "count_unit": "pixels (sampled)" if SEG else "images", "bands": [], "warnings": []}
    w = rep["warnings"]
    if MULTI:
        rep["bands"] = band_report(base)
        groups = [(g, n) for g in GROUPS for n in g["bands"]]
        for (g, _), b in zip(groups, rep["bands"]):
            if b["min"] is None or b["empty"] >= 0.999:
                w.append(f"band {b['band']} is empty in every image checked")
            elif b["std"] is not None and b["std"] < 1e-6:
                w.append(f"band {b['band']} has the same value everywhere ({b['min']:g}): the band order in the Dataset tab may be wrong")
            elif b["empty"] > 0.2:
                w.append(f"band {b['band']}: {b['empty']:.0%} of the pixels are empty (NaN)")
            if g["sensor"] == "s1" and b["min"] is not None:
                if b["min"] >= 0 and not g["db"]:
                    w.append(f"{b['band']} looks like linear backscatter (no negative values): tick 'convert to decibels'")
            if g["sensor"] == "s2" and b["max"] is not None and b["max"] > 20000:
                w.append(f"{b['band']} reaches {b['max']:.0f}: Sentinel-2 reflectance is usually 0-10000; check the band order or scaling")
    if REGRESS:
        labels = labels_of(train)
        if labels is None:
            labels = [main_y(train[i][1]) for i in random.Random(0).sample(range(len(train)), min(500, len(train)))]
        ys = torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in labels]).double()
        rep["targets"] = [{"target": c, "min": float(ys[:, k].min()), "max": float(ys[:, k].max()), "mean": float(ys[:, k].mean()),
                           "std": float(ys[:, k].std(unbiased=False))} for k, c in enumerate(CLASSES)]
        for t in rep["targets"]:
            if t["std"] < 1e-12:
                w.append(f"the target {t['target']} has the same value ({t['min']:g}) for every image: nothing to learn")
        if len(train) < 50:
            w.append(f"only {len(train)} training images for a regression")
        return rep
    present = [c for c in counts if c > 0]
    missing = [CLASSES[i] for i, c in enumerate(counts) if c == 0]
    if missing:
        w.append(f"no {'pixels' if SEG else 'images'} of {', '.join(missing[:6])}{' …' if len(missing) > 6 else ''} in the training data checked")
    if present and max(present) > 3 * min(present):
        w.append(f"unbalanced: the largest class has {max(present) / min(present):.1f}× the {rep['count_unit']} of the smallest")
    if len(train) < 10 * len(CLASSES):
        w.append(f"only {len(train)} training images for {len(CLASSES)} classes")
    return rep


def check_only(model, train, val, test, first_batch, device):
    """Check data: load the data, report it, run the one-batch shape check, show a few images; no training"""
    emit("info", message="checking the data only (no training)")
    rep = data_report(train, val, test)
    emit("data_report", **rep)
    one_batch_check(model, first_batch, device)
    res = {"task": TASK, "check_only": True, "classes": CLASSES, "data_report": rep, "samples": []}
    if MULTI or not SEG:
        stats_m, stats_s = stats() or (None, None)
        for i in random.Random(2).sample(range(len(test)), min(12, len(test))):
            x, y = test[i]
            y = main_y(y)
            img = composite(x) if MULTI else (x * torch.tensor(stats_s).view(-1, 1, 1) + torch.tensor(stats_m).view(-1, 1, 1)).clamp(0, 1)
            label = ([round(float(v), 5) for v in y] if REGRESS else [c for c in range(len(CLASSES)) if float(y[c]) > 0.5] if MULTILABEL
                     else None if SEG else int(y))
            res["samples"].append({"image": small(img, 64), "label": label})
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    emit("done", check_only=True, warnings=len(rep["warnings"]))


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
    if cfg.get("pretrained"):
        loaded = load_pretrained(model, cfg["pretrained"])
        if cfg.get("freeze") == "backbone":
            # train only what was not loaded (the new output head)
            names = set(loaded)
            for k, prm in model.named_parameters():
                prm.requires_grad = k not in names
            emit("info", message=f"training only {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters (the backbone is frozen)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train, val, test = load_data(cfg)
    # data loading: one worker per CPU core (at most 8) on a GPU machine, each preparing batches ahead
    nw = cfg.get("num_workers", "auto")
    workers = (min(8, os.cpu_count() or 2) if nw == "auto" else int(nw)) if device.type == "cuda" else 0
    ahead = {"prefetch_factor": 4} if workers else {}
    bs = cfg["batch_size"]

    def loader(epoch, skip_batches=0):
        return DataLoader(train, batch_size=bs, sampler=EpochOrder(len(train), cfg["seed"], epoch, skip_batches * bs), num_workers=workers,
                          pin_memory=device.type == "cuda", **ahead)

    # evaluation batches: halved when they do not fit (evaluate_safe)
    eval_bs = {"v": 64 if SEG else 256}
    # per-epoch validation: the held-out part of the training data, or (without one) the test data as before
    val_data = val if val is not None else test

    def evaluate_safe(data):
        while True:
            try:
                return evaluate(model, DataLoader(data, batch_size=eval_bs["v"], shuffle=False, num_workers=workers, **ahead), device)
            except Exception as e:
                if not is_oom(e) or eval_bs["v"] <= 1:
                    raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            eval_bs["v"] //= 2
            emit("memory_split", what="evaluation", batch=eval_bs["v"])
    # early stopping: the monitored value, the best epoch so far and how many epochs since then (kept in the checkpoint)
    monitor = cfg.get("monitor", "val_loss")
    if REGRESS and monitor == "val_miou":
        monitor = "val_loss"   # val_acc is R² for regression; there is no mIoU
    early, patience = bool(cfg.get("early_stop")), max(1, int(cfg.get("patience", 5)))
    es = {"best": None, "best_epoch": 0, "bad": 0}
    best_path = os.path.join(OUT, "best_model.pt")
    # the training recipe: optimizer with weight decay, learning-rate schedule, label smoothing, gradient clipping
    wd = float(cfg.get("weight_decay") or 0)
    base_lr = cfg["lr"]
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=wd)
    elif cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(params, lr=base_lr, weight_decay=wd)
    smoothing = float(cfg.get("label_smoothing") or 0)
    clip = float(cfg.get("grad_clip") or 0)
    schedule = cfg.get("schedule", "constant")
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    # bfloat16 on GPUs that have it (A100, A6000, RTX 30xx and newer: compute capability 8+): the range of float32, so no loss scaling
    bf16 = amp and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    precision = ("bf16" if bf16 else "fp16") if amp else "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not bf16)
    # channels-last memory layout: faster convolutions with mixed precision on NVIDIA GPUs
    # (not with Conv3D: its 5-D weights have no channels-last layout)
    channels_last = device.type == "cuda" and any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in model.modules()) \
        and not any(isinstance(m, nn.Conv3d) for m in model.modules())
    if channels_last:
        model.to(memory_format=torch.channels_last)
    # a batch that does not fit is processed in `parts` pieces whose gradients add up to the batch's (kept in the checkpoint)
    auto_split = bool(cfg.get("auto_split", True))
    split = {"parts": 1, "reported": False}
    ck_path = os.path.join(OUT, "checkpoint.pt")
    steps = (len(train) + bs - 1) // bs
    total_steps, warm_steps = cfg["epochs"] * steps, int(float(cfg.get("warmup_epochs") or 0) * steps)
    # reduce-on-plateau state (halve the learning rate after `plateau_patience` epochs without a lower validation loss)
    plat = {"mult": 1.0, "best": None, "bad": 0}
    plateau_patience = max(2, int(cfg.get("plateau_patience", 3)))

    def lr_at(i):
        """the learning rate for optimizer step i (counted from 0 over the whole run): linear warm-up, then the schedule"""
        if warm_steps and i < warm_steps:
            return base_lr * (i + 1) / warm_steps
        if schedule == "cosine":
            prog = min(1.0, (i - warm_steps) / max(1, total_steps - warm_steps))
            return base_lr * 0.5 * (1 + math.cos(math.pi * prog))
        return base_lr * plat["mult"]
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
        plat = ck.get("plat") or plat
        split["parts"] = int(ck.get("parts") or 1)
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
                "steps": steps, "sums": pos["sums"], "rng": rng, "resumes": resumes, "es": es, "plat": plat, "parts": split["parts"], "version": 5}

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
         early_stop=early, patience=patience if early else None, monitor=monitor if early else None,
         recipe={"optimizer": cfg["optimizer"], "lr": base_lr, "weight_decay": wd, "schedule": schedule, "warmup_epochs": cfg.get("warmup_epochs", 0),
                 "label_smoothing": smoothing, "grad_clip": clip, "loss": LOSS or "cross_entropy", "focal_gamma": GAMMA if LOSS == "focal" else None},
         amp=amp, augmentation={k: v for k, v in AUG.items() if v},
         precision=precision, channels_last=channels_last, workers=workers, save_memory=bool(getattr(model, "save_memory", False)),
         auto_split=auto_split)
    if cfg.get("check_only"):
        check_only(model, train, val, test, loader(1), device)
        return
    if REGRESS:
        target_stats(train)
    elif cfg.get("class_weights") in ("inverse", "sqrt"):
        class_weights(train, cfg["class_weights"])
    if MULTITASK:
        head_stats(train)
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
            waited, mark, began = 0.0, time.time(), time.time()  # time spent waiting for the next batch (data loading too slow?)
            for step, (x, y) in enumerate(loader(epoch, pos["step"]), pos["step"] + 1):
                waited += time.time() - mark
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last) if channels_last and x.dim() == 4 else x.to(device, non_blocking=True)
                yall = y_to(y, device, non_blocking=True)
                y = main_y(yall)   # several outputs: the main task's labels (sizes, accuracy); the loss gets them all
                lr_now = lr_at((epoch - 1) * steps + step - 1)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                while True:
                    # the batch in `parts` pieces (1 = all at once); each piece's loss is weighted by its share, so the gradients add up to
                    # exactly the whole batch's (BatchNorm sees the smaller pieces, the only difference)
                    try:
                        opt.zero_grad(set_to_none=True)
                        batch_loss, batch_correct = 0.0, 0
                        for xs, ys_all in zip(x.tensor_split(split["parts"]), y_split(yall, split["parts"])):
                            ys = main_y(ys_all)
                            if TEST_OOM_ABOVE and len(ys) > TEST_OOM_ABOVE:
                                raise OOM("CUDA out of memory (simulated for the test)")
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                                outs = model(xs)
                                out = main_out(outs)
                                loss = loss_of(outs, ys_all, smoothing) * (len(ys) / len(y))
                                if SEG and not bool((ys >= 0).any()):
                                    loss = out.float().sum() * 0.0  # no labelled pixel in this piece: nothing to learn (and no NaN)
                            scaler.scale(loss).backward()
                            batch_loss += loss.item()
                            if not REGRESS:
                                batch_correct += int(((out > 0) == (ys > 0.5)).sum()) if MULTILABEL else int((out.argmax(1) == ys).sum())
                        break
                    except Exception as e:
                        if not (auto_split and is_oom(e)) or split["parts"] >= len(y):
                            if is_oom(e):
                                raise SystemExit("MEMORY: even one image at a time does not fit on this GPU. Use a smaller design or input, "
                                                 "switch on \"save memory\" in the Train tab, or a GPU with more memory.")
                            raise
                    # out of memory (handled outside `except`, so that the failed attempt's tensors are freed): twice as many pieces
                    out = outs = loss = None
                    opt.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    split["parts"] = min(len(y), split["parts"] * 2)
                    emit("memory_split", what="training", batch=len(y), parts=split["parts"], part_size=-(-len(y) // split["parts"]),
                         epoch=epoch, step=step)
                if clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, clip)
                scaler.step(opt)
                scaler.update()
                sums["loss_sum"] += batch_loss * len(y)
                sums["n"] += len(y)
                sums["correct"] += batch_correct
                sums["units"] += int((y >= 0).sum()) if SEG else y.numel()
                pos.update(step=step, sums=sums)
                dirty["v"] = True
                if step % every == 0 or step == steps:
                    emit("batch", epoch=epoch, step=step, steps=steps, loss=round(batch_loss, 5))
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
                mark = time.time()
            data_wait = waited / max(time.time() - began, 1e-9)
            if device.type == "cuda" and not split["reported"]:
                # the real peak of this run (the Train tab's figure is an estimate), once, after the first epoch it trained
                split["reported"] = True
                emit("memory", peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
                     total_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), batch=bs, parts=split["parts"],
                     precision=precision, save_memory=bool(getattr(model, "save_memory", False)))
            val_loss, val_acc, _, miou, _, val_top5 = evaluate_safe(val_data)
            ev = dict(epoch=epoch, train_loss=round(sums["loss_sum"] / max(sums["n"], 1), 5), train_acc=round(sums["correct"] / max(sums["units"], 1), 5),
                      val_loss=round(val_loss, 5), val_acc=round(val_acc, 5), seconds=round(sums["seconds"] + time.time() - t0, 2), partial=stopped,
                      lr=float(f"{opt.param_groups[0]['lr']:.3g}"), data_wait=round(data_wait, 3))
            if SEG:
                ev["val_miou"] = round(miou, 5)
            if MULTILABEL:
                ev["val_map"] = round(miou, 5)
            if REGRESS:
                ev["val_mae"] = round(miou, 6)
                del ev["train_acc"]   # no running R² while training
            if val_top5 is not None:
                ev["val_top5"] = round(val_top5, 5)
            if HEAD_SCORES:
                ev["heads"] = [{"name": h["name"], "metric": h["metric"], "score": round(h["score"], 5)} for h in HEAD_SCORES]
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
            if schedule == "plateau" and not stopped:
                if plat["best"] is None or val_loss < plat["best"] - 1e-4:
                    plat.update(best=val_loss, bad=0)
                else:
                    plat["bad"] += 1
                    if plat["bad"] >= plateau_patience and plat["mult"] > 1e-3:
                        plat.update(mult=plat["mult"] * 0.5, bad=0)
                        emit("lr_drop", epoch=epoch, lr=float(f"{base_lr * plat['mult']:.3g}"))
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
        if MULTILABEL and cfg.get("tune_thresholds"):
            if val is None:
                emit("warning", message="per-class thresholds need validation data (Train tab: validation share); 0.5 is used")
            else:
                tune_thresholds(model, val, device)
        val_loss, val_acc, confusion, miou, iou, test_top5 = evaluate_safe(test)
    except Interrupted:
        interrupted()
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    res = {"task": TASK, "test_acc": val_acc, "test_loss": val_loss, "classes": CLASSES,
           "confusion": None if MULTILABEL or REGRESS else confusion}
    if REGRESS:
        res.update(confusion)   # r2 (= test_acc), mae, rmse, per_target, scatter
        res["target_stats"] = {"mean": [float(v) for v in TARGET_MEAN], "std": [float(v) for v in TARGET_STD]}
        with open(os.path.join(OUT, "target_stats.json"), "w", encoding="utf-8") as f:
            # model.pt outputs standardised numbers: value = output × std + mean
            json.dump({"targets": CLASSES, **res["target_stats"]}, f)
    if MULTILABEL:
        res.update(confusion)   # micro_f1 (= test_acc), macro_f1, map, exact, per_class
        if THRESH is not None:
            res["thresholds"] = [float(t) for t in THRESH]
    if HEAD_SCORES:
        res["heads"] = HEAD_SCORES
    if LOSS:
        res["loss"] = {"kind": LOSS, "gamma": GAMMA} if LOSS == "focal" else {"kind": LOSS}
    if LOSS_WEIGHT is not None:
        res["class_weights"] = [round(float(v), 4) for v in LOSS_WEIGHT]
    if test_top5 is not None:
        res["test_top5"] = test_top5
    if SEG:
        res.update(miou=miou, iou=iou)
    res.update(pictures(model, test, device))
    if cfg.get("export"):
        export_model(model, torch.stack([test[0][0]]))
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    emit("stopped" if stopped else "done", test_acc=round(val_acc, 5), **({"miou": round(miou, 5)} if SEG else {}),
         **({"test_map": round(confusion["map"], 5), "test_macro_f1": round(confusion["macro_f1"], 5), "test_exact": round(confusion["exact"], 5)}
            if MULTILABEL else {}),
         **({"test_mae": round(confusion["mae"], 6), "test_rmse": round(confusion["rmse"], 6)} if REGRESS else {}),
         **({"heads": [{"name": h["name"], "metric": h["metric"], "score": round(h["score"], 5)} for h in HEAD_SCORES]} if HEAD_SCORES else {}),
         **({"best_epoch": es["best_epoch"]} if early else {}), **({"test_top5": round(test_top5, 5)} if test_top5 is not None else {}))
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

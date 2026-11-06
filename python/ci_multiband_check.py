"""CI only (GitHub Actions; never on the owner's Mac): multispectral and radar images with real PyTorch on the CPU.

Generates class folders like EuroSAT-MS (13 Sentinel-2 bands, half the files GeoTIFF (bands last, as most tools write them) and half
.npy (bands first)) and like a Sentinel-1 set (VV, VH in linear backscatter), trains the generated train.py on each, and checks: the
chosen bands in the chosen order reached the network, per-band statistics were measured (and in dB for Sentinel-1), the network learnt the
two classes, and the result pictures are 3-channel composites.

Then the owner's own files as Google Earth Engine exports them: Sentinel-2 named B1 … B12 with empty (NaN) pixels in class folders
(found in <data>/local/<key>, where the app copies them), Sentinel-1 already in dB (not converted twice), a scene split into two files with
a label image (no-data 255, an empty tile) cut into tiles for segmentation, and multiband images + masks.

usage: python ci_multiband_check.py <folder from the multiband_case example>
"""
import json
import os
import subprocess
import sys

import numpy as np

S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


def make_s2(root):
    import tifffile
    rng = np.random.default_rng(0)
    for k, cls in enumerate(["forest", "water"]):
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for i in range(120):
            a = rng.normal(1000, 200, size=(13, 40, 40)).astype(np.float32)
            # the classes differ in B08 (near infrared) and B11 only: a network that sees the chosen bands can learn them
            a[S2.index("B08")] += 1500 if cls == "forest" else -400
            a[S2.index("B11")] += 300 if cls == "water" else 0
            if i % 2:
                tifffile.imwrite(os.path.join(root, cls, f"{i}.tif"), a.transpose(1, 2, 0))  # bands last
            else:
                np.save(os.path.join(root, cls, f"{i}.npy"), a)


def make_s1(root):
    rng = np.random.default_rng(1)
    for cls in ["flood", "land"]:
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for i in range(120):
            # water is dark in radar: about -20 dB (0.01) against -8 dB (0.16) on land
            level = 0.01 if cls == "flood" else 0.16
            a = (rng.gamma(4.0, level / 4.0, size=(2, 32, 32))).astype(np.float32)
            np.save(os.path.join(root, cls, f"{i}.npy"), a)


GEE = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


def make_gee(root):
    """class folders of 12-band GeoTIFFs (bands last) with Earth Engine's names; every image has an empty 4×4 corner (NaN in all bands)"""
    import tifffile
    rng = np.random.default_rng(2)
    for cls in ["forest", "water"]:
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for i in range(100):
            a = rng.normal(1000, 200, size=(12, 40, 40)).astype(np.float32)
            a[GEE.index("B8")] += 1500 if cls == "forest" else -400
            a[GEE.index("B11")] += 300 if cls == "water" else 0
            a[:, :4, :4] = np.nan
            tifffile.imwrite(os.path.join(root, cls, f"{cls}_{i}.tif"), a.transpose(1, 2, 0))


def make_s1_db(root):
    """Sentinel-1 as Earth Engine's S1_GRD gives it: VV, VH already in decibels, plus the incidence angle"""
    rng = np.random.default_rng(3)
    for cls in ["flood", "land"]:
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for i in range(100):
            level = -20.0 if cls == "flood" else -8.0
            a = np.stack([rng.normal(level, 1.5, (32, 32)), rng.normal(level - 6, 1.5, (32, 32)), rng.normal(35, 3, (32, 32))]).astype(np.float32)
            np.save(os.path.join(root, cls, f"{i}.npy"), a)


def make_scene(root):
    """a scene exported in two files (scene-0000000000-0000000000.tif, scene-0000000000-0000000128.tif) and a label image with the same
    names: classes land 0 / water 1 / forest 2 in 8×8 patches, the first two rows without a class (255), one 16×16 tile without data"""
    import tifffile
    rng = np.random.default_rng(4)
    for d in ("nnb_image", "nnb_label"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for name in ("scene-0000000000-0000000000", "scene-0000000000-0000000128"):
        lab = np.kron(rng.integers(0, 3, (16, 16)), np.ones((8, 8), dtype=np.int64)).astype(np.uint8)
        img = rng.normal(1000, 150, size=(4, 128, 128)).astype(np.float32)
        img[3] += np.choose(lab, [0.0, -700.0, 1800.0])   # B8 (near infrared): water dark, forest bright
        img[2] += np.choose(lab, [400.0, 0.0, -300.0])    # B4 (red)
        lab[:2] = 255
        img[:, 16:32, 16:32] = np.nan
        tifffile.imwrite(os.path.join(root, "nnb_image", name + ".tif"), img.transpose(1, 2, 0))
        tifffile.imwrite(os.path.join(root, "nnb_label", name + ".tif"), lab)


def make_segfiles(root):
    """multiband images (.npy, 5 bands first) and GeoTIFF masks with the same names: a field where b5 is high; a few pixels 255"""
    import tifffile
    rng = np.random.default_rng(5)
    for d in ("images", "masks"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for i in range(120):
        m = np.zeros((16, 16), dtype=np.uint8)
        y, x = rng.integers(0, 10, 2)
        m[y:y + 6, x:x + 6] = 1
        a = rng.normal(0, 1, size=(5, 16, 16)).astype(np.float32)
        a[4] += 3 * m
        m[0, :3] = 255
        np.save(os.path.join(root, "images", f"p{i}.npy"), a)
        tifffile.imwrite(os.path.join(root, "masks", f"p{i}_mask.tif"), m)


def make_fusion(root):
    """Sentinel-2 (4 bands, 40×40 GeoTIFF) in s2/<class>/ and Sentinel-1 (VV, VH linear, 20×20 .npy) with the same names in s1/<class>/:
    the classes differ only in the radar, so the network must use the second band group"""
    import tifffile
    rng = np.random.default_rng(6)
    for cls in ["flood", "land"]:
        for d in ("s2", "s1"):
            os.makedirs(os.path.join(root, d, cls), exist_ok=True)
        for i in range(100):
            tifffile.imwrite(os.path.join(root, "s2", cls, f"t{i}.tif"), rng.normal(1000, 200, size=(40, 40, 4)).astype(np.float32))
            level = 0.01 if cls == "flood" else 0.16
            np.save(os.path.join(root, "s1", cls, f"t{i}.npy"), rng.gamma(4.0, level / 4.0, size=(2, 20, 20)).astype(np.float32))


def make_multilabel(root):
    """BigEarthNet-like: S2 patches (.npy) and S1 patches with the same names, and labels.csv with any of forest / water / urban per
    patch (forest raises B8, water lowers VV, urban raises B4)"""
    rng = np.random.default_rng(7)
    for d in ("s2", "s1"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    rows = ["patch_id,labels"]
    for i in range(400):
        has = [c for c in ("forest", "water", "urban") if rng.random() < 0.4]
        a = rng.normal(1000, 150, size=(4, 24, 24)).astype(np.float32)
        a[3] += 900 if "forest" in has else 0
        a[2] += 700 if "urban" in has else 0
        r = rng.normal(-8, 1.0, size=(2, 24, 24)).astype(np.float32)   # already in dB
        r[0] -= 10 if "water" in has else 0
        np.save(os.path.join(root, "s2", f"p{i}.npy"), a)
        np.save(os.path.join(root, "s1", f"p{i}.npy"), r)
        rows.append(f"p{i},\"{'; '.join(has)}\"")
    rows.append("missing_patch,forest")
    with open(os.path.join(root, "labels.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def run(folder, data, slug):
    env = {**os.environ, "NNB_DATA": data, "PYTHONUNBUFFERED": "1"}
    with open(os.path.join(folder, "log.txt"), "w") as log:
        code = subprocess.run([sys.executable, "train.py"], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
    log = open(os.path.join(folder, "log.txt")).read()
    assert code == 0, f"{slug}: train.py failed ({code}):\n{log[-3000:]}"
    with open(os.path.join(folder, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(root):
    root = os.path.abspath(root)
    data = os.path.join(root, "data")
    make_s2(os.path.join(data, "kaggle", "ci__sentinel2"))
    make_s1(os.path.join(data, "kaggle", "ci__sentinel1"))
    for name, slug, bands in (("s2", "sentinel2", ["B02", "B03", "B04", "B08", "B11", "B12"]), ("s1", "sentinel1", ["VV", "VH"])):
        folder = os.path.join(root, name)
        ev = run(folder, data, slug)
        check = next(e for e in ev if e["event"] == "check")
        assert check["batch"][1:] == [len(bands), 32, 32], check
        st = json.load(open(os.path.join(folder, "band_stats.json")))
        assert st["bands"] == bands, st
        if name == "s2":
            # B08's mean (1000 + the class offsets +1500 / -400, averaged: 1550) shows the right band was picked by name
            assert 1450 < st["mean"][bands.index("B08")] < 1650, st
            assert 900 < st["mean"][bands.index("B02")] < 1100, st
        else:
            assert all(-25 < m < -5 for m in st["mean"]), f"Sentinel-1 not in dB: {st}"
        done = next(e for e in ev if e["event"] == "done")
        assert done["test_acc"] >= 0.9, f"{name}: test accuracy {done['test_acc']}"
        res = json.load(open(os.path.join(folder, "results.json")))
        assert res["bands"] == bands and len(res["samples"][0]["image"]) == 3, "composite pictures"
        print(f"{name}: bands {bands} -> input {check['batch'][1:]}, per-band mean {[round(m, 1) for m in st['mean']]}, "
              f"test accuracy {done['test_acc']}")


def own_files(root):
    root = os.path.abspath(root)
    data = os.path.join(root, "data")
    key = open(os.path.join(root, "gee", "local_key.txt")).read().strip()
    make_gee(os.path.join(data, "local", key))
    make_s1_db(os.path.join(root, "own", "s1"))
    make_scene(os.path.join(root, "own", "scene"))
    make_segfiles(os.path.join(root, "own", "segfiles"))

    ev = run(os.path.join(root, "gee"), data, "gee")
    st = json.load(open(os.path.join(root, "gee", "band_stats.json")))
    bands = ["B04", "B03", "B02", "B8", "B11"]
    assert st["bands"] == bands, st
    assert 1450 < st["mean"][3] < 1650 and all(np.isfinite(st["mean"] + st["std"])), f"B8 by Earth Engine name, NaN left out: {st}"
    assert all(np.isfinite(e["train_loss"]) for e in ev if e["event"] == "epoch"), "empty pixels must not give NaN losses"
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.9, done
    print(f"gee: names B1 … B12, NaN corners -> B8 mean {st['mean'][3]:.1f}, test accuracy {done['test_acc']}")

    ev = run(os.path.join(root, "s1gee"), data, "s1gee")
    st = json.load(open(os.path.join(root, "s1gee", "band_stats.json")))
    assert all(-25 < m < -5 for m in st["mean"]), f"already-dB values converted again: {st}"
    assert any(e["event"] == "warning" and "already in decibels" in e.get("message", "") for e in ev)
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.9, done
    print(f"s1gee: already in dB, mean {[round(m, 1) for m in st['mean']]}, test accuracy {done['test_acc']}")

    ev = run(os.path.join(root, "scene"), data, "scene")
    tiles = next(e for e in ev if e["event"] == "info" and "tiles of 16×16" in e.get("message", ""))
    assert tiles["message"].startswith("126 tiles"), tiles  # 2 × 64 tiles, one without data in each
    check = next(e for e in ev if e["event"] == "check")
    assert check["batch"][1:] == [2, 16, 16], check
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.75 and done["miou"] >= 0.5, done
    res = json.load(open(os.path.join(root, "scene", "results.json")))
    assert len(res["samples"][0]["image"]) == 3, "composite pictures"
    print(f"scene: {tiles['message']}, pixel accuracy {done['test_acc']}, mean IoU {done['miou']}")

    make_fusion(os.path.join(root, "own", "fusion"))
    ev = run(os.path.join(root, "fusion"), data, "fusion")
    st = json.load(open(os.path.join(root, "fusion", "band_stats.json")))
    assert st["bands"] == ["B2", "B3", "B4", "B8", "VV", "VH"], st
    assert all(-25 < m < -5 for m in st["mean"][4:]) and all(900 < m < 1100 for m in st["mean"][:4]), f"S1 group in dB: {st}"
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.9, f"fusion: the radar group must reach the network: {done}"
    print(f"fusion: S2 4 bands + S1 2 bands (20×20 resized), mid fusion, per-band mean {[round(m, 1) for m in st['mean']]}, "
          f"test accuracy {done['test_acc']}")

    make_multilabel(os.path.join(root, "own", "ml"))
    ev = run(os.path.join(root, "multilabel"), data, "multilabel")
    assert any(e["event"] == "warning" and "not in the dataset" in e.get("message", "") for e in ev), "the missing row is reported"
    check = next(e for e in ev if e["event"] == "check")
    assert check["batch"][1:] == [5, 24, 24], check
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.85 and done["test_map"] >= 0.85, f"multi-label: {done}"
    res = json.load(open(os.path.join(root, "multilabel", "results.json")))
    assert res["task"] == "multilabel" and len(res["per_class"]) == 3 and "labels" in res["samples"][0], res.keys()
    print(f"multilabel: micro F1 {done['test_acc']}, macro F1 {done['test_macro_f1']}, mAP {done['test_map']}, exact {done['test_exact']}")

    ev = run(os.path.join(root, "segfiles"), data, "segfiles")
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.8, done
    print(f"segfiles: images + masks, pixel accuracy {done['test_acc']}, mean IoU {done['miou']}")


if __name__ == "__main__":
    main(sys.argv[1])
    own_files(sys.argv[1])

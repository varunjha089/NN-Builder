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
    strip = next(e for e in ev if e["event"] == "info" and e.get("message", "").startswith("test strip"))
    # 8 columns of tiles per image: the eastern one (8 rows × 2 images) is test, the one next to it is left out, the rest trains
    assert strip["message"] == "test strip: the eastern 16 tiles; 16 tiles next to it left out; 94 for training", strip
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
    th = next(e for e in ev if e["event"] == "thresholds")["thresholds"]
    assert len(th) == 3 and all(0.05 <= t <= 0.95 for t in th) and res["thresholds"] == th, th
    cw = next(e for e in ev if e["event"] == "class_weights")
    assert cw["kind"] == "pos_weight" and len(cw["weights"]) == 3, cw
    print(f"multilabel: micro F1 {done['test_acc']}, macro F1 {done['test_macro_f1']}, mAP {done['test_map']}, exact {done['test_exact']}, "
          f"thresholds {th}, pos_weight {cw['weights']}")

    ev = run(os.path.join(root, "segfiles"), data, "segfiles")
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.8, done
    print(f"segfiles: images + masks, pixel accuracy {done['test_acc']}, mean IoU {done['miou']}")


def make_imbalanced(root):
    """train/: 200 'common' and 20 'rare' images (2 bands, 16×16), the rare class brighter in b2; test/: 30 of each, so the rare class's
    recall is measured on enough images"""
    rng = np.random.default_rng(8)
    for part, sizes in (("train", (("common", 200), ("rare", 20))), ("test", (("common", 30), ("rare", 30)))):
        for cls, n in sizes:
            os.makedirs(os.path.join(root, part, cls), exist_ok=True)
            for i in range(n):
                a = rng.normal(0, 1, size=(2, 16, 16)).astype(np.float32)
                a[1] += 2.5 if cls == "rare" else 0
                np.save(os.path.join(root, part, cls, f"{i}.npy"), a)


def imbalanced(root):
    """class weights from the training counts (inverse frequency: about 10× for the rare class), and the rare class still learnt"""
    root = os.path.abspath(root)
    make_imbalanced(os.path.join(root, "own", "imb"))
    ev = run(os.path.join(root, "imbalanced"), os.path.join(root, "data"), "imbalanced")
    cw = next(e for e in ev if e["event"] == "class_weights")
    c, w = cw["counts"], cw["weights"]
    assert c == [200, 20] and abs(w[1] / w[0] - 10) < 0.1, cw
    res = json.load(open(os.path.join(root, "imbalanced", "results.json")))
    conf = res["confusion"]
    rare_recall = conf[1][1] / max(1, sum(conf[1]))
    assert rare_recall >= 0.5, f"the rare class was not learnt: {conf}"
    print(f"imbalanced: counts {c}, weights {w}, rare-class recall {rare_recall:.2f}, test accuracy {res['test_acc']:.3f}")


def make_ben(root):
    """BigEarthNet-like: S2 patch folders <tile>/<patch>/<patch>_<band>.tif (10 m bands 24×24, B05 12×12), S1 folders named differently,
    metadata.parquet with label lists, split and s1_name"""
    import pandas as pd
    import tifffile
    rng = np.random.default_rng(9)
    classes = ["Beaches, dunes, sands", "Coniferous forest", "Inland waters"]
    rows = []
    for i in range(240):
        has = [c for c in classes if rng.random() < 0.4]
        patch, s1 = f"S2A_T{i % 3}_{i:04d}", f"S1B_X_{i:04d}"
        pdir, sdir = os.path.join(root, "BigEarthNet-S2", f"tile{i % 3}", patch), os.path.join(root, "BigEarthNet-S1", f"tile{i % 3}", s1)
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(sdir, exist_ok=True)
        for b in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]:
            size = 24 if b in ("B02", "B03", "B04", "B08") else 12 if b != "B01" and b != "B09" else 4
            a = rng.normal(1000, 120, size=(size, size)).astype(np.uint16 if b != "B05" else np.float32)
            if b == "B08" and "Coniferous forest" in has:
                a = a + 1500
            if b == "B05" and classes[0] in has:
                a = a + 800
            tifffile.imwrite(os.path.join(pdir, f"{patch}_{b}.tif"), a.astype(np.float32))
        for b in ("VV", "VH"):
            r = rng.normal(-9, 1, size=(24, 24)).astype(np.float32)
            if "Inland waters" in has:
                r -= 10
            tifffile.imwrite(os.path.join(sdir, f"{s1}_{b}.tif"), r)
        rows.append({"patch_id": patch, "labels": has, "split": ["train", "train", "train", "validation", "test"][i % 5], "s1_name": s1})
    pd.DataFrame(rows).to_parquet(os.path.join(root, "metadata.parquet"))


def ben(root):
    root = os.path.abspath(root)
    make_ben(os.path.join(root, "own", "ben"))
    ev = run(os.path.join(root, "ben"), os.path.join(root, "data"), "ben")
    sp = next(e for e in ev if e["event"] == "info" and e.get("message", "").startswith("split from the labels table"))
    assert sp["message"] == "split from the labels table: 144 train, 48 validation, 48 test", sp
    check = next(e for e in ev if e["event"] == "check")
    assert check["batch"][1:] == [7, 24, 24], check
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_map"] >= 0.85, f"BigEarthNet-like: {done}"
    print(f"ben: patch folders + parquet + s1_name fusion, {sp['message']}, mAP {done['test_map']}, micro F1 {done['test_acc']}")


def pretrained_ms(root):
    root = os.path.abspath(root)
    ev = run(os.path.join(root, "pretrained_ms"), os.path.join(root, "data"), "pretrained_ms")
    widened = next(e for e in ev if e["event"] == "info" and "widened" in e.get("message", ""))
    assert "from 3 to 4 channels (3 from red / green / blue" in widened["message"], widened
    pre = next(e for e in ev if e["event"] == "pretrained")
    assert pre["loaded"] >= 0.8 * pre["total"] - 10, pre
    print(f"pretrained_ms: {widened['message']}; {pre['loaded']} of {pre['total']} tensors loaded")


def hfbands(root):
    """a Hugging Face dataset (a local parquet copy) whose 'bands' column holds 4-band arrays; 'bright' is brighter in b4"""
    import pandas as pd
    root = os.path.abspath(root)
    rng = np.random.default_rng(10)
    d = os.path.join(root, "hfbands", "ci", "hfbands")
    os.makedirs(d, exist_ok=True)
    for split, n in (("train", 160), ("test", 40)):
        rows = []
        for i in range(n):
            y = i % 2
            a = rng.normal(0, 1, size=(4, 16, 16)).astype(np.float32)
            a[3] += 3 * y
            rows.append({"bands": a.tolist(), "label": y})
        pd.DataFrame(rows).to_parquet(os.path.join(d, f"{split}.parquet"))
    ev = run(os.path.join(root, "hfbands"), os.path.join(root, "data"), "hfbands")
    check = next(e for e in ev if e["event"] == "check")
    assert check["batch"][1:] == [2, 16, 16], check
    done = next(e for e in ev if e["event"] == "done")
    assert done["test_acc"] >= 0.9, done
    print(f"hfbands: Hugging Face band arrays, bands b2 b4 -> {check['batch'][1:]}, test accuracy {done['test_acc']}")


def cloud(root):
    """a cloud location (file:// stands in for gs:// / Drive): fetched into <data>/cloud/<key> once, then used like own files"""
    import shutil
    root = os.path.abspath(root)
    shutil.copytree(os.path.join(root, "own", "imb"), os.path.join(root, "own", "cloudsrc"), dirs_exist_ok=True)
    ev = run(os.path.join(root, "cloud"), os.path.join(root, "data"), "cloud")
    got = next(e for e in ev if e["event"] == "info" and e.get("message", "").startswith("data ready in"))
    assert "/data/cloud/nnb-cloudsrc-" in got["message"] and "files fetched" in got["message"], got
    assert next(e for e in ev if e["event"] == "done")["test_acc"] >= 0.8
    ev = run(os.path.join(root, "cloud"), os.path.join(root, "data"), "cloud")   # a second run: already there
    print(f"cloud: {got['message']}")


def check_data(root):
    """Check data: no training, a report that sees the linear Sentinel-1 values and counts the classes, and the one-batch check"""
    root = os.path.abspath(root)
    ev = run(os.path.join(root, "check"), os.path.join(root, "data"), "check")
    assert not any(e["event"] == "epoch" for e in ev), "a data check must not train"
    rep = next(e for e in ev if e["event"] == "data_report")
    assert rep["counts"] == [108, 108] or sum(rep["counts"]) == rep["sizes"]["train"], rep
    assert any("linear backscatter" in w for w in rep["warnings"]), rep["warnings"]
    assert [b["band"] for b in rep["bands"]] == ["VV", "VH"] and rep["bands"][0]["min"] >= 0, rep["bands"]
    assert any(e["event"] == "check" for e in ev), "the one-batch check ran"
    done = next(e for e in ev if e["event"] == "done")
    assert done["check_only"] and done["warnings"] >= 1, done
    res = json.load(open(os.path.join(root, "check", "results.json")))
    assert res["check_only"] and len(res["samples"]) == 12 and len(res["samples"][0]["image"]) == 3, res.keys()
    print(f"check: no training; counts {rep['counts']}, warnings {rep['warnings']}")


if __name__ == "__main__":
    main(sys.argv[1])
    own_files(sys.argv[1])
    check_data(sys.argv[1])
    imbalanced(sys.argv[1])
    cloud(sys.argv[1])
    ben(sys.argv[1])
    pretrained_ms(sys.argv[1])
    hfbands(sys.argv[1])

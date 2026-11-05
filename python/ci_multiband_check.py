"""CI only (GitHub Actions; never on the owner's Mac): multispectral and radar images with real PyTorch on the CPU.

Generates class folders like EuroSAT-MS (13 Sentinel-2 bands, half the files GeoTIFF (bands last, as most tools write them) and half
.npy (bands first)) and like a Sentinel-1 set (VV, VH in linear backscatter), trains the generated train.py on each, and checks: the
chosen bands in the chosen order reached the network, per-band statistics were measured (and in dB for Sentinel-1), the network learnt the
two classes, and the result pictures are 3-channel composites.

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


if __name__ == "__main__":
    main(sys.argv[1])

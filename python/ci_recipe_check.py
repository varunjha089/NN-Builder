"""CI only (GitHub Actions; never on the owner's Mac): the training recipe with real PyTorch.

The resume_case example (with "recipe") writes train.py with AdamW (weight decay 0.05), one warm-up epoch then cosine decay over 4 epochs,
label smoothing 0.1 and gradient clipping at 1.0. The run must finish, report the recipe, and the learning rate it reports after each epoch
must follow the schedule: the full rate at the end of the warm-up, then decreasing towards 0.

usage: python ci_recipe_check.py <run folder>
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_resume_check import events, make_images  # noqa: E402


def main(run):
    run = os.path.abspath(run)
    data = os.path.join(run, "data")
    make_images(data, classes=6, per_class=100)
    env = {**os.environ, "NNB_DATA": data, "PYTHONUNBUFFERED": "1"}
    with open(os.path.join(run, "log.txt"), "w") as log:
        code = subprocess.call([sys.executable, "train.py"], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        sys.exit(f"FAIL: exit code {code}\n" + open(os.path.join(run, "log.txt")).read()[-3000:])
    ev = events(run)
    start = next(e for e in ev if e["event"] == "start")
    r = start["recipe"]
    assert (r["optimizer"], r["weight_decay"], r["schedule"], r["warmup_epochs"], r["label_smoothing"], r["grad_clip"]) == \
        ("adamw", 0.05, "cosine", 1.0, 0.1, 1.0), r
    lrs = [e["lr"] for e in ev if e["event"] == "epoch"]
    assert len(lrs) == 4, lrs
    assert abs(lrs[0] - 1e-3) < 1e-6, f"end of warm-up should be the full rate: {lrs}"
    assert lrs[0] > lrs[1] > lrs[2] > lrs[3] and lrs[3] < 1e-4, f"cosine decay: {lrs}"
    assert ev[-1]["event"] == "done", ev[-1]
    # six classes: top-5 accuracy per epoch and at the end, never below top-1
    assert all("val_top5" in e for e in ev if e["event"] == "epoch"), "val_top5 missing"
    assert ev[-1]["test_top5"] >= ev[-1]["test_acc"], ev[-1]
    # results pictures: Grad-CAM maps for the samples, misclassified images
    import json
    import torch
    res = json.load(open(os.path.join(run, "results.json")))
    cams = [s for s in res["samples"] if "cam" in s]
    assert res.get("cam_layer") and len(cams) == len(res["samples"]) and len(cams[0]["cam"]) > 0, res.get("cam_layer")
    assert all(0 <= v <= 1 for row in cams[0]["cam"] for v in row), "Grad-CAM values in 0..1"
    assert "mistakes" in res and all(m["pred"] != m["label"] for m in res["mistakes"]), "mistakes are misclassified"
    # exports: TorchScript gives the same outputs as the saved weights; the ONNX file exists
    assert any(e["event"] == "exported" and "model.ts" in e["files"] for e in ev), "no exported event"
    ts = torch.jit.load(os.path.join(run, "model.ts")).eval()
    sys.path.insert(0, run)
    import importlib.util
    spec = importlib.util.spec_from_file_location("trained", os.path.join(run, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.Net().eval()
    net.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu"))
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert float((ts(x) - net(x)).abs().max()) < 1e-4, "TorchScript differs from model.pt"
    assert os.path.getsize(os.path.join(run, "model.onnx")) > 1000, "ONNX file"
    print(f"recipe ran: learning rate per epoch {lrs}; test accuracy {ev[-1]['test_acc']}, top-5 {ev[-1]['test_top5']}; "
          f"Grad-CAM on {res['cam_layer']}, {len(res['mistakes'])} mistakes shown; TorchScript = model.pt, ONNX written")


if __name__ == "__main__":
    main(sys.argv[1])

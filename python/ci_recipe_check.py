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
    make_images(data)
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
    print(f"recipe ran: learning rate per epoch {lrs}; test accuracy {ev[-1]['test_acc']}")


if __name__ == "__main__":
    main(sys.argv[1])

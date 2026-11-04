"""CI only (GitHub Actions; never on the owner's Mac): early stopping with real PyTorch.

The resume_case example (with "early") writes train.py for a small CNN on a generated two-class image folder: 25 epochs at most,
early stopping on validation accuracy with patience 2, 20% of the training data held out. The task is easy, so validation accuracy
stops improving and the run must stop early, report its best epoch, keep validation and test data apart, and leave model.pt = the
best epoch's weights.

usage: python ci_early_stop_check.py <run folder>   (images from ci_resume_check.make_images)
"""
import json
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
    ev = events(run)
    if code != 0:
        sys.exit(f"FAIL: exit code {code}\n" + open(os.path.join(run, "log.txt")).read()[-3000:])
    start = next(e for e in ev if e["event"] == "start")
    # 600 images: 10% test (60), then 20% of the remaining 540 for validation (108), 432 for training
    assert (start["train_size"], start["val_size"], start["test_size"]) == (432, 108, 60), start
    assert start["early_stop"] and start["patience"] == 2 and start["monitor"] == "val_acc", start
    stops = [e for e in ev if e["event"] == "early_stop"]
    assert stops, "the run did not stop early: " + json.dumps([e for e in ev if e["event"] == "epoch"])[-500:]
    stop = stops[0]
    epochs = [e["epoch"] for e in ev if e["event"] == "epoch"]
    assert epochs[-1] == stop["epoch"] < 25 and stop["epoch"] - stop["best_epoch"] == 2, (epochs, stop)
    best = [e for e in ev if e["event"] == "best"]
    assert best and best[-1]["epoch"] == stop["best_epoch"], (best, stop)
    done = ev[-1]
    assert done["event"] == "done" and done["best_epoch"] == stop["best_epoch"], done
    assert os.path.exists(os.path.join(run, "best_model.pt")) and os.path.exists(os.path.join(run, "model.pt"))
    print(f"stopped early after epoch {stop['epoch']} (best epoch {stop['best_epoch']}, validation accuracy {stop['best']}); "
          f"split 432/108/60; test accuracy of the best epoch {done['test_acc']}")


if __name__ == "__main__":
    main(sys.argv[1])

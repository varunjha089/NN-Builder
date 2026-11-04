"""CI only (GitHub Actions; never on the owner's Mac): checks the spot-interruption path of the training harness with real PyTorch.

1. a two-class image folder is generated (no download), train.py (from the resume_case example) starts;
2. in epoch 2 it gets SIGUSR1 (the interruption notice): it must save a mid-epoch checkpoint and exit 75;
3. started again it must resume at that batch (same epoch), take a SIGUSR2 (rebalance: checkpoint, keep going) and finish.

usage: python ci_resume_check.py <run folder>
"""
import json
import os
import signal
import subprocess
import sys
import time


def events(run):
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_images(data):
    from PIL import Image, ImageDraw
    root = os.path.join(data, "kaggle", "ci__shapes")
    for label, shape in (("a", "box"), ("b", "dot")):
        os.makedirs(os.path.join(root, label), exist_ok=True)
        for i in range(300):
            im = Image.new("RGB", (32, 32), (i % 50, 20, 40))
            d = ImageDraw.Draw(im)
            x, y = 4 + i % 16, 4 + (i * 7) % 16
            (d.rectangle if shape == "box" else d.ellipse)([x, y, x + 10, y + 10], fill=(220, 200, 60))
            im.save(os.path.join(root, label, f"{i}.png"))


def wait_for(proc, run, test, what, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = events(run)
        if test(ev):
            return ev
        if proc.poll() is not None:
            sys.exit(f"FAIL: train.py ended (code {proc.returncode}) before {what}\n" + open(os.path.join(run, "log.txt")).read()[-3000:])
        time.sleep(0.05)
    sys.exit(f"FAIL: timed out waiting for {what}")


def start(run, env):
    log = open(os.path.join(run, "log.txt"), "a")
    return subprocess.Popen([sys.executable, "train.py"], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)


def main(run):
    run = os.path.abspath(run)  # train.py runs inside the run folder
    data = os.path.join(run, "data")
    make_images(data)
    env = {**os.environ, "NNB_DATA": data, "PYTHONUNBUFFERED": "1"}

    # 1st start: interrupted in the middle of epoch 2
    p = start(run, env)
    wait_for(p, run, lambda ev: any(e["event"] == "batch" and e["epoch"] == 2 and 0 < e["step"] < e["steps"] // 2 for e in ev), "a batch of epoch 2")
    p.send_signal(signal.SIGUSR1)
    code = p.wait(timeout=300)
    ev = events(run)
    assert code == 75, f"exit code {code}, wanted 75"
    inter = [e for e in ev if e["event"] == "interrupted"]
    ck = [e for e in ev if e["event"] == "checkpoint" and e.get("reason") == "spot interruption"]
    assert inter and ck, "no interrupted / checkpoint event"
    assert ck[-1]["epoch"] == 2 and 0 < ck[-1]["step"] < ck[-1]["steps"], f"not a mid-epoch checkpoint: {ck[-1]}"
    assert not os.path.exists(os.path.join(run, "checkpoint.pt.tmp")), "a temporary checkpoint was left"
    assert not any(e["event"] in ("done", "stopped") for e in ev), "an interruption must not end the run"
    step = ck[-1]["step"]
    print(f"interrupted at epoch 2, batch {step}/{ck[-1]['steps']}: exit 75, checkpoint saved")

    # 2nd start: resumes at that batch; a rebalance recommendation in epoch 3 checkpoints without stopping
    n_before = len(ev)
    p = start(run, env)
    ev = wait_for(p, run, lambda ev: any(e["event"] == "resume" for e in ev[n_before:]), "the resume event")
    res = [e for e in ev[n_before:] if e["event"] == "resume"][0]
    assert res["epoch"] == 2 and res["step"] == step and res["resumes"] == 1, f"resumed at the wrong place: {res}"
    wait_for(p, run, lambda ev: any(e["event"] == "batch" and e["epoch"] == 3 for e in ev[n_before:]), "a batch of epoch 3")
    p.send_signal(signal.SIGUSR2)
    code = p.wait(timeout=600)
    ev = events(run)[n_before:]
    assert code == 0, f"exit code {code} after resuming"
    first = next(e for e in ev if e["event"] == "batch")
    assert first["epoch"] == 2 and first["step"] > step, f"the first batch after resuming was {first}, not after batch {step}"
    assert sum(1 for e in ev if e["event"] == "epoch" and e["epoch"] == 2) == 1, "epoch 2 must finish once"
    assert any(e["event"] == "checkpoint" and e.get("reason") == "rebalance recommendation" for e in ev), "no checkpoint for the rebalance"
    assert ev[-1]["event"] == "done", f"last event {ev[-1]}"
    assert [e["epoch"] for e in events(run) if e["event"] == "epoch"] == [1, 2, 3], "epochs 1, 2, 3 each once"
    print(f"resumed at batch {step} of epoch 2, rebalance checkpoint taken, finished: test accuracy {ev[-1]['test_acc']}")


if __name__ == "__main__":
    main(sys.argv[1])

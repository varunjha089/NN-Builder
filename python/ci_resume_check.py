"""CI only (GitHub Actions; never on the owner's Mac): checks the spot-interruption path of the training harness with real PyTorch.

1. a two-class image folder is generated (no download), train.py (from the resume_case example) starts;
2. in epoch 2 it gets SIGUSR1 (the interruption notice): it must save a mid-epoch checkpoint and exit 75;
3. started again it must resume at that batch (same epoch), take a SIGUSR2 (rebalance: checkpoint, keep going) and finish;
4. a fresh copy with data-loading workers gets Ctrl-C on its whole process group (the app's Stop for a tmux run): it must stop
   cleanly ("stopped", results written), not die because its workers died first.

usage: python ci_resume_check.py <run folder>
"""
import json
import os
import shutil
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


def make_images(data, classes=2, per_class=300):
    """a generated image folder: class a = a box, b = a dot, and (for more classes) c-f = other colours of box and dot"""
    from PIL import Image, ImageDraw
    root = os.path.join(data, "kaggle", "ci__shapes")
    colours = [(220, 200, 60), (220, 200, 60), (60, 200, 220), (60, 200, 220), (220, 60, 160), (220, 60, 160)]
    for c in range(classes):
        label, shape = "abcdef"[c], ("box", "dot")[c % 2]
        os.makedirs(os.path.join(root, label), exist_ok=True)
        for i in range(per_class):
            im = Image.new("RGB", (32, 32), (i % 50, 20, 40))
            d = ImageDraw.Draw(im)
            x, y = 4 + i % 16, 4 + (i * 7) % 16
            (d.rectangle if shape == "box" else d.ellipse)([x, y, x + 10, y + 10], fill=colours[c])
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

    # a fresh copy with 2 loader workers: Stop = Ctrl-C to the whole process group (tmux send-keys C-c reaches every process)
    stop = os.path.join(run, "stop")
    os.makedirs(stop, exist_ok=True)
    for f in os.listdir(run):
        if f.endswith(".py") or f.endswith(".json") and f not in ("results.json",):
            shutil.copy(os.path.join(run, f), stop)
    log = open(os.path.join(stop, "log.txt"), "a")
    p = subprocess.Popen([sys.executable, "train.py"], cwd=stop, env={**env, "NNB_TEST_WORKERS": "2"}, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    wait_for(p, stop, lambda ev: any(e["event"] == "batch" and e["epoch"] == 2 for e in ev), "a batch of epoch 2 (with workers)")
    os.killpg(p.pid, signal.SIGINT)
    code = p.wait(timeout=300)
    ev = events(stop)
    errors = [e for e in ev if e["event"] == "error"]
    assert not errors, f"Stop with loader workers failed: {errors[0].get('message')}"
    assert ev[-1]["event"] == "stopped", f"last event {ev[-1]}, exit code {code}"
    assert os.path.exists(os.path.join(stop, "results.json")), "no results after Stop"
    print(f"Ctrl-C to the process group with 2 loader workers: stopped cleanly (exit {code}), results written")

    # a spot run: TERM while the machine shuts down (NNB_TEST_SHUTDOWN pretends it) is an interruption (exit 75, checkpoint, resumable);
    # the same TERM while it runs normally is the app's Stop
    for shutdown in (True, False):
        d = os.path.join(run, "spot_shutdown" if shutdown else "spot_stop")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(run):
            if f.endswith(".py") or f.endswith(".json") and f != "results.json":
                shutil.copy(os.path.join(run, f), d)
        e = {**env, "NNB_SPOT": "1", **({"NNB_TEST_SHUTDOWN": "1"} if shutdown else {})}
        log = open(os.path.join(d, "log.txt"), "a")
        p = subprocess.Popen([sys.executable, "train.py"], cwd=d, env=e, stdout=log, stderr=subprocess.STDOUT)
        wait_for(p, d, lambda ev: any(x["event"] == "batch" and x["epoch"] == 2 for x in ev), "a batch of epoch 2 (spot)")
        p.send_signal(signal.SIGTERM)
        code = p.wait(timeout=300)
        ev = events(d)
        if shutdown:
            assert code == 75, f"a shutdown on spot must exit 75 (resume), got {code}"
            assert any(x["event"] == "interrupted" for x in ev) and not any(x["event"] in ("stopped", "done") for x in ev), ev[-3:]
            assert os.path.exists(os.path.join(d, "checkpoint.pt")), "no checkpoint to resume from"
        else:
            assert ev[-1]["event"] == "stopped", f"TERM without a shutdown is a Stop: {ev[-1]}"
    print("spot: TERM during a shutdown -> checkpoint + exit 75 (resumable); TERM otherwise -> stopped with results")


if __name__ == "__main__":
    main(sys.argv[1])

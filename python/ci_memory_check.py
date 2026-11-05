"""CI only (GitHub Actions; never on the owner's Mac): the GPU-memory features of the training harness, with real PyTorch on the CPU.

1. "save memory" (activation checkpointing): for demos of every graph shape, the checkpointed forward (in parts) gives the same outputs
   and gradients as the plain forward of the same network.
2. automatic splitting: the same small run twice, once normally and once pretending that more than 8 images do not fit
   (NNB_TEST_OOM_ABOVE=8), so every batch of 32 is processed in 4 parts; both must end with the same weights and test accuracy.

usage: python ci_memory_check.py <folder from the memory_case example> <run folder A> <run folder B>
"""
import importlib.util
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_resume_check import events, make_images  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines Net and the harness; training runs only under __main__
    return mod


def parts_equal(root):
    for key in sorted(os.listdir(root)):
        mod = load(os.path.join(root, key, "train.py"), f"mem_{key}")
        torch.manual_seed(0)
        net = mod.Net().train()
        assert hasattr(net, "_part1"), f"{key}: no parts were generated"
        x = torch.randn(2, *mod.EXPECTED["input"], requires_grad=False)
        results = []
        for checkpointed in (True, False):
            net.save_memory = checkpointed
            net.zero_grad(set_to_none=True)
            torch.manual_seed(1)  # the same dropout masks in both passes
            out = net(x)
            out.float().square().mean().backward()
            results.append((out.detach().clone(), {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}))
        (a, ga), (b, gb) = results
        assert torch.allclose(a, b, atol=1e-5), f"{key}: outputs differ by {float((a - b).abs().max())}"
        assert ga.keys() == gb.keys() and ga, f"{key}: different parameters got gradients"
        worst = max(float((ga[n] - gb[n]).abs().max() / (gb[n].abs().max() + 1e-12)) for n in ga)
        assert worst < 1e-4, f"{key}: gradients differ (relative {worst:.2e})"
        parts = sum(1 for n in dir(net) if n.startswith("_part"))
        print(f"save memory, {key}: {parts} parts; outputs and gradients equal to the plain forward (relative difference {worst:.1e})")


def run(folder, extra):
    folder = os.path.abspath(folder)
    data = os.path.join(folder, "data")
    make_images(data)
    env = {**os.environ, "NNB_DATA": data, "PYTHONUNBUFFERED": "1", **extra}
    with open(os.path.join(folder, "log.txt"), "w") as log:
        code = subprocess.run([sys.executable, "train.py"], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
    assert code == 0, f"train.py failed ({code}):\n" + open(os.path.join(folder, "log.txt")).read()[-3000:]
    return events(folder)


def split_equal(a, b):
    ev_a = run(a, {})
    ev_b = run(b, {"NNB_TEST_OOM_ABOVE": "8"})
    splits = [e for e in ev_b if e["event"] == "memory_split" and e["what"] == "training"]
    assert not any(e["event"] == "memory_split" for e in ev_a), "the normal run split its batches"
    assert splits and splits[-1]["parts"] == 4 and splits[-1]["part_size"] == 8, f"expected 4 parts of 8: {splits}"
    assert len(splits) == 2, f"the split should be found once (1 -> 2 -> 4 parts), then kept: {splits}"
    wa = torch.load(os.path.join(a, "model.pt"), map_location="cpu")
    wb = torch.load(os.path.join(b, "model.pt"), map_location="cpu")
    worst = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    assert worst < 1e-3, f"the weights differ by {worst}"
    acc = [next(e for e in ev if e["event"] == "done")["test_acc"] for ev in (ev_a, ev_b)]
    assert acc[0] == acc[1], f"test accuracy {acc[0]} vs {acc[1]}"
    epoch = [e for e in ev_b if e["event"] == "epoch"]
    assert all("data_wait" in e for e in epoch), "no data_wait in the epoch events"
    start = next(e for e in ev_b if e["event"] == "start")
    assert start["precision"] == "fp32" and start["auto_split"] is True, start
    print(f"automatic split: batches of 32 in 4 parts of 8; same weights (max difference {worst:.1e}) and test accuracy {acc[0]}")


if __name__ == "__main__":
    parts_equal(sys.argv[1])
    split_equal(sys.argv[2], sys.argv[3])

"""Checks the app's shape engine against PyTorch (GitHub Actions; never on the owner's Mac): every random design from dump_cases must build,
every module must have the output shape the app computed (EXPECTED), and the parameter count must match.
    python python/verify_shapes.py cases.json"""
import json
import sys

import torch
import torch.nn as nn

cases = json.load(open(sys.argv[1]))
ok = [c for c in cases if c["ok"]]
assert len(ok) >= len(cases) // 4, f"too few valid designs: {len(ok)}"
bad = 0
for c in ok:
    ns = {"torch": torch, "nn": nn, "json": json}
    exec(c["code"], ns)
    net = ns["Net"]().eval()
    seen = {}
    for name, _, _ in ns["LAYERS"]:
        getattr(net, name).register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, list(o.shape[1:])))
    with torch.no_grad():
        net(torch.zeros(2, *c["input"]))
    exp = ns["EXPECTED"]
    wrong = [f"{n}: app {exp[n]} torch {seen.get(n)}" for n, _, _ in ns["LAYERS"] if seen.get(n) != exp[n]]
    params = sum(p.numel() for p in net.parameters())
    if wrong or params != c["params"]:
        bad += 1
        print(f"case {c['case']}: {wrong} params app {c['params']} torch {params}\n{c['code']}")
print(f"{len(ok)} designs checked, {bad} differ")
sys.exit(1 if bad else 0)

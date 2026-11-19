"""CI only (GitHub Actions; never on the owner's Mac): the built-in datasets from Hugging Face's copies, with the harness's own code.

builtin_fast() downloads MNIST, Fashion-MNIST and CIFAR-10 from Hugging Face (ylecun/mnist, zalando-datasets/fashion_mnist,
uoft-cs/cifar10); MNIST and CIFAR-10 must be image for image and label for label what torchvision's own download gives, so a run gets
the same data whichever way it came. A second call must use the cache (no download).

usage: python ci_builtin_check.py <scratch folder>
"""
import ast
import os
import sys
import time

import numpy as np


def harness_functions():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.py")).read()
    names = {"BUILTIN_HF", "TV_FILES", "tv_copy", "fetch", "builtin_fast"}
    code = [n for n in ast.parse(src).body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in names)]
    events = []
    ns = {"os": os, "time": time, "emit": lambda e, **k: events.append(k.get("message", "")), "DATA": os.path.abspath(sys.argv[1])}
    exec(compile(ast.Module(code, []), "harness.py", "exec"), ns)
    return ns, events


def main(work):
    from torchvision import datasets
    ns, events = harness_functions()
    root = os.path.join(os.path.abspath(work), "hf")
    for name, cls in (("mnist", datasets.MNIST), ("fashion_mnist", None), ("cifar10", datasets.CIFAR10)):
        got = ns["builtin_fast"](root, name)
        assert got is not None, f"{name}: Hugging Face's copy failed: {events[-1:]}"
        (xs, ys), (xt, yt) = got
        print(f"{name}: {xs.shape} {xt.shape}; {events[-1]}")
        if cls is None:
            assert xs.shape == (60000, 28, 28) and xt.shape == (10000, 28, 28)
            continue
        tv = os.path.join(os.path.abspath(work), "tv")
        for train, (x, y) in ((True, (xs, ys)), (False, (xt, yt))):
            ref = cls(tv, train=train, download=True)
            rx, ry = np.asarray(ref.data), np.asarray(ref.targets)
            assert x.shape == rx.shape, f"{name}: {x.shape} vs torchvision {rx.shape}"
            assert (y == ry).all(), f"{name}: labels differ from torchvision's"
            assert (x == rx).all(), f"{name}: images differ from torchvision's ({int((x != rx).any(axis=tuple(range(1, x.ndim))).sum())} differ)"
        print(f"{name}: identical to torchvision's download (train and test, images and labels)")
        n = len(events)
        ns["builtin_fast"](root, name)
        assert len(events) == n, f"{name}: downloaded again instead of using the cache"


if __name__ == "__main__":
    main(sys.argv[1])

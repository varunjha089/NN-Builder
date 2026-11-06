"""CI only (GitHub Actions; never on the owner's Mac): ImageNet weights in the demos, with real PyTorch.

For ResNet-18, VGG-16 and ViT-Base/16 (1000 classes): import the train.py the app generated, load torchvision's weights into its Net with
the harness's load_pretrained (tensor by tensor, by order and shape), and compare its outputs with torchvision's own model on the same
random images. They must agree (up to LayerNorm's epsilon, 1e-5 here and 1e-6 in torchvision's ViT).

usage: python ci_pretrained_check.py <folder from the pretrained_case example>
"""
import importlib.util
import os
import sys

import torch
import torchvision


def main(root):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    for folder in ("resnet18", "vgg16", "vit_b_16", "vit_b_16_lora"):
        name = folder.replace("_lora", "")
        path = os.path.join(os.path.abspath(root), folder, "train.py")
        spec = importlib.util.spec_from_file_location(f"train_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # defines Net and the harness; training runs only under __main__
        net = mod.Net().eval()
        loaded = mod.load_pretrained(net, name)
        ref = getattr(torchvision.models, name)(weights="DEFAULT").eval()
        with torch.no_grad():
            a, b = net(x), ref(x)
        diff = float((a - b).abs().max())
        # (with LoRA the adapters A and B have no counterpart: B starts at zero, so the outputs are still torchvision's)
        new = [k for k in net.state_dict() if k.endswith((".a", ".b"))] if folder.endswith("_lora") else []
        assert len(loaded) + len(new) == len(net.state_dict()), f"{folder}: {len(loaded)} of {len(net.state_dict())} tensors loaded"
        assert diff < (5e-3 if name.startswith("vit") else 1e-4), f"{name}: outputs differ by {diff}"
        assert bool((a.argmax(1) == b.argmax(1)).all()), f"{name}: different predictions"
        print(f"{folder}: all {len(loaded)} tensors loaded ({len(new)} new adapters); outputs equal to torchvision's (max difference {diff:.2e})")


if __name__ == "__main__":
    main(sys.argv[1])

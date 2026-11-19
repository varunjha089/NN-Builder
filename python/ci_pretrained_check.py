"""CI only (GitHub Actions; never on the owner's Mac): ImageNet weights in the demos, with real PyTorch.

For ResNet-18, VGG-16 and ViT-Base/16 (1000 classes): import the train.py the app generated, load torchvision's weights into its Net with
the harness's load_pretrained (tensor by tensor, by order and shape), and compare its outputs with torchvision's own model on the same
random images. They must agree (up to LayerNorm's epsilon, 1e-5 here and 1e-6 in torchvision's ViT).

Then a U-Net whose encoder is ResNet-18's layers (doc/examples/unet_resnet18.nnb) on 13 bands (radar, Sentinel-2, indices): with ImageNet's
and with SSL4EO-S12's Sentinel-2 weights, the encoder must compute exactly what the reference ResNet-18 computes (given the first-layer
filters the harness chose by band name), the first layer must take each named band's filter, and the decoder must stay untouched.

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

    for name in ("resnet18", "ssl4eo_s2"):
        encoder(root, name)


def load(root, folder):
    path = os.path.join(os.path.abspath(root), folder, "train.py")
    spec = importlib.util.spec_from_file_location(f"train_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encoder(root, name):
    mod = load(root, f"unet_{name}")
    torch.manual_seed(0)
    net = mod.Net().eval()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    loaded = mod.load_pretrained(net, name)
    ref, ref_bands, _, _ = mod.reference(name)
    ref = ref.eval()
    after = net.state_dict()
    # the first layer: each band with a name the reference knows takes that band's filter, the others the mean (all scaled by n / 13)
    first = next(k for k, v in after.items() if v.dim() == 4 and v.shape[1] == 13)
    w, rw = after[first], ref.conv1.weight.detach()
    scale = rw.shape[1] / 13
    at = {b: i for i, b in enumerate(ref_bands)}
    for c, b in enumerate(mod.BANDS):
        k = mod.band_key(b)
        want = rw[:, at[k]] if k in at else rw.mean(1)
        assert torch.allclose(w[:, c], want * scale, atol=1e-6), f"{name}: band {b} has the wrong first-layer filter"
    # the encoder: the design's own layers compute what the reference ResNet-18 computes, up to layer4
    ref.conv1.weight.data = w.clone()
    x = torch.randn(2, 13, 64, 64)
    outs = []
    hooks = [m.register_forward_hook(lambda m, i, o: outs.append(o)) for m in net.modules() if m is not net]
    with torch.no_grad():
        net(x)
        r = ref.layer4(ref.layer3(ref.layer2(ref.layer1(ref.maxpool(ref.relu(ref.bn1(ref.conv1(x))))))))
    for h in hooks:
        h.remove()
    same = [o for o in outs if torch.is_tensor(o) and o.shape == r.shape and float((o - r).abs().max()) < 1e-4]
    assert same, f"{name}: no layer of the design computes ResNet-18's layer4 output"
    # 20 convolutions + 20 BatchNorms (weight, bias, running mean / var each) of the encoder; nothing in the decoder
    convs = [k for k in loaded if after[k].dim() == 4]
    assert len(convs) == 20, f"{name}: {len(convs)} convolutions loaded"
    untouched = [k for k in after if k not in loaded and not k.endswith("num_batches_tracked")]
    assert untouched and all(torch.equal(after[k], before[k]) for k in untouched), f"{name}: a tensor that was not loaded changed"
    print(f"unet_{name}: encoder = ResNet-18 ({len(loaded)} tensors, {len(convs)} convolutions), {len(untouched)} decoder tensors left random; "
          f"first layer by band name ({sum(mod.band_key(b) in at for b in mod.BANDS)} of 13 bands)")


if __name__ == "__main__":
    main(sys.argv[1])

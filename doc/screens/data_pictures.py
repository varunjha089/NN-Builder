"""Pictures of each kind of data NN Builder reads, for the manual (doc/figures/kind_*.png). The data are GENERATED here (a random land
cover map turned into Sentinel-2 reflectances and Sentinel-1 backscatter with typical values per class), so nothing is downloaded and
nothing real is shown; the figures say so. Needs numpy and matplotlib.
    python doc/screens/data_pictures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
rng = np.random.default_rng(3)
CLASSES = ["water", "forest", "cropland", "built-up"]
COLOURS = ["#2b6cb0", "#2f855a", "#d69e2e", "#c53030"]
# typical surface reflectance (×10000) per class for B2 B3 B4 B8 B11 B12, and Sentinel-1 VV / VH in dB
S2 = {"water": [600, 700, 500, 300, 150, 100], "forest": [300, 550, 350, 3200, 1500, 700],
      "cropland": [500, 800, 700, 2600, 2200, 1300], "built-up": [900, 950, 1000, 1800, 2400, 2100]}
S1 = {"water": [-21, -27], "forest": [-8, -14], "cropland": [-11, -17], "built-up": [-3, -10]}
NOTE = "generated example data (not a real image)"


def smooth(shape, scale):
    """a smooth random field (Gaussian filter in Fourier space)"""
    f = np.fft.fft2(rng.normal(size=shape))
    ky, kx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij")
    return np.real(np.fft.ifft2(f * np.exp(-(kx ** 2 + ky ** 2) * (scale ** 2))))


def landcover(h, w, scale=12.0):
    fields = np.stack([smooth((h, w), scale) for _ in CLASSES])
    fields[0] -= 0.25 * fields.std()  # a little less water
    return fields.argmax(0)


def to_s2(lc):
    a = np.stack([np.choose(lc, [S2[c][b] for c in CLASSES]) for b in range(6)]).astype(float)
    return a * (1 + 0.08 * rng.normal(size=a.shape)) + 40 * smooth(lc.shape, 3)[None]


def to_s1(lc):
    a = np.stack([np.choose(lc, [S1[c][b] for c in CLASSES]) for b in range(2)]).astype(float)
    return a + 1.8 * rng.normal(size=a.shape)  # speckle


def stretch(x):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def true_colour(a):
    """B4 B3 B2 stretched together (keeps the colours' balance), slightly brightened"""
    rgb = np.stack([a[2], a[1], a[0]], -1)
    lo, hi = np.percentile(rgb, [1, 99])
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8


def radar_colour(r):
    return np.stack([stretch(r[0]), stretch(r[1]), stretch(r[0] - r[1])], -1)


def lc_image(lc, alpha=None):
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in COLOURS])[lc]
    return rgb


def finish(fig, name):
    fig.text(0.995, 0.01, NOTE, ha="right", va="bottom", fontsize=7, color="#777")
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(os.path.join("doc/figures", name))


def off(ax, title=""):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def pictures():
    """ordinary pictures in class folders"""
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1]})
    for r, (cls, top) in enumerate([("forest", 1), ("water", 0)]):
        axes[r, 0].text(0.0, 0.5, f"train/\n  {cls}/\n    img_001.jpg\n    img_002.png\n    …", family="monospace", fontsize=8, va="center")
        axes[r, 0].axis("off")
        for k in range(4):
            lc = landcover(48, 48, 6)
            lc[lc == 0] = 3 if top else 0
            lc = np.where(rng.random(lc.shape) < 0.6, top, lc)
            off(axes[r, k + 1], f"{cls} {k + 1}")
            axes[r, k + 1].imshow(true_colour(to_s2(lc)))
    fig.suptitle("Pictures (PNG / JPEG, grey or colour), one folder per class", fontsize=10)
    finish(fig, "kind_pictures.png")


def bands():
    """a Sentinel-2 patch band by band, its true-colour composite, and a Sentinel-1 patch"""
    lc = landcover(64, 64, 9)
    a, r = to_s2(lc), to_s1(lc)
    names = ["B2 blue", "B3 green", "B4 red", "B8 near infrared", "B11 SWIR", "B12 SWIR"]
    fig, axes = plt.subplots(2, 6, figsize=(10, 3.9))
    for b in range(6):
        off(axes[0, b], names[b])
        axes[0, b].imshow(a[b], cmap="gray")
    off(axes[1, 0], "true colour (B4 B3 B2)")
    axes[1, 0].imshow(true_colour(a))
    off(axes[1, 1], "false colour (B8 B4 B3)")
    axes[1, 1].imshow(np.stack([stretch(a[3]), stretch(a[2]), stretch(a[1])], -1))
    off(axes[1, 2], "S1 VV (dB)")
    axes[1, 2].imshow(r[0], cmap="gray")
    off(axes[1, 3], "S1 VH (dB)")
    axes[1, 3].imshow(r[1], cmap="gray")
    off(axes[1, 4], "S1 VV / VH / VV−VH")
    axes[1, 4].imshow(radar_colour(r))
    off(axes[1, 5], "land cover (for reference)")
    axes[1, 5].imshow(lc_image(lc))
    fig.suptitle("Satellite images: one GeoTIFF / .npy file holds all bands of a patch (Sentinel-2 top, Sentinel-1 below)", fontsize=10)
    finish(fig, "kind_bands.png")


def fusion():
    """the same place in Sentinel-2 and Sentinel-1, in two folders with the same file names, stacked into one input"""
    lc = landcover(64, 64, 9)
    a, r = to_s2(lc), to_s1(lc)
    fig = plt.figure(figsize=(10, 3.4))
    ax1 = fig.add_axes([0.02, 0.12, 0.22, 0.7])
    off(ax1, "s2/forest/patch_17.tif\n(6 bands, true colour shown)")
    ax1.imshow(true_colour(a))
    ax2 = fig.add_axes([0.27, 0.12, 0.22, 0.7])
    off(ax2, "s1/forest/patch_17.npy\n(VV, VH; same name)")
    ax2.imshow(radar_colour(r))
    ax3 = fig.add_axes([0.53, 0.05, 0.45, 0.85])
    ax3.axis("off")
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 10)
    for k, (lab, col) in enumerate([("B2", "#4a6fa5"), ("B3", "#4f9a5a"), ("B4", "#c0504d"), ("B8", "#7f5aa2"), ("B11", "#9c6b30"),
                                     ("B12", "#8a8a8a"), ("VV", "#222"), ("VH", "#555")]):
        ax3.add_patch(Rectangle((0.3 + k * 0.42, 1.5 + k * 0.42), 3.2, 3.2, facecolor=col, edgecolor="white", alpha=0.85))
        ax3.text(0.45 + k * 0.42, 4.45 + k * 0.42, lab, color="white", fontsize=7)
    ax3.text(0.3, 0.6, "stacked: 8 channels (group 1 = S2, group 2 = S1)", fontsize=8)
    ax3.text(7.4, 7.6, "early fusion: one Input takes all 8", fontsize=8)
    ax3.text(7.4, 6.4, "mid fusion: Input 1 (6 bands) and\nInput 2 (2 bands), a branch each,\njoined by Concat", fontsize=8, va="top")
    fig.suptitle("Sentinel-2 + Sentinel-1 fusion: two band groups for the same places", fontsize=10)
    finish(fig, "kind_fusion.png")


def segmentation():
    """images and masks with the same names"""
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for k in range(4):
        lc = landcover(64, 64, 8)
        mask = (lc == 2).astype(int)  # fields
        off(axes[0, k], f"images/tile_{k}.tif")
        axes[0, k].imshow(true_colour(to_s2(lc)))
        off(axes[1, k], f"masks/tile_{k}_mask.png")
        axes[1, k].imshow(mask, cmap="gray", vmin=0, vmax=1)
    fig.suptitle("Segmentation: an image and a mask (class number per pixel: 0 background, 1 field) with the same name", fontsize=10)
    finish(fig, "kind_segmentation.png")


def scene():
    """a large export cut into tiles, with its label image; empty tiles are left out"""
    lc = landcover(256, 384, 34)
    a = to_s2(lc)
    rgb = true_colour(a)
    rgb[:, :52] = 1.0  # outside the region: no data
    lab = lc_image(lc)
    lab[:, :52] = 0.92
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, img, t in [(axes[0], rgb, "nnb_image/scene-0000000000-0000000000.tif"), (axes[1], lab, "nnb_label/scene-0000000000-0000000000.tif")]:
        off(ax, t)
        ax.imshow(img)
        for y in range(0, 256, 64):
            for x in range(0, 384, 64):
                empty = x < 52
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 64, 64, fill=empty, facecolor="#00000018", edgecolor="white" if not empty else "#bbb",
                                       linewidth=0.8, linestyle="-" if not empty else ":"))
    for c, col in zip(CLASSES, COLOURS):
        axes[1].plot([], [], "s", color=col, label=c)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle("A large image + a label image on the same grid, cut into tiles of the image size (dotted: mostly empty, left out)", fontsize=10)
    finish(fig, "kind_scene.png")


def multilabel():
    """patches with several classes each, and the CSV that lists them"""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.2]})
    rows = []
    for k in range(4):
        lc = landcover(48, 48, 7)
        present = [CLASSES[c] for c in range(4) if (lc == c).mean() > 0.08]
        off(axes[k], f"p{k + 1}\n" + ", ".join(present))
        axes[k].imshow(true_colour(to_s2(lc)))
        rows.append(f'p{k + 1},"{"; ".join(present)}"')
    axes[4].axis("off")
    axes[4].text(0.02, 0.5, "labels.csv\n\npatch_id,labels\n" + "\n".join(rows), family="monospace", fontsize=8, va="center")
    fig.suptitle("Multi-label: several classes per image, listed in a CSV", fontsize=10)
    finish(fig, "kind_multilabel.png")


if __name__ == "__main__":
    for f in (pictures, bands, fusion, segmentation, scene, multilabel):
        f()

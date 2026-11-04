//! Famous architectures as starting designs. Each comes with the input it was made for; loading one sets the dataset's image size (built-in
//! sets are resized) and, for U-Net-style networks, the segmentation task. Layer settings follow the original papers, with two simplifications
//! noted in the descriptions (no auxiliary classifiers in GoogLeNet; padding "same" instead of cropping in U-Net).

use crate::model::{self, Edge, Graph, Node};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct Template {
    pub key: &'static str,
    pub title: &'static str,
    pub year: u16,
    pub task: &'static str,
    /// channels, height, width the network was designed for
    pub input: [usize; 3],
    pub about: &'static str,
}

pub const TEMPLATES: &[Template] = &[
    Template { key: "lenet5", title: "LeNet-5", year: 1998, task: "classify", input: [1, 28, 28],
        about: "LeCun et al. Two 5×5 convolutions with average pooling and tanh, then three dense layers: the first CNN for digits. Runs on MNIST as it is." },
    Template { key: "small_cnn", title: "Small CNN (CIFAR-10)", year: 2015, task: "classify", input: [3, 32, 32],
        about: "A VGG-style network for 32×32 colour images: three blocks of two 3×3 convolutions with BatchNorm, max pooling and dropout. A good quick baseline." },
    Template { key: "alexnet", title: "AlexNet", year: 2012, task: "classify", input: [3, 224, 224],
        about: "Krizhevsky et al. 11×11 stride-4 convolution, overlapping 3×3 stride-2 max pooling, ReLU, dropout and two 4096-unit layers (about 60 million parameters, mostly in the dense layers)." },
    Template { key: "vgg16", title: "VGG-16", year: 2014, task: "classify", input: [3, 224, 224],
        about: "Simonyan & Zisserman. Only 3×3 convolutions in five blocks (64→512 filters), each followed by 2×2 max pooling; channels double as the image halves." },
    Template { key: "googlenet", title: "GoogLeNet (Inception v1)", year: 2014, task: "classify", input: [3, 224, 224],
        about: "Szegedy et al. Nine Inception modules: 1×1, 3×3 and 5×5 convolutions and a pooling branch side by side, joined with Concat; 1×1 convolutions reduce channels first. Global average pooling instead of large dense layers. (Without the two auxiliary classifiers.)" },
    Template { key: "resnet18", title: "ResNet-18", year: 2015, task: "classify", input: [3, 224, 224],
        about: "He et al. Residual blocks: two 3×3 convolutions with BatchNorm, added to a shortcut (a 1×1 stride-2 convolution where the size or channels change). Skip connections let very deep networks train." },
    Template { key: "mobilenet", title: "MobileNet v1", year: 2017, task: "classify", input: [3, 224, 224],
        about: "Howard et al. Depthwise separable convolutions: a 3×3 depthwise convolution per channel, then a 1×1 convolution to mix channels; about 9× fewer computations than normal convolutions." },
    Template { key: "densenet_small", title: "DenseNet (small, CIFAR)", year: 2017, task: "classify", input: [3, 32, 32],
        about: "Huang et al. In each dense block every layer receives the Concat of all earlier layers' outputs (growth 12); transition layers (1×1 convolution + average pooling) shrink the image between blocks." },
    Template { key: "transformer", title: "Transformer encoder (image rows as tokens)", year: 2017, task: "classify", input: [1, 28, 28],
        about: "Vaswani et al. The Transformer's encoder: each block is LayerNorm → multi-head self-attention → Add, then LayerNorm → a two-layer MLP (GELU) → Add. The app trains on images, so every row of a 28×28 image is one token (28 tokens of 28 values), embedded to 64 dimensions with a class token and position embeddings; 2 blocks, 4 heads." },
    Template { key: "vit_b16", title: "ViT-Base/16", year: 2020, task: "classify", input: [3, 224, 224],
        about: "Dosovitskiy et al. The image is cut into 16×16 patches (a 16×16 stride-16 convolution), giving 196 tokens of 768 values; a class token and position embeddings are added; 12 Transformer encoder blocks (12 heads, MLP 3072); the class token gives the prediction. About 86 million parameters (the same count as torchvision's vit_b_16)." },
    Template { key: "vit_tiny", title: "ViT-Tiny (CIFAR-10)", year: 2020, task: "classify", input: [3, 32, 32],
        about: "A small Vision Transformer for 32×32 images: 4×4 patches (64 tokens of 192 values), 6 encoder blocks with 3 heads and MLP 384, dropout 0.1. Trains in minutes on a GPU; needs augmentation to do well on small datasets." },
    Template { key: "unet", title: "U-Net", year: 2015, task: "segment", input: [3, 128, 128],
        about: "Ronneberger et al. An encoder (two 3×3 convolutions + max pooling, 32→256 filters) and a decoder (2×2 transposed convolution, Concat with the encoder's skip connection, two 3×3 convolutions), ending with a 1×1 convolution giving a class for every pixel. Padding 'same' keeps sizes equal (the paper crops instead)." },
    Template { key: "attention_unet", title: "Attention U-Net", year: 2018, task: "segment", input: [3, 128, 128],
        about: "Oktay et al. U-Net with attention gates on the skip connections: the decoder signal and the skip features are each projected by a 1×1 convolution, added, passed through ReLU, a 1×1 convolution and a sigmoid, and the resulting map multiplies the skip features before the Concat." },
];

struct B {
    g: Graph,
}

impl B {
    fn new() -> B {
        let mut b = B { g: Graph::default() };
        b.add("input", json!({}), &[]);
        b
    }
    fn add(&mut self, kind: &str, over: Value, from: &[&str]) -> String {
        let id = format!("n{}", self.g.nodes.len() + 1);
        let mut p = model::defaults(kind).as_object().cloned().unwrap_or_default();
        for (k, v) in over.as_object().unwrap() {
            p.insert(k.clone(), v.clone());
        }
        self.g.nodes.push(Node { id: id.clone(), kind: kind.into(), params: p, x: 0.0, y: 0.0 });
        for f in from {
            let eid = format!("e{}", self.g.edges.len() + 1);
            self.g.edges.push(Edge { id: eid, source: f.to_string(), target: id.clone() });
        }
        id
    }
    fn input(&self) -> String {
        "n1".into()
    }
    /// convolution (no activation, no bias) -> BatchNorm -> ReLU
    fn cbr(&mut self, x: &str, filters: usize, k: usize, s: usize) -> String {
        let c = self.add("conv2d", json!({"filters": filters, "kernel": [k, k], "stride": [s, s], "padding": "same", "activation": "none", "bias": false}), &[x]);
        let n = self.add("batchnorm", json!({}), &[&c]);
        self.add("activation", json!({"fn": "relu"}), &[&n])
    }
    fn conv(&mut self, x: &str, filters: usize, k: usize, s: usize, act: &str) -> String {
        self.add("conv2d", json!({"filters": filters, "kernel": [k, k], "stride": [s, s], "padding": "same", "activation": act}), &[x])
    }
    /// a pre-norm Transformer encoder block: x + Attention(LN(x)), then + MLP(LN(.))
    fn encoder(&mut self, x: &str, d: usize, heads: usize, mlp: usize, drop: f64) -> String {
        let ln1 = self.add("layernorm", json!({}), &[x]);
        let mut a = self.add("attention", json!({"heads": heads, "dropout": drop}), &[&ln1]);
        if drop > 0.0 {
            a = self.add("dropout", json!({"rate": drop}), &[&a]);
        }
        let s1 = self.add("add", json!({}), &[x, &a]);
        let ln2 = self.add("layernorm", json!({}), &[&s1]);
        let mut m = self.add("dense", json!({"units": mlp, "activation": "gelu"}), &[&ln2]);
        if drop > 0.0 {
            m = self.add("dropout", json!({"rate": drop}), &[&m]);
        }
        let mut m2 = self.add("dense", json!({"units": d, "activation": "none"}), &[&m]);
        if drop > 0.0 {
            m2 = self.add("dropout", json!({"rate": drop}), &[&m2]);
        }
        self.add("add", json!({}), &[&s1, &m2])
    }
    fn maxpool(&mut self, x: &str, size: usize, stride: usize, pad: usize) -> String {
        let st = if size == stride { json!("size") } else { json!([stride, stride]) };
        self.add("pool2d", json!({"mode": "max", "size": [size, size], "stride": st, "padding": pad}), &[x])
    }
    /// place the nodes in columns by depth (longest path from the Input), left to right; branches are stacked in a column
    fn layout(mut self) -> Graph {
        let mut depth: HashMap<String, usize> = HashMap::new();
        for n in &self.g.nodes {
            // nodes are added in computation order, so every input already has its depth
            let d = self.g.edges.iter().filter(|e| e.target == n.id).filter_map(|e| depth.get(&e.source)).max().map(|d| d + 1).unwrap_or(0);
            depth.insert(n.id.clone(), d);
        }
        let mut per_row: HashMap<usize, usize> = HashMap::new();
        let widest = {
            let mut c: HashMap<usize, usize> = HashMap::new();
            for d in depth.values() {
                *c.entry(*d).or_default() += 1;
            }
            c.values().copied().max().unwrap_or(1)
        };
        let row_count: HashMap<usize, usize> = depth.values().fold(HashMap::new(), |mut m, d| {
            *m.entry(*d).or_default() += 1;
            m
        });
        for n in &mut self.g.nodes {
            let d = depth[&n.id];
            let i = per_row.entry(d).or_default();
            let offset = (widest - row_count[&d]) as f64 * 75.0;
            n.x = 60.0 + d as f64 * 300.0;
            n.y = 60.0 + offset + *i as f64 * 150.0;
            *i += 1;
        }
        self.g
    }
}

pub fn get(key: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.key == key)
}

pub fn build(key: &str) -> Option<Graph> {
    let mut b = B::new();
    let x = b.input();
    match key {
        "lenet5" => {
            let c1 = b.add("conv2d", json!({"filters": 6, "kernel": [5, 5], "padding": 2, "activation": "tanh"}), &[&x]);
            let p1 = b.add("pool2d", json!({"mode": "avg"}), &[&c1]);
            let c2 = b.add("conv2d", json!({"filters": 16, "kernel": [5, 5], "padding": "valid", "activation": "tanh"}), &[&p1]);
            let p2 = b.add("pool2d", json!({"mode": "avg"}), &[&c2]);
            let f = b.add("flatten", json!({}), &[&p2]);
            let d1 = b.add("dense", json!({"units": 120, "activation": "tanh"}), &[&f]);
            let d2 = b.add("dense", json!({"units": 84, "activation": "tanh"}), &[&d1]);
            b.add("output", json!({}), &[&d2]);
        }
        "small_cnn" => {
            let mut h = x;
            for (i, f) in [32, 64, 128].into_iter().enumerate() {
                h = b.cbr(&h, f, 3, 1);
                h = b.cbr(&h, f, 3, 1);
                h = b.maxpool(&h, 2, 2, 0);
                h = b.add("dropout", json!({"rate": 0.2 + 0.1 * i as f64}), &[&h]);
            }
            let f = b.add("flatten", json!({}), &[&h]);
            let d = b.add("dense", json!({"units": 128, "activation": "relu"}), &[&f]);
            let d = b.add("dropout", json!({"rate": 0.5}), &[&d]);
            b.add("output", json!({}), &[&d]);
        }
        "alexnet" => {
            let c1 = b.add("conv2d", json!({"filters": 96, "kernel": [11, 11], "stride": [4, 4], "padding": 2, "activation": "relu"}), &[&x]);
            let p1 = b.maxpool(&c1, 3, 2, 0);
            let c2 = b.add("conv2d", json!({"filters": 256, "kernel": [5, 5], "padding": "same", "activation": "relu"}), &[&p1]);
            let p2 = b.maxpool(&c2, 3, 2, 0);
            let c3 = b.conv(&p2, 384, 3, 1, "relu");
            let c4 = b.conv(&c3, 384, 3, 1, "relu");
            let c5 = b.conv(&c4, 256, 3, 1, "relu");
            let p3 = b.maxpool(&c5, 3, 2, 0);
            let f = b.add("flatten", json!({}), &[&p3]);
            let d = b.add("dropout", json!({"rate": 0.5}), &[&f]);
            let d = b.add("dense", json!({"units": 4096, "activation": "relu"}), &[&d]);
            let d = b.add("dropout", json!({"rate": 0.5}), &[&d]);
            let d = b.add("dense", json!({"units": 4096, "activation": "relu"}), &[&d]);
            b.add("output", json!({}), &[&d]);
        }
        "vgg16" => {
            let mut h = x;
            for (f, n) in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)] {
                for _ in 0..n {
                    h = b.conv(&h, f, 3, 1, "relu");
                }
                h = b.maxpool(&h, 2, 2, 0);
            }
            let f = b.add("flatten", json!({}), &[&h]);
            let d = b.add("dense", json!({"units": 4096, "activation": "relu"}), &[&f]);
            let d = b.add("dropout", json!({"rate": 0.5}), &[&d]);
            let d = b.add("dense", json!({"units": 4096, "activation": "relu"}), &[&d]);
            let d = b.add("dropout", json!({"rate": 0.5}), &[&d]);
            b.add("output", json!({}), &[&d]);
        }
        "googlenet" => {
            let c1 = b.add("conv2d", json!({"filters": 64, "kernel": [7, 7], "stride": [2, 2], "padding": 3, "activation": "relu"}), &[&x]);
            let p1 = b.maxpool(&c1, 3, 2, 1);
            let c2 = b.conv(&p1, 64, 1, 1, "relu");
            let c3 = b.conv(&c2, 192, 3, 1, "relu");
            let mut h = b.maxpool(&c3, 3, 2, 1);
            // (1×1, 3×3 reduce, 3×3, 5×5 reduce, 5×5, pool projection) per module; max pooling after 3b and 4e
            let mods: [(usize, usize, usize, usize, usize, usize, bool); 9] = [
                (64, 96, 128, 16, 32, 32, false), (128, 128, 192, 32, 96, 64, true),
                (192, 96, 208, 16, 48, 64, false), (160, 112, 224, 24, 64, 64, false), (128, 128, 256, 24, 64, 64, false),
                (112, 144, 288, 32, 64, 64, false), (256, 160, 320, 32, 128, 128, true),
                (256, 160, 320, 32, 128, 128, false), (384, 192, 384, 48, 128, 128, false),
            ];
            for (c1x1, r3, c3x3, r5, c5x5, pp, pool_after) in mods {
                let a = b.conv(&h, c1x1, 1, 1, "relu");
                let r = b.conv(&h, r3, 1, 1, "relu");
                let bb = b.conv(&r, c3x3, 3, 1, "relu");
                let r = b.conv(&h, r5, 1, 1, "relu");
                let c = b.conv(&r, c5x5, 5, 1, "relu");
                let p = b.maxpool(&h, 3, 1, 1);
                let d = b.conv(&p, pp, 1, 1, "relu");
                h = b.add("concat", json!({}), &[&a, &bb, &c, &d]);
                if pool_after {
                    h = b.maxpool(&h, 3, 2, 1);
                }
            }
            let g = b.add("global_pool", json!({"mode": "avg"}), &[&h]);
            let d = b.add("dropout", json!({"rate": 0.4}), &[&g]);
            b.add("output", json!({}), &[&d]);
        }
        "resnet18" => {
            let c = b.cbr(&x, 64, 7, 2);
            let mut h = b.maxpool(&c, 3, 2, 1);
            let mut cin = 64;
            for (f, first_stride) in [(64, 1), (128, 2), (256, 2), (512, 2)] {
                for i in 0..2 {
                    let s = if i == 0 { first_stride } else { 1 };
                    let a = b.cbr(&h, f, 3, s);
                    let c2 = b.add("conv2d", json!({"filters": f, "kernel": [3, 3], "padding": "same", "activation": "none", "bias": false}), &[&a]);
                    let n2 = b.add("batchnorm", json!({}), &[&c2]);
                    let short = if s != 1 || cin != f {
                        let p = b.add("conv2d", json!({"filters": f, "kernel": [1, 1], "stride": [s, s], "padding": "valid", "activation": "none", "bias": false}), &[&h]);
                        b.add("batchnorm", json!({}), &[&p])
                    } else {
                        h.clone()
                    };
                    let sum = b.add("add", json!({}), &[&n2, &short]);
                    h = b.add("activation", json!({"fn": "relu"}), &[&sum]);
                    cin = f;
                }
            }
            let g = b.add("global_pool", json!({"mode": "avg"}), &[&h]);
            b.add("output", json!({}), &[&g]);
        }
        "mobilenet" => {
            let mut h = b.cbr(&x, 32, 3, 2);
            let blocks = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)];
            for (f, s) in blocks {
                let d = b.add("dwconv2d", json!({"multiplier": 1, "kernel": [3, 3], "stride": [s, s], "padding": "same", "activation": "none", "bias": false}), &[&h]);
                let n = b.add("batchnorm", json!({}), &[&d]);
                let a = b.add("activation", json!({"fn": "relu"}), &[&n]);
                h = b.cbr(&a, f, 1, 1);
            }
            let g = b.add("global_pool", json!({"mode": "avg"}), &[&h]);
            b.add("output", json!({}), &[&g]);
        }
        "densenet_small" => {
            let mut h = b.add("conv2d", json!({"filters": 24, "kernel": [3, 3], "padding": "same", "activation": "none", "bias": false}), &[&x]);
            let mut ch = 24;
            for block in 0..3 {
                for _ in 0..4 {
                    let n = b.add("batchnorm", json!({}), &[&h]);
                    let a = b.add("activation", json!({"fn": "relu"}), &[&n]);
                    let c = b.add("conv2d", json!({"filters": 12, "kernel": [3, 3], "padding": "same", "activation": "none", "bias": false}), &[&a]);
                    h = b.add("concat", json!({}), &[&h, &c]);
                    ch += 12;
                }
                if block < 2 {
                    let n = b.add("batchnorm", json!({}), &[&h]);
                    let c = b.add("conv2d", json!({"filters": ch / 2, "kernel": [1, 1], "padding": "valid", "activation": "none", "bias": false}), &[&n]);
                    h = b.add("pool2d", json!({"mode": "avg", "size": [2, 2], "stride": "size"}), &[&c]);
                    ch /= 2;
                }
            }
            let n = b.add("batchnorm", json!({}), &[&h]);
            let a = b.add("activation", json!({"fn": "relu"}), &[&n]);
            let g = b.add("global_pool", json!({"mode": "avg"}), &[&a]);
            b.add("output", json!({}), &[&g]);
        }
        "transformer" => {
            let t = b.add("to_tokens", json!({"mode": "rows"}), &[&x]);
            let e = b.add("dense", json!({"units": 64, "activation": "none"}), &[&t]);
            let mut h = b.add("cls_pos", json!({"cls": true}), &[&e]);
            for _ in 0..2 {
                h = b.encoder(&h, 64, 4, 128, 0.1);
            }
            let n = b.add("layernorm", json!({}), &[&h]);
            let p = b.add("token_pool", json!({"mode": "cls"}), &[&n]);
            b.add("output", json!({}), &[&p]);
        }
        "vit_b16" | "vit_tiny" => {
            let (patch, d, depth, heads, mlp, drop) = if key == "vit_b16" { (16, 768, 12, 12, 3072, 0.0) } else { (4, 192, 6, 3, 384, 0.1) };
            let c = b.add("conv2d", json!({"filters": d, "kernel": [patch, patch], "stride": [patch, patch], "padding": "valid", "activation": "none"}), &[&x]);
            let t = b.add("to_tokens", json!({"mode": "patches"}), &[&c]);
            let mut h = b.add("cls_pos", json!({"cls": true}), &[&t]);
            for _ in 0..depth {
                h = b.encoder(&h, d, heads, mlp, drop);
            }
            let n = b.add("layernorm", json!({}), &[&h]);
            let p = b.add("token_pool", json!({"mode": "cls"}), &[&n]);
            b.add("output", json!({}), &[&p]);
        }
        "unet" | "attention_unet" => {
            let attention = key == "attention_unet";
            let double = |b: &mut B, x: &str, f: usize| {
                let a = b.conv(x, f, 3, 1, "relu");
                b.conv(&a, f, 3, 1, "relu")
            };
            let mut skips = vec![];
            let mut h = x;
            for f in [32, 64, 128, 256] {
                let d = double(&mut b, &h, f);
                skips.push((d.clone(), f));
                h = b.maxpool(&d, 2, 2, 0);
            }
            h = double(&mut b, &h, 512);
            for (skip, f) in skips.into_iter().rev() {
                let up = b.add("convtranspose2d", json!({"filters": f, "kernel": [2, 2], "stride": [2, 2], "activation": "none"}), &[&h]);
                let joined_skip = if attention {
                    // attention gate: g = up (decoder), x = skip (encoder)
                    let wg = b.add("conv2d", json!({"filters": f / 2, "kernel": [1, 1], "padding": "valid", "activation": "none"}), &[&up]);
                    let wx = b.add("conv2d", json!({"filters": f / 2, "kernel": [1, 1], "padding": "valid", "activation": "none"}), &[&skip]);
                    let s = b.add("add", json!({}), &[&wg, &wx]);
                    let r = b.add("activation", json!({"fn": "relu"}), &[&s]);
                    let psi = b.add("conv2d", json!({"filters": 1, "kernel": [1, 1], "padding": "valid", "activation": "sigmoid"}), &[&r]);
                    b.add("multiply", json!({}), &[&skip, &psi])
                } else {
                    skip
                };
                let c = b.add("concat", json!({}), &[&joined_skip, &up]);
                h = double(&mut b, &c, f);
            }
            b.add("seg_output", json!({}), &[&h]);
        }
        _ => return None,
    }
    Some(b.layout())
}

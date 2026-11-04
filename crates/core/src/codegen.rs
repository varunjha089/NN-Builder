//! From design to Python: the `Net` class (a forward that follows the graph) plus the training harness (python/harness.py), as one
//! standalone train.py that needs only torch/torchvision (+ `datasets` for Hugging Face, `kaggle` for Kaggle data off Kaggle).

use crate::dataset::Dataset;
use crate::model::{Graph, Node};
use crate::shapes::{self, activation, int, pair, Analysis};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const HARNESS: &str = include_str!("../../../python/harness.py");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainSettings {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// "adam" | "sgd"
    pub optimizer: String,
    pub seed: u64,
    /// mixed precision on CUDA
    pub amp: bool,
    /// 0 = all training images
    pub limit_train: usize,
    /// college / AWS: the run is ended after this many hours
    pub max_hours: f64,
    /// AWS: stop the instance when the run ends
    pub stop_instance: bool,
    /// data augmentation, on the training images only (segmentation: flips and rotation are applied to image and mask together)
    #[serde(default)]
    pub aug_hflip: bool,
    #[serde(default)]
    pub aug_vflip: bool,
    /// random crop after padding by this many pixels (0 = off; classification only)
    #[serde(default)]
    pub aug_crop: usize,
    /// random rotation up to ± this many degrees (0 = off)
    #[serde(default)]
    pub aug_rotate: f64,
    /// random brightness / contrast / saturation (±20%)
    #[serde(default)]
    pub aug_color: bool,
    /// also save a checkpoint every this many minutes inside an epoch (0 = only at the end of each epoch; spot runs then use SPOT_CHECKPOINT_MINUTES)
    #[serde(default)]
    pub checkpoint_minutes: f64,
    /// EC2 spot: how many times a run interrupted by AWS is started again automatically
    #[serde(default = "default_max_resumes")]
    pub max_resumes: usize,
    /// part of the training data held out for per-epoch validation (0 = validate on the test data, as before); needed for early stopping
    #[serde(default)]
    pub val_fraction: f64,
    /// stop when the monitored validation value has not improved for `patience` epochs; the best epoch's weights become the model
    #[serde(default)]
    pub early_stop: bool,
    #[serde(default = "default_patience")]
    pub patience: usize,
    /// "val_loss" (lower is better) | "val_acc" | "val_miou" (higher is better)
    #[serde(default = "default_monitor")]
    pub monitor: String,
}

fn default_patience() -> usize {
    5
}

fn default_monitor() -> String {
    "val_loss".into()
}

fn default_max_resumes() -> usize {
    20
}

/// checkpoint interval for a run on a spot instance when none is set: at most this much training is lost to a hard interruption
pub const SPOT_CHECKPOINT_MINUTES: f64 = 10.0;

impl Default for TrainSettings {
    fn default() -> Self {
        TrainSettings { epochs: 10, batch_size: 64, lr: 0.001, optimizer: "adam".into(), seed: 1, amp: true, limit_train: 0, max_hours: 6.0, stop_instance: true,
            aug_hflip: false, aug_vflip: false, aug_crop: 0, aug_rotate: 0.0, aug_color: false, checkpoint_minutes: 0.0, max_resumes: default_max_resumes(),
            val_fraction: 0.0, early_stop: false, patience: default_patience(), monitor: default_monitor() }
    }
}

impl TrainSettings {
    pub fn clean(mut self) -> Self {
        self.epochs = self.epochs.clamp(1, 1000);
        self.batch_size = self.batch_size.clamp(1, 8192);
        self.lr = if self.lr.is_finite() { self.lr.clamp(1e-7, 1.0) } else { 0.001 };
        if self.optimizer != "sgd" {
            self.optimizer = "adam".into();
        }
        self.max_hours = if self.max_hours.is_finite() { self.max_hours.clamp(0.1, 72.0) } else { 6.0 };
        self.aug_crop = self.aug_crop.min(16);
        self.aug_rotate = if self.aug_rotate.is_finite() { self.aug_rotate.clamp(0.0, 45.0) } else { 0.0 };
        self.checkpoint_minutes = if self.checkpoint_minutes.is_finite() { self.checkpoint_minutes.clamp(0.0, 240.0) } else { 0.0 };
        self.max_resumes = self.max_resumes.min(100);
        self.val_fraction = if self.val_fraction.is_finite() { self.val_fraction.clamp(0.0, 0.5) } else { 0.0 };
        if self.early_stop && self.val_fraction < 0.01 {
            self.val_fraction = 0.1; // early stopping must not look at the test data
        }
        self.patience = self.patience.clamp(1, 100);
        if !["val_loss", "val_acc", "val_miou"].contains(&self.monitor.as_str()) {
            self.monitor = default_monitor();
        }
        self
    }
}

fn act(name: &str) -> Option<&'static str> {
    Some(match name {
        "relu" => "nn.ReLU()",
        "leaky_relu" => "nn.LeakyReLU(0.01)",
        "gelu" => "nn.GELU()",
        "silu" => "nn.SiLU()",
        "sigmoid" => "nn.Sigmoid()",
        "tanh" => "nn.Tanh()",
        _ => return None,
    })
}

fn k2(v: [usize; 2]) -> String {
    if v[0] == v[1] {
        v[0].to_string()
    } else {
        format!("({}, {})", v[0], v[1])
    }
}

/// small modules for transformers (written into train.py only when the design uses them)
const TOKEN_MODULES: &str = r#"class ToTokens(nn.Module):
    """an image (C, H, W) as a sequence: "patches" -> H*W tokens of C values; "rows" -> H tokens of C*W values"""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return x.flatten(2).transpose(1, 2) if self.mode == "patches" else x.permute(0, 2, 1, 3).flatten(2)


class ClsPos(nn.Module):
    """a learnable class token in front (optional) and a learnable position embedding added to every token"""
    def __init__(self, tokens, dim, cls=True):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls else None
        self.pos = nn.Parameter(torch.randn(1, tokens + (1 if cls else 0), dim) * 0.02)

    def forward(self, x):
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos


class SelfAttention(nn.Module):
    """multi-head self-attention over the tokens"""
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


class TokenPool(nn.Module):
    """one vector per image: the class token ("cls") or the mean of all tokens ("mean")"""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return x[:, 0] if self.mode == "cls" else x.mean(dim=1)


"#;

/// the modules of one node as Python source (empty for Input / Add / Concat, which are written in forward)
fn modules(n: &Node, a: &Analysis, classes: usize) -> Vec<String> {
    let r = a.get(&n.id).unwrap();
    let p = &n.params;
    let inp = r.inputs.first().cloned().unwrap_or_default();
    let mut m = vec![];
    match n.kind.as_str() {
        "conv2d" | "dwconv2d" => {
            let c = inp[0];
            let out = r.out.as_ref().unwrap()[0];
            let mut args = vec![format!("in_channels={c}"), format!("out_channels={out}"), format!("kernel_size={}", k2(pair(p, "kernel", 3, 1, 32)))];
            let st = r.stride.unwrap();
            if st != [1, 1] {
                args.push(format!("stride={}", k2(st)));
            }
            let pd = r.padding.unwrap();
            if pd != [0, 0] {
                args.push(format!("padding={}", k2(pd)));
            }
            if n.kind == "conv2d" {
                let d = int(p, "dilation", 1, 1, 6);
                if d != 1 {
                    args.push(format!("dilation={d}"));
                }
            } else {
                args.push(format!("groups={c}"));
            }
            if !p.get("bias").and_then(|v| v.as_bool()).unwrap_or(true) {
                args.push("bias=False".into());
            }
            m.push(format!("nn.Conv2d({})", args.join(", ")));
            m.extend(act(activation(p, "activation")).map(String::from));
        }
        "convtranspose2d" => {
            let mut args = vec![format!("in_channels={}", inp[0]), format!("out_channels={}", r.out.as_ref().unwrap()[0]),
                format!("kernel_size={}", k2(pair(p, "kernel", 2, 1, 32))), format!("stride={}", k2(r.stride.unwrap()))];
            let pd = r.padding.unwrap();
            if pd != [0, 0] {
                args.push(format!("padding={}", k2(pd)));
            }
            let op = int(p, "output_padding", 0, 0, 7);
            if op != 0 {
                args.push(format!("output_padding={op}"));
            }
            m.push(format!("nn.ConvTranspose2d({})", args.join(", ")));
            m.extend(act(activation(p, "activation")).map(String::from));
        }
        "pool2d" => {
            let cls = if shapes::text(p, "mode", &["max", "avg"], "max") == "avg" { "nn.AvgPool2d" } else { "nn.MaxPool2d" };
            let mut args = vec![format!("kernel_size={}", k2(pair(p, "size", 2, 1, 8))), format!("stride={}", k2(r.stride.unwrap()))];
            let pd = r.padding.unwrap()[0];
            if pd != 0 {
                args.push(format!("padding={pd}"));
            }
            m.push(format!("{cls}({})", args.join(", ")));
        }
        "global_pool" => {
            m.push(if shapes::text(p, "mode", &["avg", "max"], "avg") == "max" { "nn.AdaptiveMaxPool2d(1)".into() } else { "nn.AdaptiveAvgPool2d(1)".into() });
            m.push("nn.Flatten()".into());
        }
        "upsample" => {
            let s = int(p, "scale", 2, 2, 8);
            m.push(if shapes::text(p, "mode", &["nearest", "bilinear"], "nearest") == "bilinear" {
                format!("nn.Upsample(scale_factor={s}, mode=\"bilinear\", align_corners=False)")
            } else {
                format!("nn.Upsample(scale_factor={s}, mode=\"nearest\")")
            });
        }
        "batchnorm" => m.push(if inp.len() == 3 { format!("nn.BatchNorm2d({})", inp[0]) } else { format!("nn.BatchNorm1d({})", inp[0]) }),
        "layernorm" => m.push(if inp.len() == 3 { format!("nn.GroupNorm(1, {})   # LayerNorm over (channels, height, width)", inp[0]) } else { format!("nn.LayerNorm({})", inp.last().unwrap()) }),
        "to_tokens" => m.push(format!("ToTokens({:?})", shapes::text(p, "mode", &["patches", "rows"], "patches"))),
        "cls_pos" => {
            let cls = p.get("cls").and_then(|v| v.as_bool()).unwrap_or(true);
            m.push(format!("ClsPos(tokens={}, dim={}, cls={})", inp[0], inp[1], if cls { "True" } else { "False" }));
        }
        "attention" => {
            let drop = p.get("dropout").and_then(|v| v.as_f64()).unwrap_or(0.0).clamp(0.0, 0.9);
            m.push(format!("SelfAttention(dim={}, heads={}, dropout={})", inp[1], int(p, "heads", 4, 1, 64), (drop * 100.0).round() / 100.0));
        }
        "token_pool" => m.push(format!("TokenPool({:?})", shapes::text(p, "mode", &["cls", "mean"], "cls"))),
        "dropout" => {
            let rate = p.get("rate").and_then(|v| v.as_f64()).unwrap_or(0.25).clamp(0.0, 0.95);
            m.push(format!("nn.Dropout(p={})", (rate * 100.0).round() / 100.0));
        }
        "flatten" => m.push("nn.Flatten()".into()),
        "activation" => m.push(act(activation(p, "fn")).unwrap_or("nn.Identity()").into()),
        "dense" => {
            m.push(format!("nn.Linear(in_features={}, out_features={})", inp.last().unwrap(), int(p, "units", 128, 1, 1 << 20)));
            m.extend(act(activation(p, "activation")).map(String::from));
        }
        "output" => m.push(format!("nn.Linear(in_features={}, out_features={classes})", inp[0])),
        "seg_output" => m.push(format!("nn.Conv2d(in_channels={}, out_channels={classes}, kernel_size=1)", inp[0])),
        _ => {}
    }
    m
}

fn fmt(s: &[usize]) -> String {
    format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

/// The `Net` class, LAYERS (module name, label, is a convolution) and EXPECTED (module name -> output shape). Fails for a design with problems.
pub fn model_code(g: &Graph, a: &Analysis) -> Result<String> {
    if !a.ok {
        bail!("the design has problems; fix the red blocks first");
    }
    let order = a.order();
    let var: std::collections::HashMap<&str, String> = order.iter().enumerate().map(|(i, id)| (*id, format!("t{i}"))).collect();
    let mut init = vec![];
    let mut fwd = vec![];
    let mut layers = vec![];
    let mut expected = serde_json::Map::new();
    expected.insert("input".into(), json!(a.input));
    let mut result_var = String::new();
    for (i, id) in order.iter().enumerate() {
        let n = g.node(id).unwrap();
        let r = a.get(id).unwrap();
        let ins: Vec<String> = g.inputs_of(id).iter().filter_map(|s| var.get(s).cloned()).collect();
        let v = &var[id];
        let label = format!("{i}. {}", r.title);
        match n.kind.as_str() {
            "input" => fwd.push(format!("        {v} = x")),
            "add" => fwd.push(format!("        {v} = {}   # {label} -> {}", ins.join(" + "), fmt(r.out.as_ref().unwrap()))),
            "multiply" => {
                let shp: Vec<&Vec<usize>> = r.inputs.iter().collect();
                let side = |i: usize| if shp[i].len() == 1 && shp[1 - i].len() == 3 { format!("{}[:, :, None, None]", ins[i]) } else { ins[i].clone() };
                fwd.push(format!("        {v} = {} * {}   # {label} -> {}", side(0), side(1), fmt(r.out.as_ref().unwrap())));
            }
            "concat" => fwd.push(format!("        {v} = torch.cat([{}], dim=1)   # {label} -> {}", ins.join(", "), fmt(r.out.as_ref().unwrap()))),
            _ => {
                let mods = modules(n, a, a.classes);
                let name = format!("l{i}");
                let note = match n.kind.as_str() {
                    "output" => " (one score per class; the loss applies softmax)",
                    "seg_output" => " (a score per class for every pixel)",
                    _ => "",
                };
                init.push(format!("        # {label}: {} -> {}, {} parameters{note}", r.inputs.iter().map(|s| fmt(s)).collect::<Vec<_>>().join(" + "),
                    fmt(r.out.as_ref().unwrap()), r.params));
                init.push(if mods.len() == 1 { format!("        self.{name} = {}", mods[0]) } else { format!("        self.{name} = nn.Sequential(\n            {},\n        )", mods.join(",\n            ")) });
                fwd.push(format!("        {v} = self.{name}({})", ins.join(", ")));
                let conv = matches!(n.kind.as_str(), "conv2d" | "dwconv2d" | "convtranspose2d");
                layers.push(json!([name, label, conv]));
                expected.insert(name, json!(r.out));
            }
        }
        if n.kind == "output" || n.kind == "seg_output" {
            result_var = v.clone();
        }
    }
    let mut s = String::new();
    if g.nodes.iter().any(|n| order.contains(&n.id.as_str()) && matches!(n.kind.as_str(), "to_tokens" | "cls_pos" | "attention" | "token_pool")) {
        s += TOKEN_MODULES;
    }
    s += "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n";
    s += &init.join("\n");
    s += "\n\n    def forward(self, x):\n";
    s += &fwd.join("\n");
    s += &format!("\n        return {result_var}\n\n\n");
    s += &format!("# {} trainable parameters\n", a.total_params);
    s += &format!("LAYERS = {}   # module name, label, is a convolution\n", py_json(&Value::Array(layers)));
    s += &format!("EXPECTED = {}   # output shape of each module (without the batch)\n", py_json(&Value::Object(expected)));
    Ok(s)
}

/// a JSON value as a readable Python literal (strings use JSON escapes, which are valid Python string escapes)
fn py_json(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), py_json(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// the whole standalone train.py
pub fn train_script(g: &Graph, ds: &Dataset, a: &Analysis, t: &TrainSettings, title: &str) -> Result<String> {
    if let Some(p) = ds.problems().first() {
        bail!("{p}");
    }
    let model = model_code(g, a)?;
    let ds_json = serde_json::to_value(ds)?;
    let title: String = title.chars().filter(|c| !matches!(c, '"' | '\\' | '\n' | '\r')).collect();
    let t_json = json!({"epochs": t.epochs, "batch_size": t.batch_size, "lr": t.lr, "optimizer": t.optimizer, "seed": t.seed, "amp": t.amp,
        "limit_train": t.limit_train, "num_workers": 2, "task": ds.task, "checkpoint_minutes": t.checkpoint_minutes,
        "val_fraction": t.val_fraction, "early_stop": t.early_stop, "patience": t.patience, "monitor": t.monitor,
        "aug": {"hflip": t.aug_hflip, "vflip": t.aug_vflip, "crop": t.aug_crop, "rotate": t.aug_rotate, "color": t.aug_color}});
    let head = format!(
        "\"\"\"{title}: generated by NN Builder {}. Run: python train.py\n\
         Needs torch and torchvision{}{}. Writes metrics.jsonl, checkpoint.pt, model.pt and results.json next to this file\n\
         (in /kaggle/working on Kaggle). A run that is started again resumes from checkpoint.pt.\"\"\"\n\
         import json\n\nimport torch\nimport torch.nn as nn\nimport torch.utils.data\n\n\
         DATASET = {}\nTRAIN = {}\nCLASSES = {}\n\n\n",
        env!("CARGO_PKG_VERSION"),
        if ds.source == "huggingface" { ", datasets and pillow" } else { "" },
        if ds.source == "kaggle" { " (and the kaggle package with a token when not on Kaggle)" } else { "" },
        py_json(&ds_json),
        py_json(&t_json),
        py_json(&json!(ds.class_names())),
    );
    Ok(head + &model + "\n\n" + HARNESS)
}

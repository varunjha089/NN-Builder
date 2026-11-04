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
}

impl Default for TrainSettings {
    fn default() -> Self {
        TrainSettings { epochs: 10, batch_size: 64, lr: 0.001, optimizer: "adam".into(), seed: 1, amp: true, limit_train: 0, max_hours: 6.0, stop_instance: true }
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
            let mut args = vec![format!("in_channels={c}"), format!("out_channels={out}"), format!("kernel_size={}", k2(pair(p, "kernel", 3, 1, 15)))];
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
                format!("kernel_size={}", k2(pair(p, "kernel", 2, 1, 15))), format!("stride={}", k2(r.stride.unwrap()))];
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
        "layernorm" => m.push(if inp.len() == 3 { format!("nn.GroupNorm(1, {})   # LayerNorm over (channels, height, width)", inp[0]) } else { format!("nn.LayerNorm({})", inp[0]) }),
        "dropout" => {
            let rate = p.get("rate").and_then(|v| v.as_f64()).unwrap_or(0.25).clamp(0.0, 0.95);
            m.push(format!("nn.Dropout(p={})", (rate * 100.0).round() / 100.0));
        }
        "flatten" => m.push("nn.Flatten()".into()),
        "activation" => m.push(act(activation(p, "fn")).unwrap_or("nn.Identity()").into()),
        "dense" => {
            m.push(format!("nn.Linear(in_features={}, out_features={})", inp[0], int(p, "units", 128, 1, 1 << 20)));
            m.extend(act(activation(p, "activation")).map(String::from));
        }
        "output" => m.push(format!("nn.Linear(in_features={}, out_features={classes})", inp[0])),
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
            "concat" => fwd.push(format!("        {v} = torch.cat([{}], dim=1)   # {label} -> {}", ins.join(", "), fmt(r.out.as_ref().unwrap()))),
            _ => {
                let mods = modules(n, a, a.classes);
                let name = format!("l{i}");
                let note = if n.kind == "output" { " (one score per class; the loss applies softmax)" } else { "" };
                init.push(format!("        # {label}: {} -> {}, {} parameters{note}", r.inputs.iter().map(|s| fmt(s)).collect::<Vec<_>>().join(" + "),
                    fmt(r.out.as_ref().unwrap()), r.params));
                init.push(if mods.len() == 1 { format!("        self.{name} = {}", mods[0]) } else { format!("        self.{name} = nn.Sequential(\n            {},\n        )", mods.join(",\n            ")) });
                fwd.push(format!("        {v} = self.{name}({})", ins.join(", ")));
                let conv = matches!(n.kind.as_str(), "conv2d" | "dwconv2d" | "convtranspose2d");
                layers.push(json!([name, label, conv]));
                expected.insert(name, json!(r.out));
            }
        }
        if n.kind == "output" {
            result_var = v.clone();
        }
    }
    let mut s = String::from("class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n");
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
        "limit_train": t.limit_train, "num_workers": 2});
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

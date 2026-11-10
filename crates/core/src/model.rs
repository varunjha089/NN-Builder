//! The design: a graph of layer nodes joined by edges (tensors), as drawn on the canvas. Saved as `model.json` in the project.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Node {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    /// the block's label in the design text (dsl.rs): "5", or "6.2.b" inside a repeat / function; set by the text sync
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Graph {
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub edges: Vec<Edge>,
    /// the design as text (dsl.rs), kept in step with the blocks: written by the user (with functions and loops) or printed from the canvas
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// the design's own PyTorch code: nn.Module classes that Custom blocks name (custom.rs); pasted into train.py as written
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub python: String,
}

/// Every node kind the canvas offers, with its title, its default parameters and how many inputs it takes (None = two or more).
pub const KINDS: &[(&str, &str)] = &[
    ("input", "Input"),
    ("conv2d", "Conv2D"),
    ("dwconv2d", "Depthwise Conv2D"),
    ("convtranspose2d", "ConvTranspose2D"),
    ("pool2d", "Pool"),
    ("global_pool", "Global pool"),
    ("upsample", "Upsample"),
    ("batchnorm", "BatchNorm"),
    ("layernorm", "LayerNorm"),
    ("groupnorm", "GroupNorm"),
    ("dropout", "Dropout"),
    ("flatten", "Flatten"),
    ("dense", "Dense"),
    ("activation", "Activation"),
    ("add", "Add"),
    ("concat", "Concat"),
    ("multiply", "Multiply"),
    ("subtract", "Subtract"),
    ("to_tokens", "To tokens"),
    ("cls_pos", "Class token + position"),
    ("attention", "Self-attention"),
    ("token_pool", "Token pool"),
    ("cross_attention", "Cross-attention"),
    ("rmsnorm", "RMSNorm"),
    ("swiglu", "SwiGLU (gated MLP)"),
    ("conv3d", "Conv3D (time × space)"),
    ("temporal_pool", "Temporal pool"),
    ("to_sequence", "To sequence"),
    ("lstm", "LSTM"),
    ("gru", "GRU"),
    ("conv1d", "Conv1D (over time)"),
    ("output", "Output"),
    ("seg_output", "Segmentation output"),
    ("detect_output", "Detection output"),
    ("exit_output", "Early exit"),
    ("moe", "Mixture of experts"),
    ("custom", "Custom (PyTorch)"),
    ("shared", "Shared weights"),
];

pub const ACTIVATIONS: &[&str] = &["relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "none"];

pub fn title(kind: &str) -> &'static str {
    KINDS.iter().find(|(k, _)| *k == kind).map(|(_, t)| *t).unwrap_or("?")
}

pub fn defaults(kind: &str) -> Value {
    match kind {
        "conv2d" => json!({"filters": 32, "kernel": [3, 3], "stride": [1, 1], "padding": "same", "dilation": 1, "activation": "relu", "bias": true}),
        "dwconv2d" => json!({"multiplier": 1, "kernel": [3, 3], "stride": [1, 1], "padding": "same", "activation": "relu", "bias": true}),
        "convtranspose2d" => json!({"filters": 32, "kernel": [2, 2], "stride": [2, 2], "padding": 0, "output_padding": 0, "activation": "relu"}),
        "pool2d" => json!({"mode": "max", "size": [2, 2], "stride": "size", "padding": 0}),
        "global_pool" => json!({"mode": "avg"}),
        "upsample" => json!({"scale": 2, "mode": "nearest"}),
        "dropout" => json!({"rate": 0.25}),
        "groupnorm" => json!({"groups": 8}),
        // lora: the rank of a low-rank adapter (0 = none): the layer's own weights are frozen and only the adapter learns
        "dense" => json!({"units": 128, "activation": "relu", "lora": 0}),
        "activation" => json!({"fn": "relu"}),
        "to_tokens" => json!({"mode": "patches"}),
        // pos: "learned" (ViT) | "sinusoidal" (fixed, the original Transformer's) | "none" (e.g. with rotary attention)
        "cls_pos" => json!({"cls": true, "pos": "learned"}),
        // causal: each token sees only itself and the tokens before it; rotary: positions rotate the queries and keys (RoPE)
        "attention" => json!({"heads": 4, "dropout": 0.0, "causal": false, "rotary": false, "lora": 0}),
        // experts: small MLPs (units hidden); a router sends every token to its top_k experts (Switch / Mixtral). GShard: capacity = the
        // most tokens an expert takes, × its fair share (0 = no limit); random_second = the second expert only by chance (2 × its weight)
        "moe" => json!({"experts": 4, "top_k": 2, "units": 128, "capacity": 0, "random_second": false}),
        // an extra classifier part way through the network, trained with the main task's labels (its loss × weight)
        "exit_output" => json!({"weight": 0.3}),
        // the first input's tokens ask (queries), the second input's tokens answer (keys and values)
        "cross_attention" => json!({"heads": 4, "dropout": 0.0}),
        // a gated MLP (LLaMA): out = W2 (silu(W1 x) * W3 x), back to the input's width; units = the hidden width
        "swiglu" => json!({"units": 128}),
        "token_pool" => json!({"mode": "cls"}),
        // time series (channels, dates, height, width): a convolution over dates and pixels; tkernel = dates per window
        "conv3d" => json!({"filters": 16, "kernel": [3, 3], "tkernel": 3, "stride": [1, 1], "padding": "same", "activation": "relu", "bias": true}),
        // (C, T, H, W) -> (C, H, W): the dates of every pixel combined (mean, max, or attention weights learnt per pixel)
        "temporal_pool" => json!({"mode": "attention"}),
        "lstm" => json!({"units": 64, "bidirectional": false}),
        "gru" => json!({"units": 64, "bidirectional": false}),
        "conv1d" => json!({"filters": 64, "kernel": 3, "stride": 1, "padding": "same", "activation": "relu", "bias": true}),
        // 0 = all channels; 1 / 2 = one band group of a fused dataset (Sentinel-2 + Sentinel-1: one Input per sensor)
        "input" => json!({"group": 0}),
        // the name of a class in the design's PyTorch code (Graph::python)
        "custom" => json!({"class": ""}),
        // which output of a multi-task network: 0 = the main task, 1, 2, ... = the Dataset tab's extra outputs
        "output" => json!({"head": 0}),
        // a - b, or |a - b| (Siamese change detection: the difference of the two branches)
        "subtract" => json!({"abs": false}),
        // runs another block's layer with that block's weights (Siamese / twin networks): "of" = that block's id
        "shared" => json!({"of": ""}),
        _ => json!({}),
    }
}

/// Number of inputs a kind takes: Some(0) for Input, Some(1) for ordinary layers, None for Add/Concat (two or more).
pub fn arity(kind: &str) -> Option<usize> {
    match kind {
        "input" => Some(0),
        "add" | "concat" | "multiply" | "subtract" | "cross_attention" => None,
        _ => Some(1),
    }
}

impl Graph {
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
    /// sources of the edges into `id`, in edge order (the order matters for Concat)
    pub fn inputs_of(&self, id: &str) -> Vec<&str> {
        self.edges.iter().filter(|e| e.target == id).map(|e| e.source.as_str()).collect()
    }
    /// the inputs of every block at once (in edge order), for code that looks at many blocks (inputs_of searches all edges each time)
    pub fn input_map(&self) -> std::collections::HashMap<&str, Vec<&str>> {
        let mut m: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
        for e in &self.edges {
            m.entry(e.target.as_str()).or_default().push(e.source.as_str());
        }
        m
    }
    pub fn outputs_of(&self, id: &str) -> Vec<&str> {
        self.edges.iter().filter(|e| e.source == id).map(|e| e.target.as_str()).collect()
    }
}

/// place the nodes in columns by depth (longest path from the inputs), left to right; branches are stacked in a column. The nodes must be
/// in computation order (every input before its user).
pub fn layout(g: &mut Graph) {
    use std::collections::HashMap;
    let mut depth: HashMap<String, usize> = HashMap::new();
    for n in &g.nodes {
        let d = g.edges.iter().filter(|e| e.target == n.id).filter_map(|e| depth.get(&e.source)).max().map(|d| d + 1).unwrap_or(0);
        depth.insert(n.id.clone(), d);
    }
    let mut count: HashMap<usize, usize> = HashMap::new();
    for d in depth.values() {
        *count.entry(*d).or_default() += 1;
    }
    let widest = count.values().copied().max().unwrap_or(1);
    let mut per_col: HashMap<usize, usize> = HashMap::new();
    for n in &mut g.nodes {
        let d = depth[&n.id];
        let i = per_col.entry(d).or_default();
        n.x = 60.0 + d as f64 * 300.0;
        n.y = 60.0 + (widest - count[&d]) as f64 * 75.0 + *i as f64 * 150.0;
        *i += 1;
    }
}

/// A small CNN as the design of a new project.
pub fn starter() -> Graph {
    let layers: Vec<(&str, Value)> = vec![
        ("input", json!({})),
        ("conv2d", json!({"filters": 16})),
        ("pool2d", json!({})),
        ("conv2d", json!({"filters": 32})),
        ("pool2d", json!({})),
        ("flatten", json!({})),
        ("dense", json!({"units": 64})),
        ("dropout", json!({})),
        ("output", json!({})),
    ];
    let mut g = Graph::default();
    for (i, (kind, over)) in layers.into_iter().enumerate() {
        let mut p = defaults(kind).as_object().cloned().unwrap_or_default();
        for (k, v) in over.as_object().unwrap() {
            p.insert(k.clone(), v.clone());
        }
        g.nodes.push(Node { id: format!("n{}", i + 1), kind: kind.into(), params: p, x: 40.0 + 300.0 * i as f64, y: 80.0, ..Default::default() });
        if i > 0 {
            g.edges.push(Edge { id: format!("e{}", i), source: format!("n{i}"), target: format!("n{}", i + 1) });
        }
    }
    g
}

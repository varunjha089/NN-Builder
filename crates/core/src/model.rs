//! The design: a graph of layer nodes joined by edges (tensors), as drawn on the canvas. Saved as `model.json` in the project.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
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
    ("dropout", "Dropout"),
    ("flatten", "Flatten"),
    ("dense", "Dense"),
    ("activation", "Activation"),
    ("add", "Add"),
    ("concat", "Concat"),
    ("output", "Output"),
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
        "dense" => json!({"units": 128, "activation": "relu"}),
        "activation" => json!({"fn": "relu"}),
        _ => json!({}),
    }
}

/// Number of inputs a kind takes: Some(0) for Input, Some(1) for ordinary layers, None for Add/Concat (two or more).
pub fn arity(kind: &str) -> Option<usize> {
    match kind {
        "input" => Some(0),
        "add" | "concat" => None,
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
    pub fn outputs_of(&self, id: &str) -> Vec<&str> {
        self.edges.iter().filter(|e| e.source == id).map(|e| e.target.as_str()).collect()
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
        g.nodes.push(Node { id: format!("n{}", i + 1), kind: kind.into(), params: p, x: 80.0, y: 40.0 + 110.0 * i as f64 });
        if i > 0 {
            g.edges.push(Edge { id: format!("e{}", i), source: format!("n{i}"), target: format!("n{}", i + 1) });
        }
    }
    g
}

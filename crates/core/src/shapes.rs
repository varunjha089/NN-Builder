//! The shape check: every node's input and output shape, its parameters, receptive field and plain-words problems, computed from the graph
//! and the dataset without PyTorch. The code generator (codegen.rs) uses the same numbers, and the GitHub Actions job compares them with a
//! real PyTorch forward pass.
//!
//! Output size of a convolution or pooling window, per side (PyTorch, ceil_mode = false):
//!     out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1
//! Transposed convolution:  out = (in - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1

use crate::dataset::Dataset;
use crate::model::{self, Graph, Node};
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

pub const BIG_DENSE: u64 = 5_000_000;

fn is_zero(v: &u64) -> bool {
    *v == 0
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct NodeResult {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub inputs: Vec<Vec<usize>>,
    pub out: Option<Vec<usize>>,
    pub params: u64,
    /// of `params`, the ones that are not trained (the block is frozen, or a LoRA layer's own weights)
    #[serde(skip_serializing_if = "is_zero")]
    pub frozen: u64,
    /// ok | warn | error | blocked (an input has an error) | unconnected (not reached from Input: ignored)
    pub status: String,
    pub messages: Vec<String>,
    pub formula: Vec<String>,
    pub rf: Option<[usize; 2]>,
    /// the padding and stride actually used (conv/pool), for the code and the sliding preview
    pub padding: Option<[usize; 2]>,
    pub stride: Option<[usize; 2]>,
    /// multiply-accumulate operations for one image (convolutions, dense layers, attention; normalisation, activations, pooling and
    /// joins are counted as 0)
    pub macs: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    pub input: Vec<usize>,
    pub classes: usize,
    /// reached nodes in computation order, then the unconnected ones
    pub nodes: Vec<NodeResult>,
    /// edge id -> the shape it carries
    pub edges: BTreeMap<String, Vec<usize>>,
    pub total_params: u64,
    /// of total_params, the ones that are not trained (frozen blocks, LoRA layers' own weights)
    pub frozen_params: u64,
    pub total_macs: u64,
    pub problems: Vec<String>,
    pub ok: bool,
    /// the dataset's extra outputs (multi-task), for the Output block's head choice: head k = head_names[k - 1]
    pub head_names: Vec<String>,
}

impl Analysis {
    pub fn get(&self, id: &str) -> Option<&NodeResult> {
        self.nodes.iter().find(|n| n.id == id)
    }
    /// ids in computation order (only nodes that were computed)
    pub fn order(&self) -> Vec<&str> {
        self.nodes.iter().filter(|n| n.out.is_some()).map(|n| n.id.as_str()).collect()
    }
}

// ---------------------------------------------------------------- reading parameters (the canvas sends them; never trust them blindly)
/// the highest number of filters (Conv2D, ConvTranspose2D) and attention heads: enough for the largest published models (ViT-22B has
/// 6144 values per token and 48 heads; LLM-sized widths reach 16384+)
pub const MAX_FILTERS: i64 = 65536;
pub const MAX_HEADS: i64 = 512;

/// the allowed range of every number setting, per block kind (the same limits the shape engine and the code generator read)
pub fn limits(kind: &str) -> &'static [(&'static str, i64, i64)] {
    match kind {
        "conv2d" => &[("filters", 1, MAX_FILTERS), ("kernel", 1, 32), ("stride", 1, 32), ("dilation", 1, 6), ("padding", 0, 10)],
        "dwconv2d" => &[("multiplier", 1, 8), ("kernel", 1, 32), ("stride", 1, 32), ("padding", 0, 10)],
        "convtranspose2d" => &[("filters", 1, MAX_FILTERS), ("kernel", 1, 32), ("stride", 1, 32), ("padding", 0, 10), ("output_padding", 0, 7)],
        "pool2d" => &[("size", 1, 8), ("stride", 1, 32), ("padding", 0, 4)],
        "upsample" => &[("scale", 2, 8)],
        "groupnorm" => &[("groups", 1, MAX_FILTERS)],
        "attention" | "cross_attention" => &[("heads", 1, MAX_HEADS)],
        "swiglu" => &[("units", 1, 1 << 20)],
        "dense" => &[("units", 1, 1 << 20)],
        "conv3d" => &[("filters", 1, MAX_FILTERS), ("kernel", 1, 15), ("tkernel", 1, 15), ("stride", 1, 8)],
        "lstm" | "gru" => &[("units", 1, 1 << 16)],
        "conv1d" => &[("filters", 1, MAX_FILTERS), ("kernel", 1, 31), ("stride", 1, 16)],
        _ => &[],
    }
}

/// the allowed range of the fraction settings (dropout rates)
pub fn float_limits(kind: &str) -> &'static [(&'static str, f64, f64)] {
    match kind {
        "dropout" => &[("rate", 0.0, 0.95)],
        "attention" | "cross_attention" => &[("dropout", 0.0, 0.9)],
        _ => &[],
    }
}

/// a number setting outside its range: the message (the value would otherwise be changed silently, giving a different network)
fn out_of_range(node: &Node) -> Option<String> {
    for (key, lo, hi) in float_limits(&node.kind) {
        if let Some(v) = node.params.get(*key).and_then(|v| v.as_f64()) {
            if v < *lo || v > *hi {
                return Some(format!("{key} is {v} but must be from {lo} to {hi}."));
            }
        }
    }
    for (key, lo, hi) in limits(&node.kind) {
        let vals: Vec<f64> = match node.params.get(*key) {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_f64()).collect(),
            Some(v) => v.as_f64().into_iter().collect(),
            None => vec![],
        };
        if let Some(v) = vals.iter().find(|v| v.round() < *lo as f64 || v.round() > *hi as f64) {
            return Some(format!("{key} is {v} but must be from {lo} to {hi}."));
        }
    }
    None
}
pub fn int(p: &Map<String, Value>, key: &str, default: i64, lo: i64, hi: i64) -> usize {
    // (a pair where one number is meant: its first value, never the default in silence)
    let v = p.get(key).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
        .or_else(|| v.as_array().and_then(|a| a.first()).and_then(|x| x.as_f64()))).map(|f| f.round() as i64).unwrap_or(default);
    v.clamp(lo, hi) as usize
}

pub fn pair(p: &Map<String, Value>, key: &str, default: usize, lo: i64, hi: i64) -> [usize; 2] {
    match p.get(key) {
        Some(Value::Array(a)) if a.len() == 2 => {
            let one = |v: &Value| v.as_f64().map(|f| f.round() as i64).unwrap_or(default as i64).clamp(lo, hi) as usize;
            [one(&a[0]), one(&a[1])]
        }
        Some(_) => {
            let v = int(p, key, default as i64, lo, hi);
            [v, v]
        }
        None => [default, default],
    }
}

pub fn text<'a>(p: &'a Map<String, Value>, key: &str, allowed: &[&'a str], default: &'a str) -> &'a str {
    let v = p.get(key).and_then(|v| v.as_str()).unwrap_or(default);
    allowed.iter().find(|a| **a == v).copied().unwrap_or(default)
}

pub fn activation(p: &Map<String, Value>, key: &str) -> &'static str {
    let v = p.get(key).and_then(|v| v.as_str()).unwrap_or("relu");
    model::ACTIVATIONS.iter().find(|a| **a == v).copied().unwrap_or("relu")
}

/// conv padding: "same" | "valid" | a number
pub enum Pad {
    Same,
    Num(usize),
}

pub fn padding(p: &Map<String, Value>) -> Pad {
    match p.get("padding") {
        Some(Value::String(s)) if s == "same" => Pad::Same,
        Some(Value::String(s)) if s == "valid" => Pad::Num(0),
        _ => Pad::Num(int(p, "padding", 0, 0, 10)),
    }
}

fn fmt(s: &[usize]) -> String {
    format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

/// one side of a window: (out, in + 2p - effective kernel, effective kernel); out 0 means "does not fit"
fn window(n: usize, k: usize, s: usize, p: usize, d: usize) -> (usize, i64, usize) {
    let keff = d * (k - 1) + 1;
    let num = n as i64 + 2 * p as i64 - keff as i64;
    (if num >= 0 { (num as usize) / s + 1 } else { 0 }, num, keff)
}

// ---------------------------------------------------------------- the analysis
pub fn analyze(g: &Graph, ds: &Dataset) -> Analysis {
    let input_shape = ds.input_shape();
    let classes = ds.class_names().len();
    let py = crate::custom::classes(&g.python);
    // regression: the "classes" are the target columns; detection may look for a single kind of object
    let need = if ds.task == "regress" || ds.task == "detect" { 1 } else { 2 };
    let mut problems: Vec<String> = vec![];
    let ids: HashSet<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    let edges: Vec<_> = g.edges.iter().filter(|e| ids.contains(e.source.as_str()) && ids.contains(e.target.as_str()) && e.source != e.target).collect();
    // lookups by id (a design can have thousands of blocks: no search through all blocks or edges per block)
    let node_of: HashMap<&str, &Node> = g.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut out_edges: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_edges: HashMap<&str, Vec<&str>> = HashMap::new(); // in edge order (Concat joins in this order)
    for e in &edges {
        out_edges.entry(e.source.as_str()).or_default().push(e.target.as_str());
        in_edges.entry(e.target.as_str()).or_default().push(e.source.as_str());
    }
    let has_any_output: HashSet<&str> = g.edges.iter().map(|e| e.source.as_str()).collect();
    let no_targets: Vec<&str> = vec![];

    let inputs: Vec<&Node> = g.nodes.iter().filter(|n| n.kind == "input").collect();
    // which band group an Input takes: 0 = all channels; 1 / 2 = the first / second band group of a fused dataset (mid fusion)
    let group_of = |n: &Node| n.params.get("group").and_then(|v| v.as_u64()).unwrap_or(0);
    let groups: Vec<u64> = inputs.iter().map(|n| group_of(n)).collect();
    if inputs.is_empty() {
        problems.push("Add an Input block.".into());
    } else if inputs.len() > 1 {
        if !ds.fused() {
            problems.push("There must be exactly one Input block (two are for two band groups, e.g. Sentinel-2 + Sentinel-1: Dataset tab).".into());
        } else if groups.contains(&0) || groups.iter().enumerate().any(|(i, g)| groups[..i].contains(g)) {
            problems.push("With two Input blocks, one takes band group 1 and the other band group 2.".into());
        }
    } else if ds.fused() && groups[0] != 0 {
        problems.push(format!("The Input takes only band group {}: the other group is not used (add an Input for it, or take all bands).", groups[0]));
    }
    // reachable from the Inputs
    let mut reach: HashSet<&str> = HashSet::new();
    let mut q: VecDeque<&str> = if inputs.len() == 1 || ds.fused() { inputs.iter().map(|n| n.id.as_str()).collect() } else { inputs.iter().take(1).map(|n| n.id.as_str()).collect() };
    while let Some(id) = q.pop_front() {
        if reach.insert(id) {
            q.extend(out_edges.get(id).unwrap_or(&no_targets).iter().copied());
        }
    }
    // computation order (Kahn) over the reached part; a Shared-weights block also waits for the block whose weights it uses
    let shared_of = |n: &Node| n.params.get("of").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let mut after: HashMap<&str, Vec<&str>> = HashMap::new();
    for n in g.nodes.iter().filter(|n| n.kind == "shared" && reach.contains(n.id.as_str())) {
        let of = shared_of(n);
        if let Some(src) = reach.get(of.as_str()) {
            if *src != n.id.as_str() {
                after.entry(*src).or_default().push(n.id.as_str());
            }
        }
    }
    let mut indeg: HashMap<&str, usize> = reach.iter().map(|id| (*id, 0)).collect();
    for e in edges.iter().filter(|e| reach.contains(e.source.as_str()) && reach.contains(e.target.as_str())) {
        *indeg.get_mut(e.target.as_str()).unwrap() += 1;
    }
    for t in after.values().flatten() {
        *indeg.get_mut(t).unwrap() += 1;
    }
    let mut ready: VecDeque<&str> = g.nodes.iter().map(|n| n.id.as_str()).filter(|id| indeg.get(id) == Some(&0)).collect();
    let mut order: Vec<&str> = vec![];
    while let Some(id) = ready.pop_front() {
        order.push(id);
        for t in out_edges.get(id).unwrap_or(&no_targets).iter().copied().filter(|t| reach.contains(t)).chain(after.get(id).cloned().unwrap_or_default()) {
            let d = indeg.get_mut(t).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push_back(t);
            }
        }
    }
    let ordered: HashSet<&str> = order.iter().copied().collect();
    let looped: Vec<&str> = reach.iter().copied().filter(|id| !ordered.contains(id)).collect();
    if !looped.is_empty() {
        problems.push("The design has a loop (an arrow leads back to an earlier layer). Networks must flow one way.".into());
    }

    let mut results: HashMap<&str, NodeResult> = HashMap::new();
    let mut rfs: HashMap<&str, ([usize; 2], [usize; 2])> = HashMap::new();
    let is_out = |k: &str| k == "output" || k == "seg_output" || k == "detect_output";
    let n_out = order.iter().filter(|id| node_of.get(*id).map(|n| is_out(&n.kind)).unwrap_or(false)).count();
    // several outputs (multi-task): one Output per head, head 0 = the main task
    let head_of = |n: &Node| n.params.get("head").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let mut per_head: HashMap<usize, usize> = HashMap::new();
    for id in &order {
        if let Some(n) = node_of.get(id).filter(|n| is_out(&n.kind)) {
            *per_head.entry(head_of(n)).or_default() += 1;
        }
    }
    let mut total = 0u64;
    let mut frozen_total = 0u64;
    for id in &order {
        let node = node_of[id];
        let srcs: Vec<&str> = in_edges.get(id).unwrap_or(&no_targets).iter().copied().filter(|s| reach.contains(s)).collect();
        let mut r = NodeResult { id: id.to_string(), kind: node.kind.clone(), title: model::title(&node.kind).into(), status: "ok".into(), ..Default::default() };
        let blocked = srcs.iter().any(|s| results.get(s).map(|x| x.out.is_none()).unwrap_or(true));
        if node.kind == "input" {
            match ds.group_shape(group_of(node)) {
                Some(_) if group_of(node) == 0 && ds.separate_inputs() => {
                    err(&mut r, "The band groups have different sizes (Dataset tab: the second group's own size), so they cannot be stacked: \
                        give this Input band group 1 and add an Input for band group 2, each with its own branch.");
                }
                Some(shp) if !input_shape.is_empty() => {
                    r.out = Some(shp);
                }
                _ if group_of(node) == 0 => r.out = Some(input_shape.clone()),
                _ => {
                    r.status = "error".into();
                    r.messages.push(format!("Band group {} needs a second band group in the Dataset tab (e.g. Sentinel-1 next to Sentinel-2).", group_of(node)));
                }
            }
            rfs.insert(id, ([1, 1], [1, 1]));
        } else if blocked {
            r.status = "blocked".into();
            r.messages.push("Fix the layer above first.".into());
        } else {
            r.inputs = srcs.iter().map(|s| results[s].out.clone().unwrap()).collect();
            let rf_in = srcs.iter().filter_map(|s| rfs.get(s)).fold(([1, 1], [1, 1]), |a, b| {
                ([a.0[0].max(b.0[0]), a.0[1].max(b.0[1])], [a.1[0].max(b.1[0]), a.1[1].max(b.1[1])])
            });
            let out = match out_of_range(node) {
                Some(m) => {
                    err(&mut r, &m);
                    None
                }
                None if node.kind == "shared" => shared_step(node, &mut r, &node_of, &results, rf_in, &mut rfs, &py, ds, classes),
                None => {
                    // an Output block gives as many values as its head has classes / targets
                    let k = if node.kind == "output" { ds.head_size(head_of(node)).unwrap_or(0) } else { classes };
                    step(node, &mut r, k, rf_in, &mut rfs, &py)
                }
            };
            if r.status != "error" {
                r.out = out;
            }
            if r.out.is_none() && r.status != "error" {
                r.status = "error".into();
            }
        }
        if is_out(&node.kind) {
            let want = match ds.task.as_str() { "segment" => "seg_output", "detect" => "detect_output", _ => "output" };
            if node.kind != want && (node.kind != "output" || ds.task == "segment" || ds.task == "detect") {
                err(&mut r, &match want {
                    "detect_output" => "The dataset has boxes (object detection): end with a Detection output on a feature map.".to_string(),
                    "seg_output" => "The dataset is a segmentation dataset (a class for every pixel): end with a Segmentation output instead of Output.".to_string(),
                    _ => format!("{} is for {}, but the dataset has one label per image: use Output, or change the task in the Dataset tab.",
                        model::title(&node.kind), if node.kind == "seg_output" { "segmentation" } else { "object detection" }),
                });
                r.out = None;
            }
            if node.kind == "detect_output" {
                if let Some(o) = r.out.clone() {
                    let (ih, iw) = (input_shape[input_shape.len() - 2], input_shape[input_shape.len() - 1]);
                    if !ih.is_multiple_of(o[1]) || !iw.is_multiple_of(o[2]) || ih / o[1] != iw / o[2] {
                        err(&mut r, &format!("The images are {ih}×{iw} and this map is {}×{}: the image must be a whole number of map cells \
                            (the same stride, e.g. 4 or 8, both ways).", o[1], o[2]));
                        r.out = None;
                    }
                }
            }
            if node.kind == "seg_output" {
                if let Some(o) = r.out.clone() {
                    // the masks have the images' height and width (the last two sizes, also of a time series)
                    let (mh, mw) = (input_shape[input_shape.len() - 2], input_shape[input_shape.len() - 1]);
                    if o[1] != mh || o[2] != mw {
                        err(&mut r, &format!("The masks are {}×{} but this output is {}×{}: the decoder must return to the input size (Upsample / ConvTranspose2D).",
                            mh, mw, o[1], o[2]));
                        r.out = None;
                    }
                }
            }
            let head = head_of(node);
            if node.kind == "output" && head > ds.heads.len() {
                err(&mut r, &format!("This Output is for output {head}, but the Dataset tab has {} extra output{}.", ds.heads.len(),
                    if ds.heads.len() == 1 { "" } else { "s" }));
                r.out = None;
            } else if per_head.get(&head).copied().unwrap_or(0) > 1 {
                err(&mut r, &if ds.heads.is_empty() { "There is more than one Output block.".to_string() } else {
                    format!("There is more than one Output block for {}.", if head == 0 { "the main task".to_string() } else { ds.heads[head - 1].name.clone() })
                });
                r.out = None;
            }
            if has_any_output.contains(id) {
                err(&mut r, "Output must be the last block: nothing can come after it.");
                r.out = None;
            }
        } else if node.kind == "exit_output" {
            if r.out.is_some() && ds.task != "classify" {
                err(&mut r, "Early exits are for classification (one class per image) for now.");
                r.out = None;
            }
        } else if r.out.is_some() && !out_edges.contains_key(id) {
            warn(&mut r, "The result of this layer is not used: connect it to a following layer.");
        }
        if r.out.is_some() && node.params.get("frozen").and_then(|v| v.as_bool()).unwrap_or(false) {
            r.frozen = r.params;
        }
        if r.out.is_some() {
            total += r.params;
            frozen_total += r.frozen;
        }
        results.insert(id, r);
    }
    for (k, h) in ds.heads.iter().enumerate() {
        if n_out > 0 && !per_head.contains_key(&(k + 1)) {
            problems.push(format!("The Dataset tab has an extra output \"{}\": add an Output block for it (Output: head {}).", h.name, k + 1));
        }
    }
    if n_out > 0 && !per_head.contains_key(&0) {
        problems.push("Add an Output block for the main task (Output: head 0).".into());
    }
    if inputs.len() == 1 && n_out == 0 {
        problems.push(if ds.task == "detect" {
            "Add a Detection output at the end, on a feature map (e.g. a quarter of the image size): it finds the boxes.".into()
        } else if ds.task == "segment" {
            "Add a Segmentation output at the end, connected to the rest: it gives a class for every pixel.".into()
        } else {
            "Add an Output block at the end, connected to the rest: it gives one score per class.".into()
        });
    }
    let mut nodes: Vec<NodeResult> = order.iter().map(|id| results.remove(id).unwrap()).collect();
    for n in &g.nodes {
        if !order.contains(&n.id.as_str()) {
            let mut r = NodeResult { id: n.id.clone(), kind: n.kind.clone(), title: model::title(&n.kind).into(), ..Default::default() };
            if looped.contains(&n.id.as_str()) {
                r.status = "error".into();
                r.messages.push("This layer is part of a loop.".into());
            } else if n.kind == "input" {
                r.status = "error".into();
                r.messages.push("Only one Input block is allowed.".into());
            } else {
                r.status = "unconnected".into();
                r.messages.push("Not connected to the Input: this block is ignored.".into());
            }
            nodes.push(r);
        }
    }
    for r in nodes.iter_mut().filter(|r| r.out.is_some() && !matches!(r.status.as_str(), "error" | "blocked" | "unconnected")) {
        let bias = node_of.get(r.id.as_str()).map(|n| n.params.get("bias").and_then(|b| b.as_bool()).unwrap_or(true)).unwrap_or(true);
        r.macs = macs(r, bias);
    }
    let total_macs = nodes.iter().map(|r| r.macs).sum();
    let mut edge_shapes = BTreeMap::new();
    for e in &g.edges {
        if let Some(s) = nodes.iter().find(|n| n.id == e.source).and_then(|n| n.out.clone()) {
            edge_shapes.insert(e.id.clone(), s);
        }
    }
    let ok = problems.is_empty() && !nodes.iter().any(|n| n.status == "error" || n.status == "blocked") && classes >= need;
    if classes < need {
        problems.push(if need == 1 { "The dataset has no target columns yet (Dataset tab).".into() } else { "The dataset has no classes yet (Dataset tab).".into() });
    }
    Analysis { input: input_shape, classes, nodes, edges: edge_shapes, total_params: total, frozen_params: frozen_total, total_macs, problems, ok: ok && classes >= need,
        head_names: ds.heads.iter().map(|h| h.name.clone()).collect() }
}

/// multiply-accumulates of one layer for one image, from its weights and its input / output size
fn macs(r: &NodeResult, bias: bool) -> u64 {
    let out = r.out.as_ref().unwrap();
    let inp = r.inputs.first();
    let prod = |v: &[usize]| v.iter().map(|&x| x as u64).product::<u64>();
    match r.kind.as_str() {
        // weights (without the bias) × output positions; the output's first dimension is the channels
        "conv2d" | "dwconv2d" | "conv3d" => {
            let w = r.params.saturating_sub(if bias { out[0] as u64 } else { 0 });
            w * prod(&out[1..])
        }
        // a weight per (step, feature) pair along the sequence
        "conv1d" => r.params.saturating_sub(if bias { out[1] as u64 } else { 0 }) * out[0] as u64,
        "lstm" | "gru" => r.params * out[0] as u64,
        "seg_output" => r.params.saturating_sub(out[0] as u64) * prod(&out[1..]),
        // every input position meets every weight
        "convtranspose2d" => r.params.saturating_sub(out[0] as u64) * inp.map(|x| prod(&x[1..])).unwrap_or(0),
        // per token on a sequence: (in × units) for each of the N tokens
        "dense" | "output" => {
            let units = *out.last().unwrap() as u64;
            let tokens = if out.len() == 2 { out[0] as u64 } else { 1 };
            r.params.saturating_sub(units) * tokens
        }
        // query/key/value/output projections 4·N·d², attention scores and mixing 2·N²·d
        "attention" => {
            let (n, d) = (out[0] as u64, out[1] as u64);
            4 * n * d * d + 2 * n * n * d
        }
        // query and output projections 2·N·d², key and value projections 2·M·dk·d, scores and mixing 2·N·M·d
        "cross_attention" => {
            let (n, d) = (out[0] as u64, out[1] as u64);
            let (m, dk) = r.inputs.get(1).map(|s| (s[0] as u64, s[1] as u64)).unwrap_or((0, 0));
            2 * n * d * d + 2 * m * dk * d + 2 * n * m * d
        }
        "swiglu" => r.params * if out.len() == 2 { out[0] as u64 } else { 1 },
        // computed by the step, which has the settings: per token the router and top_k experts' two layers
        "moe" => r.macs,
        "exit_output" => r.params.saturating_sub(out[0] as u64),
        _ => 0,
    }
}

fn err(r: &mut NodeResult, m: &str) {
    r.messages.push(m.into());
    r.status = "error".into();
}

fn warn(r: &mut NodeResult, m: &str) {
    r.messages.push(m.into());
    if r.status == "ok" {
        r.status = "warn".into();
    }
}

/// the kinds whose layer has weights another block can share
pub const SHAREABLE: &[&str] = &["moe", "conv2d", "dwconv2d", "convtranspose2d", "dense", "batchnorm", "layernorm", "groupnorm", "attention", "cls_pos",
    "custom", "output", "seg_output", "conv3d", "lstm", "gru", "conv1d", "temporal_pool", "rmsnorm", "swiglu"];

/// a Shared-weights block: the shape the shared block's layer gives on this input (which must have the shape that block's input has,
/// or the weights would not fit); no parameters of its own
#[allow(clippy::too_many_arguments)]
fn shared_step<'a>(node: &'a Node, r: &mut NodeResult, node_of: &HashMap<&str, &Node>, results: &HashMap<&str, NodeResult>,
    rf_in: ([usize; 2], [usize; 2]), rfs: &mut HashMap<&'a str, ([usize; 2], [usize; 2])>, py: &[crate::custom::PyClass], ds: &Dataset, classes: usize) -> Option<Vec<usize>> {
    let of = node.params.get("of").and_then(|v| v.as_str()).unwrap_or("");
    let Some(src) = node_of.get(of).copied() else {
        err(r, if of.is_empty() { "Choose the block whose weights this one uses (Shared weights: of)." } else { "The block whose weights this one uses is not in the design." });
        return None;
    };
    if src.kind == "shared" {
        err(r, "Share the original block's weights, not another Shared-weights block's.");
        return None;
    }
    if !SHAREABLE.contains(&src.kind.as_str()) {
        err(r, &format!("{} has no weights to share.", model::title(&src.kind)));
        return None;
    }
    let Some(sr) = results.get(src.id.as_str()).filter(|s| s.out.is_some()) else {
        err(r, &format!("The {} whose weights this block uses has a problem (or is not connected).", model::title(&src.kind)));
        return None;
    };
    if r.inputs.len() != 1 {
        err(r, if r.inputs.is_empty() { "Connect an input to this block." } else { "A Shared-weights block takes one input." });
        return None;
    }
    if r.inputs[0] != sr.inputs[0] {
        err(r, &format!("The shared {}'s weights are for an input {}, but this block's input is {}.", model::title(&src.kind), fmt(&sr.inputs[0]), fmt(&r.inputs[0])));
        return None;
    }
    let mut twin = src.clone();
    twin.id = node.id.clone();
    let k = if src.kind == "output" { ds.head_size(src.params.get("head").and_then(|v| v.as_u64()).unwrap_or(0) as usize).unwrap_or(0) } else { classes };
    let mut probe = NodeResult { inputs: r.inputs.clone(), title: r.title.clone(), ..Default::default() };
    // (the receptive field is the shared block's computed on this block's input; kept under this block's id)
    let mut tmp: HashMap<&str, ([usize; 2], [usize; 2])> = HashMap::new();
    let out = step(&twin, &mut probe, k, rf_in, &mut tmp, py);
    if let Some(v) = tmp.get(twin.id.as_str()) {
        rfs.insert(node.id.as_str(), *v);
    }
    if probe.status == "error" {
        for m in &probe.messages {
            err(r, m);
        }
        return None;
    }
    r.formula = probe.formula;
    r.padding = probe.padding;
    r.stride = probe.stride;
    r.rf = probe.rf;
    r.params = 0;
    out
}

/// the output shape of one node from its inputs (or None with an error in r)
fn step<'a>(node: &'a Node, r: &mut NodeResult, classes: usize, rf_in: ([usize; 2], [usize; 2]), rfs: &mut HashMap<&'a str, ([usize; 2], [usize; 2])>,
    py: &[crate::custom::PyClass]) -> Option<Vec<usize>> {
    let p = &node.params;
    let k = node.kind.as_str();
    let ins = r.inputs.clone();
    match model::arity(k) {
        Some(1) if ins.len() != 1 => {
            err(r, if ins.is_empty() { "Connect an input to this block." } else { "This block takes one input; use Add or Concat to join branches." });
            return None;
        }
        None if ins.len() < 2 => {
            err(r, "Connect two or more branches to this block.");
            return None;
        }
        _ => {}
    }
    let x = ins[0].clone();
    let is3 = x.len() == 3;
    let is2 = x.len() == 2;
    let is4 = x.len() == 4;
    let need3 = |r: &mut NodeResult| {
        if x.len() == 4 {
            err(r, &format!("{} needs an image (channels, height, width), but its input is a time series (channels, dates, height, width) {}. \
                Put Temporal pool (or To sequence) before it, or use Conv3D.", r.title, fmt(&x)));
        } else if x.len() == 2 {
            err(r, &format!("{} needs an image (channels, height, width), but its input is a sequence of tokens {}. Put it before To tokens.", r.title, fmt(&x)));
        } else {
            err(r, &format!("{} needs an image (channels, height, width), but its input is already flat {}. Put it before Flatten / Global pool.", r.title, fmt(&x)));
        }
    };
    let need2 = |r: &mut NodeResult| {
        err(r, &format!("{} needs a sequence of tokens (tokens, dimension), but its input is {}. Put To tokens before it.", r.title, fmt(&x)));
    };
    let window_node = |r: &mut NodeResult, kernel: [usize; 2], stride: [usize; 2], pad: [usize; 2], dil: usize| -> Option<[usize; 2]> {
        let (h, w) = (x[1], x[2]);
        let (oh, nh, keh) = window(h, kernel[0], stride[0], pad[0], dil);
        let (ow, nw, kew) = window(w, kernel[1], stride[1], pad[1], dil);
        let f = |n: usize, k: usize, s: usize, p: usize, o: usize| format!("floor(({n} + 2×{p} − {dil}×({k}−1) − 1) / {s}) + 1 = {o}");
        r.formula = vec![format!("height: {}", f(h, kernel[0], stride[0], pad[0], oh)), format!("width:  {}", f(w, kernel[1], stride[1], pad[1], ow))];
        r.padding = Some(pad);
        r.stride = Some(stride);
        if oh < 1 || ow < 1 {
            err(r, &format!("The image is {h}×{w} here, too small for a {}×{} window{}. Use a smaller kernel, padding, or fewer pooling layers before this.",
                kernel[0], kernel[1], if dil > 1 { format!(" with dilation {dil}") } else { String::new() }));
            return None;
        }
        for (side, num, s) in [("rows", nh, stride[0]), ("columns", nw, stride[1])] {
            if !(num as usize).is_multiple_of(s) {
                warn(r, &format!("The {side}: in + 2×padding − window = {num} does not divide evenly by stride {s}, so the last {} {side} are ignored.", num as usize % s));
            }
        }
        if stride[0] > keh || stride[1] > kew {
            warn(r, &format!("Stride {}×{} is larger than the window {keh}×{kew}: pixels between windows are skipped.", stride[0], stride[1]));
        }
        Some([oh, ow])
    };
    let rf_after = |kernel: [usize; 2], stride: [usize; 2], dil: usize| {
        let (rf, jump) = rf_in;
        let ke = [dil * (kernel[0] - 1) + 1, dil * (kernel[1] - 1) + 1];
        ([rf[0] + (ke[0] - 1) * jump[0], rf[1] + (ke[1] - 1) * jump[1]], [jump[0] * stride[0], jump[1] * stride[1]])
    };
    let id: &'a str = node.id.as_str();
    rfs.insert(id, rf_in);
    // a time series (C, T, H, W) goes through these blocks only; the others need an image, tokens or a vector
    if is4 && !matches!(k, "conv3d" | "temporal_pool" | "to_sequence" | "batchnorm" | "dropout" | "activation" | "add" | "subtract" | "multiply"
        | "concat" | "flatten" | "global_pool" | "custom") {
        need3(r);
        return None;
    }
    let out = match k {
        "conv3d" => {
            if !is4 {
                err(r, &format!("Conv3D needs a time series (channels, dates, height, width), but its input is {}.", fmt(&x)));
                return None;
            }
            let kernel = pair(p, "kernel", 3, 1, 15);
            let tk = int(p, "tkernel", 3, 1, 15);
            let stride = pair(p, "stride", 1, 1, 8);
            let same = text(p, "padding", &["same", "valid"], "same") == "same";
            if same && (tk.is_multiple_of(2) || kernel.iter().any(|k| k.is_multiple_of(2))) {
                err(r, "'same' keeps the dates and the size with odd kernels only (3, 5, ...): use an odd kernel, or valid.");
                return None;
            }
            let (pt, ph, pw) = if same { (tk / 2, kernel[0] / 2, kernel[1] / 2) } else { (0, 0, 0) };
            let side = |n: usize, k: usize, s: usize, p: usize| ((n + 2 * p) as i64 - k as i64) / s as i64 + 1;
            let (ot, oh, ow) = (side(x[1], tk, 1, pt), side(x[2], kernel[0], stride[0], ph), side(x[3], kernel[1], stride[1], pw));
            r.formula = vec![format!("dates:  {} + 2×{pt} − {tk} + 1 = {ot}", x[1]),
                format!("height: floor(({} + 2×{ph} − {}) / {}) + 1 = {oh}", x[2], kernel[0], stride[0]),
                format!("width:  floor(({} + 2×{pw} − {}) / {}) + 1 = {ow}", x[3], kernel[1], stride[1])];
            r.padding = Some([ph, pw]);
            r.stride = Some(stride);
            if ot < 1 || oh < 1 || ow < 1 {
                err(r, &format!("The input {} is too small for a {tk}×{}×{} window (dates × height × width).", fmt(&x), kernel[0], kernel[1]));
                return None;
            }
            let f = int(p, "filters", 16, 1, MAX_FILTERS);
            let bias = p.get("bias").and_then(|v| v.as_bool()).unwrap_or(true) as u64;
            r.params = (x[0] * tk * kernel[0] * kernel[1]) as u64 * f as u64 + bias * f as u64;
            vec![f, ot as usize, oh as usize, ow as usize]
        }
        "temporal_pool" => {
            if !is4 {
                err(r, &format!("Temporal pool needs a time series (channels, dates, height, width), but its input is {}.", fmt(&x)));
                return None;
            }
            if text(p, "mode", &["mean", "max", "attention"], "attention") == "attention" {
                r.params = x[0] as u64 + 1; // a score per date and pixel from the channels (a 1×1×1 convolution)
            }
            vec![x[0], x[2], x[3]]
        }
        "to_sequence" => {
            if !is4 {
                err(r, &format!("To sequence needs a time series (channels, dates, height, width), but its input is {}.", fmt(&x)));
                return None;
            }
            // the mean over the pixels of each date: one vector of the channels per date
            vec![x[1], x[0]]
        }
        "lstm" | "gru" => {
            if !is2 {
                err(r, &format!("{} needs a sequence (steps, features), e.g. after To sequence, but its input is {}.", r.title, fmt(&x)));
                return None;
            }
            let h = int(p, "units", 64, 1, 1 << 16);
            let dirs = if p.get("bidirectional").and_then(|v| v.as_bool()).unwrap_or(false) { 2 } else { 1 };
            let gates = if k == "lstm" { 4 } else { 3 };
            // per direction: input and hidden weights of every gate, and PyTorch's two bias vectors
            r.params = (dirs * gates * (h * (x[1] + h) + 2 * h)) as u64;
            vec![x[0], h * dirs]
        }
        "conv1d" => {
            if !is2 {
                err(r, &format!("Conv1D works along a sequence (steps, features), e.g. after To sequence, but its input is {}.", fmt(&x)));
                return None;
            }
            let kk = int(p, "kernel", 3, 1, 31);
            let s = int(p, "stride", 1, 1, 16);
            let same = text(p, "padding", &["same", "valid"], "same") == "same";
            if same && kk.is_multiple_of(2) {
                err(r, "'same' keeps the length with an odd kernel only: use 3, 5, ... or valid.");
                return None;
            }
            let pd = if same { kk / 2 } else { 0 };
            let ot = ((x[0] + 2 * pd) as i64 - kk as i64) / s as i64 + 1;
            r.formula = vec![format!("steps: floor(({} + 2×{pd} − {kk}) / {s}) + 1 = {ot}", x[0])];
            if ot < 1 {
                err(r, &format!("The sequence has {} steps, too few for a kernel of {kk}.", x[0]));
                return None;
            }
            let f = int(p, "filters", 64, 1, MAX_FILTERS);
            let bias = p.get("bias").and_then(|v| v.as_bool()).unwrap_or(true) as u64;
            r.params = (x[1] * kk) as u64 * f as u64 + bias * f as u64;
            vec![ot as usize, f]
        }
        "conv2d" | "dwconv2d" => {
            if !is3 {
                need3(r);
                return None;
            }
            let kernel = pair(p, "kernel", 3, 1, 32);
            let stride = pair(p, "stride", 1, 1, 32);
            let dil = if k == "conv2d" { int(p, "dilation", 1, 1, 6) } else { 1 };
            let pad = match padding(p) {
                Pad::Same => {
                    if kernel.iter().any(|kk| dil * (kk - 1) % 2 == 1) {
                        warn(r, "'same' cannot keep the size exactly with an even kernel: PyTorch pads both sides equally, so the size changes by one.");
                    }
                    [dil * (kernel[0] - 1) / 2, dil * (kernel[1] - 1) / 2]
                }
                Pad::Num(n) => [n, n],
            };
            let hw = window_node(r, kernel, stride, pad, dil)?;
            let c = x[0];
            let bias = p.get("bias").and_then(|v| v.as_bool()).unwrap_or(true) as u64;
            let f = if k == "conv2d" { int(p, "filters", 32, 1, MAX_FILTERS) } else { c * int(p, "multiplier", 1, 1, 8) };
            r.params = if k == "conv2d" { (kernel[0] * kernel[1] * c) as u64 * f as u64 + bias * f as u64 } else { (kernel[0] * kernel[1]) as u64 * f as u64 + bias * f as u64 };
            let next = rf_after(kernel, stride, dil);
            rfs.insert(id, next);
            r.rf = Some(next.0);
            vec![f, hw[0], hw[1]]
        }
        "pool2d" => {
            if !is3 {
                need3(r);
                return None;
            }
            let size = pair(p, "size", 2, 1, 8);
            let stride = if p.get("stride").and_then(|v| v.as_str()) == Some("size") || p.get("stride").is_none() { size } else { pair(p, "stride", 2, 1, 32) };
            let pd = int(p, "padding", 0, 0, 4);
            if pd * 2 > size[0].min(size[1]) {
                err(r, &format!("Padding {pd} is more than half the pool size {}×{} (PyTorch does not allow that).", size[0], size[1]));
                return None;
            }
            let hw = window_node(r, size, stride, [pd, pd], 1)?;
            let next = rf_after(size, stride, 1);
            rfs.insert(id, next);
            r.rf = Some(next.0);
            vec![x[0], hw[0], hw[1]]
        }
        "convtranspose2d" => {
            if !is3 {
                need3(r);
                return None;
            }
            let kernel = pair(p, "kernel", 2, 1, 32);
            let stride = pair(p, "stride", 2, 1, 32);
            let pd = int(p, "padding", 0, 0, 10);
            let op = int(p, "output_padding", 0, 0, 7);
            if op >= stride[0].max(1) || op >= stride[1].max(1) {
                err(r, &format!("Output padding {op} must be smaller than the stride (PyTorch rule)."));
                return None;
            }
            let f = int(p, "filters", 32, 1, MAX_FILTERS);
            let side = |n: usize, k: usize, s: usize| (n as i64 - 1) * s as i64 - 2 * pd as i64 + (k as i64 - 1) + op as i64 + 1;
            let (oh, ow) = (side(x[1], kernel[0], stride[0]), side(x[2], kernel[1], stride[1]));
            r.formula = vec![
                format!("height: ({} − 1)×{} − 2×{pd} + ({}−1) + {op} + 1 = {oh}", x[1], stride[0], kernel[0]),
                format!("width:  ({} − 1)×{} − 2×{pd} + ({}−1) + {op} + 1 = {ow}", x[2], stride[1], kernel[1]),
            ];
            r.padding = Some([pd, pd]);
            r.stride = Some(stride);
            if oh < 1 || ow < 1 {
                err(r, "The padding is too large for this input: the output would be empty.");
                return None;
            }
            r.params = (kernel[0] * kernel[1] * x[0] * f + f) as u64;
            vec![f, oh as usize, ow as usize]
        }
        "global_pool" => {
            if !is3 && !is4 {
                need3(r);
                return None;
            }
            vec![x[0]]
        }
        "custom" => {
            // the shape the class declares (# out:); its parameters are counted on the training machine
            let name = p.get("class").and_then(|v| v.as_str()).unwrap_or("");
            let found: Vec<&crate::custom::PyClass> = py.iter().filter(|c| c.name == name).collect();
            if name.is_empty() {
                err(r, "Choose the PyTorch class this block runs (Custom block: class), or write one in the design's PyTorch code.");
                return None;
            }
            if found.is_empty() {
                let names: Vec<&str> = py.iter().map(|c| c.name.as_str()).collect();
                err(r, &format!("There is no class {name} in the design's PyTorch code{}.",
                    if names.is_empty() { String::new() } else { format!(" (it has {})", names.join(", ")) }));
                return None;
            }
            if found.len() > 1 {
                err(r, &format!("The design's PyTorch code defines {name} {} times; keep one.", found.len()));
                return None;
            }
            match crate::custom::out_shape(found[0], &x) {
                Ok(s) => s,
                Err(m) => {
                    err(r, &m);
                    return None;
                }
            }
        }
        "upsample" => {
            if !is3 {
                need3(r);
                return None;
            }
            let s = int(p, "scale", 2, 2, 8);
            vec![x[0], x[1] * s, x[2] * s]
        }
        "groupnorm" => {
            if !is3 {
                need3(r);
                return None;
            }
            let g = int(p, "groups", 8, 1, MAX_FILTERS);
            if !x[0].is_multiple_of(g) {
                let fits: Vec<String> = [32, 16, 8, 4, 2, 1].iter().filter(|d| x[0].is_multiple_of(**d)).map(|d| d.to_string()).collect();
                err(r, &format!("GroupNorm splits the {} channels into {g} groups, but {} does not divide by {g}; for example {} groups.", x[0], x[0], fits.join(", ")));
                return None;
            }
            r.params = 2 * x[0] as u64;
            x.clone()
        }
        "batchnorm" | "layernorm" => {
            if k == "layernorm" && is4 {
                need3(r);
                return None;
            }
            if k == "batchnorm" && is2 {
                err(r, "BatchNorm is for images and vectors; use LayerNorm on a sequence of tokens.");
                return None;
            }
            r.params = 2 * if is2 { x[1] } else { x[0] } as u64;
            x.clone()
        }
        "to_tokens" => {
            if !is3 {
                need3(r);
                return None;
            }
            if text(p, "mode", &["patches", "rows"], "patches") == "rows" {
                vec![x[1], x[0] * x[2]]
            } else {
                vec![x[1] * x[2], x[0]]
            }
        }
        "cls_pos" => {
            if !is2 {
                need2(r);
                return None;
            }
            let cls = p.get("cls").and_then(|v| v.as_bool()).unwrap_or(true) as usize;
            let learned = text(p, "pos", &["learned", "sinusoidal", "none"], "learned") == "learned";
            if text(p, "pos", &["learned", "sinusoidal", "none"], "learned") == "sinusoidal" && !x[1].is_multiple_of(2) {
                err(r, &format!("Sinusoidal positions pair sines and cosines: the token dimension {} must be even.", x[1]));
                return None;
            }
            r.params = (cls * x[1] + if learned { (x[0] + cls) * x[1] } else { 0 }) as u64;
            vec![x[0] + cls, x[1]]
        }
        "attention" => {
            if !is2 {
                need2(r);
                return None;
            }
            let h = int(p, "heads", 4, 1, MAX_HEADS);
            if !x[1].is_multiple_of(h) {
                err(r, &format!("The token dimension {} must divide evenly by the number of heads {h}.", x[1]));
                return None;
            }
            if p.get("rotary").and_then(|v| v.as_bool()).unwrap_or(false) && !(x[1] / h).is_multiple_of(2) {
                err(r, &format!("Rotary positions rotate pairs of values: each head's width {} / {h} = {} must be even.", x[1], x[1] / h));
                return None;
            }
            let d = x[1] as u64;
            r.params = 4 * d * d + 4 * d;
            let rank = int(p, "lora", 0, 0, 1024) as u64;
            if rank > 0 {
                // LoRA on the query/key/value projection (d → 3d) and the output projection (d → d)
                r.frozen = r.params;
                r.params += rank * (d + 3 * d) + rank * (d + d);
            }
            x.clone()
        }
        "cross_attention" => {
            if ins.len() != 2 || ins.iter().any(|s| s.len() != 2) {
                err(r, &if ins.len() != 2 { "Cross-attention takes two inputs: the tokens that ask (first arrow) and the tokens they look at (second).".to_string() } else {
                    format!("Cross-attention works on token sequences (tokens, dimension), but its inputs are {}.", ins.iter().map(|s| fmt(s)).collect::<Vec<_>>().join(" and "))
                });
                return None;
            }
            let (q, kv) = (&ins[0], &ins[1]);
            let h = int(p, "heads", 4, 1, MAX_HEADS);
            if !q[1].is_multiple_of(h) {
                err(r, &format!("The asking tokens' dimension {} must divide evenly by the number of heads {h}.", q[1]));
                return None;
            }
            let (d, dk) = (q[1] as u64, kv[1] as u64);
            // queries d×d, keys and values dk×d each, the output d×d, with their biases
            r.params = 2 * d * d + 2 * dk * d + 4 * d;
            q.clone()
        }
        "rmsnorm" => {
            if !(is2 || x.len() == 1) {
                err(r, &format!("RMSNorm normalises each token (or vector) over its values, but its input is {}: use it on tokens or vectors.", fmt(&x)));
                return None;
            }
            r.params = *x.last().unwrap() as u64;
            x.clone()
        }
        "swiglu" => {
            if !(is2 || x.len() == 1) {
                err(r, &format!("SwiGLU is a gated MLP for tokens (or vectors), but its input is {}.", fmt(&x)));
                return None;
            }
            let (d, u) = (*x.last().unwrap() as u64, int(p, "units", 128, 1, 1 << 20) as u64);
            r.params = 3 * d * u;
            x.clone()
        }
        "token_pool" => {
            if !is2 {
                need2(r);
                return None;
            }
            text(p, "mode", &["cls", "mean", "last"], "cls");
            vec![x[1]]
        }
        "dropout" | "activation" => x.clone(),
        "flatten" => {
            if is2 {
                vec![x[0] * x[1]]
            } else if is4 {
                vec![x[0] * x[1] * x[2] * x[3]]
            } else if !is3 {
                warn(r, "The input is already flat: Flatten does nothing here.");
                x.clone()
            } else {
                vec![x[0] * x[1] * x[2]]
            }
        }
        "moe" => {
            if !(is2 || x.len() == 1) {
                err(r, &format!("Mixture of experts works on tokens (or vectors), but its input is {}.", fmt(&x)));
                return None;
            }
            let (e, kk, h) = (int(p, "experts", 4, 1, 256), int(p, "top_k", 2, 1, 256), int(p, "units", 128, 1, 1 << 20));
            if kk > e {
                err(r, &format!("Each token goes to its top {kk} experts, but there are only {e}."));
                return None;
            }
            let d = *x.last().unwrap();
            // the router (d → experts) and every expert's two layers (d → units → d, with biases)
            r.params = (d * e + e + e * (2 * d * h + h + d)) as u64;
            let tokens = if is2 { x[0] } else { 1 };
            r.macs = (tokens * (d * e + kk * 2 * d * h)) as u64;
            x.clone()
        }
        "exit_output" => {
            if !(x.len() == 1) {
                err(r, &format!("An early exit gives one score per class from a vector, but its input is {}: add Global pool (or Token pool) before it.", fmt(&x)));
                return None;
            }
            r.params = (x[0] as u64 + 1) * classes.max(1) as u64;
            vec![classes.max(1)]
        }
        "dense" | "output" => {
            let units = if k == "dense" { int(p, "units", 128, 1, 1 << 20) } else { classes.max(1) };
            if is3 {
                err(r, &format!("{} needs a flat input, but its input is an image {}: add Flatten or Global pool before it.", r.title, fmt(&x)));
                return None;
            }
            if is2 && k == "output" {
                err(r, &format!("Output needs one vector per image, but its input is a sequence of tokens {}: add Token pool before it.", fmt(&x)));
                return None;
            }
            // on tokens, Dense acts on every token (the last dimension)
            r.params = (*x.last().unwrap() as u64 + 1) * units as u64;
            let rank = if k == "dense" { int(p, "lora", 0, 0, 1024) as u64 } else { 0 };
            if rank > 0 {
                // LoRA: the layer's own weights stay as they are; A (rank × in) and B (units × rank) learn the change
                r.frozen = r.params;
                r.params += rank * (*x.last().unwrap() as u64 + units as u64);
            }
            // the warning is about Flatten -> huge Dense; a transformer MLP on tokens is large by design
            if k == "dense" && !is2 && r.params > BIG_DENSE {
                warn(r, &format!("This Dense layer has {} parameters: try Global pool instead of Flatten before it.", r.params));
            }
            if is2 { vec![x[0], units] } else { vec![units] }
        }
        "add" => {
            if ins.iter().any(|s| *s != x) {
                err(r, &format!("Add needs inputs of the same shape, but they are {}. Use a 1×1 Conv2D to match channels, or a stride to match sizes.",
                    ins.iter().map(|s| fmt(s)).collect::<Vec<_>>().join(" and ")));
                return None;
            }
            x.clone()
        }
        "multiply" => {
            if ins.len() != 2 {
                err(r, "Multiply takes exactly two inputs.");
                return None;
            }
            let (a, b) = (&ins[0], &ins[1]);
            if a == b {
                a.clone()
            } else if a.len() == 3 && b.len() == 3 && a[1] == b[1] && a[2] == b[2] && (a[0] == 1 || b[0] == 1) {
                // a one-channel map (an attention gate) scales every channel
                vec![a[0].max(b[0]), a[1], a[2]]
            } else if a.len() == 3 && b.len() == 1 && b[0] == a[0] {
                a.clone()
            } else if b.len() == 3 && a.len() == 1 && a[0] == b[0] {
                b.clone()
            } else {
                err(r, &format!("Multiply needs equal shapes, a one-channel map of the same size (attention gate), or one weight per channel; got {} and {}.",
                    fmt(a), fmt(b)));
                return None;
            }
        }
        "subtract" => {
            if ins.len() != 2 || ins[0] != ins[1] {
                err(r, &if ins.len() != 2 { "Subtract takes exactly two inputs (the first minus the second).".to_string() } else {
                    format!("Subtract needs two inputs of the same shape, but they are {} and {}.", fmt(&ins[0]), fmt(&ins[1]))
                });
                return None;
            }
            x.clone()
        }
        "seg_output" => {
            if !is3 {
                need3(r);
                return None;
            }
            r.params = (x[0] as u64 + 1) * classes.max(1) as u64;
            vec![classes.max(1), x[1], x[2]]
        }
        // CenterNet-style: one heatmap per class, then box width and height, then the centre's offset in its cell
        "detect_output" => {
            if !is3 {
                need3(r);
                return None;
            }
            let k = classes.max(1) + 4;
            r.params = (x[0] as u64 + 1) * k as u64;
            vec![k, x[1], x[2]]
        }
        "concat" => {
            if is2 {
                // token sequences are joined one after the other: (N1 + N2, D)
                if ins.iter().any(|s| s.len() != 2 || s[1] != x[1]) {
                    err(r, &format!("Concat joins token sequences one after the other, so their dimension must match, but they are {}.",
                        ins.iter().map(|s| fmt(s)).collect::<Vec<_>>().join(" and ")));
                    return None;
                }
                return Some(vec![ins.iter().map(|s| s[0]).sum(), x[1]]);
            }
            if ins.iter().any(|s| s.len() != x.len()) {
                err(r, "Concat cannot join an image and a flat vector.");
                return None;
            }
            if is4 {
                if ins.iter().any(|s| s[1..] != x[1..]) {
                    err(r, &format!("Concat joins time series along channels, so dates, height and width must match, but they are {}.",
                        ins.iter().map(|s| fmt(&s[1..])).collect::<Vec<_>>().join(" and ")));
                    return None;
                }
                vec![ins.iter().map(|s| s[0]).sum(), x[1], x[2], x[3]]
            } else if is3 {
                if ins.iter().any(|s| s[1] != x[1] || s[2] != x[2]) {
                    err(r, &format!("Concat joins along channels, so height and width must match, but they are {}.",
                        ins.iter().map(|s| format!("{}×{}", s[1], s[2])).collect::<Vec<_>>().join(" and ")));
                    return None;
                }
                vec![ins.iter().map(|s| s[0]).sum(), x[1], x[2]]
            } else {
                vec![ins.iter().map(|s| s[0]).sum()]
            }
        }
        "input" => x.clone(),
        other => {
            err(r, &format!("Unknown block type {other:?}."));
            return None;
        }
    };
    Some(out)
}

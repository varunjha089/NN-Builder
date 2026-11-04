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

#[derive(Debug, Clone, Serialize, Default)]
pub struct NodeResult {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub inputs: Vec<Vec<usize>>,
    pub out: Option<Vec<usize>>,
    pub params: u64,
    /// ok | warn | error | blocked (an input has an error) | unconnected (not reached from Input: ignored)
    pub status: String,
    pub messages: Vec<String>,
    pub formula: Vec<String>,
    pub rf: Option<[usize; 2]>,
    /// the padding and stride actually used (conv/pool), for the code and the sliding preview
    pub padding: Option<[usize; 2]>,
    pub stride: Option<[usize; 2]>,
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
    pub problems: Vec<String>,
    pub ok: bool,
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
pub fn int(p: &Map<String, Value>, key: &str, default: i64, lo: i64, hi: i64) -> usize {
    let v = p.get(key).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))).map(|f| f.round() as i64).unwrap_or(default);
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
    let mut problems: Vec<String> = vec![];
    let ids: HashSet<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    let edges: Vec<_> = g.edges.iter().filter(|e| ids.contains(e.source.as_str()) && ids.contains(e.target.as_str()) && e.source != e.target).collect();

    let inputs: Vec<&Node> = g.nodes.iter().filter(|n| n.kind == "input").collect();
    if inputs.len() != 1 {
        problems.push(if inputs.is_empty() { "Add an Input block.".into() } else { "There must be exactly one Input block.".into() });
    }
    // reachable from the (first) Input
    let mut reach: HashSet<&str> = HashSet::new();
    if let Some(inp) = inputs.first() {
        let mut q = VecDeque::from([inp.id.as_str()]);
        while let Some(id) = q.pop_front() {
            if reach.insert(id) {
                for e in edges.iter().filter(|e| e.source == id) {
                    q.push_back(e.target.as_str());
                }
            }
        }
    }
    // computation order (Kahn) over the reached part
    let mut indeg: HashMap<&str, usize> = reach.iter().map(|id| (*id, 0)).collect();
    for e in edges.iter().filter(|e| reach.contains(e.source.as_str()) && reach.contains(e.target.as_str())) {
        *indeg.get_mut(e.target.as_str()).unwrap() += 1;
    }
    let mut ready: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).filter(|id| indeg.get(id) == Some(&0)).collect();
    let mut order: Vec<&str> = vec![];
    while let Some(id) = (!ready.is_empty()).then(|| ready.remove(0)) {
        order.push(id);
        for e in edges.iter().filter(|e| e.source == id && reach.contains(e.target.as_str())) {
            let d = indeg.get_mut(e.target.as_str()).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push(e.target.as_str());
            }
        }
    }
    let looped: Vec<&str> = reach.iter().copied().filter(|id| !order.contains(id)).collect();
    if !looped.is_empty() {
        problems.push("The design has a loop (an arrow leads back to an earlier layer). Networks must flow one way.".into());
    }

    let mut results: HashMap<&str, NodeResult> = HashMap::new();
    let mut rfs: HashMap<&str, ([usize; 2], [usize; 2])> = HashMap::new();
    let is_out = |k: &str| k == "output" || k == "seg_output";
    let n_out = order.iter().filter(|id| g.node(id).map(|n| is_out(&n.kind)).unwrap_or(false)).count();
    let mut total = 0u64;
    for id in &order {
        let node = g.node(id).unwrap();
        let srcs: Vec<&str> = edges.iter().filter(|e| e.target == *id && reach.contains(e.source.as_str())).map(|e| e.source.as_str()).collect();
        let mut r = NodeResult { id: id.to_string(), kind: node.kind.clone(), title: model::title(&node.kind).into(), status: "ok".into(), ..Default::default() };
        let blocked = srcs.iter().any(|s| results.get(s).map(|x| x.out.is_none()).unwrap_or(true));
        if node.kind == "input" {
            r.out = Some(input_shape.clone());
            rfs.insert(id, ([1, 1], [1, 1]));
        } else if blocked {
            r.status = "blocked".into();
            r.messages.push("Fix the layer above first.".into());
        } else {
            r.inputs = srcs.iter().map(|s| results[s].out.clone().unwrap()).collect();
            let rf_in = srcs.iter().filter_map(|s| rfs.get(s)).fold(([1, 1], [1, 1]), |a, b| {
                ([a.0[0].max(b.0[0]), a.0[1].max(b.0[1])], [a.1[0].max(b.1[0]), a.1[1].max(b.1[1])])
            });
            let out = step(node, &mut r, classes, rf_in, &mut rfs);
            if r.status != "error" {
                r.out = out;
            }
            if r.out.is_none() && r.status != "error" {
                r.status = "error".into();
            }
        }
        if is_out(&node.kind) {
            if (node.kind == "seg_output") != (ds.task == "segment") {
                err(&mut r, if ds.task == "segment" {
                    "The dataset is a segmentation dataset (a class for every pixel): end with a Segmentation output instead of Output."
                } else {
                    "Segmentation output gives a class for every pixel, but the dataset is for classification: use Output, or choose a segmentation dataset."
                });
                r.out = None;
            }
            if node.kind == "seg_output" {
                if let Some(o) = r.out.clone() {
                    if o[1] != input_shape[1] || o[2] != input_shape[2] {
                        err(&mut r, &format!("The masks are {}×{} but this output is {}×{}: the decoder must return to the input size (Upsample / ConvTranspose2D).",
                            input_shape[1], input_shape[2], o[1], o[2]));
                        r.out = None;
                    }
                }
            }
            if n_out > 1 {
                err(&mut r, "There is more than one Output block.");
                r.out = None;
            }
            if !g.outputs_of(id).is_empty() {
                err(&mut r, "Output must be the last block: nothing can come after it.");
                r.out = None;
            }
        } else if r.out.is_some() && !edges.iter().any(|e| e.source == *id) {
            warn(&mut r, "The result of this layer is not used: connect it to a following layer.");
        }
        if r.out.is_some() {
            total += r.params;
        }
        results.insert(id, r);
    }
    if inputs.len() == 1 && n_out == 0 {
        problems.push(if ds.task == "segment" {
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
    let mut edge_shapes = BTreeMap::new();
    for e in &g.edges {
        if let Some(s) = nodes.iter().find(|n| n.id == e.source).and_then(|n| n.out.clone()) {
            edge_shapes.insert(e.id.clone(), s);
        }
    }
    let ok = problems.is_empty() && !nodes.iter().any(|n| n.status == "error" || n.status == "blocked") && classes >= 2;
    if classes < 2 {
        problems.push("The dataset has no classes yet (Dataset tab).".into());
    }
    Analysis { input: input_shape, classes, nodes, edges: edge_shapes, total_params: total, problems, ok: ok && classes >= 2 }
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

/// the output shape of one node from its inputs (or None with an error in r)
fn step<'a>(node: &'a Node, r: &mut NodeResult, classes: usize, rf_in: ([usize; 2], [usize; 2]), rfs: &mut HashMap<&'a str, ([usize; 2], [usize; 2])>) -> Option<Vec<usize>> {
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
    let need3 = |r: &mut NodeResult| {
        if x.len() == 2 {
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
    let out = match k {
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
            let f = if k == "conv2d" { int(p, "filters", 32, 1, 4096) } else { c * int(p, "multiplier", 1, 1, 8) };
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
            let f = int(p, "filters", 32, 1, 4096);
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
            if !is3 {
                need3(r);
                return None;
            }
            vec![x[0]]
        }
        "upsample" => {
            if !is3 {
                need3(r);
                return None;
            }
            let s = int(p, "scale", 2, 2, 8);
            vec![x[0], x[1] * s, x[2] * s]
        }
        "batchnorm" | "layernorm" => {
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
            r.params = (cls * x[1] + (x[0] + cls) * x[1]) as u64;
            vec![x[0] + cls, x[1]]
        }
        "attention" => {
            if !is2 {
                need2(r);
                return None;
            }
            let h = int(p, "heads", 4, 1, 64);
            if !x[1].is_multiple_of(h) {
                err(r, &format!("The token dimension {} must divide evenly by the number of heads {h}.", x[1]));
                return None;
            }
            let d = x[1] as u64;
            r.params = 4 * d * d + 4 * d;
            x.clone()
        }
        "token_pool" => {
            if !is2 {
                need2(r);
                return None;
            }
            vec![x[1]]
        }
        "dropout" | "activation" => x.clone(),
        "flatten" => {
            if is2 {
                vec![x[0] * x[1]]
            } else if !is3 {
                warn(r, "The input is already flat: Flatten does nothing here.");
                x.clone()
            } else {
                vec![x[0] * x[1] * x[2]]
            }
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
            if k == "dense" && r.params > BIG_DENSE {
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
        "seg_output" => {
            if !is3 {
                need3(r);
                return None;
            }
            r.params = (x[0] as u64 + 1) * classes.max(1) as u64;
            vec![classes.max(1), x[1], x[2]]
        }
        "concat" => {
            if is2 {
                err(r, "Concat on token sequences is not supported yet.");
                return None;
            }
            if ins.iter().any(|s| s.len() != x.len()) {
                err(r, "Concat cannot join an image and a flat vector.");
                return None;
            }
            if is3 {
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

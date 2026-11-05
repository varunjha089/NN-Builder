//! Writes random designs as JSON for python/verify_shapes.py (run in GitHub Actions, never on the owner's Mac):
//! for each: the generated model code, the input shape, every module's expected output shape and the parameter count.
//!     cargo run -p nnb-core --example dump_cases -- 200 > cases.json
use nnb_core::codegen::model_code;
use nnb_core::dataset::Dataset;
use nnb_core::model::{self, Edge, Graph, Node};
use nnb_core::shapes::analyze;
use serde_json::{json, Value};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

fn main() {
    let count: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut r = Lcg(7);
    let mut cases = vec![];
    for i in 0..count {
        let ds = match r.next(3) {
            0 => Dataset::default(),
            1 => Dataset { builtin: "cifar10".into(), ..Default::default() },
            _ => Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [20 + r.next(40) as usize, 20 + r.next(40) as usize],
                channels: if r.next(2) == 0 { 1 } else { 3 }, classes: vec!["a".into(), "b".into(), "c".into()], ..Default::default() },
        };
        let mut g = Graph::default();
        let add = |g: &mut Graph, kind: &str, p: Value, from: &[String]| -> String {
            let id = format!("n{}", g.nodes.len());
            let mut params = model::defaults(kind).as_object().cloned().unwrap_or_default();
            for (k, v) in p.as_object().unwrap() { params.insert(k.clone(), v.clone()); }
            g.nodes.push(Node { id: id.clone(), kind: kind.into(), params, ..Default::default() });
            for f in from { let eid = format!("e{}", g.edges.len()); g.edges.push(Edge { id: eid, source: f.clone(), target: id.clone() }); }
            id
        };
        let mut last = add(&mut g, "input", json!({}), &[]);
        for _ in 0..(1 + r.next(5)) {
            last = match r.next(9) {
                0 | 1 => { let k = 1 + r.next(5); add(&mut g, "conv2d", json!({"filters": 1 + r.next(12), "kernel": [k, if r.next(2) == 0 { k } else { 1 + r.next(5) }],
                    "stride": [1 + r.next(3), 1 + r.next(3)], "padding": (["same", "valid", "1", "2"][r.next(4) as usize]), "dilation": 1 + r.next(2),
                    "activation": (model::ACTIVATIONS[r.next(7) as usize]), "bias": r.next(3) != 0}), &[last]) }
                2 => { let k = 1 + r.next(3); add(&mut g, "pool2d", json!({"mode": (["max", "avg"][r.next(2) as usize]), "size": [k, k],
                    "stride": if r.next(2) == 0 { json!("size") } else { json!([1 + r.next(3), 1 + r.next(3)]) }, "padding": r.next(2)}), &[last]) }
                3 => add(&mut g, "dwconv2d", json!({"multiplier": 1 + r.next(2), "kernel": [3, 3], "stride": [1 + r.next(2), 1 + r.next(2)]}), &[last]),
                4 => add(&mut g, "convtranspose2d", json!({"filters": 1 + r.next(8), "kernel": [2 + r.next(2), 2 + r.next(2)], "stride": [2, 2], "padding": r.next(2), "output_padding": r.next(2)}), &[last]),
                5 => { let a = add(&mut g, "conv2d", json!({"filters": 6}), &[last.clone()]); let b = add(&mut g, "conv2d", json!({"filters": 6, "kernel": [1, 1]}), &[last]);
                       if r.next(2) == 0 { add(&mut g, "add", json!({}), &[a, b]) } else { add(&mut g, "concat", json!({}), &[a, b]) } }
                6 => add(&mut g, ["batchnorm", "layernorm", "dropout", "activation", "groupnorm"][r.next(5) as usize], json!({"fn": "gelu", "groups": 2}), &[last]),
                7 => add(&mut g, "upsample", json!({"scale": 2, "mode": (["nearest", "bilinear"][r.next(2) as usize])}), &[last]),
                _ => add(&mut g, "conv2d", json!({"filters": 4}), &[last]),
            };
        }
        last = add(&mut g, if r.next(2) == 0 { "flatten" } else { "global_pool" }, json!({}), &[last]);
        if r.next(2) == 0 { last = add(&mut g, "dense", json!({"units": 1 + r.next(30)}), &[last]); }
        add(&mut g, "output", json!({}), &[last]);
        let a = analyze(&g, &ds);
        cases.push(if a.ok {
            json!({"case": i, "ok": true, "code": model_code(&g, &a).unwrap(), "input": a.input, "params": a.total_params})
        } else {
            json!({"case": i, "ok": false})
        });
    }
    // the famous architectures too, at the input they were made for (2 classes for segmentation)
    for t in nnb_core::templates::TEMPLATES {
        let g = nnb_core::templates::build(t.key).unwrap();
        let ds = Dataset { source: "huggingface".into(), task: t.task.into(), hf_repo: "x/y".into(), image_size: [t.input[1], t.input[2]], channels: t.input[0],
            classes: if t.task == "segment" { vec!["a".into(), "b".into()] } else { (0..10).map(|i| i.to_string()).collect() }, ..Default::default() };
        let a = analyze(&g, &ds);
        assert!(a.ok, "template {}", t.key);
        cases.push(json!({"case": t.key, "ok": true, "code": model_code(&g, &a).unwrap(), "input": a.input, "params": a.total_params}));
    }
    println!("{}", serde_json::to_string(&cases).unwrap());
}

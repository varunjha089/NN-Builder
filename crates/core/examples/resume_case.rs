//! Writes a small train.py for python/ci_resume_check.py (GitHub Actions only, never on the owner's Mac): a little CNN on a two-class
//! image folder that CI generates itself, so the spot-interruption checkpoint and mid-epoch resume run with real PyTorch.
//!     cargo run -p nnb-core --example resume_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings};
use nnb_core::dataset::Dataset;
use nnb_core::model::{Edge, Graph, Node};
use nnb_core::shapes::analyze;
use serde_json::json;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    let node = |id: &str, kind: &str, p: serde_json::Value| Node { id: id.into(), kind: kind.into(), params: nnb_core::model::defaults(kind).as_object().cloned()
        .map(|mut d| { d.extend(p.as_object().unwrap().clone()); d }).unwrap_or_default(), x: 0.0, y: 0.0 };
    let g = Graph {
        nodes: vec![node("n1", "input", json!({})), node("n2", "conv2d", json!({"filters": 32})), node("n3", "pool2d", json!({})),
                    node("n4", "conv2d", json!({"filters": 32})), node("n5", "global_pool", json!({})), node("n6", "output", json!({}))],
        edges: (1..6).map(|i| Edge { id: format!("e{i}"), source: format!("n{i}"), target: format!("n{}", i + 1) }).collect(),
    };
    let ds = Dataset { source: "kaggle".into(), kaggle_slug: "ci/shapes".into(), image_size: [32, 32], channels: 3,
        classes: vec!["a".into(), "b".into()], ..Default::default() };
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let t = TrainSettings { epochs: 3, batch_size: 4, amp: false, ..Default::default() };
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(format!("{dir}/train.py"), train_script(&g, &ds, &a, &t, "ci resume check").unwrap()).unwrap();
}

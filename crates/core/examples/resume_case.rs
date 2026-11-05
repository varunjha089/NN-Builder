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
    // "early": a script with a validation split and early stopping (python/ci_early_stop_check.py)
    let mode = std::env::args().nth(2).unwrap_or_default();
    let early = mode == "early";
    let node = |id: &str, kind: &str, p: serde_json::Value| Node { id: id.into(), kind: kind.into(), params: nnb_core::model::defaults(kind).as_object().cloned()
        .map(|mut d| { d.extend(p.as_object().unwrap().clone()); d }).unwrap_or_default(), ..Default::default() };
    let g = Graph {
        nodes: vec![node("n1", "input", json!({})), node("n2", "conv2d", json!({"filters": 32})), node("n3", "pool2d", json!({})),
                    node("n4", "conv2d", json!({"filters": 32})), node("n5", "global_pool", json!({})), node("n6", "output", json!({}))],
        edges: (1..6).map(|i| Edge { id: format!("e{i}"), source: format!("n{i}"), target: format!("n{}", i + 1) }).collect(),
        ..Default::default()
    };
    // the recipe check uses six classes, so that top-5 accuracy is measured too
    let names: &[&str] = if mode == "recipe" { &["a", "b", "c", "d", "e", "f"] } else { &["a", "b"] };
    let ds = Dataset { source: "kaggle".into(), kaggle_slug: "ci/shapes".into(), image_size: [32, 32], channels: 3,
        classes: names.iter().map(|s| s.to_string()).collect(), ..Default::default() };
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let t = if mode == "recipe" {
        // python/ci_recipe_check.py: AdamW + weight decay, 1 warm-up epoch then cosine, label smoothing, gradient clipping
        TrainSettings { epochs: 4, batch_size: 32, amp: false, optimizer: "adamw".into(), weight_decay: 0.05, schedule: "cosine".into(),
            warmup_epochs: 1.0, label_smoothing: 0.1, grad_clip: 1.0, export: true, ..Default::default() }
    } else if mode == "split" {
        // python/ci_memory_check.py: the same run with and without batches split for memory must end with the same weights
        TrainSettings { epochs: 2, batch_size: 32, amp: false, ..Default::default() }
    } else if early {
        TrainSettings { epochs: 25, batch_size: 16, amp: false, early_stop: true, patience: 2, monitor: "val_acc".into(), val_fraction: 0.2, ..Default::default() }
    } else {
        TrainSettings { epochs: 3, batch_size: 4, amp: false, ..Default::default() }
    }.clean();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(format!("{dir}/train.py"), train_script(&g, &ds, &a, &t, "ci resume check").unwrap()).unwrap();
}

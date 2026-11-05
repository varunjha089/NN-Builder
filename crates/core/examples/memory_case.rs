//! Writes train.py files with "save memory" (activation checkpointing) for python/ci_memory_check.py (GitHub Actions only, never on the
//! owner's Mac): demos of every shape of graph (chain, ResNet skips, DenseNet / U-Net concatenations, Inception branches, transformers),
//! whose checkpointed forward must give exactly the outputs and gradients of the plain one.
//!     cargo run -p nnb-core --example memory_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings};
use nnb_core::dataset::Dataset;
use nnb_core::shapes::analyze;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    let t = TrainSettings { save_memory: true, ..Default::default() };
    for key in ["lenet5", "resnet18", "densenet_small", "googlenet", "unet", "vit_tiny", "msg_former"] {
        let tp = nnb_core::templates::get(key).unwrap();
        let g = nnb_core::templates::build(key).unwrap();
        let ds = if tp.task == "segment" {
            Dataset { source: "huggingface".into(), task: "segment".into(), hf_repo: "x/y".into(), image_size: [tp.input[1], tp.input[2]],
                channels: tp.input[0], classes: vec!["background".into(), "object".into()], ..Default::default() }
        } else {
            Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [tp.input[1], tp.input[2]], channels: tp.input[0],
                classes: (0..10).map(|i| format!("c{i}")).collect(), ..Default::default() }
        };
        let a = analyze(&g, &ds);
        assert!(a.ok, "{key}: {:?}", a.problems);
        let d = format!("{dir}/{key}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, &format!("ci memory {key}")).unwrap()).unwrap();
    }
}

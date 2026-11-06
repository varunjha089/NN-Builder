//! Writes train.py files for python/ci_pretrained_check.py (GitHub Actions only, never on the owner's Mac): the ResNet-18, VGG-16 and
//! ViT-Base/16 demos with 1000 classes and torchvision's ImageNet weights, so the check can load the weights and compare the outputs with
//! torchvision's own model.
//!     cargo run -p nnb-core --example pretrained_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings, PRETRAINED};
use nnb_core::dataset::Dataset;
use nnb_core::shapes::analyze;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    for (weights, template, _) in PRETRAINED {
        let g = nnb_core::templates::build(template).unwrap();
        let ds = Dataset { source: "huggingface".into(), hf_repo: "x/imagenet".into(), image_size: [224, 224], channels: 3,
            classes: (0..1000).map(|i| format!("c{i}")).collect(), ..Default::default() };
        let a = analyze(&g, &ds);
        assert!(a.ok, "{template}: {:?}", a.problems);
        let t = TrainSettings { pretrained: weights.to_string(), ..Default::default() }.clean();
        let d = format!("{dir}/{weights}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, &format!("ci {weights}")).unwrap()).unwrap();
        if *weights == "vit_b_16" {
            // the same with a LoRA adapter (rank 4) on every attention: the weights must still load, and B = 0 leaves the outputs unchanged
            let mut g = g.clone();
            for n in g.nodes.iter_mut().filter(|n| n.kind == "attention") {
                n.params.insert("lora".into(), serde_json::json!(4));
            }
            let a = analyze(&g, &ds);
            let d = format!("{dir}/vit_b_16_lora");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, "ci vit_b_16 lora").unwrap()).unwrap();
        }
    }
}

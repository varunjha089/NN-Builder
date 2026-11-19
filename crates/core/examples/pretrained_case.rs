//! Writes train.py files for python/ci_pretrained_check.py (GitHub Actions only, never on the owner's Mac): the ResNet-18, VGG-16 and
//! ViT-Base/16 demos with 1000 classes and torchvision's ImageNet weights, so the check can load the weights and compare the outputs with
//! torchvision's own model. And doc/examples/unet_resnet18.nnb (a U-Net with a ResNet-18 encoder) on 13 bands of radar, optical and indices,
//! with ImageNet's and SSL4EO-S12's Sentinel-2 ResNet-18 weights in its encoder (folders unet_resnet18 and unet_ssl4eo_s2).
//!     cargo run -p nnb-core --example pretrained_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings, PRETRAINED};
use nnb_core::dataset::Dataset;
use nnb_core::shapes::analyze;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    for (weights, template, _) in PRETRAINED.iter().filter(|(w, _, _)| !w.starts_with("ssl4eo")) {
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
    // the U-Net with a ResNet-18 encoder, 13 bands at 64×64: the pretrained weights fill only its encoder
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/examples/unet_resnet18.nnb")).unwrap();
    let g = nnb_core::dsl::apply(&text, &Default::default()).unwrap();
    let bands: Vec<String> = ["VV", "VH", "VV_pre", "VH_pre", "B2", "B3", "B4", "B8", "B11", "B12", "NDWI", "MNDWI", "present"].iter().map(|b| b.to_string()).collect();
    let ds = Dataset { source: "local".into(), task: "segment".into(), local_dir: "/data/x".into(), local_on_machine: true, kaggle_train_dir: "images".into(),
        kaggle_mask_dir: "masks".into(), image_size: [64, 64], channels: 13, sensor: "other".into(), band_order: bands.clone(), bands,
        classes: vec!["land".into(), "water".into()], ..Default::default() };
    let a = analyze(&g, &ds);
    assert!(a.ok, "unet_resnet18: {:?}", a.problems);
    for weights in ["resnet18", "ssl4eo_s2"] {
        let t = TrainSettings { pretrained: weights.into(), ..Default::default() }.clean();
        let d = format!("{dir}/unet_{weights}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, &format!("ci unet {weights}")).unwrap()).unwrap();
    }
}

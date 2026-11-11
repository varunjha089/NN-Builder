//! For the manual's Dataset-tab pictures (sec:data-kinds): prints, as JSON, what the window's app_state and project_load would return for
//! a project set up for one kind of data (the real dataset, design and shape analysis). doc/screens/render.sh shows the real UI with
//! these answers in headless Chrome, so no app window has to come to the front.
//!     cargo run -p nnb-core --example doc_screens -- <builtin | huggingface | folders | s2 | fusion | scene | multilabel | segfiles
//!         | build_classifier | build_hybrid | build_segment | build_change>
use nnb_core::dataset::{Dataset, BUILTINS, SENTINEL2_GEE};
use nnb_core::{codegen::TrainSettings, model, shapes};
use serde_json::json;

/// a design text from doc/examples (the manual's worked examples)
fn example(name: &str) -> &'static str {
    let path = format!("{}/../../doc/examples/{name}.nnb", env!("CARGO_MANIFEST_DIR"));
    Box::leak(std::fs::read_to_string(path).expect("example").into_boxed_str())
}

fn main() {
    let case = std::env::args().nth(1).expect("case");
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let data = "/Users/Shared/nnb-screens/data";
    let worldcover = s(&["tree cover", "shrubland", "grassland", "cropland", "built-up", "bare / sparse vegetation", "snow and ice",
        "permanent water", "herbaceous wetland", "mangroves", "moss and lichen"]);
    let base = Dataset { source: "local".into(), image_size: [64, 64], ..Default::default() };
    let s2 = Dataset { sensor: "s2".into(), band_order: s(SENTINEL2_GEE), bands: s(&["B2", "B3", "B4", "B8", "B11", "B12"]), ..base.clone() };
    let (ds, text) = match case.as_str() {
        "builtin" => (Dataset { source: "builtin".into(), builtin: "cifar10".into(), ..Default::default() },
            "1 input\n2 conv 32 3x3\n3 maxpool 2x2\n4 conv 64 3x3\n5 global_pool\n6 output\n"),
        "huggingface" => (Dataset { source: "huggingface".into(), hf_repo: "timm/eurosat-rgb".into(), hf_test_split: "test".into(),
            classes: s(&["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]),
            ..base.clone() }, "1 input\n2 conv 32 3x3\n3 maxpool 2x2\n4 conv 64 3x3\n5 global_pool\n6 output\n"),
        "folders" => (Dataset { local_dir: format!("{data}/flowers"), kaggle_train_dir: "train".into(), kaggle_test_dir: "test".into(),
            classes: s(&["daisy", "dandelion", "rose", "sunflower", "tulip"]), ..base.clone() }, "1 input\n2 conv 32 3x3\n3 maxpool 2x2\n4 conv 64 3x3\n5 global_pool\n6 output\n"),
        "s2" => (Dataset { local_dir: format!("{data}/sentinel2_patches"), classes: s(&["cropland", "forest", "urban", "water"]), ..s2.clone() },
            "1 input\n2 conv 32 3x3\n3 maxpool 2x2\n4 conv 64 3x3\n5 global_pool\n6 output\n"),
        "fusion" => (Dataset { local_dir: format!("{data}/flood_s1_s2"), kaggle_train_dir: "s2".into(), sensor2: "s1".into(),
            band_order2: s(&["VV", "VH"]), bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(), classes: s(&["flooded", "dry"]), ..s2.clone() },
            "net \"S1 + S2 mid fusion\"\n1 input 1\n2 conv 32 3x3\n3 input 2\n4 conv 16 3x3\n5 concat <- 2, 4\n6 conv 64 3x3 stride 2\n7 global_pool\n8 output\n"),
        "scene" => (Dataset { task: "segment".into(), layout: "scene".into(), local_dir: format!("{data}/gee_export"), scene_image: "nnb_image".into(),
            scene_label: "nnb_label".into(), classes: worldcover.clone(), image_size: [128, 128], ..s2.clone() },
            "1 input\n2 conv 32 3x3\n3 conv 32 3x3\n4 conv 32 3x3 dil 2\n5 seg_output\n"),
        "segfiles" => (Dataset { task: "segment".into(), local_dir: format!("{data}/field_boundaries"), kaggle_train_dir: "images".into(),
            kaggle_mask_dir: "masks".into(), classes: s(&["background", "field"]), ..s2.clone() }, "1 input\n2 conv 32 3x3\n3 conv 32 3x3\n4 seg_output\n"),
        "multilabel" => (Dataset { task: "multilabel".into(), local_dir: format!("{data}/bigearthnet_like"), labels_csv: "labels.csv".into(),
            kaggle_train_dir: "s2".into(), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]), bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(),
            classes: s(&["arable land", "coniferous forest", "inland waters", "pastures", "urban fabric"]), image_size: [120, 120], ..s2.clone() },
            "1 input\n2 conv 32 3x3\n3 maxpool 2x2\n4 conv 64 3x3\n5 global_pool\n6 output\n"),
        // the worked examples of sec:build, drawn on the Design tab (doc/screens/render.sh designs)
        "build_classifier" => (Dataset { source: "huggingface".into(), hf_repo: "timm/eurosat-rgb".into(), hf_test_split: "test".into(),
            classes: s(&["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]),
            ..base.clone() }, example(&case)),
        "build_hybrid" => (Dataset { source: "builtin".into(), builtin: "cifar10".into(), ..Default::default() }, example(&case)),
        "build_segment" => (Dataset { task: "segment".into(), local_dir: format!("{data}/water_masks"), kaggle_train_dir: "images".into(),
            kaggle_mask_dir: "masks".into(), classes: s(&["not water", "water"]), image_size: [128, 128], ..s2.clone() }, example(&case)),
        "build_change" => (Dataset { local_dir: format!("{data}/two_dates"), kaggle_train_dir: "before".into(), bands: s(&["B2", "B3", "B4", "B8"]),
            sensor2: "s2".into(), band_order2: s(SENTINEL2_GEE), bands2: s(&["B2", "B3", "B4", "B8"]), fuse_dir: "after".into(),
            classes: s(&["changed", "unchanged"]), ..s2.clone() }, example(&case)),
        other => panic!("unknown case {other}"),
    };
    assert!(ds.problems().is_empty(), "{case}: {:?}", ds.problems());
    let g = nnb_core::dsl::apply(text, &Default::default()).unwrap();
    let a = shapes::analyze(&g, &ds);
    let name = format!("{} data", case);
    let path = format!("~/NN_Builder_Projects/{name}");
    let state = json!({
        "version": env!("CARGO_PKG_VERSION"), "home": "~/Library/Application Support/NN Builder", "projects": [{"path": path, "name": name, "exists": true}],
        "projects_dir": "~/NN_Builder_Projects", "design_size": {"max_blocks": 5000, "recommended": 5000, "own": false, "ram_gb": 16.0},
        "project": {"name": name, "path": path, "draft": false}, "tools_ready": true,
        "secrets": {"kaggle": "your-kaggle-name", "huggingface": true, "aws": null}, "machines": [],
        "builtins": BUILTINS.iter().map(|b| json!({"key": b.key, "label": b.label, "shape": b.shape, "classes": b.classes})).collect::<Vec<_>>(),
        "kinds": model::KINDS.iter().map(|(k, t)| json!({"kind": k, "title": t, "defaults": model::defaults(k)})).collect::<Vec<_>>(),
        "activations": model::ACTIVATIONS,
    });
    let project = json!({"name": name, "path": path, "graph": g, "dataset": ds, "train": TrainSettings::default(), "analysis": a,
        "dataset_problems": ds.problems()});
    println!("{}", json!({"app_state": state, "project_load": project}));
}

//! Writes train.py files for python/ci_multiband_check.py (GitHub Actions only, never on the owner's Mac): a small CNN on generated
//! Sentinel-2 (13 bands, GeoTIFF and .npy) and Sentinel-1 (VV, VH in linear units, converted to dB) image folders; and the owner's own
//! files as Google Earth Engine exports them: Sentinel-2 with B1 … B12 names and empty (NaN) pixels in class folders (copied to
//! <data>/local/<key>), Sentinel-1 already in dB, a split scene + label image (segmentation tiles) and multiband images + masks.
//!     cargo run -p nnb-core --example multiband_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings};
use nnb_core::dataset::{Dataset, SENTINEL1_BANDS, SENTINEL2_BANDS, SENTINEL2_GEE};
use nnb_core::shapes::analyze;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    std::fs::create_dir_all(&dir).unwrap();
    // own files are given by their full path, as the Dataset tab requires
    let dir = std::fs::canonicalize(&dir).unwrap().to_string_lossy().to_string();
    let text = "net \"multiband ci\"\n1 input\n2 conv 16 3x3\n3 maxpool 2x2\n4 conv 32 3x3\n5 global_pool\n6 output\n";
    let g = nnb_core::dsl::apply(text, &Default::default()).unwrap();
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let cases = [
        ("s2", Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel2".into(), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B08", "B11", "B12"]), classes: s(&["forest", "water"]), ..Default::default() }),
        ("s1", Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel1".into(), image_size: [32, 32], sensor: "s1".into(), s1_db: true,
            band_order: s(SENTINEL1_BANDS), bands: s(SENTINEL1_BANDS), classes: s(&["flood", "land"]), ..Default::default() }),
        ("gee", Dataset { source: "local".into(), local_dir: format!("{dir}/own/gee s2"), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_GEE), bands: s(&["B04", "B03", "B02", "B8", "B11"]), classes: s(&["forest", "water"]), ..Default::default() }),
        ("s1gee", Dataset { source: "local".into(), local_dir: format!("{dir}/own/s1"), local_on_machine: true, image_size: [32, 32], sensor: "s1".into(),
            s1_db: true, band_order: s(&["VV", "VH", "angle"]), bands: s(SENTINEL1_BANDS), classes: s(&["flood", "land"]), ..Default::default() }),
        ("scene", Dataset { source: "local".into(), task: "segment".into(), layout: "scene".into(), local_dir: format!("{dir}/own/scene"),
            local_on_machine: true, scene_image: "nnb_image".into(), scene_label: "nnb_label".into(), scene_split: "strip".into(), image_size: [16, 16], sensor: "s2".into(),
            band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B4", "B8"]), classes: s(&["land", "water", "forest"]), ..Default::default() }),
        ("segfiles", Dataset { source: "local".into(), task: "segment".into(), local_dir: format!("{dir}/own/segfiles"), local_on_machine: true,
            kaggle_train_dir: "images".into(), kaggle_mask_dir: "masks".into(), image_size: [16, 16], sensor: "other".into(),
            band_order: s(&["b1", "b2", "b3", "b4", "b5"]), bands: s(&["b2", "b5"]), classes: s(&["background", "field"]), ..Default::default() }),
        // S1 + S2 fusion, mid: one Input per band group (the classes differ only in the radar group)
        ("fusion", Dataset { source: "local".into(), local_dir: format!("{dir}/own/fusion"), local_on_machine: true, kaggle_train_dir: "s2".into(),
            image_size: [32, 32], sensor: "s2".into(), band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B3", "B4", "B8"]),
            sensor2: "s1".into(), band_order2: s(&["VV", "VH"]), bands2: s(&["VV", "VH"]), s1_db2: true, fuse_dir: "s1".into(),
            classes: s(&["flood", "land"]), ..Default::default() }),
        // multi-label (BigEarthNet-like): a labels CSV, S2 + S1 stacked into one Input (early fusion)
        ("multilabel", Dataset { source: "local".into(), task: "multilabel".into(), local_dir: format!("{dir}/own/ml"), local_on_machine: true,
            labels_csv: "labels.csv".into(), kaggle_train_dir: "s2".into(), image_size: [24, 24], sensor: "s2".into(),
            band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B4", "B8"]), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]),
            bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(), classes: s(&["forest", "water", "urban"]), ..Default::default() }),
        // unbalanced classes (10 : 1), trained with inverse-frequency class weights
        ("imbalanced", Dataset { source: "local".into(), local_dir: format!("{dir}/own/imb"), local_on_machine: true, image_size: [16, 16],
            kaggle_train_dir: "train".into(), kaggle_test_dir: "test".into(),
            sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), classes: s(&["common", "rare"]), ..Default::default() }),
        // BigEarthNet v2's layout: a folder of single-band GeoTIFFs per patch (10 m and 20 m sizes), metadata.parquet with label lists, a
        // split column and s1_name pairing the Sentinel-1 patch folders (fusion); a class name with commas
        ("ben", Dataset { source: "local".into(), task: "multilabel".into(), local_dir: format!("{dir}/own/ben"), local_on_machine: true,
            labels_csv: "metadata.parquet".into(), kaggle_train_dir: "BigEarthNet-S2".into(), image_size: [24, 24], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B05", "B08"]), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]),
            bands2: s(&["VV", "VH"]), fuse_dir: "BigEarthNet-S1".into(),
            classes: s(&["Beaches, dunes, sands", "Coniferous forest", "Inland waters"]), ..Default::default() }),
        // a Hugging Face dataset whose image column holds band arrays (a local copy in parquet, so CI needs no network for it)
        ("hfbands", Dataset { source: "huggingface".into(), hf_repo: "ci/hfbands".into(), hf_test_split: "test".into(), hf_image_col: "bands".into(),
            image_size: [16, 16], sensor: "other".into(), band_order: s(&["b1", "b2", "b3", "b4"]), bands: s(&["b2", "b4"]),
            classes: s(&["dark", "bright"]), ..Default::default() }),
        // files "in the cloud": the machine fetches them itself (file:// here; gs:// and Drive links take the same path into the cache)
        ("cloud", Dataset { source: "local".into(), cloud_url: format!("file://{dir}/own/cloudsrc"), image_size: [16, 16], sensor: "other".into(),
            band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), classes: s(&["common", "rare"]), kaggle_train_dir: "train".into(), ..Default::default() }),
        // Check data on Sentinel-1 left in linear units (the report must say so), with an extra constant band
        ("check", Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel1".into(), image_size: [32, 32], sensor: "s1".into(),
            band_order: s(SENTINEL1_BANDS), bands: s(SENTINEL1_BANDS), classes: s(&["flood", "land"]), ..Default::default() }),
    ];
    let mid = nnb_core::dsl::apply("net \"mid fusion ci\"\n1 input 1\n2 conv 16 3x3\n3 maxpool 2x2\n4 input 2\n5 conv 8 3x3\n6 maxpool 2x2\n\
        7 concat <- 3, 6\n8 conv 32 3x3\n9 global_pool\n10 output\n", &Default::default()).unwrap();
    let seg = nnb_core::dsl::apply("net \"multiband seg ci\"\n1 input\n2 conv 16 3x3\n3 conv 16 3x3\n4 seg_output\n", &Default::default()).unwrap();
    // ImageNet's ResNet-18 on 4 Sentinel-2 bands: its first convolution widened from 3 to 4 channels
    {
        let ds = Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel2".into(), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B08"]), classes: s(&["forest", "water"]), ..Default::default() };
        let g = nnb_core::templates::build("resnet18").unwrap();
        let a = analyze(&g, &ds);
        assert!(a.ok, "resnet18 on 4 bands: {:?}", a.problems);
        let t = TrainSettings { pretrained: "resnet18".into(), epochs: 1, batch_size: 16, amp: false, ..Default::default() }.clean();
        let d = format!("{dir}/pretrained_ms");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, "pretrained_ms").unwrap()).unwrap();
    }
    for (name, ds) in cases {
        assert!(ds.problems().is_empty(), "{name}: {:?}", ds.problems());
        let g = if ds.task == "segment" { &seg } else if name == "fusion" { &mid } else { &g };
        let a = analyze(g, &ds);
        assert!(a.ok, "{name}: {:?}", a.problems);
        assert_eq!(a.input, vec![ds.bands.len() + ds.bands2.len(), ds.image_size[0], ds.image_size[1]]);
        let t = TrainSettings { check_only: name == "check",
            class_weights: match name { "imbalanced" => "inverse".into(), "multilabel" => "sqrt".into(), _ => String::new() },
            tune_thresholds: name == "multilabel", val_fraction: if name == "multilabel" { 0.15 } else { 0.0 }, epochs: match ds.task.as_str() { "segment" => 10, "multilabel" => 8, _ => 3 }, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() };
        let d = format!("{dir}/{name}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(g, &ds, &a, &t, name).unwrap()).unwrap();
        if ds.source == "local" && !ds.local_on_machine {
            std::fs::write(format!("{d}/local_key.txt"), ds.local_key()).unwrap(); // where the app would copy the folder: <data>/local/<key>
        }
    }
}

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
            local_on_machine: true, scene_image: "nnb_image".into(), scene_label: "nnb_label".into(), image_size: [16, 16], sensor: "s2".into(),
            band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B4", "B8"]), classes: s(&["land", "water", "forest"]), ..Default::default() }),
        ("segfiles", Dataset { source: "local".into(), task: "segment".into(), local_dir: format!("{dir}/own/segfiles"), local_on_machine: true,
            kaggle_train_dir: "images".into(), kaggle_mask_dir: "masks".into(), image_size: [16, 16], sensor: "other".into(),
            band_order: s(&["b1", "b2", "b3", "b4", "b5"]), bands: s(&["b2", "b5"]), classes: s(&["background", "field"]), ..Default::default() }),
    ];
    let seg = nnb_core::dsl::apply("net \"multiband seg ci\"\n1 input\n2 conv 16 3x3\n3 conv 16 3x3\n4 seg_output\n", &Default::default()).unwrap();
    for (name, ds) in cases {
        assert!(ds.problems().is_empty(), "{name}: {:?}", ds.problems());
        let g = if ds.task == "segment" { &seg } else { &g };
        let a = analyze(g, &ds);
        assert!(a.ok, "{name}: {:?}", a.problems);
        assert_eq!(a.input, vec![ds.bands.len(), ds.image_size[0], ds.image_size[1]]);
        let t = TrainSettings { epochs: if ds.task == "segment" { 10 } else { 3 }, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() };
        let d = format!("{dir}/{name}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(g, &ds, &a, &t, name).unwrap()).unwrap();
        if ds.source == "local" && !ds.local_on_machine {
            std::fs::write(format!("{d}/local_key.txt"), ds.local_key()).unwrap(); // where the app would copy the folder: <data>/local/<key>
        }
    }
}

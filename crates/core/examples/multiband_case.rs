//! Writes train.py files for python/ci_multiband_check.py (GitHub Actions only, never on the owner's Mac): a small CNN on generated
//! Sentinel-2 (13 bands, GeoTIFF and .npy) and Sentinel-1 (VV, VH in linear units, converted to dB) image folders.
//!     cargo run -p nnb-core --example multiband_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings};
use nnb_core::dataset::{Dataset, SENTINEL1_BANDS, SENTINEL2_BANDS};
use nnb_core::shapes::analyze;

fn main() {
    let dir = std::env::args().nth(1).expect("folder");
    let text = "net \"multiband ci\"\n1 input\n2 conv 16 3x3\n3 maxpool 2x2\n4 conv 32 3x3\n5 global_pool\n6 output\n";
    let g = nnb_core::dsl::apply(text, &Default::default()).unwrap();
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let cases = [
        ("s2", Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel2".into(), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B08", "B11", "B12"]), classes: s(&["forest", "water"]), ..Default::default() }),
        ("s1", Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel1".into(), image_size: [32, 32], sensor: "s1".into(), s1_db: true,
            band_order: s(SENTINEL1_BANDS), bands: s(SENTINEL1_BANDS), classes: s(&["flood", "land"]), ..Default::default() }),
    ];
    for (name, ds) in cases {
        assert!(ds.problems().is_empty(), "{name}: {:?}", ds.problems());
        let a = analyze(&g, &ds);
        assert!(a.ok, "{name}: {:?}", a.problems);
        assert_eq!(a.input, vec![ds.bands.len(), 32, 32]);
        let t = TrainSettings { epochs: 3, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() };
        let d = format!("{dir}/{name}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, name).unwrap()).unwrap();
    }
}

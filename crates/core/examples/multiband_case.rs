//! Writes train.py files for python/ci_multiband_check.py (GitHub Actions only, never on the owner's Mac): a small CNN on generated
//! Sentinel-2 (13 bands, GeoTIFF and .npy) and Sentinel-1 (VV, VH in linear units, converted to dB) image folders; and the owner's own
//! files as Google Earth Engine exports them: Sentinel-2 with B1 … B12 names and empty (NaN) pixels in class folders (copied to
//! <data>/local/<key>), Sentinel-1 already in dB, a split scene + label image (segmentation tiles) and multiband images + masks.
//!     cargo run -p nnb-core --example multiband_case -- <folder>
use nnb_core::codegen::{train_script, TrainSettings};
use nnb_core::dataset::{Dataset, Head, SENTINEL1_BANDS, SENTINEL2_BANDS, SENTINEL2_GEE};
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
        // the same, with Sentinel-1 at its own size (16×16 next to S2's 32×32): two separate inputs, not stacked on one grid
        ("fusion_sizes", Dataset { source: "local".into(), local_dir: format!("{dir}/own/fusion"), local_on_machine: true, kaggle_train_dir: "s2".into(),
            image_size: [32, 32], sensor: "s2".into(), band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B3", "B4", "B8"]),
            sensor2: "s1".into(), band_order2: s(&["VV", "VH"]), bands2: s(&["VV", "VH"]), s1_db2: true, fuse_dir: "s1".into(),
            image_size2: vec![16, 16], classes: s(&["flood", "land"]), ..Default::default() }),
        // multi-label (BigEarthNet-like): a labels CSV, S2 + S1 stacked into one Input (early fusion)
        ("multilabel", Dataset { source: "local".into(), task: "multilabel".into(), local_dir: format!("{dir}/own/ml"), local_on_machine: true,
            labels_csv: "labels.csv".into(), kaggle_train_dir: "s2".into(), image_size: [24, 24], sensor: "s2".into(),
            band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B4", "B8"]), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]),
            bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(), classes: s(&["forest", "water", "urban"]), ..Default::default() }),
        // unbalanced classes (10 : 1), trained with inverse-frequency class weights
        ("imbalanced", Dataset { source: "local".into(), local_dir: format!("{dir}/own/imb"), local_on_machine: true, image_size: [16, 16],
            kaggle_train_dir: "train".into(), kaggle_test_dir: "test".into(),
            sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), classes: s(&["common", "rare"]), ..Default::default() }),
        // regression: two numbers per image from columns of a labels CSV (one row has no number), trained with Huber loss
        ("regress", Dataset { source: "local".into(), task: "regress".into(), local_dir: format!("{dir}/own/reg"), local_on_machine: true,
            labels_csv: "targets.csv".into(), image_size: [16, 16], sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]),
            classes: s(&["biomass", "height"]), ..Default::default() }),
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
    // S1 at half S2's size: only the S2 branch is pooled, so both reach the join at 16×16
    let mid_sizes = nnb_core::dsl::apply("net \"mid fusion, own sizes ci\"\n1 input 1\n2 conv 16 3x3\n3 maxpool 2x2\n4 input 2\n5 conv 8 3x3\n\
        6 concat <- 3, 5\n7 conv 32 3x3\n8 global_pool\n9 output\n", &Default::default()).unwrap();
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
    // the design's own PyTorch code: a squeeze-and-excitation class and a strided down-sampling class (# out: declares the shape), with
    // a custom loss; and the same with a wrong # out: line, which the one-batch check on the machine must catch
    {
        let ds = Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel2".into(), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B08", "B11", "B12"]), classes: s(&["forest", "water"]), ..Default::default() };
        let code = "class SE(nn.Module):\n    # out: c, h, w\n    def __init__(self, c, h, w):\n        super().__init__()\n\
            \x20       self.fc = nn.Sequential(nn.Linear(c, c // 4), nn.ReLU(), nn.Linear(c // 4, c), nn.Sigmoid())\n\n\
            \x20   def forward(self, x):\n        return x * self.fc(x.mean((2, 3)))[:, :, None, None]\n\n\n\
            class Down(nn.Module):\n    # out: c * 2, h // 2, w // 2\n    def __init__(self, c, h, w):\n        super().__init__()\n\
            \x20       self.conv = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)\n\n\
            \x20   def forward(self, x):\n        return torch.relu(self.conv(x))\n";
        let text = format!("net \"custom ci\"\npython \"\"\"\n{code}\"\"\"\n1 input\n2 conv 16 3x3\n3 custom SE\n4 custom Down\n5 global_pool\n6 output\n");
        let loss = "def custom_loss(out, y):\n    # cross-entropy plus a small penalty on large scores\n    return nn.functional.cross_entropy(out, y) + 1e-3 * (out ** 2).mean()\n";
        for (name, text) in [("custom", text.clone()), ("custom_bad", text.replace("c * 2, h // 2, w // 2", "c * 2, h, w"))] {
            let g = nnb_core::dsl::apply(&text, &Default::default()).unwrap();
            let a = analyze(&g, &ds);
            assert!(a.ok, "{name}: {:?}", a.problems);
            let t = TrainSettings { epochs: 3, batch_size: 16, amp: false, loss: "custom".into(), custom_loss: loss.into(), ..Default::default() }.clean();
            let d = format!("{dir}/{name}");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, name).unwrap()).unwrap();
        }
    }
    // several outputs: the class from the image's folder, plus biomass and height (regression, Huber) and dry / wet (a class) from
    // labels.csv, each with its own Output block
    {
        let ds = Dataset { source: "local".into(), local_dir: format!("{dir}/own/mt"), local_on_machine: true, labels_csv: "labels.csv".into(),
            image_size: [16, 16], sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), classes: s(&["forest", "water"]),
            heads: vec![
                Head { name: "biomass".into(), task: "regress".into(), columns: s(&["biomass", "height"]), loss: "huber".into(), weight: 0.5, ..Default::default() },
                Head { name: "wet".into(), task: "classify".into(), columns: s(&["wet"]), classes: s(&["dry", "wet"]), weight: 1.0, ..Default::default() },
            ], ..Default::default() };
        let g = nnb_core::dsl::apply("net \"multi-task ci\"\n1 input\n2 conv 16 3x3\n3 maxpool 2x2\n4 conv 32 3x3\n5 global_pool\n6 output\n\
            7 output 1 <- 5\n8 output 2 <- 5\n", &Default::default()).unwrap();
        let a = analyze(&g, &ds);
        assert!(a.ok && ds.problems().is_empty(), "multitask: {:?} {:?}", a.problems, ds.problems());
        let t = TrainSettings { epochs: 12, batch_size: 16, amp: false, val_fraction: 0.1, aug_hflip: true, ..Default::default() }.clean();
        let d = format!("{dir}/multitask");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, "multitask").unwrap()).unwrap();
    }
    // change detection: before / after pairs (the same two bands in both groups), a Siamese encoder (shared weights) and |a - b|
    {
        let ds = Dataset { source: "local".into(), local_dir: format!("{dir}/own/cd"), local_on_machine: true, kaggle_train_dir: "before".into(),
            image_size: [24, 24], sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), sensor2: "other".into(),
            band_order2: s(&["b1", "b2"]), bands2: s(&["b1", "b2"]), fuse_dir: "after".into(), classes: s(&["changed", "same"]), ..Default::default() };
        let g = nnb_core::dsl::apply("net \"siamese ci\"\n1 input 1\n2 conv 16 3x3\n3 conv 16 3x3\n4 input 2\n5 shared 2\n6 shared 3\n\
            7 subtract abs <- 3, 6\n8 global_pool max\n9 output\n", &Default::default()).unwrap();
        let a = analyze(&g, &ds);
        assert!(a.ok && ds.problems().is_empty(), "siamese: {:?} {:?}", a.problems, ds.problems());
        let t = TrainSettings { epochs: 12, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() }.clean();
        let d = format!("{dir}/siamese");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, "siamese").unwrap()).unwrap();
    }
    // time series: 6 dates of 2 bands. Classes that differ only in when they green up (the season's mean is the same): a sequence path
    // (.npy class folders); and a crop-type map per pixel from GeoTIFFs whose bands go date by date, with masks (Conv3D + temporal attention)
    {
        let ts = |name: &str, task: &str| Dataset { source: "local".into(), task: task.into(), local_dir: format!("{dir}/own/{name}"), local_on_machine: true,
            image_size: [16, 16], sensor: "other".into(), band_order: s(&["b1", "b2"]), bands: s(&["b1", "b2"]), time_steps: 6,
            classes: s(&["early", "late"]), ..Default::default() };
        let seq = Dataset { kaggle_train_dir: "train".into(), ..ts("ts_seq", "classify") };
        let seg = Dataset { kaggle_train_dir: "images".into(), kaggle_mask_dir: "masks".into(), classes: s(&["background", "early", "late"]), ..ts("ts_seg", "segment") };
        for (name, ds, text) in [
            ("ts_seq", seq, "net \"time series ci\"\n1 input\n2 to_sequence\n3 conv1d 16 3\n4 gru 16\n5 token_pool mean\n6 output\n"),
            ("ts_seg", seg, "net \"crop map ci\"\n1 input\n2 conv3d 16 3x3 time 3\n3 conv3d 16 3x3 time 3\n4 temporal_pool attention\n5 conv 16 3x3\n6 seg_output\n"),
        ] {
            let g = nnb_core::dsl::apply(text, &Default::default()).unwrap();
            let a = analyze(&g, &ds);
            assert!(a.ok && ds.problems().is_empty(), "{name}: {:?} {:?}", a.problems, ds.problems());
            assert_eq!(a.input, vec![2, 6, 16, 16]);
            let t = TrainSettings { epochs: if name == "ts_seg" { 20 } else { 10 }, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() }.clean();
            let d = format!("{dir}/{name}");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, name).unwrap()).unwrap();
        }
    }
    // transformer additions, for the behaviour checks in ci_multiband_check (causal masks, rotary positions, sinusoidal positions, RMSNorm)
    {
        let ds = Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [16, 16], channels: 3, classes: s(&["a", "b"]), ..Default::default() };
        let g = nnb_core::dsl::apply("1 input\n2 conv 32 2x2 stride 2 valid none\n3 to_tokens\n4 cls_pos sinusoidal\n5 attention heads 4 rotary causal\n\
            6 conv 48 4x4 stride 4 valid none <- 1\n7 to_tokens\n8 cross_attention heads 4 <- 5, 7\n9 rmsnorm\n10 swiglu 64\n11 concat <- 10, 5\n\
            12 token_pool mean\n13 output\n", &Default::default()).unwrap();
        let a = analyze(&g, &ds);
        assert!(a.ok, "tokens: {:?}", a.problems);
        let d = format!("{dir}/tokens");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &TrainSettings::default(), "tokens").unwrap()).unwrap();
    }
    // object detection: pictures with red squares and blue discs, YOLO label files, a CenterNet-style head on a quarter-size map
    {
        let ds = Dataset { source: "local".into(), task: "detect".into(), local_dir: format!("{dir}/own/det"), local_on_machine: true,
            kaggle_train_dir: "images".into(), kaggle_mask_dir: "labels".into(), image_size: [64, 64], channels: 3,
            classes: s(&["square", "disc"]), ..Default::default() };
        let g = nnb_core::dsl::apply("net \"detector ci\"\n1 input\n2 conv 16 3x3\n3 conv 16 3x3\n4 maxpool 2x2\n5 conv 32 3x3\n6 conv 32 3x3\n\
            7 maxpool 2x2\n8 conv 32 3x3\n9 detect\n", &Default::default()).unwrap();
        let a = analyze(&g, &ds);
        assert!(a.ok && ds.problems().is_empty(), "detect: {:?} {:?}", a.problems, ds.problems());
        let t = TrainSettings { epochs: 30, batch_size: 16, amp: false, aug_hflip: true, aug_vflip: true, ..Default::default() }.clean();
        let d = format!("{dir}/detect");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, "detect").unwrap()).unwrap();
    }
    // several GPUs (CI: two CPU processes with gloo, NNB_TEST_PROCESSES=2): a whole run, and one interrupted by a spot notice and resumed
    {
        let ds = Dataset { source: "kaggle".into(), kaggle_slug: "ci/sentinel2".into(), image_size: [32, 32], sensor: "s2".into(),
            band_order: s(SENTINEL2_BANDS), bands: s(&["B02", "B03", "B04", "B08", "B11", "B12"]), classes: s(&["forest", "water"]), ..Default::default() };
        let a = analyze(&g, &ds);
        for (name, epochs) in [("ddp", 4), ("ddp_resume", 8)] {
            let t = TrainSettings { epochs, batch_size: 16, amp: false, gpus: 2, aug_hflip: true, ..Default::default() }.clean();
            let d = format!("{dir}/{name}");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(format!("{d}/train.py"), train_script(&g, &ds, &a, &t, name).unwrap()).unwrap();
        }
    }
    for (name, ds) in cases {
        assert!(ds.problems().is_empty(), "{name}: {:?}", ds.problems());
        let g = if ds.task == "segment" { &seg } else if name == "fusion" { &mid } else if name == "fusion_sizes" { &mid_sizes } else { &g };
        let a = analyze(g, &ds);
        assert!(a.ok, "{name}: {:?}", a.problems);
        assert_eq!(a.input, vec![ds.bands.len() + ds.bands2.len(), ds.image_size[0], ds.image_size[1]]);
        if name == "fusion_sizes" {
            let i2 = g.nodes.iter().find(|n| n.kind == "input" && n.params["group"] == 2).unwrap();
            assert_eq!(a.get(&i2.id).unwrap().out, Some(vec![2, 16, 16]), "group 2 at its own size");
        }
        let t = TrainSettings { check_only: name == "check",
            class_weights: match name { "imbalanced" => "inverse".into(), "multilabel" => "sqrt".into(), _ => String::new() },
            loss: match name { "imbalanced" | "multilabel" => "focal".into(), "segfiles" => "dice_ce".into(), "scene" => "dice".into(), "regress" => "huber".into(), _ => String::new() },
            tune_thresholds: name == "multilabel", val_fraction: match name { "multilabel" => 0.15, "regress" => 0.1, _ => 0.0 },
            early_stop: name == "regress", patience: 6, monitor: if name == "regress" { "val_acc".into() } else { "val_loss".into() },
            epochs: match ds.task.as_str() { "segment" => 10, "multilabel" => 8, "regress" => 15, _ => 3 }, batch_size: 16, amp: false, aug_hflip: true, aug_rotate: 10.0, ..Default::default() };
        let d = format!("{dir}/{name}");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(format!("{d}/train.py"), train_script(g, &ds, &a, &t, name).unwrap()).unwrap();
        if ds.source == "local" && !ds.local_on_machine {
            std::fs::write(format!("{d}/local_key.txt"), ds.local_key()).unwrap(); // where the app would copy the folder: <data>/local/<key>
        }
    }
}

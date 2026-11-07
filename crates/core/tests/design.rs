//! The shape engine and the code generator (no network, no PyTorch: the generated Python is only compiled, never run here).

use nnb_core::codegen::{model_code, train_script, TrainSettings};
use nnb_core::dataset::Dataset;
use nnb_core::model::{self, Edge, Graph, Node};
use nnb_core::remote::{self, Machine};
use nnb_core::shapes::analyze;
use serde_json::{json, Value};

fn node(id: &str, kind: &str, over: Value) -> Node {
    let mut p = model::defaults(kind).as_object().cloned().unwrap_or_default();
    for (k, v) in over.as_object().unwrap() {
        p.insert(k.clone(), v.clone());
    }
    Node { id: id.into(), kind: kind.into(), params: p, ..Default::default() }
}

fn chain(layers: Vec<(&str, Value)>) -> Graph {
    let mut g = Graph::default();
    for (i, (k, p)) in layers.into_iter().enumerate() {
        g.nodes.push(node(&format!("n{i}"), k, p));
        if i > 0 {
            g.edges.push(Edge { id: format!("e{i}"), source: format!("n{}", i - 1), target: format!("n{i}") });
        }
    }
    g
}

fn edge(g: &mut Graph, a: &str, b: &str) {
    let id = format!("e{}", g.edges.len() + 100);
    g.edges.push(Edge { id, source: a.into(), target: b.into() });
}

/// compile a Python file with python3 (or python on Windows); None when no Python is installed (the check is skipped)
fn py_compile(f: &std::path::Path) -> Option<std::process::Output> {
    ["python3", "python"].iter().find_map(|py| std::process::Command::new(py).args(["-m", "py_compile"]).arg(f).output().ok())
}

fn mnist() -> Dataset {
    Dataset::default()
}

#[test]
fn starter_design_is_ok() {
    let a = analyze(&model::starter(), &mnist());
    assert!(a.ok, "{:?}", a.problems);
    let shapes: Vec<_> = a.nodes.iter().map(|n| n.out.clone().unwrap()).collect();
    assert_eq!(shapes[1], vec![16, 28, 28]);
    assert_eq!(shapes[2], vec![16, 14, 14]);
    assert_eq!(shapes[5], vec![32 * 7 * 7]);
    assert_eq!(shapes.last().unwrap(), &vec![10]);
    assert_eq!(a.nodes[1].params, (3 * 3 + 1) * 16);
    assert_eq!(a.nodes[2].rf, Some([4, 4]));
    assert_eq!(a.total_params, a.nodes.iter().map(|n| n.params).sum::<u64>());
}

#[test]
fn messages_in_plain_words() {
    let g = chain(vec![
        ("input", json!({})),
        ("conv2d", json!({"kernel": [5, 5], "padding": "valid", "stride": [2, 2]})),
        ("pool2d", json!({})),
        ("conv2d", json!({"kernel": [7, 7], "padding": "valid"})),
        ("dense", json!({})),
        ("output", json!({})),
    ]);
    let a = analyze(&g, &mnist());
    assert_eq!(a.nodes[1].out, Some(vec![32, 12, 12]));
    assert_eq!(a.nodes[1].status, "warn"); // 28 - 5 = 23: one row dropped by stride 2
    assert!(a.nodes[1].formula[0].contains("= 12"));
    assert_eq!(a.nodes[3].status, "error");
    assert!(a.nodes[3].messages[0].contains("too small"));
    assert_eq!(a.nodes[4].status, "blocked");
    assert!(!a.ok);
    let g = chain(vec![("input", json!({})), ("dense", json!({})), ("output", json!({}))]);
    assert!(analyze(&g, &mnist()).nodes[1].messages[0].contains("Flatten"));
}

#[test]
fn residual_add_and_mismatch() {
    // input -> conv(16) -> conv(16) -> add(conv1, conv2) -> gap -> output
    let mut g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 16})), ("conv2d", json!({"filters": 16})), ("add", json!({})),
        ("global_pool", json!({})), ("output", json!({}))]);
    edge(&mut g, "n1", "n3");
    let a = analyze(&g, &mnist());
    assert!(a.ok, "{:?} {:?}", a.problems, a.nodes);
    assert_eq!(a.get("n3").unwrap().out, Some(vec![16, 28, 28]));
    // different channels -> error with advice
    g.nodes[2].params.insert("filters".into(), json!(8));
    let a = analyze(&g, &mnist());
    assert_eq!(a.get("n3").unwrap().status, "error");
    assert!(a.get("n3").unwrap().messages[0].contains("1×1 Conv2D"));
}

#[test]
fn small_unet_with_concat_and_transpose() {
    let mut g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 8})), ("pool2d", json!({})), ("conv2d", json!({"filters": 16})),
        ("convtranspose2d", json!({"filters": 8})), ("concat", json!({})), ("global_pool", json!({})), ("output", json!({}))]);
    edge(&mut g, "n1", "n5");
    let a = analyze(&g, &mnist());
    assert!(a.ok, "{:?}", a.nodes.iter().map(|n| (&n.id, &n.messages)).collect::<Vec<_>>());
    assert_eq!(a.get("n4").unwrap().out, Some(vec![8, 28, 28]));
    assert_eq!(a.get("n5").unwrap().out, Some(vec![16, 28, 28]));
    let code = model_code(&g, &a).unwrap();
    assert!(code.contains("torch.cat([t4, t1], dim=1)") || code.contains("torch.cat([t1, t4], dim=1)"), "{code}");
    assert!(code.contains("nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)"), "{code}");
}

#[test]
fn loops_unconnected_and_output_rules() {
    let mut g = model::starter();
    g.nodes.push(node("lonely", "dense", json!({})));
    let a = analyze(&g, &mnist());
    assert!(a.ok);
    assert_eq!(a.get("lonely").unwrap().status, "unconnected");
    edge(&mut g, "n4", "n3"); // back edge -> loop
    let a = analyze(&g, &mnist());
    assert!(!a.ok);
    assert!(a.problems[0].contains("loop"));
    let mut g = model::starter();
    g.nodes.retain(|n| n.kind != "output");
    g.edges.retain(|e| e.target != "n9");
    let a = analyze(&g, &mnist());
    assert!(a.problems.iter().any(|p| p.contains("Output")));
}

#[test]
fn custom_dataset_shape_and_classes() {
    let ds = Dataset { source: "huggingface".into(), hf_repo: "someone/flood-images".into(), image_size: [64, 96], channels: 3,
        classes: vec!["flooded".into(), "dry".into()], ..Default::default() };
    assert!(ds.problems().is_empty());
    let a = analyze(&model::starter(), &ds);
    assert_eq!(a.input, vec![3, 64, 96]);
    assert_eq!(a.nodes.last().unwrap().out, Some(vec![2]));
    let bad = Dataset { source: "kaggle".into(), kaggle_slug: "no-slash".into(), ..Default::default() };
    assert!(!bad.problems().is_empty());
}

#[test]
fn generated_python_compiles() {
    let g = model::starter();
    for ds in [mnist(), Dataset { source: "kaggle".into(), kaggle_slug: "owner/flood-area".into(), kaggle_train_dir: "data/train".into(),
        classes: vec!["a\"b".into(), "c'''d".into(), "e".into()], ..Default::default() }] {
        let a = analyze(&g, &ds);
        let code = train_script(&g, &ds, &a, &TrainSettings::default(), "nnb Test \"\"\" 1").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("train.py");
        std::fs::write(&f, &code).unwrap();
        if let Some(out) = py_compile(&f) {
            assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stderr), code);
        }
        assert!(code.contains("nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)") || ds.source != "builtin");
        assert!(code.contains("EXPECTED = {\"input\": ["), "{code}");
    }
}

#[test]
fn early_stopping_settings() {
    // early stopping always gets a validation part of the training data (never the test data)
    let t = TrainSettings { early_stop: true, val_fraction: 0.0, patience: 0, monitor: "loss?".into(), ..Default::default() }.clean();
    assert_eq!((t.val_fraction, t.patience, t.monitor.as_str()), (0.1, 1, "val_loss"));
    let off = TrainSettings::default().clean();
    assert!(!off.early_stop && off.val_fraction == 0.0, "off by default: earlier runs behave as before");
    // old train.json files without the new fields still load
    let old: TrainSettings = serde_json::from_str(r#"{"epochs":3,"batch_size":64,"lr":0.001,"optimizer":"adam","seed":1,"amp":true,"limit_train":0,"max_hours":6.0,"stop_instance":true}"#).unwrap();
    assert!(!old.early_stop && old.patience == 5 && old.monitor == "val_loss");
    assert!(old.schedule == "constant" && old.weight_decay == 0.0 && old.label_smoothing == 0.0 && old.grad_clip == 0.0 && old.warmup_epochs == 0.0,
        "old projects keep plain constant-rate training");
    let r = TrainSettings { epochs: 10, optimizer: "adamw".into(), schedule: "warp".into(), warmup_epochs: 50.0, weight_decay: -1.0,
        label_smoothing: 0.9, grad_clip: f64::NAN, ..Default::default() }.clean();
    assert_eq!((r.optimizer.as_str(), r.schedule.as_str(), r.warmup_epochs, r.weight_decay, r.label_smoothing, r.grad_clip),
        ("adamw", "constant", 5.0, 0.0, 0.5, 0.0));
    let code = train_script(&model::starter(), &mnist(), &analyze(&model::starter(), &mnist()),
        &TrainSettings { optimizer: "adamw".into(), schedule: "cosine".into(), warmup_epochs: 2.0, ..Default::default() }.clean(), "r").unwrap();
    assert!(code.contains("\"schedule\": \"cosine\"") && code.contains("\"optimizer\": \"adamw\""));
    let g = model::starter();
    let ds = mnist();
    let code = train_script(&g, &ds, &analyze(&g, &ds), &t, "es").unwrap();
    assert!(code.contains("\"early_stop\": True") && code.contains("\"val_fraction\": 0.1"), "{code}");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(out) = py_compile(&dir.path().join("train.py")) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn machine_validation_and_commands() {
    assert!(Machine { name: "x".into(), host: "a; rm -rf /".into(), ..Default::default() }.clean().is_err());
    assert!(Machine { name: "Kaggle".into(), host: "h".into(), ..Default::default() }.clean().is_err());
    assert!(Machine { name: "ec2".into(), kind: "aws".into(), region: "ap-south-1".into(), instance_id: "bad".into(), key: "/k.pem".into(), ..Default::default() }.clean().is_err());
    let m = Machine { name: "ec2".into(), kind: "aws".into(), region: "ap-south-1".into(), instance_id: "i-0abc".into(), key: "/k.pem".into(), ..Default::default() }
        .clean().unwrap();
    assert_eq!((m.user.as_str(), m.auth.as_str(), m.workdir.as_str()), ("ubuntu", "key", ".nn_builder"));
    let c = Machine { name: "college".into(), host: "gpu01.example.edu".into(), user: "s1".into(), auth: "password".into(), jump: "s1@gate.example.edu".into(),
        ..Default::default() }.clean().unwrap();
    let cc = c.connect_command();
    assert_eq!(&cc[..4], &["ssh", "-M", "-f", "-N"]);
    assert!(!cc.iter().any(|x| x.contains("BatchMode")));
    assert!(cc.contains(&"-J".to_string()));
    assert!(c.ssh_options().contains(&"BatchMode=yes".to_string()));
    assert_eq!(remote::remote_path(".nn_builder/runs/x"), "\"$HOME\"/.nn_builder/runs/x");
    assert_eq!(remote::sh_quote("a b'c"), "'a b'\\''c'");
    let p = remote::parse_prepare(&["NNB_GPU 0, Tesla T4, 15360 MiB, 0 MiB".into(), "NNB_TMUX 1".into(), "NNB_SYSTEMD 1".into(),
        "NNB_PYTHON /home/u/.nn_builder/venv/bin/python".into(), "NNB_TORCH 2.9.0 cuda 3.12.3".into()]);
    assert!(p.cuda && p.tmux && !p.linger);
    assert_eq!(remote::choose_launch(&c, &p), "tmux");
    assert!(remote::launch_note("tmux", &p).contains("lingering"));
    let secret = remote::put_secret_script(".kaggle/kaggle.json", "{\"key\":\"x\"}");
    assert!(secret.contains("chmod 600") && secret.contains("umask 077"));
}

/// the dataset a template is shown with: its own input size; segmentation templates get two classes
fn dataset_for(t: &nnb_core::templates::Template) -> Dataset {
    if t.task == "segment" {
        Dataset { source: "huggingface".into(), task: "segment".into(), hf_repo: "x/y".into(), image_size: [t.input[1], t.input[2]], channels: t.input[0],
            classes: vec!["background".into(), "flood".into()], ..Default::default() }
    } else {
        let builtin = if t.input == [1, 28, 28] { "mnist" } else { "cifar10" };
        let native = if builtin == "mnist" { [1, 28, 28] } else { [3, 32, 32] };
        Dataset { builtin: builtin.into(), resize: t.input != native, image_size: [t.input[1], t.input[2]], channels: t.input[0], ..Default::default() }
    }
}

#[test]
fn every_template_is_valid_and_compiles() {
    for t in nnb_core::templates::TEMPLATES {
        let g = nnb_core::templates::build(t.key).unwrap();
        let ds = dataset_for(t);
        let a = analyze(&g, &ds);
        let errs: Vec<_> = a.nodes.iter().filter(|n| n.status == "error" || n.status == "blocked").map(|n| (&n.title, &n.messages)).collect();
        assert!(a.ok, "{}: {:?} {:?}", t.key, a.problems, errs);
        assert_eq!(a.input, t.input.to_vec(), "{}", t.key);
        let code = train_script(&g, &ds, &a, &TrainSettings::default(), t.title).unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("t.py"), &code).unwrap();
        if let Some(out) = py_compile(&dir.path().join("t.py")) {
            assert!(out.status.success(), "{}: {}", t.key, String::from_utf8_lossy(&out.stderr));
        }
        // positions are spread out (no two nodes on the same spot)
        let mut pos: Vec<(i64, i64)> = g.nodes.iter().map(|n| (n.x as i64, n.y as i64)).collect();
        pos.sort();
        pos.dedup();
        assert_eq!(pos.len(), g.nodes.len(), "{}", t.key);
    }
    // known parameter counts (no bias in the BN-followed convolutions, as in torchvision)
    let p = |k: &str| analyze(&nnb_core::templates::build(k).unwrap(), &dataset_for(nnb_core::templates::get(k).unwrap())).total_params;
    assert_eq!(p("lenet5"), 61_706); // LeNet-5 with 10 classes
    assert_eq!(p("vgg16"), 134_301_514); // torchvision VGG-16 with 10 classes
    assert_eq!(p("resnet18"), 11_181_642); // torchvision ResNet-18 with 10 classes
    assert_eq!(p("vit_b16"), 85_806_346); // torchvision vit_b_16 (86,567,656 with 1000 classes) with a 10-class head
    // MSG-Former, counted by hand: stem 928, gated blocks 38,288 + 152,352, class token + positions 8,448, 2 encoders 264,960,
    // final LayerNorm 256, head 1,290
    assert_eq!(p("msg_former"), 466_522);
    // ViT-g/14, counted by hand: 40 blocks × 25,250,176 (attention 4d²+4d, MLP 2·d·6144 + 6144 + d, two LayerNorms) + patch convolution
    // 829,312 + class token and 257 positions 363,264 + final LayerNorm 2,816 + head 14,090
    assert_eq!(p("vit_g14"), 1_011_216_522);
    // multiply-accumulates per image. MSG-Former by hand: stem 884,736; block 1 36,702,208 (branches 3·9·32·32·32², fuse, shortcut,
    // gate); block 2 36,708,352; two encoders 19,202,560 (4Nd² + 2N²d attention, 2·N·d·256 MLP, N=65, d=128); head 1,280
    let m = |k: &str| analyze(&nnb_core::templates::build(k).unwrap(), &dataset_for(nnb_core::templates::get(k).unwrap())).total_macs;
    assert_eq!(m("msg_former"), 93_499_136);
    // ResNet-18 at 224×224: the usual figure is 1.8 G multiply-accumulates (BatchNorm and ReLU not counted)
    let r18 = m("resnet18") as f64;
    assert!((1.80e9..1.83e9).contains(&r18), "resnet18 {r18}");
    // a transformer's MLP layers are large by design: no "large Dense after Flatten" warning on tokens (VGG's first Dense still warns)
    let warns = |k: &str| analyze(&nnb_core::templates::build(k).unwrap(), &dataset_for(nnb_core::templates::get(k).unwrap())).nodes.iter()
        .filter(|n| n.status == "warn" && n.messages.iter().any(|m| m.contains("instead of Flatten"))).count();
    assert_eq!(warns("vit_g14"), 0);
    assert_eq!(warns("vit_b16"), 0);
    assert!(warns("vgg16") > 0);
}

#[test]
fn segmentation_and_multiply_rules() {
    let t = nnb_core::templates::get("attention_unet").unwrap();
    let g = nnb_core::templates::build("attention_unet").unwrap();
    let a = analyze(&g, &dataset_for(t));
    assert_eq!(a.nodes.iter().find(|n| n.kind == "seg_output").unwrap().out, Some(vec![2, 128, 128]));
    let m = a.nodes.iter().find(|n| n.kind == "multiply").unwrap();
    assert_eq!(m.inputs[1][0], 1, "the attention map has one channel");
    // the same network on a classification dataset is refused with advice
    let a = analyze(&g, &Dataset::default());
    assert!(!a.ok);
    // Output on a segmentation dataset is refused too
    let a = analyze(&model::starter(), &dataset_for(t));
    assert!(a.nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("Segmentation output"))));
}

#[test]
fn transformer_blocks() {
    let t = nnb_core::templates::get("vit_tiny").unwrap();
    let g = nnb_core::templates::build("vit_tiny").unwrap();
    let a = analyze(&g, &dataset_for(t));
    let shape = |k: &str| a.nodes.iter().find(|n| n.kind == k).unwrap().out.clone().unwrap();
    assert_eq!(shape("to_tokens"), vec![64, 192]);
    assert_eq!(shape("cls_pos"), vec![65, 192]);
    assert_eq!(shape("token_pool"), vec![192]);
    let code = model_code(&g, &a).unwrap();
    assert!(code.contains("class SelfAttention(nn.Module)") && code.contains("SelfAttention(dim=192, heads=3, dropout=0.1)"), "{code}");
    // heads must divide the dimension; Output needs a pooled vector; BatchNorm is refused on tokens
    let mut g2 = g.clone();
    for n in g2.nodes.iter_mut().filter(|n| n.kind == "attention") {
        n.params.insert("heads".into(), json!(5));
    }
    assert!(analyze(&g2, &dataset_for(t)).nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("divide evenly"))));
    let g3 = chain(vec![("input", json!({})), ("to_tokens", json!({})), ("batchnorm", json!({})), ("token_pool", json!({})), ("output", json!({}))]);
    assert!(!analyze(&g3, &Dataset::default()).ok);
    let g4 = chain(vec![("input", json!({})), ("to_tokens", json!({"mode": "rows"})), ("output", json!({}))]);
    assert!(analyze(&g4, &Dataset::default()).nodes[2].messages[0].contains("Token pool"));
}

#[test]
fn groupnorm_block() {
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 32})), ("groupnorm", json!({"groups": 8})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &mnist());
    assert!(a.ok, "{:?}", a.problems);
    let gn = a.nodes.iter().find(|n| n.kind == "groupnorm").unwrap();
    assert_eq!((gn.params, gn.out.clone()), (64, Some(vec![32, 28, 28])));
    assert!(model_code(&g, &a).unwrap().contains("nn.GroupNorm(8, 32)"));
    let bad = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 32})), ("groupnorm", json!({"groups": 5})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&bad, &mnist());
    let gn = a.nodes.iter().find(|n| n.kind == "groupnorm").unwrap();
    assert!(!a.ok && gn.status == "error" && gn.messages[0].contains("32 channels into 5 groups"), "{:?}", gn.messages);
}

#[test]
fn save_memory_parts() {
    use nnb_core::codegen::checkpoint_cuts;
    // a chain of 9 lines: cuts every ~3 lines, never in the last two
    let chain: Vec<Vec<usize>> = (0..9).map(|j| if j == 0 { vec![] } else { vec![j - 1] }).collect();
    assert_eq!(checkpoint_cuts(&chain, 8), vec![2, 5]);
    // a skip from line 1 to line 5 (an Add): nothing between 1 and 4 may be a cut, since line 1's output crosses it
    let mut skip = chain.clone();
    skip[5] = vec![4, 1];
    assert!(checkpoint_cuts(&skip, 8).iter().all(|&c| !(1..5).contains(&c)), "{:?}", checkpoint_cuts(&skip, 8));
    // the result line counts as used at the end: no cut after it, even when lines follow
    assert!(checkpoint_cuts(&chain, 3).iter().all(|&c| c < 3));

    let t = TrainSettings { save_memory: true, ..TrainSettings::default() };
    for tp in nnb_core::templates::TEMPLATES {
        let g = nnb_core::templates::build(tp.key).unwrap();
        let ds = dataset_for(tp);
        let a = analyze(&g, &ds);
        let code = train_script(&g, &ds, &a, &t, tp.title).unwrap();
        if ["resnet18", "vit_tiny", "vgg16"].contains(&tp.key) {
            assert!(code.contains("def _part1(self, ") && code.contains("torch.utils.checkpoint.checkpoint(self._part0, x"), "{}", tp.key);
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("t.py"), &code).unwrap();
        if let Some(out) = py_compile(&dir.path().join("t.py")) {
            assert!(out.status.success(), "{}: {}", tp.key, String::from_utf8_lossy(&out.stderr));
        }
    }
    // without the setting the code is unchanged (no parts)
    let g = nnb_core::templates::build("resnet18").unwrap();
    let a = analyze(&g, &dataset_for(nnb_core::templates::get("resnet18").unwrap()));
    assert!(!model_code(&g, &a).unwrap().contains("_part0"));
}

#[test]
fn large_widths_and_out_of_range_values() {
    // 8192 filters and 64 heads are kept as written (a 25-billion-parameter width), not cut to a smaller network
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 8192, "kernel": [16, 16], "stride": [16, 16], "padding": "valid"})),
        ("to_tokens", json!({})), ("attention", json!({"heads": 64})), ("token_pool", json!({"mode": "mean"})), ("output", json!({}))]);
    let ds = Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [64, 64], channels: 3, classes: vec!["a".into(), "b".into()], ..Default::default() };
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.nodes.iter().map(|n| &n.messages).collect::<Vec<_>>());
    assert_eq!(a.nodes.iter().find(|n| n.kind == "conv2d").unwrap().out, Some(vec![8192, 4, 4]));
    // a value beyond the limit is an error on its block, not silently changed
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 100000})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &ds);
    assert!(!a.ok);
    let c = a.nodes.iter().find(|n| n.kind == "conv2d").unwrap();
    assert!(c.status == "error" && c.messages[0].contains("filters is 100000 but must be from 1 to 65536"), "{:?}", c.messages);
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"kernel": [40, 40]})), ("global_pool", json!({})), ("output", json!({}))]);
    assert!(analyze(&g, &ds).nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("kernel is 40"))));
}

#[test]
fn multiband_datasets() {
    use nnb_core::dataset::{SENTINEL1_BANDS, SENTINEL2_BANDS};
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let ds = Dataset { source: "kaggle".into(), kaggle_slug: "owner/eurosat-ms".into(), image_size: [64, 64], sensor: "s2".into(),
        band_order: s(SENTINEL2_BANDS), bands: s(SENTINEL2_BANDS), classes: s(&["a", "b"]), ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    assert_eq!(ds.input_shape(), vec![13, 64, 64]);
    // a conv on 13 bands: the shape engine and the code see 13 input channels
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 16})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &ds);
    assert!(a.ok);
    assert_eq!(a.nodes.iter().find(|n| n.kind == "conv2d").unwrap().params, 13 * 9 * 16 + 16);
    assert!(model_code(&g, &a).unwrap().contains("nn.Conv2d(in_channels=13"));
    // problems: a band not in the files, a band twice, not from Kaggle
    let bad = |f: &dyn Fn(&mut Dataset)| { let mut d = ds.clone(); f(&mut d); d.problems().join(" ") };
    assert!(bad(&|d| d.bands = s(&["B04", "B99"])).contains("B99 is not in the bands of the files"));
    assert!(bad(&|d| d.bands = s(&["B04", "B04"])).contains("chosen twice"));
    // Hugging Face band arrays are allowed; the built-in sets are not
    assert!(!bad(&|d| { d.source = "huggingface".into(); d.hf_repo = "owner/bands".into(); }).contains("Multispectral"));
    assert!(bad(&|d| { d.source = "builtin".into(); d.resize = true; }).contains("Hugging Face dataset with band arrays"));
    // Sentinel-1: two bands
    let s1 = Dataset { sensor: "s1".into(), s1_db: true, band_order: s(SENTINEL1_BANDS), bands: s(SENTINEL1_BANDS), ..ds.clone() };
    assert!(s1.problems().is_empty());
    assert_eq!(s1.input_shape(), vec![2, 64, 64]);
    // ordinary images are unchanged: 1 or 3 channels
    let plain = Dataset { bands: vec![], channels: 2, ..ds };
    assert!(plain.problems().join(" ").contains("Channels must be 1"));
}

/// a little-endian TIFF with one IFD: width × height, `bands` float samples and GDAL's band names (tag 42112)
fn tiny_tiff(path: &std::path::Path, width: u32, height: u32, names: &[&str], big_endian: bool) {
    let xml = format!("<GDALMetadata>{}</GDALMetadata>\0",
        names.iter().enumerate().map(|(i, n)| format!("<Item name=\"DESCRIPTION\" sample=\"{i}\" role=\"description\">{n}</Item>")).collect::<String>());
    let w16 = |v: u16| if big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
    let w32 = |v: u32| if big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
    let mut f: Vec<u8> = if big_endian { b"MM".to_vec() } else { b"II".to_vec() };
    f.extend(w16(42));
    f.extend(w32(8));
    let tags: Vec<(u16, u16, u32, u32)> = vec![(256, 4, 1, width), (257, 4, 1, height), (258, 3, 1, 32), (277, 3, 1, names.len() as u32),
        (339, 3, 1, 3), (42112, 2, xml.len() as u32, 8 + 2 + 5 * 12 + 4 + 2 * 12)];
    let tags: Vec<_> = tags.into_iter().filter(|t| t.0 != 42112 || !names.is_empty()).collect();
    f.extend(w16(tags.len() as u16));
    for (tag, typ, count, value) in &tags {
        f.extend(w16(*tag));
        f.extend(w16(*typ));
        f.extend(w32(*count));
        if *typ == 3 {
            f.extend(w16(*value as u16));
            f.extend([0, 0]);
        } else {
            f.extend(w32(*value));
        }
    }
    f.extend(w32(0));
    while f.len() < 8 + 2 + 5 * 12 + 4 + 2 * 12 {
        f.push(0);
    }
    f.extend(xml.as_bytes());
    std::fs::write(path, f).unwrap();
}

#[test]
fn own_files_and_earth_engine_exports() {
    use nnb_core::dataset::{band_key, SENTINEL2_GEE};
    use nnb_core::localdata;
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    // Earth Engine's B4 and the app's B04 are the same band; B8A stays B8A
    assert_eq!((band_key("B4"), band_key("b04"), band_key("B8A"), band_key("B11"), band_key("vv")), ("B04".into(), "B04".into(), "B8A".into(), "B11".into(), "VV".into()));
    let ds = Dataset { source: "local".into(), local_dir: "/data/Sentinel 2 export".into(), sensor: "s2".into(), band_order: s(SENTINEL2_GEE),
        bands: s(&["B04", "B03", "B02", "B8"]), classes: s(&["forest", "water"]), image_size: [64, 64], ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    assert_eq!(ds.input_shape(), vec![4, 64, 64]);
    let bad = |f: &dyn Fn(&mut Dataset)| { let mut d = ds.clone(); f(&mut d); d.problems().join(" ") };
    assert!(bad(&|d| d.local_dir = "".into()).contains("full path"));
    assert!(bad(&|d| d.local_dir = "relative/path".into()).contains("full path"));
    assert!(bad(&|d| d.local_dir = "/a/$(rm)".into()).contains("may not contain"));
    assert!(bad(&|d| d.bands = s(&["B4", "B04"])).contains("chosen twice"));
    assert!(bad(&|d| d.band_order = s(&["B4", "B04"])).contains("appears twice"));
    assert!(bad(&|d| d.layout = "scene".into()).contains("for segmentation"));
    // segmentation of multispectral images: images + masks, or a scene with a label image
    let seg = Dataset { task: "segment".into(), kaggle_train_dir: "images".into(), kaggle_mask_dir: "masks".into(), ..ds.clone() };
    assert!(seg.problems().is_empty(), "{:?}", seg.problems());
    let scene = Dataset { layout: "scene".into(), scene_image: "nnb_image".into(), scene_label: "nnb_label".into(), kaggle_train_dir: "".into(), ..seg.clone() };
    assert!(scene.problems().is_empty(), "{:?}", scene.problems());
    assert!(Dataset { scene_label: "".into(), ..scene.clone() }.problems().join(" ").contains("label image"));
    // the key: stable, in Kaggle's letters, 3-50 characters, different for two folders with the same name
    let k = ds.local_key();
    assert!(k.starts_with("nnb-sentinel-2-export-") && k.len() <= 50 && k.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'), "{k}");
    assert_eq!(k, ds.clone().local_key());
    assert_ne!(k, Dataset { local_dir: "/other/Sentinel 2 export".into(), ..ds.clone() }.local_key());
    assert!(Dataset { local_dir: "/".into(), ..ds.clone() }.local_key().len() >= 6);
    // train.py for a scene: knows the key, and compiles
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 2, "kernel": 1})), ("seg_output", json!({}))]);
    let a = analyze(&g, &scene);
    assert!(a.ok, "{:?}", a.problems);
    let code = train_script(&g, &scene, &a, &TrainSettings::default(), "t").unwrap();
    assert!(code.contains(&format!("\"local_key\": \"{k}\"")) && code.contains("\"layout\": \"scene\""));
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(out) = py_compile(&dir.path().join("train.py")) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }

    // the folder on the Mac: class folders, and the band names Earth Engine / GDAL write into a GeoTIFF (little and big endian)
    let root = dir.path().join("export");
    for c in ["forest", "water"] {
        std::fs::create_dir_all(root.join(c)).unwrap();
    }
    tiny_tiff(&root.join("forest/a.tif"), 64, 48, &["B4", "B3", "B2", "B8"], false);
    tiny_tiff(&root.join("water/b.tif"), 64, 48, &[], true);
    std::fs::write(root.join(".DS_Store"), "x").unwrap();
    let i = localdata::tiff_info(&root.join("forest/a.tif")).unwrap();
    assert_eq!((i.width, i.height, i.bands, i.bits, i.float), (64, 48, 4, 32, true));
    assert_eq!(i.names, s(&["B4", "B3", "B2", "B8"]));
    let j = localdata::tiff_info(&root.join("water/b.tif")).unwrap();
    assert_eq!((j.width, j.bands, j.names.len()), (64, 0, 0), "big endian; no names (and no samples tag value here)");
    assert!(localdata::tiff_info(&dir.path().join("train.py")).is_err());
    let d = localdata::describe(&root).unwrap();
    assert_eq!(d["files"], 2, "hidden files left out: {d}");
    assert_eq!(d["tifs"], 2);
    assert_eq!(d["folders"][0]["classes"], json!(["forest", "water"]));
    assert_eq!(d["tif"]["names"], json!(["B4", "B3", "B2", "B8"]));
    let sig = localdata::signature(&root).unwrap();
    assert!(sig.starts_with("2 files"), "{sig}");
    assert!(localdata::describe(&dir.path().join("missing")).is_err());
}

#[test]
fn multilabel_and_fusion() {
    use nnb_core::localdata;
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    // multi-label: a CSV of labels; built-in sets refused; the Output gives one score per class
    let ml = Dataset { source: "local".into(), task: "multilabel".into(), local_dir: "/data/ben".into(), labels_csv: "labels.csv".into(),
        classes: s(&["forest", "water", "urban"]), image_size: [32, 32], ..Default::default() };
    assert!(ml.problems().is_empty(), "{:?}", ml.problems());
    assert!(Dataset { labels_csv: "".into(), ..ml.clone() }.problems().join(" ").contains("CSV file with the labels"));
    assert!(Dataset { source: "builtin".into(), ..ml.clone() }.problems().join(" ").contains("one label per image"));
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 4})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &ml);
    assert!(a.ok, "{:?}", a.problems);
    assert_eq!(a.nodes.last().unwrap().out, Some(vec![3]));

    // fusion: Sentinel-2 (4 bands) + Sentinel-1 (2) stacked = 6 channels
    let fused = Dataset { source: "local".into(), local_dir: "/data/fuse".into(), kaggle_train_dir: "s2".into(), sensor: "s2".into(),
        band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B3", "B4", "B8"]), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]),
        bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(), classes: s(&["a", "b"]), image_size: [32, 32], ..Default::default() };
    assert!(fused.problems().is_empty(), "{:?}", fused.problems());
    assert!(fused.fused());
    assert_eq!(fused.input_shape(), vec![6, 32, 32]);
    let bad = |f: &dyn Fn(&mut Dataset)| { let mut d = fused.clone(); f(&mut d); d.problems().join(" ") };
    assert!(bad(&|d| d.fuse_dir = "".into()).contains("second group's files"));
    assert!(bad(&|d| d.bands2 = s(&["VV", "HH"])).contains("HH is not in the bands of the second files"));
    assert!(bad(&|d| d.band_order2 = s(&["B2", "VH"])).contains("share a band name") || bad(&|d| { d.band_order2 = s(&["B02", "VH"]); d.bands2 = s(&["B02"]); }).contains("share a band name"));
    assert!(bad(&|d| d.kaggle_test_dir = "test".into()).contains("leave the test folder empty"));
    // early fusion: one Input takes all 6
    let a = analyze(&g, &fused);
    assert!(a.ok && a.nodes[0].out == Some(vec![6, 32, 32]), "{:?}", a.problems);
    // mid fusion, written as text: an Input per group, a branch each, joined by Concat
    let text = "net \"fusion\"\n1 input 1\n2 conv 16 3x3\n3 input 2\n4 conv 8 3x3\n5 concat <- 2, 4\n6 global_pool\n7 output\n";
    let mid = nnb_core::dsl::apply(text, &Default::default()).unwrap();
    let a = analyze(&mid, &fused);
    assert!(a.ok, "{:?} {:?}", a.problems, a.nodes.iter().map(|n| (&n.kind, &n.messages)).collect::<Vec<_>>());
    let convs: Vec<_> = a.nodes.iter().filter(|n| n.kind == "conv2d").collect();
    assert_eq!(convs.iter().map(|n| n.params).collect::<Vec<_>>(), vec![4 * 9 * 16 + 16, 2 * 9 * 8 + 8]);
    assert!(a.nodes.iter().any(|n| n.kind == "concat" && n.out == Some(vec![24, 32, 32])));
    let code = model_code(&mid, &a).unwrap();
    assert!(code.contains("= x[:, :4]   # band group 1") && code.contains("= x[:, 4:]   # band group 2"), "{code}");
    // the text prints back the same groups
    let printed = nnb_core::dsl::print(&mid).0;
    assert!(printed.contains("input 1") && printed.contains("input 2"), "{printed}");
    // two Inputs without a second group, or two of the same group, or one Input with only one group: explained
    assert!(analyze(&mid, &Dataset { bands2: vec![], band_order2: vec![], sensor2: "".into(), ..fused.clone() }).problems.join(" ").contains("exactly one Input"));
    let same = nnb_core::dsl::apply(&text.replace("3 input 2", "3 input 1"), &Default::default()).unwrap();
    assert!(analyze(&same, &fused).problems.join(" ").contains("one takes band group 1"));
    let only1 = chain(vec![("input", json!({"group": 1})), ("conv2d", json!({"filters": 4})), ("global_pool", json!({})), ("output", json!({}))]);
    assert!(analyze(&only1, &fused).problems.join(" ").contains("is not used"));
    assert!(!analyze(&only1, &Dataset { bands2: vec![], ..fused.clone() }).ok, "group 1 without a second group");
    // a train.py for mid fusion + multi-label compiles
    let both = Dataset { task: "multilabel".into(), labels_csv: "labels.csv".into(), classes: s(&["a", "b", "c"]), ..fused.clone() };
    let a = analyze(&mid, &both);
    assert!(a.ok, "{:?}", a.problems);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), train_script(&mid, &both, &a, &TrainSettings::default(), "t").unwrap()).unwrap();
    if let Some(out) = py_compile(&dir.path().join("train.py")) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }

    // the labels CSV on the Mac: names in one column (quoted, with spaces), or one 0/1 column per class
    let c1 = dir.path().join("labels.csv");
    std::fs::write(&c1, "\u{feff}patch_id,labels\na,\"tree cover; water\"\nb,water\nc,\n").unwrap();
    let r = localdata::csv_classes(&c1).unwrap();
    assert_eq!((r["rows"].clone(), r["file_column"].clone(), r["classes"].clone(), r["counts"].clone()),
        (json!(3), json!("patch_id"), json!(["tree cover", "water"]), json!([1, 2])));
    let c2 = dir.path().join("onehot.csv");
    std::fs::write(&c2, "image,forest,water\nx.tif,1,0\ny.tif,1,1\n").unwrap();
    let r = localdata::csv_classes(&c2).unwrap();
    assert_eq!((r["label_column"].clone(), r["classes"].clone(), r["counts"].clone()), (Value::Null, json!(["forest", "water"]), json!([2, 1])));
}

#[test]
fn class_counts_weights_and_thresholds_settings() {
    let f = |p: &str| (p.to_string(), "1".to_string());
    let files = vec![f("train/a/1.png"), f("train/a/2.png"), f("train/a/x/3.png"), f("train/b/1.png"), f("test/a/1.png")];
    let classes = vec!["a".to_string(), "b".to_string()];
    assert_eq!(nnb_core::cloud::folder_counts(&files, "train", &classes), vec![2, 1]);
    assert_eq!(nnb_core::cloud::folder_counts(&files, "", &classes), vec![0, 0]);
    // unknown weighting is dropped; per-class thresholds need validation data, which clean() provides
    let t = TrainSettings { class_weights: "bogus".into(), tune_thresholds: true, val_fraction: 0.0, ..Default::default() }.clean();
    assert!(t.class_weights.is_empty() && t.val_fraction >= 0.1);
    let t = TrainSettings { class_weights: "sqrt".into(), ..Default::default() }.clean();
    assert_eq!(t.class_weights, "sqrt");
    let ds = Dataset { source: "builtin".into(), ..Default::default() };
    let g = chain(vec![("input", json!({})), ("flatten", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &ds);
    let code = train_script(&g, &ds, &a, &t, "t").unwrap();
    assert!(code.contains("\"class_weights\": \"sqrt\""), "the weighting reaches train.py");
}

#[test]
fn loss_choice() {
    let t = TrainSettings { loss: "bogus".into(), focal_gamma: 9.0, ..Default::default() }.clean();
    assert!(t.loss.is_empty() && t.focal_gamma == 5.0);
    let mut old = serde_json::to_value(TrainSettings::default()).unwrap();
    old.as_object_mut().unwrap().retain(|k, _| k != "loss" && k != "focal_gamma");
    let t: TrainSettings = serde_json::from_value(old).unwrap();
    assert!(t.loss.is_empty() && t.focal_gamma == 2.0, "old train.json: cross-entropy, γ 2");
    let g = chain(vec![("input", json!({})), ("flatten", json!({})), ("output", json!({}))]);
    let ds = Dataset { source: "builtin".into(), ..Default::default() };
    let a = analyze(&g, &ds);
    let focal = TrainSettings { loss: "focal".into(), focal_gamma: 1.5, ..Default::default() }.clean();
    let code = train_script(&g, &ds, &a, &focal, "t").unwrap();
    assert!(code.contains("\"loss\": \"focal\"") && code.contains("\"focal_gamma\": 1.5"), "the loss reaches train.py");
    // Dice needs classes per pixel or several per image: refused on the Mac for one class per image
    let dice = TrainSettings { loss: "dice_ce".into(), ..Default::default() }.clean();
    let err = train_script(&g, &ds, &a, &dice, "t").unwrap_err().to_string();
    assert!(err.contains("Dice loss needs segmentation or multi-label"), "{err}");
}

#[test]
fn regression_task() {
    let dir = std::env::temp_dir().join(format!("nnb-regress-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = dir.join("targets.csv");
    std::fs::write(&csv, "file,biomass,height,notes\na.tif,12.5,3,x\nb.tif,40,7.25,y\n").unwrap();
    let r = nnb_core::localdata::csv_classes(&csv).unwrap();
    let cols: Vec<&str> = r["numeric"].as_array().unwrap().iter().map(|n| n["column"].as_str().unwrap()).collect();
    assert_eq!(cols, vec!["biomass", "height"], "the numeric columns are offered as targets");
    assert_eq!(r["numeric"][0]["max"], json!(40.0));
    std::fs::remove_dir_all(&dir).ok();
    // one target is enough; the labels file is required; the built-in sets are refused
    let ds = Dataset { source: "local".into(), task: "regress".into(), local_dir: "/data/plots".into(), labels_csv: "targets.csv".into(),
        classes: vec!["biomass".into()], ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    let none = Dataset { classes: vec![], ..ds.clone() };
    assert!(none.problems().iter().any(|p| p.contains("target columns")), "{:?}", none.problems());
    let nocsv = Dataset { labels_csv: String::new(), ..ds.clone() };
    assert!(nocsv.problems().iter().any(|p| p.contains("target numbers")), "{:?}", nocsv.problems());
    let g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 4})), ("global_pool", json!({})), ("output", json!({}))]);
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    assert_eq!(a.get("n3").unwrap().out, Some(vec![1]), "one number per target");
    let huber = TrainSettings { loss: "huber".into(), ..Default::default() }.clean();
    let code = train_script(&g, &ds, &a, &huber, "t").unwrap();
    assert!(code.contains("\"task\": \"regress\"") && code.contains("\"loss\": \"huber\""));
    let err = train_script(&g, &ds, &a, &TrainSettings { loss: "focal".into(), ..Default::default() }, "t").unwrap_err().to_string();
    assert!(err.contains("MSE, L1 or Huber"), "{err}");
    let cls = Dataset { source: "builtin".into(), ..Default::default() };
    let err = train_script(&g, &cls, &analyze(&g, &cls), &huber, "t").unwrap_err().to_string();
    assert!(err.contains("for regression"), "{err}");
}

#[test]
fn cloud_locations() {
    use nnb_core::dataset::cloud_url_problem;
    for ok in ["gs://my-bucket/nnb_export", "gs://abc", "https://drive.google.com/drive/folders/1AbC", "rclone:gdrive:exports/s2", "file:///data/x"] {
        assert_eq!(cloud_url_problem(ok), None, "{ok}");
    }
    for bad in ["gs://ab", "http://drive.google.com/x", "https://example.com/x", "rclone::path", "s3://b/x", "gs://b/x y", "gs://b/$(id)", "rclone:x:a;rm"] {
        assert!(cloud_url_problem(bad).is_some(), "{bad}");
    }
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let ds = Dataset { source: "local".into(), cloud_url: "gs://my-bucket/nnb_export".into(), classes: s(&["a", "b"]), ..Default::default() };
    assert!(ds.problems().is_empty(), "no Mac folder needed: {:?}", ds.problems());
    assert!(ds.local_key().starts_with("nnb-nnb-export-"), "{}", ds.local_key());
    assert!(Dataset { cloud_url: "ftp://x".into(), ..ds.clone() }.problems().join(" ").contains("cloud location"));
}

const CUSTOM_CODE: &str = "class SE(nn.Module):\n    \"\"\"squeeze-and-excitation: a weight per channel from the channel means\"\"\"\n    # out: c, h, w\n    def __init__(self, c, h, w):\n        super().__init__()\n        self.fc = nn.Sequential(nn.Linear(c, max(1, c // 4)), nn.ReLU(), nn.Linear(max(1, c // 4), c), nn.Sigmoid())\n\n    def forward(self, x):\n        return x * self.fc(x.mean((2, 3)))[:, :, None, None]\n\n\nclass Down(nn.Module):\n    # out: c * 2, h // 2, w // 2\n    def __init__(self, c, h, w):\n        super().__init__()\n        self.conv = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)\n\n    def forward(self, x):\n        return torch.relu(self.conv(x))\n";

#[test]
fn custom_pytorch_blocks() {
    let mut g = chain(vec![("input", json!({})), ("conv2d", json!({"filters": 8})), ("custom", json!({"class": "SE"})),
        ("custom", json!({"class": "Down"})), ("global_pool", json!({})), ("output", json!({}))]);
    let ds = mnist();
    // without the code: the block says what is missing
    let a = analyze(&g, &ds);
    assert!(!a.ok && a.get("n2").unwrap().messages.iter().any(|m| m.contains("no class SE")), "{:?}", a.get("n2").unwrap().messages);
    g.python = CUSTOM_CODE.into();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    assert_eq!(a.get("n2").unwrap().out, Some(vec![8, 28, 28]));
    assert_eq!(a.get("n3").unwrap().out, Some(vec![16, 14, 14]), "the # out: line");
    let t = TrainSettings { loss: "custom".into(), custom_loss: "def custom_loss(out, y):\n    return nn.functional.cross_entropy(out, y)\n".into(), ..Default::default() }.clean();
    let code = train_script(&g, &ds, &a, &t, "custom").unwrap();
    assert!(code.contains("class SE(nn.Module):") && code.contains("self.l2 = SE(8, 28, 28)") && code.contains("self.l3 = Down(8, 28, 28)"), "{code}");
    assert!(code.find("class SE(").unwrap() < code.find("class Net(").unwrap(), "the classes come before Net");
    assert!(code.contains("def custom_loss(out, y):") && code.contains("\"loss\": \"custom\""));
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(out) = py_compile(&dir.path().join("train.py")) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    // a custom loss without its function is refused on the Mac
    let bad = TrainSettings { loss: "custom".into(), ..Default::default() };
    assert!(train_script(&g, &ds, &a, &bad, "x").unwrap_err().to_string().contains("def custom_loss("));
    // the wrong number of sizes for the input (a class for images after Global pool)
    let mut g2 = chain(vec![("input", json!({})), ("global_pool", json!({})), ("custom", json!({"class": "SE"})), ("output", json!({}))]);
    g2.python = CUSTOM_CODE.into();
    let a2 = analyze(&g2, &ds);
    assert!(a2.get("n2").unwrap().messages.iter().any(|m| m.contains("a vector")), "{:?}", a2.get("n2").unwrap().messages);
    // saved with the design (and older designs without code still load)
    let saved: Graph = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(saved.python, CUSTOM_CODE);
    let old: Graph = serde_json::from_str("{\"nodes\": [], \"edges\": []}").unwrap();
    assert!(old.python.is_empty());
}

#[test]
fn several_outputs_multi_task() {
    use nnb_core::dataset::Head;
    let heads = vec![
        Head { name: "biomass".into(), task: "regress".into(), columns: vec!["biomass".into(), "height".into()], loss: "huber".into(), weight: 0.5, ..Default::default() },
        Head { name: "wet".into(), task: "classify".into(), columns: vec!["wet".into()], classes: vec!["dry".into(), "wet".into(), "flooded".into()], weight: 1.0, ..Default::default() },
    ];
    let ds = Dataset { source: "local".into(), local_dir: "/data/plots".into(), labels_csv: "labels.csv".into(), classes: vec!["forest".into(), "water".into()],
        heads: heads.clone(), ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    let text = "1 input\n2 conv 8 3x3\n3 global_pool\n4 output\n5 output 1 <- 3\n6 output 2 <- 3\n";
    let g = nnb_core::dsl::apply(text, &Graph::default()).unwrap();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let out = |l: &str| a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().out.clone();
    assert_eq!((out("4"), out("5"), out("6")), (Some(vec![2]), Some(vec![2]), Some(vec![3])), "each head's own size");
    let code = train_script(&g, &ds, &a, &TrainSettings::default(), "mt").unwrap();
    assert!(code.contains("return (t3, t4, t5)") && code.contains("out_features=3"), "{code}");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(o) = py_compile(&dir.path().join("train.py")) {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // a head without its Output, two Outputs for one head, an Output for a head that does not exist
    let missing = nnb_core::dsl::apply("1 input\n2 global_pool\n3 output\n4 output 1 <- 2\n", &Graph::default()).unwrap();
    assert!(analyze(&missing, &ds).problems.iter().any(|p| p.contains("\"wet\"")), "{:?}", analyze(&missing, &ds).problems);
    let twice = nnb_core::dsl::apply("1 input\n2 global_pool\n3 output\n4 output 1 <- 2\n5 output 1 <- 2\n6 output 2 <- 2\n", &Graph::default()).unwrap();
    let a2 = analyze(&twice, &ds);
    assert!(a2.nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("more than one Output block for biomass"))));
    let beyond = nnb_core::dsl::apply("1 input\n2 global_pool\n3 output\n4 output 3 <- 2\n", &Graph::default()).unwrap();
    assert!(analyze(&beyond, &ds).nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("2 extra outputs"))));
    // dataset checks: names, tasks, columns, losses, weights, sources
    let bad = |h: Head| Dataset { heads: vec![h], ..ds.clone() }.problems();
    assert!(bad(Head { name: "x".into(), task: "classify".into(), columns: vec!["c".into()], classes: vec!["a".into()], weight: 1.0, ..Default::default() })
        .iter().any(|p| p.contains("at least two")));
    assert!(bad(Head { name: "x".into(), task: "regress".into(), weight: 1.0, ..Default::default() }).iter().any(|p| p.contains("numbers to predict")));
    assert!(bad(Head { name: "x".into(), task: "regress".into(), columns: vec!["v".into()], loss: "focal".into(), weight: 1.0, ..Default::default() })
        .iter().any(|p| p.contains("not for this task")));
    assert!(bad(Head { name: "x".into(), task: "regress".into(), columns: vec!["v".into()], weight: 500.0, ..Default::default() }).iter().any(|p| p.contains("weight")));
    let builtin = Dataset { source: "builtin".into(), heads, ..Default::default() };
    assert!(builtin.problems().iter().any(|p| p.contains("labels table")));
    // old designs and datasets load without heads
    let old: Dataset = serde_json::from_str("{\"source\": \"builtin\"}").unwrap();
    assert!(old.heads.is_empty() && old.label_column.is_empty());
}

#[test]
fn inputs_at_their_own_sizes() {
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let ds = Dataset { source: "local".into(), local_dir: "/data/s".into(), kaggle_train_dir: "s2".into(), image_size: [64, 64], sensor: "s2".into(),
        band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B3", "B4", "B8"]), sensor2: "s1".into(), band_order2: s(&["VV", "VH"]),
        bands2: s(&["VV", "VH"]), fuse_dir: "s1".into(), image_size2: vec![32, 32], classes: s(&["a", "b"]), ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    let mid = nnb_core::dsl::apply("1 input 1\n2 conv 8 3x3 stride 2\n3 input 2\n4 conv 8 3x3\n5 concat <- 2, 4\n6 global_pool\n7 output\n", &Graph::default()).unwrap();
    let a = analyze(&mid, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let out = |l: &str| a.get(&mid.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().out.clone();
    assert_eq!((out("1"), out("3"), out("5")), (Some(vec![4, 64, 64]), Some(vec![2, 32, 32]), Some(vec![16, 32, 32])));
    let code = train_script(&mid, &ds, &a, &TrainSettings::default(), "sizes").unwrap();
    assert!(code.contains("x[:, :4]") && code.contains("x[:, 4:, :32, :32]") && code.contains("\"image_size2\": [32, 32]"), "{code}");
    // one Input for both (early fusion) cannot stack groups of different sizes
    let early = nnb_core::dsl::apply("1 input\n2 global_pool\n3 output\n", &Graph::default()).unwrap();
    assert!(analyze(&early, &ds).nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("different sizes"))));
    // group 2 may not be larger than group 1, and must be two sizes
    assert!(Dataset { image_size2: vec![128, 64], ..ds.clone() }.problems().iter().any(|p| p.contains("may not be larger")));
    assert!(Dataset { image_size2: vec![32], ..ds.clone() }.problems().iter().any(|p| p.contains("two numbers")));
}

#[test]
fn siamese_shared_weights() {
    let s = |v: &[&str]| v.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    // before / after the same place: the same bands in both groups (a pair in time) is allowed
    let ds = Dataset { source: "local".into(), local_dir: "/data/cd".into(), kaggle_train_dir: "before".into(), image_size: [32, 32], sensor: "s2".into(),
        band_order: s(&["B2", "B3", "B4", "B8"]), bands: s(&["B2", "B3", "B4", "B8"]), sensor2: "s2".into(), band_order2: s(&["B2", "B3", "B4", "B8"]),
        bands2: s(&["B2", "B3", "B4", "B8"]), fuse_dir: "after".into(), classes: s(&["same", "changed"]), ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    let other = Dataset { bands2: s(&["B2", "B3"]), band_order2: s(&["B2", "B3"]), ..ds.clone() };
    assert!(other.problems().iter().any(|p| p.contains("share a band name")), "only a full pair may repeat names");
    let text = "1 input 1\n2 conv 8 3x3\n3 conv 8 3x3\n4 input 2\n5 shared 2\n6 shared 3\n7 subtract abs <- 3, 6\n8 global_pool\n9 output\n";
    let g = nnb_core::dsl::apply(text, &Graph::default()).unwrap();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let r = |l: &str| a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap();
    assert_eq!((r("5").out.clone(), r("5").params, r("6").out.clone()), (Some(vec![8, 32, 32]), 0, Some(vec![8, 32, 32])));
    // the weights are counted once: two convolutions (4->8, 8->8) and the output (8->2)
    assert_eq!(a.total_params, (4 * 9 * 8 + 8) + (8 * 9 * 8 + 8) + (8 * 2 + 2));
    let code = train_script(&g, &ds, &a, &TrainSettings::default(), "siam").unwrap();
    assert!(code.matches("= self.l2(").count() == 2 && code.matches("= self.l3(").count() == 2 && code.contains(".abs()"), "{code}");
    assert!(!code.contains("self.l4 =") && !code.contains("self.l5 ="), "no module of their own");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(o) = py_compile(&dir.path().join("train.py")) {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // problems: a different input shape, a block without weights, sharing a shared block
    let msg = |t: &str, l: &str| {
        let g = nnb_core::dsl::apply(t, &Graph::default()).unwrap();
        let a = analyze(&g, &ds);
        a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().messages.join(" ")
    };
    assert!(msg("1 input 1\n2 conv 8 3x3\n3 input 2\n4 maxpool 2x2\n5 shared 2\n6 global_pool <- 2\n7 global_pool <- 5\n8 add <- 6, 7\n9 output\n", "5").contains("weights are for an input"));
    assert!(msg("1 input 1\n2 maxpool 2x2\n3 input 2\n4 shared 2\n5 add <- 2, 4\n6 global_pool\n7 output\n", "4").contains("no weights to share"));
    assert!(msg("1 input 1\n2 conv 8 3x3\n3 input 2\n4 shared 2\n5 shared 4 <- 3\n6 add <- 2, 4\n7 add <- 6, 5\n8 global_pool\n9 output\n", "5").contains("original block"));
}

#[test]
fn transformer_additions() {
    let ds = Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [16, 16], channels: 3, classes: vec!["a".into(), "b".into()], ..Default::default() };
    // 3×16×16 -> 8×8 patches of 32 values (64 tokens); a second, coarser stream of 16 tokens of 48 values
    let text = "1 input\n2 conv 32 2x2 stride 2 valid none\n3 to_tokens\n4 cls_pos sinusoidal\n5 attention heads 4 rotary causal\n\
        6 conv 48 4x4 stride 4 valid none <- 1\n7 to_tokens\n8 cross_attention heads 4 <- 5, 7\n9 rmsnorm\n10 swiglu 64\n\
        11 concat <- 10, 5\n12 token_pool mean\n13 output\n";
    let g = nnb_core::dsl::apply(text, &Graph::default()).unwrap();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let r = |l: &str| a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().clone();
    assert_eq!((r("4").params, r("4").out.clone()), (32, Some(vec![65, 32])), "sinusoidal: only the class token is learnt");
    assert_eq!(r("5").params, 4 * 32 * 32 + 4 * 32, "rotary attention: the same projections");
    assert_eq!((r("8").params, r("8").out.clone()), (2 * 32 * 32 + 2 * 48 * 32 + 4 * 32, Some(vec![65, 32])));
    assert_eq!((r("9").params, r("10").params), (32, 3 * 32 * 64));
    assert_eq!(r("11").out, Some(vec![130, 32]), "token sequences joined one after the other");
    let code = train_script(&g, &ds, &a, &TrainSettings::default(), "tx").unwrap();
    for part in ["ClsPos(tokens=64, dim=32, cls=True, pos=\"sinusoidal\")", "SelfAttention(dim=32, heads=4, dropout=0, causal=True, rotary=True)",
        "CrossAttention(dim=32, kdim=48, heads=4, dropout=0)", "RMSNorm(32)", "SwiGLU(32, 64)"] {
        assert!(code.contains(part), "{part} in\n{code}");
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(o) = py_compile(&dir.path().join("train.py")) {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // problems: odd head width with rotary, cross-attention on an image, concat of tokens of different widths
    let msg = |t: &str, l: &str| {
        let g = nnb_core::dsl::apply(t, &Graph::default()).unwrap();
        let a = analyze(&g, &ds);
        a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().messages.join(" ")
    };
    assert!(msg("1 input\n2 conv 6 2x2 stride 2\n3 to_tokens\n4 attention heads 2 rotary\n5 token_pool mean\n6 output\n", "4").contains("must be even"));
    assert!(msg("1 input\n2 conv 8 3x3\n3 conv 8 3x3 <- 1\n4 cross_attention <- 2, 3\n5 global_pool\n6 output\n", "4").contains("token sequences"));
    assert!(msg("1 input\n2 conv 8 2x2 stride 2\n3 to_tokens\n4 conv 4 2x2 stride 2 <- 1\n5 to_tokens\n6 concat <- 3, 5\n7 token_pool mean\n8 output\n", "6").contains("dimension must match"));
}

#[test]
fn object_detection() {
    let ds = Dataset { source: "local".into(), task: "detect".into(), local_dir: "/data/boats".into(), kaggle_train_dir: "images".into(),
        kaggle_mask_dir: "labels".into(), image_size: [64, 64], channels: 3, classes: vec!["boat".into(), "car".into()], ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    assert!(Dataset { kaggle_mask_dir: String::new(), ..ds.clone() }.problems().iter().any(|p| p.contains("labels folder")));
    assert!(Dataset { classes: vec![], ..ds.clone() }.problems().iter().any(|p| p.contains("object classes")));
    let one = Dataset { classes: vec!["boat".into()], ..ds.clone() };
    let text = "1 input\n2 conv 16 3x3\n3 maxpool 2x2\n4 conv 32 3x3\n5 maxpool 2x2\n6 detect\n";
    let g = nnb_core::dsl::apply(text, &Graph::default()).unwrap();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let head = a.get(&g.nodes.iter().find(|n| n.label == "6").unwrap().id).unwrap();
    assert_eq!((head.out.clone(), head.params), (Some(vec![6, 16, 16]), 33 * 6), "2 heatmaps + width, height, x, y offset per cell");
    assert!(analyze(&g, &one).ok, "one class is enough to detect");
    let code = train_script(&g, &ds, &a, &TrainSettings::default(), "det").unwrap();
    assert!(code.contains("DetectHead(32, 2)") && code.contains("\"detect\": [6, 16, 16]") && code.contains("\"task\": \"detect\""), "{code}");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(o) = py_compile(&dir.path().join("train.py")) {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // the map must divide the image evenly; an Output or a Segmentation output is refused for boxes, and the other way round
    let msg = |t: &str, d: &Dataset| { let g = nnb_core::dsl::apply(t, &Graph::default()).unwrap(); let a = analyze(&g, d); a.nodes.iter().flat_map(|n| n.messages.clone()).collect::<Vec<_>>().join(" ") };
    assert!(msg("1 input\n2 conv 8 3x3 stride 3\n3 detect\n", &ds).contains("whole number of map cells"));
    assert!(msg("1 input\n2 global_pool\n3 output\n", &ds).contains("Detection output"));
    let cls = Dataset { task: "classify".into(), ..ds.clone() };
    assert!(msg("1 input\n2 conv 8 3x3\n3 detect\n", &cls).contains("object detection"));
}

#[test]
fn several_gpus_setting() {
    let old: TrainSettings = { let mut v = serde_json::to_value(TrainSettings::default()).unwrap(); v.as_object_mut().unwrap().remove("gpus"); serde_json::from_value(v).unwrap() };
    assert_eq!(old.gpus, 1, "older train.json: one GPU");
    assert_eq!(TrainSettings { gpus: 1000, ..Default::default() }.clean().gpus, 64);
    let g = model::starter();
    let code = train_script(&g, &mnist(), &analyze(&g, &mnist()), &TrainSettings { gpus: 0, ..Default::default() }.clean(), "all").unwrap();
    assert!(code.contains("\"gpus\": 0") && code.contains("def run_parallel(n):") && code.contains("DistributedDataParallel"));
}

#[test]
fn adapters_experts_and_exits() {
    let ds = Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [16, 16], channels: 3, classes: vec!["a".into(), "b".into(), "c".into()], ..Default::default() };
    let text = "1 input\n2 conv 16 3x3 frozen\n3 maxpool 2x2\n4 global_pool\n5 exit 0.5\n6 conv 32 3x3 <- 3\n7 to_tokens\n\
        8 attention heads 4 lora 4\n9 moe 4 top 2 hidden 64\n10 token_pool mean\n11 dense 64 lora 8\n12 output\n";
    let g = nnb_core::dsl::apply(text, &Graph::default()).unwrap();
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
    let r = |l: &str| a.get(&g.nodes.iter().find(|n| n.label == l).unwrap().id).unwrap().clone();
    assert_eq!((r("2").params, r("2").frozen), (3 * 9 * 16 + 16, 3 * 9 * 16 + 16), "a frozen convolution");
    assert_eq!((r("5").params, r("5").out.clone()), (17 * 3, Some(vec![3])), "an early exit: one score per class");
    let d = 32u64;
    assert_eq!((r("8").params, r("8").frozen), (4 * d * d + 4 * d + 4 * (d + 3 * d) + 4 * (d + d), 4 * d * d + 4 * d), "LoRA attention");
    assert_eq!(r("9").params, d * 4 + 4 + 4 * (2 * d * 64 + 64 + d), "router + 4 experts");
    assert_eq!((r("11").params, r("11").frozen), (33 * 64 + 8 * (32 + 64), 33 * 64), "LoRA dense");
    assert_eq!(a.frozen_params, r("2").frozen + r("8").frozen + r("11").frozen);
    let code = train_script(&g, &ds, &a, &TrainSettings::default(), "adapt").unwrap();
    for part in ["FROZEN = [\"l1\"]", "EXIT_WEIGHTS = [0.5]", "        exits.append(t", "self.nnb_exits = exits", "LoRALinear(32, 64, 8)",
        "SelfAttention(dim=32, heads=4, dropout=0, lora=4)", "MoE(32, 4, 2, 64)", "class LoRALinear(nn.Module):"] {
        assert!(code.contains(part), "{part} in\n{code}");
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("train.py"), &code).unwrap();
    if let Some(o) = py_compile(&dir.path().join("train.py")) {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // an early exit needs a vector and a classification; top k cannot exceed the experts
    let msg = |t: &str, d: &Dataset| { let g = nnb_core::dsl::apply(t, &Graph::default()).unwrap(); analyze(&g, d).nodes.iter().flat_map(|n| n.messages.clone()).collect::<Vec<_>>().join(" ") };
    assert!(msg("1 input\n2 conv 8 3x3\n3 exit\n4 global_pool <- 2\n5 output\n", &ds).contains("from a vector"));
    assert!(msg("1 input\n2 conv 8 2x2 stride 2\n3 to_tokens\n4 moe 2 top 3\n5 token_pool mean\n6 output\n", &ds).contains("only 2"));
    let seg = Dataset { task: "segment".into(), ..ds.clone() };
    assert!(msg("1 input\n2 conv 8 3x3\n3 global_pool\n4 exit\n5 seg_output <- 2\n", &seg).contains("classification"));
}

/// a venv whose PyTorch needs its activate script (the AWS Deep Learning AMI keeps CUDA inside /opt/pytorch): Prepare and run.sh source it
#[test]
fn prepare_and_run_activate_the_python_environment() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(bin.join("activate"), "export NNB_TEST_ACTIVATED=yes\n").unwrap();
    // the stand-in Python: every import works; the torch check reports CUDA only when the environment was activated
    let py = bin.join("python");
    std::fs::write(&py, "#!/bin/bash\n[ \"$1\" = - ] || exit 0\ncat >/dev/null\n\
        if [ \"${NNB_TEST_ACTIVATED:-}\" = yes ]; then echo 'NNB_TORCH 2.12.0 cuda 3.13.17'; else echo 'NNB_TORCH 2.12.0 cpu 3.13.17'; fi\n").unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    let m = Machine { name: "dlami".into(), host: "h".into(), python: py.to_string_lossy().into(),
        workdir: dir.path().join("w").to_string_lossy().into(), ..Default::default() };
    let out = std::process::Command::new("bash").arg("-c").arg(remote::prepare_script(&m)).env_remove("NNB_TEST_ACTIVATED").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<String> = text.lines().map(String::from).collect();
    let p = remote::parse_prepare(&lines);
    assert!(p.cuda && p.fail.is_empty(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    let sh = nnb_core::runs::run_script(&TrainSettings::default(), &p, &m, "", false);
    let (a, run) = (sh.find("/activate").expect("run.sh sources activate"), sh.find("train.py >>").unwrap());
    assert!(a < run, "{sh}");
}

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

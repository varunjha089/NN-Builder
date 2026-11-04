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
    Node { id: id.into(), kind: kind.into(), params: p, x: 0.0, y: 0.0 }
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
        let out = std::process::Command::new("python3").args(["-m", "py_compile"]).arg(&f).output().expect("python3");
        assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stderr), code);
        assert!(code.contains("nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)") || ds.source != "builtin");
        assert!(code.contains("EXPECTED = {\"input\": ["), "{code}");
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

//! The design text (dsl.rs): printing every demo and reading it back, the language (functions, for, repeat, let, expressions), errors,
//! and keeping the canvas and the text in step.

use nnb_core::dataset::Dataset;
use nnb_core::dsl::{self, apply, expand, print, same_structure, sync};
use nnb_core::model::{self, Graph, Node};
use nnb_core::shapes::analyze;
use serde_json::{json, Map};

fn dataset_for(t: &nnb_core::templates::Template) -> Dataset {
    if t.task == "segment" {
        Dataset { source: "huggingface".into(), task: "segment".into(), hf_repo: "x/y".into(), image_size: [t.input[1], t.input[2]], channels: t.input[0],
            classes: vec!["background".into(), "flood".into()], ..Default::default() }
    } else {
        Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [t.input[1], t.input[2]], channels: t.input[0],
            classes: (0..10).map(|i| format!("c{i}")).collect(), ..Default::default() }
    }
}

fn cifar() -> Dataset {
    Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [32, 32], channels: 3,
        classes: (0..10).map(|i| format!("c{i}")).collect(), ..Default::default() }
}

fn build(text: &str) -> Graph {
    match apply(text, &Graph::default()) {
        Ok(g) => g,
        Err(e) => panic!("{text}\n{e:?}"),
    }
}

fn err(text: &str) -> dsl::Diagnostic {
    match apply(text, &Graph::default()) {
        Ok(_) => panic!("expected an error:\n{text}"),
        Err(e) => e[0].clone(),
    }
}

fn by_label<'a>(g: &'a Graph, l: &str) -> &'a Node {
    g.nodes.iter().find(|n| n.label == l).unwrap_or_else(|| panic!("no block {l}: {:?}", g.nodes.iter().map(|n| &n.label).collect::<Vec<_>>()))
}

fn inputs(g: &Graph, l: &str) -> Vec<String> {
    let id = &by_label(g, l).id;
    g.inputs_of(id).iter().map(|s| g.node(s).unwrap().label.clone()).collect()
}

#[test]
fn every_demo_round_trips() {
    for t in nnb_core::templates::TEMPLATES {
        let g = nnb_core::templates::build(t.key).unwrap();
        let ds = dataset_for(t);
        let synced = sync(&g);
        // the blocks are unchanged, only labelled
        assert_eq!(synced.nodes.len(), g.nodes.len());
        assert!(synced.nodes.iter().all(|n| !n.label.is_empty()), "{}", t.key);
        // the text describes exactly these blocks
        let e = expand(&synced.text, &synced).unwrap_or_else(|d| panic!("{}: {d:?}\n{}", t.key, synced.text));
        assert!(same_structure(&e.graph, &synced), "{}:\n{}", t.key, synced.text);
        assert_eq!(analyze(&e.graph, &ds).total_params, analyze(&g, &ds).total_params, "{}", t.key);
        // printing again gives the same text, and sync keeps it
        assert_eq!(print(&e.graph).0, synced.text, "{}", t.key);
        assert_eq!(sync(&synced).text, synced.text, "{}", t.key);
        // the same text into an empty canvas builds an equivalent design
        let fresh = build(&synced.text);
        assert_eq!(analyze(&fresh, &ds).total_params, analyze(&g, &ds).total_params, "{}", t.key);
    }
}

#[test]
fn repeated_blocks_are_folded() {
    let g = sync(&nnb_core::templates::build("vit_g14").unwrap());
    assert!(g.text.contains("repeat 40 {"), "{}", g.text);
    assert!(g.text.lines().count() < 30, "ViT-g/14 in {} lines:\n{}", g.text.lines().count(), g.text);
    let g = sync(&nnb_core::templates::build("vit_b16").unwrap());
    assert!(g.text.contains("repeat 12 {"), "{}", g.text);
}

#[test]
fn every_kind_can_be_written_and_read() {
    for (kind, _) in model::KINDS {
        let params = model::defaults(kind).as_object().cloned().unwrap_or_default();
        let mut g = Graph::default();
        g.nodes.push(Node { id: "a".into(), kind: "input".into(), ..Default::default() });
        g.nodes.push(Node { id: "b".into(), kind: kind.to_string(), params: params.clone(), x: 300.0, ..Default::default() });
        if *kind != "input" {
            g.edges.push(model::Edge { id: "e".into(), source: "a".into(), target: "b".into() });
        }
        let s = sync(&g);
        let e = expand(&s.text, &s).unwrap_or_else(|d| panic!("{kind}: {d:?}\n{}", s.text));
        assert!(same_structure(&e.graph, &s), "{kind}:\n{}", s.text);
    }
    // non-default values of every short form
    let cases: &[(&str, serde_json::Value)] = &[
        ("conv2d", json!({"filters": 48, "kernel": [5, 3], "stride": [2, 1], "padding": 2, "dilation": 2, "activation": "gelu", "bias": false})),
        ("conv2d", json!({"filters": 8, "padding": "valid", "activation": "none"})),
        ("dwconv2d", json!({"multiplier": 2, "kernel": [3, 3], "stride": [2, 2], "padding": "valid", "activation": "silu"})),
        ("convtranspose2d", json!({"filters": 16, "kernel": [4, 4], "stride": [2, 2], "padding": 1, "output_padding": 1, "activation": "none"})),
        ("pool2d", json!({"mode": "avg", "size": [3, 3], "stride": [1, 1], "padding": 1})),
        ("pool2d", json!({"mode": "max", "size": [3, 3], "stride": [2, 2]})),
        ("global_pool", json!({"mode": "max"})),
        ("upsample", json!({"scale": 4, "mode": "bilinear"})),
        ("dropout", json!({"rate": 0.125})),
        ("groupnorm", json!({"groups": 4})),
        ("dense", json!({"units": 10, "activation": "tanh"})),
        ("activation", json!({"fn": "leaky_relu"})),
        ("to_tokens", json!({"mode": "rows"})),
        ("cls_pos", json!({"cls": false})),
        ("attention", json!({"heads": 6, "dropout": 0.1})),
        ("token_pool", json!({"mode": "mean"})),
    ];
    for (kind, p) in cases {
        let mut g = Graph::default();
        g.nodes.push(Node { id: "a".into(), kind: "input".into(), ..Default::default() });
        g.nodes.push(Node { id: "b".into(), kind: kind.to_string(), params: p.as_object().unwrap().clone(), x: 300.0, ..Default::default() });
        g.edges.push(model::Edge { id: "e".into(), source: "a".into(), target: "b".into() });
        let s = sync(&g);
        let e = expand(&s.text, &s).unwrap_or_else(|d| panic!("{kind}: {d:?}\n{}", s.text));
        assert!(same_structure(&e.graph, &s), "{kind} {p}:\n{}", s.text);
    }
}

#[test]
fn chains_skips_and_concat_order() {
    let g = build("1 input\n2 conv 64 3x3\n3 conv 64 3x3 none\n4 add <- 2, 3\n5 relu\n6 conv 128 3x3 stride 2\n7 concat <- 6, 6\n8 global_pool\n9 output\n");
    assert_eq!(inputs(&g, "4"), vec!["2", "3"]);
    assert_eq!(inputs(&g, "5"), vec!["4"]);
    assert_eq!(inputs(&g, "7"), vec!["6", "6"]);
    assert_eq!(by_label(&g, "6").params["stride"], json!([2, 2]));
    assert_eq!(by_label(&g, "3").params["activation"], json!("none"));
    let a = analyze(&g, &cifar());
    assert!(a.ok, "{:?}", a.problems);
}

#[test]
fn functions_loops_and_values() {
    let text = r#"
net "ResNet-ish"
let w = 32
def basic(c, s = 1) {          # a residual block with a projection when the shape changes
  a conv c 3x3 stride s
  b conv c 3x3 none
  p conv c 1x1 stride s none <- in
  d add <- b, p
  e relu
}
1 input
2 conv w 3x3
3 for c, s in (w, 1), (w * 2, 2), (w * 4, 2) {
    x basic(c, s)
    y basic(c)
  }
4 repeat 2 {
    a conv 128 * i 1x1        # i = 1, 2
  }
5 global_pool
6 output
"#;
    let g = build(text);
    // 3 passes × 2 calls × 5 blocks, + 2 + 2 + 2
    assert_eq!(g.nodes.len(), 30 + 6, "{:?}", g.nodes.iter().map(|n| &n.label).collect::<Vec<_>>());
    assert_eq!(by_label(&g, "3.2.x.a").params["filters"], json!(64));
    assert_eq!(by_label(&g, "3.2.x.a").params["stride"], json!([2, 2]));
    assert_eq!(by_label(&g, "3.3.y.a").params["filters"], json!(128));
    assert_eq!(by_label(&g, "3.3.y.a").params["stride"], json!([1, 1]));
    assert_eq!(by_label(&g, "4.2.a").params["filters"], json!(256));
    // wiring: a call takes the line above; inside, "in" is the call's input; passes are chained
    assert_eq!(inputs(&g, "3.1.x.a"), vec!["2"]);
    assert_eq!(inputs(&g, "3.1.x.p"), vec!["2"]);
    assert_eq!(inputs(&g, "3.1.y.a"), vec!["3.1.x.e"]);
    assert_eq!(inputs(&g, "3.2.x.a"), vec!["3.1.y.e"]);
    assert_eq!(inputs(&g, "4.1.a"), vec!["3.3.y.e"]);
    assert_eq!(inputs(&g, "5"), vec!["4.2.a"]);
    let a = analyze(&g, &cifar());
    assert!(a.ok, "{:?}", a.problems);
    // the hand-written text is kept while the blocks are only moved on the canvas
    let mut moved = g.clone();
    moved.nodes[3].x += 500.0;
    assert_eq!(sync(&moved).text, text);
    // a change on the canvas prints the text again (functions become plain lines and repeats) with the same blocks
    let mut changed = g.clone();
    let i = changed.nodes.iter().position(|n| n.label == "2").unwrap();
    changed.nodes[i].params.insert("filters".into(), json!(40));
    let s = sync(&changed);
    assert_ne!(s.text, text);
    assert!(s.text.contains("conv 40 3x3"), "{}", s.text);
    let e = expand(&s.text, &s).unwrap();
    assert!(same_structure(&e.graph, &s));
}

#[test]
fn references_into_groups_and_several_inputs() {
    // a small U-Net: the decoder function takes two inputs (in, in2); skips reach into a repeat by path
    let text = "def down(c) {\n  a conv c 3x3\n  b conv c 3x3\n}\ndef up(c) {\n  a convT c 2x2 stride 2\n  b concat <- a, in2\n  c conv c 3x3\n}\n\
1 input\n2 repeat 2 {\n  d down(16 * i)\n  p maxpool 2x2\n}\n3 down(64)\n4 up(32) <- 3, 2.2.d\n5 up(16) <- 4, 2.1.d\n6 seg_output\n";
    let g = build(text);
    assert_eq!(inputs(&g, "4.b"), vec!["4.a", "2.2.d.b"]);
    assert_eq!(inputs(&g, "5.b"), vec!["5.a", "2.1.d.b"]);
    let ds = Dataset { source: "huggingface".into(), task: "segment".into(), hf_repo: "x/y".into(), image_size: [64, 64], channels: 3,
        classes: vec!["a".into(), "b".into()], ..Default::default() };
    let a = analyze(&g, &ds);
    assert!(a.ok, "{:?}", a.problems);
}

#[test]
fn labels_can_be_left_out() {
    let g = build("input\nconv 32 3x3\nmaxpool 2x2\nrepeat 2 {\n  conv 32 3x3\n  bn\n}\nflatten\ndense 10 none\noutput\n");
    assert_eq!(inputs(&g, "3"), vec!["2"]);
    assert_eq!(inputs(&g, "4.2.a"), vec!["4.1.b"]);
    assert_eq!(by_label(&g, "7").kind, "output");
}

#[test]
fn errors_name_the_line() {
    let e = err("1 input\n2 conv 64 3x3 stide 2\n");
    assert_eq!(e.line, 2);
    assert!(e.message.contains("stide"), "{}", e.message);
    assert!(e.message.contains("did you mean 'stride'"), "{}", e.message);
    let e = err("1 input\n2 conv 64 3x3\n3 add <- 2, 7\n");
    assert_eq!((e.line, e.col), (3, 13));
    assert!(e.message.contains("'7'"), "{}", e.message);
    let e = err("1 input\n2 convv 64\n");
    assert!(e.message.contains("did you mean 'conv'"), "{}", e.message);
    let e = err("1 input\n2 repeat 3 {\n  a conv 8 3x3\n");
    assert!(e.message.contains("not closed"), "{}", e.message);
    let e = err("def f(c) {\n  a conv c 3x3\n}\n1 input\n2 f(1, 2)\n");
    assert_eq!(e.line, 5);
    assert!(e.message.contains("takes 1"), "{}", e.message);
    let e = err("1 input\n2 conv 8 3x3\n2 relu\n");
    assert!(e.message.contains("used twice"), "{}", e.message);
    let e = err("1 input\n2 g(3)\n");
    assert!(e.message.contains("no function 'g'"), "{}", e.message);
    let e = err("def f() {\n  a f()\n}\n1 input\n2 f()\n");
    assert!(e.message.contains("deep"), "{}", e.message);
    let e = err("1 input\n2 conv 8 3x3 99\n");
    assert!(e.message.contains("number 99"), "{}", e.message);
    // settings a block does not have are refused, not kept silently
    let e = err("1 input stride 2\n2 relu\n");
    assert_eq!(e.line, 1);
    assert!(e.message.contains("'stride' means nothing for input"), "{}", e.message);
    let e = err("1 input\n2 dense 10 pad 1\n");
    assert!(e.message.contains("'pad' means nothing for dense"), "{}", e.message);
    assert!(!e.message.contains("did you mean"), "{}", e.message);
    let e = err("1 input\n2 conv 8 3x3 filtrs=4\n");
    assert!(e.message.contains("no setting 'filtrs'") && e.message.contains("filters"), "{}", e.message);
    let e = err("1 input\n2 relu foo=1\n");
    assert!(e.message.contains("no setting 'foo'"), "{}", e.message);
    let e = err("1 input\n2 flatten size=3\n");
    assert!(e.message.contains("flatten has no settings"), "{}", e.message);
    let e = err("1 input\n2 repeat k {\n  a relu\n}\n");
    assert!(e.message.contains("'k' is not a number"), "{}", e.message);
}

#[test]
fn editing_the_text_keeps_blocks_and_positions() {
    let g = sync(&nnb_core::templates::build("lenet5").unwrap());
    // change one number in the text: every block keeps its id and place
    let edited = g.text.replacen("conv 6 ", "conv 8 ", 1);
    assert_ne!(edited, g.text);
    let h = apply(&edited, &g).unwrap();
    assert_eq!(h.nodes.len(), g.nodes.len());
    for n in &g.nodes {
        let m = h.node(&n.id).unwrap_or_else(|| panic!("block {} lost", n.id));
        assert_eq!((m.x, m.y), (n.x, n.y));
    }
    assert!(h.nodes.iter().any(|n| n.params.get("filters") == Some(&json!(8))));
    // a new line gets a new block next to its input, the others stay
    let lines: Vec<&str> = h.text.lines().collect();
    let mut added: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    added.insert(2, "2b bn".into());
    let k = apply(&(added.join("\n") + "\n"), &h).unwrap();
    assert_eq!(k.nodes.len(), h.nodes.len() + 1);
    let new = k.nodes.iter().find(|n| n.label == "2b").unwrap();
    assert!(h.node(&new.id).is_none());
    assert!(k.nodes.iter().filter(|n| n.id != new.id).all(|n| h.node(&n.id).is_some()));
    // sync keeps a just-applied text
    assert_eq!(sync(&k).text, k.text);
}

#[test]
fn cycles_and_loose_blocks_still_print() {
    let mut g = Graph::default();
    for (id, kind) in [("a", "input"), ("b", "conv2d"), ("c", "relu"), ("d", "dense")] {
        g.nodes.push(Node { id: id.into(), kind: if kind == "relu" { "activation".into() } else { kind.into() }, params: Map::new(), ..Default::default() });
    }
    g.edges.push(model::Edge { id: "1".into(), source: "a".into(), target: "b".into() });
    let s = sync(&g);
    assert!(s.text.contains("<- none"), "{}", s.text);
    // a loop on the canvas: the text names it and says what to do
    let mut l = Graph::default();
    for (id, kind) in [("a", "input"), ("b", "conv2d"), ("c", "batchnorm")] {
        l.nodes.push(Node { id: id.into(), kind: kind.into(), ..Default::default() });
    }
    for (s, t) in [("a", "b"), ("b", "c"), ("c", "b")] {
        l.edges.push(model::Edge { id: format!("{s}{t}"), source: s.into(), target: t.into() });
    }
    let ls = sync(&l);
    let d = expand(&ls.text, &ls).err().expect("a loop cannot be built");
    assert!(d[0].message.contains("go in a circle"), "{:?}", d);
    let e = expand(&s.text, &s).unwrap();
    assert!(same_structure(&e.graph, &s), "{}", s.text);
}

/// doc/examples/everything.nnb: every part of the language and every block kind but seg_output, in one valid design
#[test]
fn everything_design() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/examples/everything.nnb")).unwrap();
    let g = build(&text);
    let a = analyze(&g, &cifar());
    let bad: Vec<_> = a.nodes.iter().filter(|n| n.status == "error" || n.status == "blocked").map(|n| (g.node(&n.id).unwrap().label.clone(), n.messages.clone())).collect();
    assert!(a.ok, "{:?} {bad:?}", a.problems);
    // not even a warning (an unused block, rows lost to a stride) at 3×32×32
    let warns: Vec<_> = a.nodes.iter().filter(|n| n.status != "ok").map(|n| (g.node(&n.id).unwrap().label.clone(), n.messages.clone())).collect();
    assert!(warns.is_empty(), "{warns:?}");
    // every block kind except the segmentation output and the time-series blocks (doc/examples/timeseries.nnb has those)
    for (kind, _) in model::KINDS {
        if !matches!(*kind, "seg_output" | "detect_output" | "conv3d" | "temporal_pool" | "to_sequence" | "lstm" | "gru" | "conv1d") {
            assert!(g.nodes.iter().any(|n| n.kind == *kind), "no {kind} block");
        }
    }
    // values that came through functions, defaults, loops and arithmetic
    assert_eq!(by_label(&g, "5.2.m.gate.d1").params["units"], json!(8)); // 32 // 4
    assert_eq!(by_label(&g, "5.1.m.b3.n").params["groups"], json!(4)); // 4 + 16 % 4
    assert_eq!(by_label(&g, "5.2.d.dw").params["stride"], json!([2, 2]));
    assert_eq!(by_label(&g, "5.2.d.dw").params["kernel"], json!([4, 4]));
    assert_eq!(by_label(&g, "5.2.m.b3.dw").params["kernel"], json!([3, 3]));
    assert_eq!(by_label(&g, "5.2.d.pw").params["filters"], json!(64));
    assert_eq!(by_label(&g, "6.2.a").params["filters"], json!(64)); // 2^6 + 0*i
    assert_eq!(by_label(&g, "8.u").params["kernel"], json!([2, 2]));
    assert_eq!(by_label(&g, "16.1.b").params["heads"], json!(2)); // the setting word "heads" and the name heads
    assert_eq!(by_label(&g, "16.2.e").params["units"], json!(64));
    assert_eq!(by_label(&g, "23").params["units"], json!(64));
    // wiring: several inputs, skips by path, concat order, the label left out in the bottleneck
    assert_eq!(inputs(&g, "5.1.m.cat"), vec!["5.1.m.b1", "5.1.m.b2", "5.1.m.b3.a", "5.1.m.b5"]);
    assert_eq!(inputs(&g, "5.1.m.gate.m"), vec!["5.1.m.fuse", "5.1.m.gate.s"]);
    assert_eq!(inputs(&g, "7.j"), vec!["7.u", "5.2.m.gate.m"]);
    assert_eq!(inputs(&g, "8.j"), vec!["8.u", "5.1.m.gate.m"]);
    assert_eq!(inputs(&g, "6.2.a"), vec!["6.1.d"]);
    assert_eq!(inputs(&g, "11"), vec!["10", "8.r"]);
    assert_eq!(inputs(&g, "19"), vec!["11"]);
    assert_eq!(inputs(&g, "22"), vec!["18", "df"]);
    assert_eq!(by_label(&g, "tw").params["of"], json!(by_label(&g, "21").id.clone()), "shared 21 -> the id of block 21");
    // the generated PyTorch code compiles
    let ds = cifar();
    let code = nnb_core::codegen::train_script(&g, &ds, &a, &nnb_core::codegen::TrainSettings::default(), "everything").unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.py"), &code).unwrap();
    if let Some(out) = ["python3", "python"].iter().find_map(|py| std::process::Command::new(py).args(["-m", "py_compile"]).arg(dir.path().join("t.py")).output().ok()) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    // the hand-written text survives the canvas sync; Format prints an equivalent text that builds the same blocks
    assert_eq!(sync(&g).text, text);
    let mut cleared = g.clone();
    cleared.text.clear();
    let formatted = sync(&cleared);
    let e = expand(&formatted.text, &formatted).unwrap_or_else(|d| panic!("{d:?}\n{}", formatted.text));
    assert!(same_structure(&e.graph, &formatted), "{}", formatted.text);
    assert_eq!(analyze(&e.graph, &ds).total_params, a.total_params);
}

#[test]
fn canvas_setting_changes_keep_the_hand_written_text() {
    let text = "let w = 32\ndef pair(c) {\n  a conv c 3x3\n  b conv c 3x3\n}\n1 input\n2 conv w 3x3   # the stem\n3 pair(w * 2)\n4 dense 10 <- 3\n5 output\n";
    let g = build(text);
    // a setting of a block on its own line: only that line changes (the comment and the inputs stay)
    let mut c = g.clone();
    let i = c.nodes.iter().position(|n| n.label == "2").unwrap();
    c.nodes[i].params.insert("filters".into(), json!(48));
    let s = sync(&c);
    assert!(s.text.contains("def pair(c)") && s.text.contains("let w = 32"), "{}", s.text);
    assert!(s.text.contains("2 conv 48 3x3  # the stem"), "{}", s.text);
    let i = c.nodes.iter().position(|n| n.label == "4").unwrap();
    c.nodes[i].params.insert("units".into(), json!(20));
    let s = sync(&c);
    assert!(s.text.contains("4 dense 20 relu <- 3") || s.text.contains("4 dense 20 <- 3"), "{}", s.text);
    assert!(same_structure(&expand(&s.text, &s).unwrap().graph, &c));
    // a block made by a function: the function cannot change for one call, so the text is printed again
    let mut f = g.clone();
    let i = f.nodes.iter().position(|n| n.label == "3.a").unwrap();
    f.nodes[i].params.insert("filters".into(), json!(8));
    let s = sync(&f);
    assert!(!s.text.contains("def pair"), "{}", s.text);
    assert!(same_structure(&expand(&s.text, &s).unwrap().graph, &s));
    assert!(s.nodes.iter().any(|n| n.params.get("filters") == Some(&json!(8))));
}

#[test]
fn long_text_designs_wrap_into_rows() {
    let g = build("1 input\n2 repeat 40 {\n  a conv 8 3x3\n}\n3 global_pool\n4 output\n");
    let width = g.nodes.iter().map(|n| n.x).fold(0.0, f64::max);
    assert!(width <= 60.0 + 23.0 * 300.0, "{width}");
    let rows: std::collections::BTreeSet<i64> = g.nodes.iter().map(|n| n.y as i64).collect();
    assert!(rows.len() >= 2, "{rows:?}");
    // positions stay distinct
    let mut pos: Vec<(i64, i64)> = g.nodes.iter().map(|n| (n.x as i64, n.y as i64)).collect();
    pos.sort();
    pos.dedup();
    assert_eq!(pos.len(), g.nodes.len());
}

/// a canvas edit on the design `g`: (new blocks: id, kind, params), (edges removed: source label -> target label), (edges added, by
/// label or new id); returns the synced design
fn edit(g: &Graph, add: &[(&str, &str, serde_json::Value)], remove_edges: &[(&str, &str)], add_edges: &[(&str, &str)], remove: &[&str]) -> Graph {
    let mut h = g.clone();
    let id = |h: &Graph, l: &str| h.nodes.iter().find(|n| n.label == l || n.id == l).map(|n| n.id.clone()).unwrap_or_else(|| panic!("no {l}"));
    for (nid, kind, p) in add {
        h.nodes.push(Node { id: nid.to_string(), kind: kind.to_string(), params: p.as_object().cloned().unwrap_or_default(), x: 900.0, y: 400.0, ..Default::default() });
    }
    for (s, t) in remove_edges {
        let (s, t) = (id(&h, s), id(&h, t));
        h.edges.retain(|e| !(e.source == s && e.target == t));
    }
    for (s, t) in add_edges {
        let (s, t) = (id(&h, s), id(&h, t));
        h.edges.push(model::Edge { id: format!("x{s}{t}"), source: s, target: t });
    }
    for l in remove {
        let r = id(&h, l);
        h.nodes.retain(|n| n.id != r);
        h.edges.retain(|e| e.source != r && e.target != r);
    }
    let s = sync(&h);
    let e = expand(&s.text, &s).unwrap_or_else(|d| panic!("{d:?}\n{}", s.text));
    assert!(same_structure(&e.graph, &s), "{}", s.text);
    s
}

#[test]
fn structural_canvas_changes_keep_the_hand_written_text() {
    let text = "let w = 32\ndef pair(c) {\n  a conv c 3x3\n  b conv c 3x3\n}\n1 input\n2 conv w 3x3   # the stem\n3 pair(w * 2)\n4 dense 10\n5 output\n";
    let g = build(text);
    let kept = |s: &Graph| assert!(s.text.contains("def pair(c) {") && s.text.contains("let w = 32") && s.text.contains("pair(w * 2)") && s.text.contains("# the stem"), "{}", s.text);

    // a block inserted between the stem and the function call: a new line after the stem, the call takes it (the line above)
    let s = edit(&g, &[("nb", "batchnorm", json!({}))], &[("2", "3.a")], &[("2", "nb"), ("nb", "3.a")], &[]);
    kept(&s);
    assert!(s.text.contains("2 conv w 3x3   # the stem\n6 bn\n3 pair(w * 2)\n"), "{}", s.text);

    // a block removed in the middle: the line goes, the next one takes what is now above it
    let s = edit(&g, &[], &[("4", "5")], &[("3.b", "5")], &["4"]);
    kept(&s);
    assert!(!s.text.contains("dense"), "{}", s.text);

    // a side branch from the stem: its line goes after the stem, and the call (which took the line above) now names its input
    let s = edit(&g, &[("side", "dense", json!({"units": 5}))], &[], &[("2", "side")], &[]);
    kept(&s);
    assert!(s.text.contains("2 conv w 3x3   # the stem\n6 dense 5\n3 pair(w * 2) <- 2\n"), "{}", s.text);

    // a skip connection around the call: an add of the call's output and the stem, wired into the dense block
    let s = edit(&g, &[("sk", "add", json!({}))], &[("3.b", "4")], &[("3.b", "sk"), ("2", "sk"), ("sk", "4")], &[]);
    kept(&s);
    assert!(s.text.contains("add <- 3, 2"), "{}", s.text);

    // a block made by the function changed on the canvas: the function cannot change for one call, so the text is printed again
    let s = edit(&g, &[("nb", "batchnorm", json!({}))], &[("3.a", "3.b")], &[("3.a", "nb"), ("nb", "3.b")], &[]);
    assert!(!s.text.contains("def pair"), "{}", s.text);
}

#[test]
fn structural_change_in_the_everything_design() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/examples/everything.nnb")).unwrap();
    let g = build(&text);
    // a dropout between the decoder's add (11) and the patch embedding (12); and the CNN head (19) moved to take the dropout
    let s = edit(&g, &[("dr", "dropout", json!({"rate": 0.3}))], &[("11", "12"), ("11", "19")], &[("11", "dr"), ("dr", "12"), ("dr", "19")], &[]);
    for kept in ["def se(c, r = 4) {", "def mixer(c) {", "5 for c, s in (w, 2), (w * 2, 2) {", "6 repeat depth {", "7 up(32) <- 6, 5.2.m", "16 repeat depth {"] {
        assert!(s.text.contains(kept), "lost '{kept}':\n{}", s.text);
    }
    assert!(s.text.contains("11 add <- 10, 8\n27 dropout 0.3\n"), "{}", s.text);
    assert!(s.text.contains("19 avgpool 4x4 <- 27"), "{}", s.text);
    assert_eq!(analyze(&s, &cifar()).total_params, analyze(&g, &cifar()).total_params);
}

#[test]
fn computed_numbers_print_short_and_stay_in_step() {
    let mut g = build("1 input\n2 conv 8 3x3\n3 dropout 0.3\n4 flatten\n5 dense 10\n6 output\n");
    let i = g.nodes.iter().position(|n| n.kind == "dropout").unwrap();
    g.nodes[i].params.insert("rate".into(), json!(0.1 + 0.2));
    let s = sync(&g);
    assert!(s.text.contains("3 dropout 0.3\n"), "{}", s.text);
    // the text is kept (not printed again) although the stored number is 0.30000000000000004
    assert_eq!(s.text, g.text);
    g.text.clear();
    assert!(sync(&g).text.contains("dropout 0.3\n"));
}

#[test]
fn a_new_text_does_not_take_places_of_other_blocks() {
    let g = build("1 input\n2 conv 8 3x3\n3 maxpool 2x2\n4 conv 16 3x3\n5 maxpool 2x2\n6 flatten\n7 dense 10\n8 output\n");
    // the same numbers for different blocks: only blocks of the same kind keep their id
    let h = apply("1 input\n2 conv 8 3x3\n3 global_pool\n4 output\n", &g).unwrap();
    let same = |l: &str| g.nodes.iter().find(|n| n.label == l).map(|n| n.id.clone()) == h.nodes.iter().find(|n| n.label == l).map(|n| n.id.clone());
    assert!(same("1") && same("2"));
    assert!(!same("3") && !same("4"));
}

#[test]
fn too_many_blocks_is_reported_on_the_repeat_line() {
    let e = err("1 input\n2 repeat 9000 {\n  a ln\n  b dense 64\n  c add <- in, b\n}\n3 output\n");
    assert_eq!(e.line, 2);
    assert!(e.message.contains("27000 blocks") && e.message.contains("At most 6666 passes"), "{}", e.message);
}

#[test]
fn recommended_design_size_follows_memory() {
    use nnb_core::dsl::recommended_max_blocks;
    let gb = |g: u64| g << 30;
    assert_eq!(recommended_max_blocks(gb(8)), 6_000); // 2 GB for the canvas at 0.3 MB per block
    assert_eq!(recommended_max_blocks(gb(16)), 13_000);
    assert_eq!(recommended_max_blocks(gb(124)), 105_000);
    assert_eq!(recommended_max_blocks(gb(1)), 5_000); // never below 5,000
}

/// the manual's examples (doc/examples/*.nnb): each builds, has no error or warning at its input, compiles, and keeps its text
#[test]
fn manual_examples_build() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/examples");
    let ds = |c: usize, s: usize, task: &str| Dataset { source: "huggingface".into(), task: task.into(), hf_repo: "x/y".into(), image_size: [s, s],
        channels: c, classes: if task == "segment" { vec!["a".into(), "b".into()] } else { (0..10).map(|i| i.to_string()).collect() }, ..Default::default() };
    // two Sentinel-2 band groups with the same 4 bands (before / after) for the Siamese example
    let s2 = |b: &[&str]| b.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let two_dates = Dataset { source: "local".into(), local_dir: "/data/x".into(), image_size: [64, 64], sensor: "s2".into(),
        band_order: s2(nnb_core::dataset::SENTINEL2_GEE), bands: s2(&["B2", "B3", "B4", "B8"]), sensor2: "s2".into(),
        band_order2: s2(nnb_core::dataset::SENTINEL2_GEE), bands2: s2(&["B2", "B3", "B4", "B8"]), fuse_dir: "after".into(),
        classes: s2(&["changed", "unchanged"]), ..Default::default() };
    // six Sentinel-2 bands, 128×128 patches with masks
    let water = Dataset { source: "local".into(), task: "segment".into(), local_dir: "/data/x".into(), kaggle_train_dir: "images".into(),
        kaggle_mask_dir: "masks".into(), image_size: [128, 128], sensor: "s2".into(), band_order: s2(nnb_core::dataset::SENTINEL2_GEE),
        bands: s2(&["B2", "B3", "B4", "B8", "B11", "B12"]), classes: s2(&["not water", "water"]), ..Default::default() };
    let cases = [("lenet5", ds(1, 28, "classify"), 61_706u64), ("resnet_small", ds(3, 32, "classify"), 0), ("vit_small", ds(3, 32, "classify"), 0),
        ("unet_small", ds(3, 64, "segment"), 0), ("everything", ds(3, 32, "classify"), 0),
        // the worked examples of doc/building.tex, with the parameter counts printed there
        ("build_classifier", ds(3, 64, "classify"), 393_122), ("build_hybrid", ds(3, 32, "classify"), 718_410),
        ("build_segment", water, 488_450), ("build_change", two_dates, 241_378)];
    for (name, d, params) in cases {
        let text = std::fs::read_to_string(format!("{dir}/{name}.nnb")).unwrap();
        let g = build(&text);
        let a = analyze(&g, &d);
        let bad: Vec<_> = a.nodes.iter().filter(|n| n.status != "ok").map(|n| (g.node(&n.id).unwrap().label.clone(), n.messages.clone())).collect();
        assert!(a.ok && bad.is_empty(), "{name}: {:?} {bad:?}", a.problems);
        if params > 0 {
            assert_eq!(a.total_params, params, "{name}");
        }
        println!("{name}: {} blocks, {} parameters", g.nodes.len(), a.total_params);
        assert_eq!(sync(&g).text, text, "{name}");
        let code = nnb_core::codegen::train_script(&g, &d, &a, &Default::default(), name).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("t.py"), &code).unwrap();
        if let Some(out) = ["python3", "python"].iter().find_map(|py| std::process::Command::new(py).args(["-m", "py_compile"]).arg(tmp.path().join("t.py")).output().ok()) {
            assert!(out.status.success(), "{name}: {}", String::from_utf8_lossy(&out.stderr));
        }
    }
}

/// the short examples written into the manual's language chapter (doc/language.tex, verbatim blocks) all build
#[test]
fn manual_snippets_build() {
    // CRLF on a Windows checkout
    let tex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/language.tex")).unwrap().replace("\r\n", "\n");
    let mut n = 0;
    for part in tex.split("\\begin{verbatim}\n").skip(1) {
        let snip = part.split("\\end{verbatim}").next().unwrap();
        if snip.contains("LABEL") {
            continue; // the grammar summary
        }
        let text = if snip.contains(" input") { snip.to_string() } else { format!("1 input\n{snip}") };
        build(&text);
        n += 1;
    }
    assert!(n >= 4, "{n} snippets");
    // claims in the text: named values in a call; repeat … as NAME
    build("def basic(c, s = 1) {\n  a conv c 3x3 stride s\n}\n1 input\n2 basic(c=64, s=2)\n3 basic(32)\n");
    build("1 input\n2 repeat 12 as enc {\n  a conv 8 3x3\n}\n");
}

/// the short texts of the manual's chapter on building (doc/building.tex: the motifs) build, and no block in them is wrong
#[test]
fn building_snippets_work() {
    let tex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/building.tex")).unwrap().replace("\r\n", "\n");
    let mut n = 0;
    for part in tex.split("\\begin{verbatim}\n").skip(1) {
        let snip = part.split("\\end{verbatim}").next().unwrap();
        let g = build(&format!("1 input\n{snip}"));
        let d = if snip.contains("seg_output") { Dataset { task: "segment".into(), classes: vec!["a".into(), "b".into()], ..cifar() } } else { cifar() };
        let a = analyze(&g, &d);
        // a motif without a head ends in a block whose result is not used: that warning only
        let unused = |n: &nnb_core::shapes::NodeResult| n.status == "warn" && n.messages.iter().all(|m| m.contains("is not used"));
        let bad: Vec<_> = a.nodes.iter().filter(|n| n.status != "ok" && !unused(n)).map(|n| (g.node(&n.id).unwrap().label.clone(), n.messages.clone())).collect();
        assert!(bad.is_empty(), "{snip}\n{bad:?}");
        n += 1;
    }
    assert_eq!(n, 5, "snippets in building.tex");
}

/// every way of writing a block's settings that the manual's block reference shows, and what it sets
#[test]
fn block_reference_forms() {
    let one = |line: &str| -> Map<String, serde_json::Value> {
        let g = build(&format!("1 input\n2 {line}\n"));
        g.nodes.iter().find(|n| n.label == "2").unwrap().params.clone()
    };
    let cases: &[(&str, serde_json::Value)] = &[
        // conv: filters by the first number, kernel KxK, stride, pad / same / valid, dil, activation, nobias; or name=value
        ("conv 64 3x3", json!({"filters": 64, "kernel": [3, 3], "stride": [1, 1], "padding": "same", "activation": "relu", "bias": true})),
        ("conv 64 5x3 stride 2 pad 1 dil 2 gelu nobias", json!({"kernel": [5, 3], "stride": [2, 2], "padding": 1, "dilation": 2, "activation": "gelu", "bias": false})),
        ("conv 8 1x1 valid none", json!({"padding": "valid", "activation": "none"})),
        ("conv filters=64 kernel=7 stride=2 padding=3 dilation=1 activation=silu bias=false", json!({"filters": 64, "kernel": [7, 7], "stride": [2, 2], "padding": 3, "activation": "silu", "bias": false})),
        ("conv 32 3x3 stride=2x1", json!({"stride": [2, 1]})),
        ("Conv2D 16 3x3", json!({"filters": 16})),
        // depthwise
        ("dwconv x2 3x3 stride 2 pad 1 none nobias", json!({"multiplier": 2, "stride": [2, 2], "padding": 1, "activation": "none", "bias": false})),
        ("dwconv multiplier=3 kernel=5x5", json!({"multiplier": 3, "kernel": [5, 5]})),
        // transposed
        ("convT 64 2x2 stride 2 none", json!({"filters": 64, "kernel": [2, 2], "stride": [2, 2], "activation": "none"})),
        ("convT 32 3x3 stride 2 pad 1 outpad 1", json!({"padding": 1, "output_padding": 1})),
        // pools
        ("maxpool 2x2", json!({"mode": "max", "size": [2, 2], "stride": "size"})),
        ("avgpool 3x3 stride 1 pad 1", json!({"mode": "avg", "size": [3, 3], "stride": [1, 1], "padding": 1})),
        ("maxpool 3", json!({"size": [3, 3]})),
        ("pool avg size=2 stride=size", json!({"mode": "avg", "size": [2, 2], "stride": "size"})),
        ("global_pool max", json!({"mode": "max"})),
        ("gap", json!({"mode": "avg"})),
        ("upsample x4 bilinear", json!({"scale": 4, "mode": "bilinear"})),
        ("upsample 2", json!({"scale": 2, "mode": "nearest"})),
        // normalisation, dropout, dense, activations
        ("gn 4", json!({"groups": 4})),
        ("groupnorm groups=16", json!({"groups": 16})),
        ("dropout 0.125", json!({"rate": 0.125})),
        ("dropout rate=0.5", json!({"rate": 0.5})),
        ("dense 10 none", json!({"units": 10, "activation": "none"})),
        ("linear units=256 activation=tanh", json!({"units": 256, "activation": "tanh"})),
        ("gelu", json!({"fn": "gelu"})),
        ("activation leaky_relu", json!({"fn": "leaky_relu"})),
        ("act fn=silu", json!({"fn": "silu"})),
    ];
    for (line, want) in cases {
        let p = one(line);
        for (k, v) in want.as_object().unwrap() {
            assert_eq!(p.get(k), Some(v), "{line}: {k} in {p:?}");
        }
    }
    // transformer blocks (on tokens)
    let g = build("1 input\n2 conv 16 4x4 stride 4 valid none\n3 to_tokens rows\n4 cls_pos nocls\n5 attention 4 drop 0.1\n6 attn heads=2 dropout=0\n7 token_pool mean\n8 output\n");
    let p = |l: &str| g.nodes.iter().find(|n| n.label == l).unwrap().params.clone();
    assert_eq!(p("3")["mode"], json!("rows"));
    assert_eq!(p("4")["cls"], json!(false));
    assert_eq!((p("5")["heads"].clone(), p("5")["dropout"].clone()), (json!(4), json!(0.1)));
    assert_eq!(p("6")["heads"], json!(2));
    assert_eq!(p("7")["mode"], json!("mean"));
    // words a block does not have are errors
    for (line, msg) in [("convT 8 2x2 same", "means nothing for convT"), ("convT 8 2x2 padding=same", "padding is a number of pixels here"), ("dwconv 3x3 dil 2", "no dil setting"),
        ("conv 8 3x3 stride size", "'stride size' is for a pool"), ("convT 8 2x2 nobias", "no nobias / bias setting"), ("dense 8 3x3", "a size is for")] {
        let e = err(&format!("1 input\n2 {line}\n"));
        assert!(e.message.contains(msg), "{line}: {}", e.message);
    }
    // a fraction outside its range is an error on the block (not cut silently), and the exact rate reaches the code
    let ds = cifar();
    let g = build("1 input\n2 conv 8 3x3\n3 dropout 0.99\n4 global_pool\n5 output\n");
    assert!(analyze(&g, &ds).nodes.iter().any(|n| n.messages.iter().any(|m| m.contains("rate is 0.99 but must be from 0 to 0.95"))));
    let g = build("1 input\n2 conv 8 3x3\n3 dropout 0.125\n4 global_pool\n5 output\n");
    assert!(nnb_core::codegen::model_code(&g, &analyze(&g, &ds)).unwrap().contains("nn.Dropout(p=0.125)"));
}

#[test]
fn the_design_name_is_kept_and_used() {
    let text = "net \"My CNN\"\ndef pair(c) {\n  a conv c 3x3\n  b conv c 3x3\n}\n1 input\n2 pair(16)\n3 global_pool\n4 output\n";
    let g = build(text);
    assert_eq!(dsl::design_name(&g.text).as_deref(), Some("My CNN"));
    // Format prints the blocks again but keeps the name
    let f = dsl::reprint(&g);
    assert!(f.text.starts_with("net \"My CNN\"\n1 input\n"), "{}", f.text);
    assert!(same_structure(&expand(&f.text, &f).unwrap().graph, &f));
    // a canvas change that prints the text again keeps it too
    let mut c = g.clone();
    let i = c.nodes.iter().position(|n| n.label == "2.a").unwrap();
    c.nodes[i].params.insert("filters".into(), json!(8));
    let s = sync(&c);
    assert!(s.text.starts_with("net \"My CNN\"\n") && !s.text.contains("def pair"), "{}", s.text);
    // and it is in the generated script's first line
    let ds = cifar();
    let code = nnb_core::codegen::train_script(&g, &ds, &analyze(&g, &ds), &Default::default(), "Project X").unwrap();
    assert!(code.lines().next().unwrap().contains("Project X (design My CNN)"), "{}", code.lines().next().unwrap());
    // without a name, nothing changes
    assert_eq!(dsl::design_name("1 input\n2 output\n"), None);
}

#[test]
fn custom_blocks_and_python_section() {
    let text = "net \"custom\"\npython \"\"\"\nclass Twice(nn.Module):\n    # out: c * 2, h, w\n    def __init__(self, c, h, w):\n        super().__init__()\n        self.conv = nn.Conv2d(c, c * 2, 1)\n\n    def forward(self, x):\n        return {'a': 1} and self.conv(x)\n\"\"\"\n1 input\n2 conv 8 3x3\n3 custom Twice\n4 global_pool\n5 output\n";
    let g = build(text);
    assert!(g.python.starts_with("class Twice(nn.Module):") && g.python.contains("{'a': 1}"), "{}", g.python);
    assert_eq!(by_label(&g, "3").params["class"], json!("Twice"));
    assert_eq!(by_label(&g, "4").kind, "global_pool", "line numbers after the section are kept");
    let a = analyze(&g, &cifar());
    assert!(a.ok, "{:?}", a.problems);
    assert_eq!(a.get(&by_label(&g, "3").id).unwrap().out, Some(vec![16, 32, 32]));
    // printed from the blocks: `custom Twice`, and sync puts the code back into the text
    assert!(print(&g).0.contains("3 custom Twice"), "{}", print(&g).0);
    let mut printed = g.clone();
    printed.text = String::new();
    let s = sync(&printed);
    assert!(s.text.contains("python \"\"\"\nclass Twice") && s.text.contains("custom Twice"), "{}", s.text);
    assert_eq!(build(&s.text).python, g.python);
    // the code changed on the canvas: the text's section follows, the rest of the text stays
    let mut edited = g.clone();
    edited.python = g.python.replace("c * 2, h, w", "c * 2, h, w  ");
    let s = sync(&edited);
    assert!(s.text.contains("c * 2, h, w  ") && s.text.contains("net \"custom\"") && s.text.contains("2 conv 8 3x3"), "{}", s.text);
    // problems: an unclosed section, two sections, a bad class name
    assert!(err("python \"\"\"\nclass A(nn.Module):\n    pass\n1 input\n").message.contains("not closed"));
    assert!(err("python \"\"\"\n\"\"\"\npython \"\"\"\n\"\"\"\n1 input\n").message.contains("only one"));
    assert!(err("1 input\n2 custom 3\n").message.contains("number"));
}

#[test]
fn output_head_number() {
    let g = build("1 input\n2 global_pool\n3 output\n4 output 2 <- 2\n5 output head=1 <- 2\n");
    assert_eq!(by_label(&g, "4").params["head"], json!(2));
    assert_eq!(by_label(&g, "5").params["head"], json!(1));
    let printed = print(&g).0;
    assert!(printed.contains("3 output\n") && printed.contains("output 2 <- 2") && printed.contains("output 1 <- 2"), "{printed}");
}

#[test]
fn shared_weights_and_subtract() {
    let text = "1 input 1\n2 conv 8 3x3\n3 conv 8 3x3\n4 input 2\n5 shared 2\n6 shared 3\n7 subtract abs <- 3, 6\n8 global_pool\n9 output\n";
    let g = build(text);
    assert_eq!(by_label(&g, "5").params["of"], json!(by_label(&g, "2").id.clone()));
    assert_eq!(by_label(&g, "7").params["abs"], json!(true));
    let printed = print(&g).0;
    assert!(printed.contains("5 shared 2") && printed.contains("6 shared 3") && printed.contains("subtract abs"), "{printed}");
    assert_eq!(sync(&g).text, text, "the hand-written text is kept");
    // a function: a label inside the same call
    let f = build("def twin() {\n  a conv 8 3x3\n  b shared a <- in\n  c sub <- a, b\n}\n1 input\n2 twin()\n3 global_pool\n4 output\n");
    assert_eq!(by_label(&f, "2.b").params["of"], json!(by_label(&f, "2.a").id.clone()));
    assert!(err("1 input\n2 shared 7\n").message.contains("no block has the label 7"));
    assert!(err("1 input\n2 conv 8 3x3\n3 subtract wrong <- 1, 2\n").message.contains("abs"));
}

#[test]
fn time_series_blocks_in_text() {
    let g = build("1 input\n2 conv3d 16 3x3 time 3\n3 bn\n4 tpool attention\n5 conv 8 3x3 <- 4\n6 seq <- 3\n7 lstm 32 bi\n8 gru 16\n\
        9 conv1d 8 5 stride 2 valid gelu nobias\n10 token_pool last\n");
    let p = |l: &str| by_label(&g, l).params.clone();
    assert_eq!((p("2")["filters"].clone(), p("2")["kernel"].clone(), p("2")["tkernel"].clone()), (json!(16), json!([3, 3]), json!(3)));
    assert_eq!((p("4")["mode"].clone(), p("7")["units"].clone(), p("7")["bidirectional"].clone()), (json!("attention"), json!(32), json!(true)));
    assert_eq!((p("9")["kernel"].clone(), p("9")["stride"].clone(), p("9")["padding"].clone(), p("9")["bias"].clone()), (json!(5), json!(2), json!("valid"), json!(false)));
    assert_eq!(p("10")["mode"], json!("last"));
    let printed = print(&g).0;
    for l in ["conv3d 16 3x3 time 3", "temporal_pool", "to_sequence", "lstm 32 bi", "gru 16", "conv1d 8 5 stride 2 valid gelu nobias", "token_pool last"] {
        assert!(printed.contains(l), "{l} in\n{printed}");
    }
    assert!(err("1 input\n2 conv3d 8 3x3 pad 1\n").message.contains("same or valid"));
    assert!(err("1 input\n2 conv3d 8 3x3 dil 2\n").message.contains("no dil setting"));
}

/// doc/examples/timeseries.nnb: every time-series block on a (bands, dates, height, width) input
#[test]
fn timeseries_design() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../doc/examples/timeseries.nnb")).unwrap();
    let g = build(&text);
    let ds = Dataset { source: "local".into(), local_dir: "/x".into(), image_size: [32, 32], sensor: "s2".into(), time_steps: 6,
        band_order: vec!["B2".into(), "B3".into(), "B4".into(), "B8".into()], bands: vec!["B2".into(), "B3".into(), "B4".into(), "B8".into()],
        classes: vec!["a".into(), "b".into(), "c".into()], ..Default::default() };
    assert!(ds.problems().is_empty(), "{:?}", ds.problems());
    let a = analyze(&g, &ds);
    let bad: Vec<_> = a.nodes.iter().filter(|n| n.status != "ok").map(|n| (g.node(&n.id).unwrap().label.clone(), n.messages.clone())).collect();
    assert!(a.ok && bad.is_empty(), "{:?} {bad:?}", a.problems);
    for kind in ["conv3d", "temporal_pool", "to_sequence", "lstm", "gru", "conv1d"] {
        assert!(g.nodes.iter().any(|n| n.kind == kind), "no {kind} block");
    }
    let out = |l: &str| a.get(&by_label(&g, l).id).unwrap().out.clone().unwrap();
    assert_eq!(a.input, vec![4, 6, 32, 32]);
    assert_eq!((out("2"), out("4"), out("5"), out("9"), out("11"), out("15")),
        (vec![16, 6, 32, 32], vec![16, 6, 16, 16], vec![16, 16, 16], vec![6, 4], vec![6, 32], vec![16]));
    // parameters by hand: conv3d (4·3·3·3 + 1)·16, temporal attention 16 + 1, LSTM 2 directions · 4 gates · (16·(16 + 16) + 2·16)
    assert_eq!(a.get(&by_label(&g, "2").id).unwrap().params, (4 * 27 + 1) * 16);
    assert_eq!(a.get(&by_label(&g, "5").id).unwrap().params, 17);
    assert_eq!(a.get(&by_label(&g, "11").id).unwrap().params, 2 * 4 * (16 * 32 + 32));
    let code = nnb_core::codegen::train_script(&g, &ds, &a, &nnb_core::codegen::TrainSettings::default(), "ts").unwrap();
    assert!(code.contains("nn.Conv3d(4, 16, kernel_size=(3, 3, 3)") && code.contains("TemporalPool(\"attention\", 16)") && code.contains("Recurrent(\"lstm\", 16, 16, True)"));
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.py"), &code).unwrap();
    if let Some(out) = ["python3", "python"].iter().find_map(|py| std::process::Command::new(py).args(["-m", "py_compile"]).arg(dir.path().join("t.py")).output().ok()) {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    assert_eq!(sync(&g).text, text);
    // 2-D blocks on the time series and time-series blocks on an image say what to put between
    let msg = |t: &str, l: &str| { let g = build(t); analyze(&g, &ds).get(&by_label(&g, l).id).unwrap().messages.join(" ") };
    assert!(msg("1 input\n2 conv 8 3x3\n", "2").contains("Temporal pool"));
    assert!(msg("1 input\n2 tpool\n3 conv3d 8 3x3\n", "3").contains("needs a time series"));
    assert!(msg("1 input\n2 lstm 8\n", "2").contains("To sequence"));
}

#[test]
fn transformer_additions_in_text() {
    let g = build("1 input\n2 to_tokens\n3 cls_pos sinusoidal\n4 attention 4 causal rotary\n5 xattn heads 2 drop 0.1 <- 4, 3\n6 rms\n7 swiglu 96\n\
        8 cls_pos nocls nopos\n9 concat <- 7, 8\n10 token_pool mean\n11 output\n");
    let p = |l: &str| by_label(&g, l).params.clone();
    assert_eq!(p("3")["pos"], json!("sinusoidal"));
    assert_eq!((p("4")["causal"].clone(), p("4")["rotary"].clone(), p("4")["heads"].clone()), (json!(true), json!(true), json!(4)));
    assert_eq!((by_label(&g, "5").kind.as_str(), p("5")["heads"].clone(), p("5")["dropout"].clone()), ("cross_attention", json!(2), json!(0.1)));
    assert_eq!((by_label(&g, "6").kind.as_str(), p("7")["units"].clone(), p("8")["pos"].clone()), ("rmsnorm", json!(96), json!("none")));
    let printed = print(&g).0;
    for l in ["cls_pos sinusoidal", "attention heads 4 causal rotary", "cross_attention heads 2 drop 0.1 <- 4, 3", "rmsnorm", "swiglu 96", "cls_pos nocls nopos"] {
        assert!(printed.contains(l), "{l} in\n{printed}");
    }
}

#[test]
fn adapters_experts_exits_in_text() {
    let g = build("1 input\n2 conv 8 3x3 frozen\n3 global_pool\n4 exit 0.4\n5 dense 16 lora 4 <- 3\n6 output\n");
    assert_eq!(by_label(&g, "2").params["frozen"], json!(true));
    assert_eq!((by_label(&g, "4").kind.as_str(), by_label(&g, "4").params["weight"].clone()), ("exit_output", json!(0.4)));
    assert_eq!(by_label(&g, "5").params["lora"], json!(4));
    let t = build("1 input\n2 conv 8 2x2 stride 2\n3 to_tokens\n4 moe 8 top 1 hidden 32\n5 attention heads 2 lora 2 frozen\n6 token_pool mean\n7 output\n");
    assert_eq!((by_label(&t, "4").params["experts"].clone(), by_label(&t, "4").params["top_k"].clone(), by_label(&t, "4").params["units"].clone()),
        (json!(8), json!(1), json!(32)));
    let printed = print(&g).0 + &print(&t).0;
    for l in ["conv 8 3x3 frozen", "exit 0.4", "dense 16 lora 4", "moe 8 top 1 hidden 32", "attention heads 2 lora 2 frozen"] {
        assert!(printed.contains(l), "{l} in\n{printed}");
    }
    // GShard options: a capacity factor and the random second expert, printed back only when set
    let gs = build("1 input\n2 conv 8 2x2 stride 2\n3 to_tokens\n4 moe 8 top 2 hidden 32 cap 1.25 random\n5 token_pool mean\n6 output\n");
    assert_eq!((by_label(&gs, "4").params["capacity"].clone(), by_label(&gs, "4").params["random_second"].clone()), (json!(1.25), json!(true)));
    assert!(print(&gs).0.contains("moe 8 top 2 hidden 32 cap 1.25 random"), "{}", print(&gs).0);
    assert!(!print(&t).0.contains("cap") && !print(&t).0.contains("random"));
    // frozen only where there are weights
    assert!(err("1 input\n2 maxpool 2x2 frozen\n").message.contains("frozen"));
}

#[test]
fn renumbered_lines_keep_their_blocks() {
    // the Text tab inserted a line after 2 and moved 3.. on by one (auto numbering); the blocks keep their ids and places
    let old = build("1 input\n2 conv 8 3x3\n3 repeat 2 {\n  a conv 8 3x3\n  b add <- in, a\n}\n4 global_pool\n5 output\n");
    let new_text = "1 input\n2 conv 8 3x3\n3 bn\n4 repeat 2 {\n  a conv 8 3x3\n  b add <- in, a\n}\n5 global_pool\n6 output\n";
    let steps = vec![vec![("3".into(), "4".into()), ("4".into(), "5".into()), ("5".into(), "6".into())]];
    let g = dsl::apply_renamed(new_text, &old, &steps).unwrap();
    for (was, now) in [("2", "2"), ("3.2.a", "4.2.a"), ("3.1.b", "4.1.b"), ("4", "5"), ("5", "6")] {
        let (o, n) = (by_label(&old, was), by_label(&g, now));
        assert_eq!((&o.id, o.x, o.y), (&n.id, n.x, n.y), "{was} -> {now}");
    }
    // letters inside a group: 3.*.a -> 3.*.b ("*" = every pass)
    let g2 = dsl::apply_renamed("1 input\n2 conv 8 3x3\n3 repeat 2 {\n  a bn\n  b conv 8 3x3\n  c add <- in, b\n}\n4 global_pool\n5 output\n", &old,
        &[vec![("3.*.a".into(), "3.*.b".into()), ("3.*.b".into(), "3.*.c".into())]]).unwrap();
    assert_eq!(by_label(&g2, "3.2.b").id, by_label(&old, "3.2.a").id);
    assert_eq!(by_label(&g2, "3.1.c").id, by_label(&old, "3.1.b").id);
    // without the renames the moved blocks would be new ones
    let plain = dsl::apply(new_text, &old).unwrap();
    assert_ne!(by_label(&plain, "6").id, by_label(&old, "5").id);
}

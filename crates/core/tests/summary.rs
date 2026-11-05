//! Comparing runs: the summary (per run, and mean / sample s.d. per group of seeds) and its CSV / LaTeX export, on runs written by hand.
//! Own test binary, because it sets NN_BUILDER_HOME.

use nnb_core::runs::{self, RunInfo};
use nnb_core::store;
use serde_json::json;

fn fake_run(pr: &store::Project, id: &str, group: &str, seed: u64, acc: f64, top5: f64, epochs: usize) {
    let d = pr.runs_dir().join(id);
    std::fs::create_dir_all(&d).unwrap();
    store::write_json(&d.join("run.json"), &RunInfo { id: id.into(), target: "Kaggle".into(), kind: "kaggle".into(), status: "done".into(),
        seed, group: group.into(), dataset: "hf:uoft-cs/cifar100".into(), fetched: true, ..Default::default() }).unwrap();
    store::write_json(&d.join("train.json"), &nnb_core::codegen::TrainSettings { seed, epochs: 100, ..Default::default() }).unwrap();
    let mut m = format!("{}\n", json!({"event": "start", "params": 478132, "gpu": "Tesla T4"}));
    for e in 1..=epochs {
        m += &format!("{}\n", json!({"event": "epoch", "epoch": e, "val_acc": 0.5 + e as f64 / 100.0, "seconds": 10.0}));
    }
    m += &format!("{}\n", json!({"event": "done", "test_acc": acc, "test_top5": top5, "best_epoch": epochs - 2}));
    std::fs::write(d.join("metrics.jsonl"), m).unwrap();
    store::write_json(&d.join("results.json"), &json!({"test_acc": acc, "test_top5": top5})).unwrap();
}

#[test]
fn summary_and_export() {
    let base = tempfile::tempdir().unwrap();
    std::env::set_var("NN_BUILDER_HOME", base.path().join("settings"));
    let pr = store::create("Compare", Some(&base.path().join("projects"))).unwrap();
    fake_run(&pr, "20261105-100000-s1", "g1", 1, 0.60, 0.85, 30);
    fake_run(&pr, "20261105-100001-s2", "g1", 2, 0.62, 0.86, 28);
    fake_run(&pr, "20261105-100002-s3", "g1", 3, 0.64, 0.87, 35);
    fake_run(&pr, "20261105-110000", "", 1, 0.50, 0.80, 10);
    let ids: Vec<String> = ["20261105-100000-s1", "20261105-100001-s2", "20261105-100002-s3", "20261105-110000"].map(String::from).to_vec();
    let s = runs::summary(&pr, &ids).unwrap();
    let rows = s["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!((rows[0]["epochs_done"].as_u64(), rows[0]["best_epoch"].as_u64(), rows[0]["seconds"].as_f64()), (Some(30), Some(28), Some(300.0)));
    let g = s["groups"].as_array().unwrap();
    assert_eq!(g.len(), 2, "the group of seeds and the single run");
    let acc = &g[0]["test_acc"];
    assert!((acc["mean"].as_f64().unwrap() - 0.62).abs() < 1e-12 && (acc["std"].as_f64().unwrap() - 0.02).abs() < 1e-12, "{acc}");
    assert_eq!(g[0]["seeds"], json!([1, 2, 3]));

    let (csv, tex) = runs::export(&pr, &ids).unwrap();
    let csv = std::fs::read_to_string(csv).unwrap();
    assert_eq!(csv.lines().count(), 5);
    assert!(csv.lines().nth(1).unwrap().starts_with("20261105-100000-s1,g1,1,Kaggle,hf:uoft-cs/cifar100,adam,0.001,64,constant,100,30,28,0.6,0.85,"), "{csv}");
    let tex = std::fs::read_to_string(tex).unwrap();
    assert!(tex.contains(r"\begin{tabular}{@{}lrrrr@{}}") && tex.contains(r"Run & Seed & Epochs & Top-1 (\%) & Top-5 (\%) \\"), "{tex}");
    assert!(tex.contains(r"20261105-100000-s1 & 1 & 30 & 60.00 & 85.00 \\"), "{tex}");
    assert!(tex.contains(r"Mean $\pm$ s.d. (3 seeds) & & & 62.00 $\pm$ 2.00 & 86.00 $\pm$ 1.00 \\"), "{tex}");
    assert!(tex.trim_end().ends_with(r"\end{tabular}"), "{tex}");
    // the table compiles in a LaTeX document (when a TeX installation is present)
    if std::process::Command::new("pdflatex").arg("--version").output().is_ok() {
        let doc = base.path().join("t.tex");
        std::fs::write(&doc, format!("\\documentclass{{article}}\\usepackage{{booktabs}}\\begin{{document}}\n{tex}\\end{{document}}\n")).unwrap();
        let out = std::process::Command::new("pdflatex").args(["-interaction=nonstopmode", "-halt-on-error"]).arg(&doc).current_dir(base.path()).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    }
}

//! Demos open as the one unsaved draft: opening demos again and again leaves one folder, nothing reaches the projects folder until
//! save_as, and an unsaved draft is discarded when another project is opened or created (and when the app quits or starts).
//! Own test binary, because it sets NN_BUILDER_HOME.

use nnb_core::store;
use serde_json::Value;

#[test]
fn demos_are_one_draft_until_saved() {
    let base = tempfile::tempdir().unwrap();
    let projects = base.path().canonicalize().unwrap().join("projects"); // macOS: /var is /private/var, and open() canonicalizes
    std::env::set_var("NN_BUILDER_HOME", base.path().join("settings"));
    store::save_setting("projects_dir", Value::String(projects.to_string_lossy().into())).unwrap();
    let alexnet = |p: &store::Project| p.save_graph(&nnb_core::templates::build("alexnet").unwrap());

    for _ in 0..3 {
        assert!(store::create_draft("AlexNet demo", alexnet).unwrap().is_draft());
    }
    assert!(!projects.exists(), "opening demos writes nothing to the projects folder");
    assert!(store::listing().is_empty(), "the draft is not in the projects list");
    assert!(store::last_project().is_some_and(|p| p.is_draft() && p.graph().nodes.len() > 5));

    // quitting (or starting) discards it
    store::discard_draft().unwrap();
    assert!(store::draft().is_none() && store::last_project().is_none());

    // creating a project discards it
    store::create_draft("AlexNet demo", alexnet).unwrap();
    let other = store::create("Other", None).unwrap();
    // a typed or pasted folder: quotes and spaces around it, or a file inside the project, open the same project
    let dir = other.dir.to_string_lossy().to_string();
    // (files joined with the system's separator: Windows' \\?\ paths do not accept "/")
    let inside = |f: &str| other.dir.join(f).to_string_lossy().to_string();
    for typed in [format!("  '{dir}' "), format!("\"{dir}\""), inside("model.json"), inside("project.json")] {
        assert_eq!(store::open(std::path::Path::new(&typed)).unwrap().dir, other.dir, "{typed}");
    }
    let err = store::open(&projects).unwrap_err().to_string();
    assert!(err.contains("no project.json") && err.contains("projects"), "{err}");
    assert!(store::open(std::path::Path::new("/no/such/folder")).unwrap_err().to_string().contains("There is no folder"));
    assert!(store::draft().is_none());

    // opening a project discards it
    store::create_draft("AlexNet demo", alexnet).unwrap();
    store::open(&other.dir).unwrap();
    assert!(store::draft().is_none());

    // saving keeps it, as a normal project
    let lenet = store::create_draft("LeNet-5 demo", |p| p.save_graph(&nnb_core::templates::build("lenet5").unwrap())).unwrap();
    let saved = store::save_as(&lenet, "My LeNet", None).unwrap();
    assert_eq!(saved.dir, projects.join("My LeNet"));
    assert!(!saved.is_draft() && saved.name() == "My LeNet" && saved.graph().nodes.len() > 5);
    let meta: Value = store::read_json(&saved.dir.join("project.json")).unwrap();
    assert!(meta.get("draft").is_none());
    assert!(store::draft().is_none(), "saving moves the draft away");
    store::discard_draft().unwrap();
    assert!(saved.dir.join("model.json").exists(), "a saved demo survives the app quitting");
    assert_eq!(store::listing().len(), 2);
    assert!(store::save_as(&saved, "Again", None).is_err(), "a saved project is not saved again");
    assert!(store::create("My LeNet", None).is_err(), "names stay unique");

    // a description goes into project.json and the list; empty removes it; only listed projects take one
    let path = saved.dir.to_string_lossy().to_string();
    store::set_description(&path, "  LeNet on MNIST, try dropout next \n").unwrap();
    let row = store::listing().into_iter().find(|p| p["path"] == path.as_str()).unwrap();
    assert_eq!(row["description"], "LeNet on MNIST, try dropout next");
    assert_eq!(saved.name(), "My LeNet", "the rest of project.json stays");
    store::set_description(&path, " ").unwrap();
    let meta: Value = store::read_json(&saved.dir.join("project.json")).unwrap();
    assert!(meta.get("description").is_none());
    assert!(store::set_description(&base.path().to_string_lossy(), "x").is_err());

    // running tasks across projects: only the runs still going, newest first, with the epochs done
    let run = |id: &str, status: &str, epochs: usize| {
        let d = saved.runs_dir().join(id);
        std::fs::create_dir_all(&d).unwrap();
        store::write_json(&d.join("run.json"), &serde_json::json!({"id": id, "target": "Kaggle", "kind": "kaggle", "created": "", "status": status,
            "epochs": 10, "dataset": "mnist"})).unwrap();
        std::fs::write(d.join("metrics.jsonl"), "{\"event\": \"epoch\"}\n".repeat(epochs)).unwrap();
    };
    run("20260101-000000", "done", 10);
    run("20260102-000000", "running", 3);
    run("20260103-000000", "queued", 0);
    let act = nnb_core::runs::active_everywhere();
    assert_eq!(act.iter().map(|r| r["id"].as_str().unwrap()).collect::<Vec<_>>(), ["20260103-000000", "20260102-000000"]);
    assert_eq!((act[1]["epoch"].as_u64(), act[1]["project"].as_str()), (Some(3), Some("My LeNet")));
}

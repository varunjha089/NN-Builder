//! Demos open as the one unsaved draft: opening demos again and again leaves one folder, and nothing reaches the projects folder until
//! save_as. Own test binary, because it sets NN_BUILDER_HOME.

use nnb_core::store;
use serde_json::Value;

#[test]
fn demos_are_one_draft_until_saved() {
    let base = tempfile::tempdir().unwrap();
    let projects = base.path().join("projects");
    std::env::set_var("NN_BUILDER_HOME", base.path().join("settings"));
    store::save_setting("projects_dir", Value::String(projects.to_string_lossy().into())).unwrap();

    for _ in 0..3 {
        let d = store::create_draft("AlexNet demo", |p| p.save_graph(&nnb_core::templates::build("alexnet").unwrap())).unwrap();
        assert!(d.is_draft() && !d.draft_edited(), "a fresh demo is unedited");
    }
    assert!(!projects.exists(), "opening demos writes nothing to the projects folder");
    assert!(store::listing().is_empty(), "the draft is not in the projects list");
    let d = store::last_project().expect("the draft is reopened at the next start");
    assert!(d.is_draft() && d.graph().nodes.len() > 5);

    let mut g = d.graph();
    g.nodes.pop();
    d.save_graph(&g).unwrap();
    assert!(d.draft_edited());

    let lenet = store::create_draft("LeNet-5 demo", |p| p.save_graph(&nnb_core::templates::build("lenet5").unwrap())).unwrap();
    assert_eq!(lenet.name(), "LeNet-5 demo", "the next demo replaces the draft");
    assert!(!lenet.draft_edited());

    let saved = store::save_as(&lenet, "My LeNet", None).unwrap();
    assert_eq!(saved.dir, projects.join("My LeNet"));
    assert!(!saved.is_draft() && saved.name() == "My LeNet" && saved.graph().nodes.len() > 5);
    assert!(!saved.dir.join("draft-origin.json").exists());
    let meta: Value = store::read_json(&saved.dir.join("project.json")).unwrap();
    assert!(meta.get("draft").is_none());
    assert!(store::draft().is_none(), "saving moves the draft away");
    assert_eq!(store::listing().len(), 1);
    assert!(store::save_as(&saved, "Again", None).is_err(), "a saved project is not saved again");
    assert!(store::create("My LeNet", None).is_err(), "names stay unique");
}

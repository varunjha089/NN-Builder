//! Machine settings kept in settings.json (a scratch NN_BUILDER_HOME: its own test binary, so no other test shares the variable).
use nnb_core::remote::{self, Machine};

/// a changed Python (or place) makes the machine prepare again, so runs do not keep using the Python found before
#[test]
fn saving_a_changed_machine_forgets_what_prepare_found() {
    let home = tempfile::tempdir().unwrap();
    std::env::set_var("NN_BUILDER_HOME", home.path());
    let m = Machine { name: "lab-x".into(), host: "10.0.0.9".into(), user: "u".into(), ..Default::default() };
    remote::save_machine(m.clone()).unwrap();
    remote::save_prepared("lab-x", &remote::Prepared { python: "/w/venv/bin/python".into(), torch: "2.5".into(), ..Default::default() }).unwrap();
    remote::save_machine(Machine { group: "lab".into(), ..m.clone() }).unwrap();
    assert!(remote::prepared("lab-x").is_some(), "a group change keeps it");
    remote::save_machine(Machine { python: "/DATA/me/venv".into(), ..m }).unwrap();
    assert!(remote::prepared("lab-x").is_none(), "a new Python forgets it");
}

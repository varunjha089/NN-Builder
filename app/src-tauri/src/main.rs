//! NN Builder desktop app (Tauri 2). The window (web/) calls these commands; the work is in nnb-core. Secrets (Kaggle, Hugging Face, AWS)
//! live only in the system keychain and are handed to nnb-core for one call at a time. Nothing here trains or downloads datasets.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod update;

use nnb_core::cloud::{self, AwsCreds, KaggleCreds};
use nnb_core::codegen::TrainSettings;
use nnb_core::dataset::{Dataset, BUILTINS};
use nnb_core::jobs::{Jobs, Log};
use nnb_core::model::{self, Graph};
use nnb_core::remote::{self, Machine};
use nnb_core::runs::{self, Creds, KAGGLE};
use nnb_core::shapes;
use nnb_core::store::{self, Project};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::menu::{Menu, MenuBuilder, MenuItem, Submenu, SubmenuBuilder};
use tauri_plugin_dialog::DialogExt;
use tauri::{Emitter, Manager, State};

const SERVICE: &str = "NN Builder";

struct App {
    project: Mutex<Option<Project>>,
    jobs: Jobs,
    /// an update that is installed and waits for the restart
    update_after: std::sync::Arc<Mutex<Option<update::After>>>,
}

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

fn project(st: &State<'_, App>) -> R<Project> {
    st.project.lock().unwrap().clone().ok_or_else(|| "Open or create a project first.".to_string())
}

/// run blocking work off the window's thread
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> R<T> + Send + 'static) -> R<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(e)?
}

// ---------------------------------------------------------------- keychain
fn secret(name: &str) -> Option<String> {
    keyring::Entry::new(SERVICE, name).ok()?.get_password().ok()
}

fn set_secret(name: &str, value: &str) -> R<()> {
    keyring::Entry::new(SERVICE, name).map_err(e)?.set_password(value).map_err(e)
}

fn creds() -> Creds {
    Creds {
        kaggle: secret("kaggle").and_then(|s| serde_json::from_str(&s).ok()),
        aws: secret("aws").and_then(|s| serde_json::from_str(&s).ok()),
        hf_token: secret("huggingface"),
    }
}

// ---------------------------------------------------------------- state and projects
#[tauri::command]
async fn app_state(st: State<'_, App>) -> R<Value> {
    let pr = st.project.lock().unwrap().clone();
    let c = creds();
    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "home": store::home(),
        "projects": store::listing(),
        "projects_dir": store::default_projects_dir(),
        "project": pr.map(|p| json!({"name": p.name(), "path": p.dir})),
        "tools_ready": cloud::tools_ready(),
        "secrets": {"kaggle": c.kaggle.as_ref().map(|k| k.username.clone()), "huggingface": c.hf_token.is_some(), "aws": c.aws.as_ref().map(|a| format!("{}…", a.access_key_id.chars().take(4).collect::<String>()))},
        "machines": remote::machines().into_iter().map(|m| json!({"machine": m, "prepared": remote::prepared(&m.name)})).collect::<Vec<_>>(),
        "builtins": BUILTINS.iter().map(|b| json!({"key": b.key, "label": b.label, "shape": b.shape, "classes": b.classes})).collect::<Vec<_>>(),
        "kinds": model::KINDS.iter().map(|(k, t)| json!({"kind": k, "title": t, "defaults": model::defaults(k)})).collect::<Vec<_>>(),
        "activations": model::ACTIVATIONS,
    }))
}

#[tauri::command]
async fn project_new(st: State<'_, App>, name: String) -> R<()> {
    let p = store::create(&name, None).map_err(e)?;
    *st.project.lock().unwrap() = Some(p);
    Ok(())
}

#[tauri::command]
async fn project_open(st: State<'_, App>, path: String) -> R<()> {
    let p = store::open(&PathBuf::from(path.trim())).map_err(e)?;
    *st.project.lock().unwrap() = Some(p);
    Ok(())
}

#[tauri::command]
async fn set_projects_dir(path: String) -> R<()> {
    store::save_setting("projects_dir", json!(path.trim())).map_err(e)
}

#[tauri::command]
async fn project_load(st: State<'_, App>) -> R<Value> {
    let p = project(&st)?;
    let (g, d, t) = (p.graph(), p.dataset(), p.train());
    let a = shapes::analyze(&g, &d);
    Ok(json!({"name": p.name(), "path": p.dir, "graph": g, "dataset": d, "train": t, "analysis": a, "dataset_problems": d.problems()}))
}

#[tauri::command]
async fn save_graph(st: State<'_, App>, graph: Graph) -> R<Value> {
    let p = project(&st)?;
    p.save_graph(&graph).map_err(e)?;
    let d = p.dataset();
    serde_json::to_value(shapes::analyze(&graph, &d)).map_err(e)
}

#[tauri::command]
async fn templates_list() -> R<Value> {
    Ok(json!(nnb_core::templates::TEMPLATES))
}

/// replace the design with a famous architecture (the current design is kept as model.previous.json) and fit the dataset to it
#[tauri::command]
async fn template_apply(st: State<'_, App>, key: String) -> R<String> {
    apply_template(&project(&st)?, &key)
}

/// a new project that starts with a famous architecture (named after it; " 2", " 3" ... when that name is taken)
#[tauri::command]
async fn project_from_template(st: State<'_, App>, key: String) -> R<String> {
    new_from_template(&st, &key)
}

fn new_from_template(st: &App, key: &str) -> R<String> {
    let t = nnb_core::templates::get(key).ok_or("Unknown architecture.")?;
    let base = store::default_projects_dir();
    let mut name = format!("{} demo", t.title.split(" (").next().unwrap_or(t.title));
    let mut n = 1;
    while base.join(name.replace('/', "_")).exists() || base.join(&name).exists() {
        n += 1;
        name = format!("{} demo {n}", t.title.split(" (").next().unwrap_or(t.title));
    }
    let p = store::create(&name, None).map_err(e)?;
    let msg = apply_template(&p, key)?;
    let _ = std::fs::remove_file(p.dir.join("model.previous.json"));
    *st.project.lock().unwrap() = Some(p);
    Ok(format!("Project \"{name}\" created. {}", msg.split_once(". ").map(|x| x.1).unwrap_or("")))
}

fn apply_template(p: &Project, key: &str) -> R<String> {
    let t = nnb_core::templates::get(key).ok_or("Unknown architecture.")?;
    let g = nnb_core::templates::build(key).ok_or("Unknown architecture.")?;
    let _ = std::fs::copy(p.dir.join("model.json"), p.dir.join("model.previous.json"));
    p.save_graph(&g).map_err(e)?;
    let mut d = p.dataset();
    let mut notes = vec![format!("{} loaded (the previous design is saved as model.previous.json).", t.title)];
    if t.task == "segment" {
        if d.task != "segment" {
            d.task = "segment".into();
            notes.push("The dataset is now a segmentation dataset: choose one with masks in the Dataset tab.".into());
        }
        if d.source == "builtin" {
            d.source = "huggingface".into();
        }
        if d.classes.len() < 2 {
            d.classes = vec!["background".into(), "foreground".into()];
        }
    } else {
        d.task = "classify".into();
    }
    let want = t.input;
    if d.source == "builtin" {
        let native = d.builtin_info().map(|b| b.shape).unwrap_or([1, 28, 28]);
        d.resize = native != want;
        if d.resize {
            notes.push(format!("{} images will be resized to {}×{} with {} channel(s).", d.builtin, want[1], want[2], want[0]));
        }
    } else if d.input_shape() != want.to_vec() {
        notes.push(format!("Images will be resized to {}×{}.", want[1], want[2]));
    }
    d.image_size = [want[1], want[2]];
    d.channels = want[0];
    p.save_dataset(&d).map_err(e)?;
    Ok(notes.join(" "))
}

#[tauri::command]
async fn save_dataset(st: State<'_, App>, dataset: Dataset) -> R<Value> {
    let p = project(&st)?;
    p.save_dataset(&dataset).map_err(e)?;
    Ok(json!({"analysis": shapes::analyze(&p.graph(), &dataset), "problems": dataset.problems()}))
}

#[tauri::command]
async fn save_train(st: State<'_, App>, train: TrainSettings) -> R<TrainSettings> {
    let p = project(&st)?;
    let t = train.clean();
    p.save_train(&t).map_err(e)?;
    Ok(t)
}

#[tauri::command]
async fn code(st: State<'_, App>) -> R<Value> {
    let p = project(&st)?;
    let (g, d, t) = (p.graph(), p.dataset(), p.train());
    let a = shapes::analyze(&g, &d);
    Ok(match nnb_core::codegen::train_script(&g, &d, &a, &t, &format!("nnb {}", p.name())) {
        Ok(c) => json!({"ok": true, "code": c}),
        Err(err) => json!({"ok": false, "error": err.to_string()}),
    })
}

/// write train.py into <project>/export and show it in Finder
#[tauri::command]
async fn export_code(st: State<'_, App>) -> R<String> {
    let p = project(&st)?;
    let (g, d, t) = (p.graph(), p.dataset(), p.train());
    let a = shapes::analyze(&g, &d);
    let c = nnb_core::codegen::train_script(&g, &d, &a, &t, &format!("nnb {}", p.name())).map_err(e)?;
    let f = p.dir.join("export").join("train.py");
    std::fs::create_dir_all(f.parent().unwrap()).map_err(e)?;
    std::fs::write(&f, c).map_err(e)?;
    reveal(&f);
    Ok(f.display().to_string())
}

fn reveal(p: &std::path::Path) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg("-R").arg(p).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(format!("/select,{}", p.display())).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(p.parent().unwrap_or(p)).spawn();
}

// ---------------------------------------------------------------- Connection: secrets, tools, machines
#[tauri::command]
async fn set_kaggle(username: String, key: String) -> R<()> {
    // accept the contents of kaggle.json pasted into the key field
    let (u, k) = match serde_json::from_str::<Value>(&key) {
        Ok(v) if v["key"].is_string() => (v["username"].as_str().unwrap_or(&username).to_string(), v["key"].as_str().unwrap().to_string()),
        _ => (username.trim().to_string(), key.trim().to_string()),
    };
    if u.is_empty() || k.is_empty() {
        return Err("Give the Kaggle username and API key (or paste the contents of kaggle.json).".into());
    }
    set_secret("kaggle", &serde_json::to_string(&KaggleCreds { username: u, key: k }).map_err(e)?)
}

#[tauri::command]
async fn set_hf(token: String) -> R<()> {
    let t = token.trim();
    if !t.starts_with("hf_") {
        return Err("A Hugging Face token starts with hf_ (huggingface.co, Settings, Access Tokens; 'Read' is enough).".into());
    }
    set_secret("huggingface", t)
}

#[tauri::command]
async fn set_aws(access_key_id: String, secret_access_key: String) -> R<()> {
    if !access_key_id.trim().starts_with("AK") || secret_access_key.trim().len() < 20 {
        return Err("Give the access key id (starts with AK) and its secret.".into());
    }
    set_secret("aws", &serde_json::to_string(&AwsCreds { access_key_id: access_key_id.trim().into(), secret_access_key: secret_access_key.trim().into() }).map_err(e)?)
}

#[tauri::command]
async fn clear_secret(name: String) -> R<()> {
    if !["kaggle", "huggingface", "aws"].contains(&name.as_str()) {
        return Err("unknown".into());
    }
    if let Ok(en) = keyring::Entry::new(SERVICE, &name) {
        let _ = en.delete_credential();
    }
    Ok(())
}

#[tauri::command]
async fn test_kaggle() -> R<String> {
    blocking(|| {
        let c = creds().kaggle.ok_or("Add your Kaggle token first.")?;
        cloud::kaggle_test(&c).map_err(e)
    }).await
}

#[tauri::command]
async fn test_hf() -> R<String> {
    blocking(|| {
        let t = creds().hf_token.ok_or("Add your Hugging Face token first.")?;
        let mut r = ureq::get("https://huggingface.co/api/whoami-v2").header("Authorization", &format!("Bearer {t}")).call().map_err(|x| format!("Hugging Face refused the token: {x}"))?;
        let v: Value = r.body_mut().read_json().map_err(e)?;
        Ok(format!("Hugging Face works for {}.", v["name"].as_str().unwrap_or("?")))
    }).await
}

#[tauri::command]
async fn test_aws(name: String) -> R<String> {
    blocking(move || {
        let c = creds().aws.ok_or("Add your AWS access key first.")?;
        let m = remote::machine(&name).ok_or("Unknown machine.")?;
        let v = cloud::aws(&c, "describe", &m.region, &m.instance_id).map_err(e)?;
        let sb = v["shutdown_behavior"].as_str().unwrap_or("?");
        Ok(format!("{} is {} ({}). Shutdown behaviour: {sb}{}", m.instance_id, v["state"].as_str().unwrap_or("?"), v["type"].as_str().unwrap_or("?"),
            if sb == "terminate" { " — change it to Stop in the EC2 console, or a finished run would DELETE the instance." } else { "." }))
    }).await
}

#[tauri::command]
async fn tools_install(st: State<'_, App>) -> R<()> {
    if st.jobs.start("Install the tools (kaggle, boto3)", |log| cloud::install_tools(&log)) { Ok(()) } else { Err("Another task is running.".into()) }
}

#[tauri::command]
async fn machine_save(machine: Machine) -> R<Machine> {
    remote::save_machine(machine).map_err(e)
}

#[tauri::command]
async fn machine_delete(name: String) -> R<()> {
    remote::delete_machine(&name).map_err(e)
}

#[tauri::command]
async fn machine_test(name: String) -> R<String> {
    blocking(move || {
        let m = remote::machine(&name).ok_or("Unknown machine.")?;
        if m.kind == "aws" {
            return Err("For an AWS machine use 'Check instance' (it is started only when a run needs it).".into());
        }
        m.probe()
    }).await
}

/// password machines: open a Terminal window that logs in once (the password is typed there, never into the app)
#[tauri::command]
async fn machine_connect(name: String) -> R<String> {
    let m = remote::machine(&name).ok_or("Unknown machine.")?;
    let cmd = m.connect_command().iter().map(|a| remote::sh_quote(a)).collect::<Vec<_>>().join(" ");
    let text = format!("echo 'NN Builder: log in to {} once. You can close this window afterwards.'; {cmd} && echo 'Connected. You can close this window.'", m.name);
    #[cfg(target_os = "macos")]
    {
        let esc = text.replace('\\', "\\\\").replace('"', "\\\"");
        std::process::Command::new("osascript").args(["-e", &format!("tell application \"Terminal\" to do script \"{esc}\""), "-e", "tell application \"Terminal\" to activate"])
            .spawn().map_err(e)?;
        return Ok("A Terminal window opened: log in there once.".into());
    }
    #[allow(unreachable_code)]
    Ok(format!("Open a terminal and run: {cmd}"))
}

#[tauri::command]
async fn machine_prepare(st: State<'_, App>, name: String) -> R<()> {
    let m = remote::machine(&name).ok_or("Unknown machine.")?;
    let ok = st.jobs.start(&format!("Prepare {name}"), move |log: Log| {
        let mut m = m;
        let c = creds();
        let started = m.kind == "aws";
        if started {
            runs::aws_up(&mut m, &c, &log).map_err(e)?;
        }
        let res = runs::prepare(&m, &log);
        if started {
            log("Stopping the instance again ...".into());
            if let Some(a) = &c.aws {
                let _ = cloud::aws(a, "stop", &m.region, &m.instance_id);
            }
        }
        let p = res.map_err(e)?;
        let method = remote::choose_launch(&m, &p);
        Ok(format!("Ready: PyTorch {} ({}); {}. Runs will use: {}",
            p.torch, if p.cuda { "CUDA" } else { "CPU only" },
            if p.gpus.is_empty() { "no NVIDIA GPU".to_string() } else { p.gpus.join(" | ") }, remote::launch_note(&method, &p)))
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

// ---------------------------------------------------------------- datasets (metadata only; nothing is downloaded)
#[tauri::command]
async fn hf_info(repo: String) -> R<Value> {
    blocking(move || {
        let repo = repo.trim().to_string();
        if !repo.contains('/') || !repo.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c)) {
            return Err("Give the dataset id, like ylecun/mnist.".into());
        }
        let mut req = ureq::get(&format!("https://datasets-server.huggingface.co/info?dataset={repo}"));
        if let Some(t) = creds().hf_token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        let mut r = req.call().map_err(|x| format!("Hugging Face: {x} (a private or gated dataset needs your token, Connection tab)"))?;
        let v: Value = r.body_mut().read_json().map_err(e)?;
        let mut configs = vec![];
        for (name, info) in v["dataset_info"].as_object().cloned().unwrap_or_default() {
            let cols: Vec<Value> = info["features"].as_object().cloned().unwrap_or_default().into_iter().map(|(c, f)| {
                json!({"name": c, "type": f["_type"].as_str().unwrap_or(if f.is_object() { "?" } else { "list" }), "names": f["names"].clone()})
            }).collect();
            let splits: Vec<Value> = info["splits"].as_object().cloned().unwrap_or_default().into_iter().map(|(s, i)| json!({"name": s, "n": i["num_examples"]})).collect();
            configs.push(json!({"name": name, "columns": cols, "splits": splits}));
        }
        if configs.is_empty() {
            return Err("No information for this dataset (is the id right? Some datasets have no viewer data).".into());
        }
        Ok(json!({"configs": configs}))
    }).await
}

#[tauri::command]
async fn kaggle_files(slug: String) -> R<Value> {
    blocking(move || {
        let c = creds().kaggle.ok_or("Add your Kaggle token first (Connection tab).")?;
        let files = cloud::kaggle_files(&c, slug.trim()).map_err(e)?;
        let folders = cloud::class_folders(&files);
        Ok(json!({"files": files.len(), "sample": files.iter().take(30).map(|f| &f.0).collect::<Vec<_>>(),
            "folders": folders.iter().map(|(p, c)| json!({"folder": p, "classes": c})).collect::<Vec<_>>()}))
    }).await
}

// ---------------------------------------------------------------- runs
#[tauri::command]
async fn targets() -> R<Value> {
    let mut t = vec![json!({"name": KAGGLE, "kind": "kaggle"})];
    for m in remote::machines() {
        t.push(json!({"name": m.name, "kind": m.kind, "prepared": remote::prepared(&m.name).is_some()}));
    }
    Ok(json!(t))
}

#[tauri::command]
async fn run_start(st: State<'_, App>, target: String, gpu: String, copy_tokens: bool) -> R<()> {
    let p = project(&st)?;
    let ok = st.jobs.start(&format!("Train on {target}"), move |log: Log| {
        let id = runs::start(&p, &target, gpu.trim(), &creds(), copy_tokens, &log).map_err(e)?;
        Ok(format!("RUN {id}"))
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

#[tauri::command]
async fn runs_list(st: State<'_, App>) -> R<Value> {
    Ok(json!(runs::listing(&project(&st)?)))
}

#[tauri::command]
async fn run_status(st: State<'_, App>, id: String, since: usize) -> R<runs::Status> {
    let p = project(&st)?;
    blocking(move || runs::status(&p, &id, since, &creds()).map_err(e)).await
}

#[tauri::command]
async fn run_stop(st: State<'_, App>, id: String) -> R<String> {
    let p = project(&st)?;
    blocking(move || runs::stop(&p, &id).map_err(e)).await
}

#[tauri::command]
async fn run_fetch(st: State<'_, App>, id: String) -> R<()> {
    let p = project(&st)?;
    let ok = st.jobs.start("Fetch results", move |log: Log| {
        runs::fetch(&p, &id, &creds(), &log).map_err(e)?;
        Ok(format!("RUN {id}"))
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

#[tauri::command]
async fn run_results(st: State<'_, App>, id: String) -> R<Value> {
    runs::results(&project(&st)?, &id).map_err(e)
}

#[tauri::command]
async fn run_reveal(st: State<'_, App>, id: String) -> R<()> {
    let d = runs::run_dir(&project(&st)?, &id).map_err(e)?;
    reveal(&d.join("train.py"));
    Ok(())
}

// ---------------------------------------------------------------- automatic updates
const DAY: u64 = 24 * 3600;

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[tauri::command]
async fn update_settings() -> R<Value> {
    let s = store::settings();
    Ok(json!({"auto": s["auto_update"].as_bool().unwrap_or(true), "last_check": s["last_update_check"], "version": env!("CARGO_PKG_VERSION")}))
}

#[tauri::command]
async fn set_auto_update(on: bool) -> R<()> {
    store::save_setting("auto_update", json!(on)).map_err(e)
}

/// manual: always ask GitHub. Automatic (at start): only when switched on and not asked in the last day; null when not asked.
#[tauri::command]
async fn update_check(manual: bool) -> R<Value> {
    let s = store::settings();
    if !manual && (!s["auto_update"].as_bool().unwrap_or(true) || now_secs().saturating_sub(s["last_update_check"].as_u64().unwrap_or(0)) < DAY) {
        return Ok(Value::Null);
    }
    store::save_setting("last_update_check", json!(now_secs())).map_err(e)?;
    let r = blocking(update::check).await?;
    Ok(match r {
        Some(rel) => json!({"current": env!("CARGO_PKG_VERSION"), "release": rel}),
        None => json!({"current": env!("CARGO_PKG_VERSION"), "release": null}),
    })
}

#[tauri::command]
async fn update_install(st: State<'_, App>) -> R<()> {
    let after = st.update_after.clone();
    let ok = st.jobs.start("Update NN Builder", move |log: Log| {
        let rel = update::check()?.ok_or("No release found.")?;
        if !rel.newer {
            return Err(format!("You already have the newest version ({}).", env!("CARGO_PKG_VERSION")));
        }
        let asset = rel.asset.ok_or(format!("Version {} has no installer for this system; download it from {}", rel.version, rel.page))?;
        log(format!("Downloading {} ({:.1} MB) ...", asset.name, asset.size as f64 / 1e6));
        let file = update::download(&asset, &update::work_dir(), &*log)?;
        log("Checksum verified. Installing ...".into());
        let a = update::install(&file)?;
        *after.lock().unwrap() = Some(a);
        Ok("UPDATE_READY".into())
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

/// start the new version: restart the app, or run the Windows installer and quit
#[tauri::command]
async fn update_restart(st: State<'_, App>, app: tauri::AppHandle) -> R<()> {
    let after = st.update_after.lock().unwrap().take();
    match after {
        Some(update::After::Installer(path)) => {
            std::process::Command::new(path).spawn().map_err(e)?;
            app.exit(0);
        }
        Some(update::After::Restart) => app.restart(),
        _ => {}
    }
    Ok(())
}

#[tauri::command]
async fn job_status(st: State<'_, App>, since: usize) -> R<nnb_core::jobs::Snapshot> {
    Ok(st.jobs.snapshot(since))
}

#[tauri::command]
async fn open_url(url: String) -> R<()> {
    if !url.starts_with("https://") {
        return Err("only https links".into());
    }
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(&url).spawn().map_err(e)?;
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer").arg(&url).spawn().map_err(e)?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(&url).spawn().map_err(e)?;
    Ok(())
}

// ---------------------------------------------------------------- menus
fn item(h: &tauri::AppHandle, id: &str, text: &str, accel: Option<&str>) -> tauri::Result<MenuItem<tauri::Wry>> {
    MenuItem::with_id(h, id, text, true, accel)
}

/// App menu, File, Edit (undo/redo of the design), View, Connection, Window
fn build_menu(h: &tauri::AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let app_menu = SubmenuBuilder::new(h, "NN Builder")
        .about(None)
        .separator()
        .item(&item(h, "check-update", "Check for Updates…", None)?)
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .quit()
        .build()?;
    let demos = Submenu::with_id(h, "demos", "New from Demo", true)?;
    for t in nnb_core::templates::TEMPLATES {
        demos.append(&item(h, &format!("demo:{}", t.key), &format!("{} ({})", t.title, t.year), None)?)?;
    }
    let file = SubmenuBuilder::new(h, "File")
        .item(&item(h, "new-project", "New Project…", Some("CmdOrCtrl+N"))?)
        .item(&item(h, "open-project", "Open Project…", Some("CmdOrCtrl+O"))?)
        .item(&demos)
        .separator()
        .item(&item(h, "import-design", "Import Design…", None)?)
        .item(&item(h, "export-design", "Export Design…", None)?)
        .item(&item(h, "export-code", "Export train.py…", Some("CmdOrCtrl+E"))?)
        .separator()
        .item(&item(h, "reveal-project", "Show Project in Finder", Some("CmdOrCtrl+Shift+R"))?)
        .separator()
        .close_window()
        .build()?;
    let edit = SubmenuBuilder::new(h, "Edit")
        .item(&item(h, "undo", "Undo", Some("CmdOrCtrl+Z"))?)
        .item(&item(h, "redo", "Redo", Some("CmdOrCtrl+Shift+Z"))?)
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .separator()
        .item(&item(h, "arrange", "Arrange Left to Right", Some("CmdOrCtrl+L"))?)
        .build()?;
    let view = SubmenuBuilder::new(h, "View")
        .item(&item(h, "night", "Night Mode (neon)", Some("CmdOrCtrl+Shift+N"))?)
        .separator()
        .fullscreen()
        .build()?;
    let conn = SubmenuBuilder::new(h, "Connection")
        .item(&item(h, "connection", "Open Connection", Some("CmdOrCtrl+Shift+K"))?)
        .item(&item(h, "tools", "Install Tools (kaggle, boto3)…", None)?)
        .item(&item(h, "check-update-2", "Check for Updates…", None)?)
        .build()?;
    let window = SubmenuBuilder::new(h, "Window").minimize().maximize().separator().close_window().build()?;
    MenuBuilder::new(h).items(&[&app_menu, &file, &edit, &view, &conn, &window]).build()
}

fn toast(h: &tauri::AppHandle, msg: impl Into<String>) {
    let _ = h.emit("toast", msg.into());
}

/// file and project actions run here (they need dialogs); the rest is handed to the window as a "menu" event
fn on_menu(h: &tauri::AppHandle, id: &str) {
    let st = h.state::<App>();
    match id {
        "check-update" | "check-update-2" => {
            let _ = h.emit("check-update", ());
        }
        "connection" | "tools" => {
            let _ = h.emit("open-tab", "connection");
        }
        "open-project" => {
            let h2 = h.clone();
            h.dialog().file().set_title("Open an NN Builder project (the folder with project.json)").pick_folder(move |f| {
                let Some(path) = f.and_then(|f| f.into_path().ok()) else { return };
                match store::open(&path) {
                    Ok(p) => {
                        *h2.state::<App>().project.lock().unwrap() = Some(p);
                        let _ = h2.emit("project-changed", ());
                    }
                    Err(err) => toast(&h2, err.to_string()),
                }
            });
        }
        "import-design" => {
            let Some(p) = st.project.lock().unwrap().clone() else { return toast(h, "Open or create a project first.") };
            let h2 = h.clone();
            h.dialog().file().set_title("Import a design (model.json)").add_filter("Design", &["json"]).pick_file(move |f| {
                let Some(path) = f.and_then(|f| f.into_path().ok()) else { return };
                let res = std::fs::read_to_string(&path).map_err(e).and_then(|t| serde_json::from_str::<Graph>(&t).map_err(|x| format!("Not an NN Builder design: {x}")))
                    .and_then(|g| {
                        if !g.nodes.iter().any(|n| n.kind == "input") {
                            return Err("Not an NN Builder design (it has no Input block).".to_string());
                        }
                        let _ = std::fs::copy(p.dir.join("model.json"), p.dir.join("model.previous.json"));
                        p.save_graph(&g).map_err(e)
                    });
                match res {
                    Ok(()) => {
                        let _ = h2.emit("project-changed", ());
                        toast(&h2, "Design imported (the previous one is kept as model.previous.json).");
                    }
                    Err(err) => toast(&h2, err),
                }
            });
        }
        "export-design" => {
            let Some(p) = st.project.lock().unwrap().clone() else { return toast(h, "Open or create a project first.") };
            let h2 = h.clone();
            h.dialog().file().set_title("Export the design").set_file_name(format!("{}_design.json", p.name().replace(['/', '\\'], "_"))).add_filter("Design", &["json"])
                .save_file(move |f| {
                    let Some(path) = f.and_then(|f| f.into_path().ok()) else { return };
                    match serde_json::to_string_pretty(&p.graph()).map_err(e).and_then(|t| std::fs::write(&path, t).map_err(e)) {
                        Ok(()) => toast(&h2, format!("Design saved to {}", path.display())),
                        Err(err) => toast(&h2, err),
                    }
                });
        }
        "reveal-project" => match st.project.lock().unwrap().clone() {
            Some(p) => reveal(&p.dir.join("project.json")),
            None => toast(h, "Open or create a project first."),
        },
        d if d.starts_with("demo:") => match new_from_template(&st, &d[5..]) {
            Ok(msg) => {
                let _ = h.emit("project-changed", ());
                toast(h, msg);
            }
            Err(err) => toast(h, err),
        },
        other => {
            let _ = h.emit("menu", other.to_string());
        }
    }
    if let Some(w) = h.get_webview_window("main") {
        let _ = w.set_focus();
    }
}

fn main() {
    let app = App { project: Mutex::new(store::last_project()), jobs: Jobs::default(), update_after: Default::default() };
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(app)
        .setup(|app| {
            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, ev| on_menu(app, ev.id().as_ref()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_state, templates_list, template_apply, project_from_template, project_new, project_open, set_projects_dir, project_load, save_graph, save_dataset, save_train, code, export_code,
            set_kaggle, set_hf, set_aws, clear_secret, test_kaggle, test_hf, test_aws, tools_install,
            machine_save, machine_delete, machine_test, machine_connect, machine_prepare, hf_info, kaggle_files,
            update_settings, set_auto_update, update_check, update_install, update_restart, targets, run_start, runs_list, run_status, run_stop, run_fetch, run_results, run_reveal, job_status, open_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running NN Builder");
}

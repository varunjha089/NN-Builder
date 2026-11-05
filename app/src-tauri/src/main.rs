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
/// each item is read from the Keychain once per launch and kept in memory: every Keychain read of an unsigned (or rebuilt) app can
/// make macOS ask for the login password again, and the window and run status ask for the tokens every few seconds
static SECRETS: std::sync::Mutex<Option<std::collections::HashMap<String, Option<String>>>> = std::sync::Mutex::new(None);

fn secret(name: &str) -> Option<String> {
    let mut cache = SECRETS.lock().unwrap_or_else(|p| p.into_inner());
    let map = cache.get_or_insert_with(Default::default);
    if let Some(v) = map.get(name) {
        return v.clone();
    }
    let v = keyring::Entry::new(SERVICE, name).ok().and_then(|en| en.get_password().ok());
    map.insert(name.to_string(), v.clone());
    v
}

fn remember_secret(name: &str, value: Option<String>) {
    SECRETS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Default::default).insert(name.to_string(), value);
}

fn set_secret(name: &str, value: &str) -> R<()> {
    keyring::Entry::new(SERVICE, name).map_err(e)?.set_password(value).map_err(e)?;
    remember_secret(name, Some(value.to_string()));
    save_secrets_meta(&creds());
    Ok(())
}

/// what the window shows about the keys (the Kaggle user name, whether a Hugging Face token exists, the first characters of the AWS key
/// id): nothing secret. Kept in settings.json, so that starting the app never reads the Keychain (a Keychain prompt, as after every
/// unsigned rebuild, would otherwise hold up the window); the secrets themselves are read only when a run or a test needs them.
fn meta_of(c: &Creds) -> Value {
    json!({"kaggle": c.kaggle.as_ref().map(|k| k.username.clone()), "huggingface": c.hf_token.is_some(),
        "aws": c.aws.as_ref().map(|a| format!("{}…", a.access_key_id.chars().take(4).collect::<String>()))})
}

fn save_secrets_meta(c: &Creds) {
    let _ = store::save_setting("secrets_meta", meta_of(c));
}

fn secrets_meta() -> Value {
    let s = store::settings();
    if s["secrets_meta"].is_object() {
        return s["secrets_meta"].clone();
    }
    // first start with this version: read the Keychain once to learn what is there
    let c = creds();
    save_secrets_meta(&c);
    meta_of(&c)
}

/// the Hugging Face token for the public dataset calls: without one stored (secrets_meta) the Keychain is not asked at all
fn hf_token() -> Option<String> {
    if secrets_meta()["huggingface"] == json!(false) {
        return None;
    }
    secret("huggingface")
}

fn creds() -> Creds {
    Creds {
        kaggle: secret("kaggle").and_then(|s| serde_json::from_str(&s).ok()),
        aws: secret("aws").and_then(|s| serde_json::from_str(&s).ok()),
        hf_token: secret("huggingface"),
    }
}

// ---------------------------------------------------------------- state and projects
/// the computer's memory in bytes (asked once): macOS sysctl, Linux /proc/meminfo, Windows CIM
fn ram_bytes() -> u64 {
    static RAM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RAM.get_or_init(|| {
        #[cfg(target_os = "macos")]
        let v = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok());
        #[cfg(target_os = "linux")]
        let v = std::fs::read_to_string("/proc/meminfo").ok().and_then(|t| t.lines().find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).map(|kb| kb * 1024));
        #[cfg(target_os = "windows")]
        let v = std::process::Command::new("powershell").args(["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            .output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok());
        v.unwrap_or(8 << 30)
    })
}

/// the design-size limit: the owner's value from Settings, else the recommendation for this computer's memory
fn design_limit() -> (usize, usize, bool) {
    let rec = nnb_core::dsl::recommended_max_blocks(ram_bytes());
    match store::settings()["max_blocks"].as_u64() {
        Some(n) => (n as usize, rec, true),
        None => (rec, rec, false),
    }
}

/// Settings → Design size: a number, or null for the recommendation
#[tauri::command]
async fn set_max_blocks(n: Option<u64>) -> R<Value> {
    store::save_setting("max_blocks", n.map(|n| json!(n.clamp(100, 10_000_000))).unwrap_or(Value::Null)).map_err(e)?;
    let (limit, rec, own) = design_limit();
    nnb_core::dsl::set_max_blocks(limit);
    let ram_gb = ram_bytes() as f64 / 1073741824.0;
    Ok(json!({"max_blocks": limit, "recommended": rec, "own": own, "ram_gb": ram_gb}))
}

#[tauri::command]
async fn app_state(st: State<'_, App>) -> R<Value> {
    let pr = st.project.lock().unwrap().clone();
    let (limit, rec, own) = design_limit();
    let design_size = json!({"max_blocks": limit, "recommended": rec, "own": own, "ram_gb": ram_bytes() as f64 / 1073741824.0});
    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "home": store::home(),
        "projects": store::listing(),
        "projects_dir": store::default_projects_dir(),
        "design_size": design_size,
        "project": pr.map(|p| json!({"name": p.name(), "path": p.dir, "draft": p.is_draft()})),
        "tools_ready": cloud::tools_ready(),
        "secrets": secrets_meta(),
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

/// delete a project from the list: its folder (with its runs and results) goes to the Trash, from where it can be put back; a project
/// whose folder is missing is only taken off the list. Only folders that are NN Builder projects (project.json) are moved.
#[tauri::command]
async fn project_delete(app: tauri::AppHandle, st: State<'_, App>, path: String) -> R<String> {
    let dir = PathBuf::from(path.trim());
    if !store::listing().iter().any(|p| p["path"].as_str() == Some(path.trim())) {
        return Err("This folder is not in the projects list.".into());
    }
    let open = st.project.lock().unwrap().as_ref().is_some_and(|p| p.dir == dir);
    let msg = if dir.join("project.json").is_file() {
        let name = store::open(&dir).map(|p| p.name()).unwrap_or_else(|_| dir.display().to_string());
        #[allow(unused_mut)]
        let mut ctx = trash::TrashContext::default();
        #[cfg(target_os = "macos")]
        {
            use trash::macos::{DeleteMethod, TrashContextExtMacos};
            ctx.set_delete_method(DeleteMethod::NsFileManager);
        }
        ctx.delete(&dir).map_err(|x| format!("Could not move {} to the Trash: {x}", dir.display()))?;
        format!("Moved {name} to the Trash (put it back from there if needed).")
    } else {
        format!("Removed {} from the list (its folder was missing).", dir.display())
    };
    store::forget(&dir).map_err(e)?;
    if open {
        *st.project.lock().unwrap() = None;
        let _ = app.emit("project-changed", ());
    }
    Ok(msg)
}

#[tauri::command]
async fn set_projects_dir(path: String) -> R<()> {
    store::save_setting("projects_dir", json!(path.trim())).map_err(e)
}

#[tauri::command]
async fn project_load(st: State<'_, App>) -> R<Value> {
    let p = project(&st)?;
    let (g, d, t) = (nnb_core::dsl::sync(&p.graph()), p.dataset(), p.train());
    let a = shapes::analyze(&g, &d);
    Ok(json!({"name": p.name(), "path": p.dir, "graph": g, "dataset": d, "train": t, "analysis": a, "dataset_problems": d.problems()}))
}

#[tauri::command]
async fn save_graph(st: State<'_, App>, graph: Graph) -> R<Value> {
    let p = project(&st)?;
    // the design text follows the canvas (kept when it still fits, else printed again)
    let graph = nnb_core::dsl::sync(&graph);
    p.save_graph(&graph).map_err(e)?;
    let d = p.dataset();
    Ok(json!({"analysis": shapes::analyze(&graph, &d), "graph": graph}))
}

/// the design from its text (Text tab): saved and returned with its analysis, or the problems in the text (nothing saved)
#[tauri::command]
async fn design_from_text(st: State<'_, App>, text: String) -> R<Value> {
    let p = project(&st)?;
    match nnb_core::dsl::apply(&text, &p.graph()) {
        Ok(g) => {
            p.save_graph(&g).map_err(e)?;
            let d = p.dataset();
            Ok(json!({"ok": true, "graph": g, "analysis": shapes::analyze(&g, &d)}))
        }
        Err(errors) => Ok(json!({"ok": false, "errors": errors})),
    }
}

/// the text printed again from the blocks (Format: functions and loops written by hand become plain lines and repeats)
#[tauri::command]
async fn design_format(st: State<'_, App>) -> R<Value> {
    let p = project(&st)?;
    let g = nnb_core::dsl::reprint(&p.graph());
    p.save_graph(&g).map_err(e)?;
    let d = p.dataset();
    Ok(json!({"graph": g, "analysis": shapes::analyze(&g, &d)}))
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

/// open a famous architecture as the unsaved draft (nothing goes into the projects folder until Save Project…; an unsaved draft is replaced)
#[tauri::command]
async fn project_from_template(st: State<'_, App>, key: String) -> R<String> {
    new_from_template(&st, &key)
}

fn new_from_template(st: &App, key: &str) -> R<String> {
    let t = nnb_core::templates::get(key).ok_or("Unknown architecture.")?;
    let name = format!("{} demo", t.title.split(" (").next().unwrap_or(t.title));
    let mut msg = String::new();
    let p = store::create_draft(&name, |p| {
        msg = apply_template(p, key).map_err(|x| anyhow::anyhow!(x))?;
        let _ = std::fs::remove_file(p.dir.join("model.previous.json"));
        Ok(())
    }).map_err(e)?;
    *st.project.lock().unwrap() = Some(p);
    Ok(format!("{} opened, not saved: it is discarded unless you save it (⌘S). {}", t.title, msg.split_once(". ").map(|x| x.1).unwrap_or("")))
}

/// the unsaved demo becomes a project in the projects folder
#[tauri::command]
async fn project_save_as(st: State<'_, App>, name: String) -> R<String> {
    let p = project(&st)?;
    let saved = store::save_as(&p, &name, None).map_err(e)?;
    let msg = format!("Saved as {}", saved.dir.display());
    *st.project.lock().unwrap() = Some(saved);
    Ok(msg)
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
    // a built-in set that fits the architecture as it is (CIFAR-10 for 32×32 colour, MNIST for 28×28 grey) instead of resizing another one
    if d.source == "builtin" && t.task != "segment" {
        if let Some(fit) = BUILTINS.iter().find(|b| b.shape == want && (b.key == "cifar10" || b.key == "mnist")) {
            if d.builtin != fit.key && BUILTINS.iter().find(|b| b.key == d.builtin).map(|b| b.shape) != Some(want) {
                notes.push(format!("Dataset: {} (made for this input size).", fit.label));
                d.builtin = fit.key.into();
            }
        }
    }
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

/// open a file with the system's default app (the PDF viewer for the manual)
fn open_file(p: &std::path::Path) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(p).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(p).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(p).spawn();
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
    remember_secret(&name, None);
    save_secrets_meta(&creds());
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
        let mut msg = format!("{} is {} ({}). Shutdown behaviour: {sb}{}", m.instance_id, v["state"].as_str().unwrap_or("?"), v["type"].as_str().unwrap_or("?"),
            if sb == "terminate" { " — change it to Stop in the EC2 console, or a finished run would DELETE the instance." } else { "." });
        if v["lifecycle"] == "spot" {
            msg.push_str(&format!("\nSpot instance: {}", runs::spot_summary(&v)));
            if !m.mirror {
                msg.push_str("\nTip: turn on 'Keep a copy of checkpoints on this Mac' for this machine, so a run can continue elsewhere if AWS terminates it.");
            }
        }
        Ok(msg)
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
    if cfg!(windows) {
        return Err("Logging in once with a password needs SSH connection sharing, which Windows' OpenSSH does not have. Set up a key for \
                    this machine instead (ssh-keygen, then copy the public key to the server) and choose Login: Key.".into());
    }
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
        if let Some(t) = hf_token() {
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

/// a dataset's class balance (label counts from the dataset viewer's statistics, computed on Hugging Face's side) and, when asked, 8
/// example images (only those few images are fetched, as embedded data for the window; the dataset itself is never downloaded here)
#[tauri::command]
async fn hf_preview(repo: String, config: String, split: String, label_col: String, samples: bool) -> R<Value> {
    blocking(move || {
        let safe = |s: &str| s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c));
        if !repo.contains('/') || !safe(&repo) || !safe(&config) || !safe(&split) || !safe(&label_col) {
            return Err("Read the dataset's information first.".into());
        }
        let token = hf_token();
        let get = |url: &str| {
            let mut req = ureq::get(url);
            if let Some(t) = &token {
                req = req.header("Authorization", &format!("Bearer {t}"));
            }
            req.call()
        };
        let q = format!("dataset={repo}&config={config}&split={split}");
        let mut freq: Vec<(String, u64)> = vec![];
        if let Ok(mut r) = get(&format!("https://datasets-server.huggingface.co/statistics?{q}")) {
            let v: Value = r.body_mut().read_json().unwrap_or(Value::Null);
            if let Some(col) = v["statistics"].as_array().and_then(|a| a.iter().find(|c| c["column_name"] == label_col.as_str())) {
                freq = col["column_statistics"]["frequencies"].as_object().map(|m| m.iter().map(|(k, n)| (k.clone(), n.as_u64().unwrap_or(0))).collect())
                    .unwrap_or_default();
            }
        }
        freq.sort_by_key(|f| std::cmp::Reverse(f.1));
        let mut pictures = vec![];
        if samples {
            let mut r = get(&format!("https://datasets-server.huggingface.co/first-rows?{q}")).map_err(|x| format!("Hugging Face: {x}"))?;
            let v: Value = r.body_mut().read_json().map_err(e)?;
            for row in v["rows"].as_array().cloned().unwrap_or_default().into_iter().take(8) {
                let row = &row["row"];
                let Some(src) = row.as_object().and_then(|o| o.values().find_map(|x| x["src"].as_str().map(String::from))) else { continue };
                if !src.starts_with("https://datasets-server.huggingface.co/") {
                    continue;
                }
                let Ok(mut img) = ureq::get(&src).call() else { continue };
                let mime = img.headers().get("content-type").and_then(|h| h.to_str().ok()).unwrap_or("image/jpeg").to_string();
                let Ok(bytes) = img.body_mut().with_config().limit(512 * 1024).read_to_vec() else { continue };
                pictures.push(json!({"src": format!("data:{mime};base64,{}", base64(&bytes)), "label": row[&label_col].clone()}));
            }
        }
        Ok(json!({"frequencies": freq.into_iter().map(|(k, n)| json!([k, n])).collect::<Vec<_>>(), "samples": pictures}))
    }).await
}

/// standard base64 (for embedding a few example images)
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() { T[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
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

/// the owner's own dataset folder (e.g. GeoTIFFs from Google Earth Engine): files, class folders, the first GeoTIFF's size and band names
#[tauri::command]
async fn local_scan(dir: String) -> R<Value> {
    blocking(move || nnb_core::localdata::describe(&runs::local_path(dir.trim())).map_err(e)).await
}

#[tauri::command]
async fn pick_folder(app: tauri::AppHandle, title: String) -> R<Option<String>> {
    Ok(app.dialog().file().set_title(title).blocking_pick_folder().map(|f| f.to_string()))
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
    if p.is_draft() {
        return Err("Save the demo first (File → Save Project… ⌘S): runs are kept with a saved project.".into());
    }
    runs::preflight(&target, &creds()).map_err(e)?;
    let ok = st.jobs.start(&format!("Train on {target}"), move |log: Log| {
        let id = runs::start(&p, &target, gpu.trim(), &creds(), copy_tokens, &log).map_err(e)?;
        Ok(format!("RUN {id}"))
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

/// the same design and settings with seeds s, s+1, ... (one group of runs)
#[tauri::command]
async fn run_start_seeds(st: State<'_, App>, target: String, gpu: String, copy_tokens: bool, count: usize) -> R<()> {
    let p = project(&st)?;
    if p.is_draft() {
        return Err("Save the demo first (File → Save Project… ⌘S): runs are kept with a saved project.".into());
    }
    runs::preflight(&target, &creds()).map_err(e)?;
    let ok = st.jobs.start(&format!("Train on {target}, {count} seeds"), move |log: Log| {
        let ids = runs::start_seeds(&p, &target, gpu.trim(), &creds(), copy_tokens, count, &log).map_err(e)?;
        Ok(format!("RUN {}", ids[0]))
    });
    if ok { Ok(()) } else { Err("Another task is running.".into()) }
}

#[tauri::command]
async fn runs_summary(st: State<'_, App>, ids: Vec<String>) -> R<Value> {
    let p = project(&st)?;
    blocking(move || runs::summary(&p, &ids).map_err(e)).await
}

/// CSV + LaTeX table of the chosen runs in <project>/exports/ (shown in Finder)
#[tauri::command]
async fn runs_export(st: State<'_, App>, ids: Vec<String>) -> R<String> {
    let p = project(&st)?;
    let (csv, tex) = blocking(move || runs::export(&p, &ids).map_err(e)).await?;
    reveal(&tex);
    Ok(format!("Saved {} and {}", csv.display(), tex.display()))
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

/// continue a run on another machine from the checkpoint copy on this Mac
#[tauri::command]
async fn run_resume_on(st: State<'_, App>, id: String, target: String, gpu: String, copy_tokens: bool) -> R<()> {
    let p = project(&st)?;
    let ok = st.jobs.start(&format!("Continue on {target}"), move |log: Log| {
        let new = runs::resume_on(&p, &id, &target, gpu.trim(), &creds(), copy_tokens, &log).map_err(e)?;
        Ok(format!("RUN {new}"))
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
        .item(&item(h, "settings", "Settings…", Some("CmdOrCtrl+,"))?)
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
        .item(&item(h, "save-project", "Save Project…", Some("CmdOrCtrl+S"))?)
        .item(&item(h, "close-project", "Close Project", None)?)
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
        .build()?;
    let window = SubmenuBuilder::new(h, "Window").minimize().maximize().separator().close_window().build()?;
    let help = SubmenuBuilder::new(h, "Help")
        .item(&item(h, "documentation", "Documentation (User Manual)", Some("CmdOrCtrl+?"))?)
        .separator()
        .item(&item(h, "check-update", "Check for Updates…", None)?)
        .build()?;
    MenuBuilder::new(h).items(&[&app_menu, &file, &edit, &view, &conn, &window, &help]).build()
}

fn toast(h: &tauri::AppHandle, msg: impl Into<String>) {
    let _ = h.emit("toast", msg.into());
}

/// file and project actions run here (they need dialogs); the rest is handed to the window as a "menu" event
fn on_menu(h: &tauri::AppHandle, id: &str) {
    let st = h.state::<App>();
    match id {
        "documentation" => {
            // the user manual shipped inside the app (bundle resource); in a development build, the copy in the repository
            let bundled = h.path().resolve("nn_builder_manual.pdf", tauri::path::BaseDirectory::Resource).ok().filter(|p| p.is_file());
            let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../doc/nn_builder_manual.pdf");
            match bundled.or_else(|| dev.is_file().then_some(dev)) {
                Some(pdf) => open_file(&pdf),
                None => toast(h, "The user manual is not in this build: see doc/nn_builder_manual.pdf in the repository."),
            }
        }
        "check-update" => {
            let _ = h.emit("check-update", ());
        }
        "settings" => {
            let _ = h.emit("open-tab", "settings");
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
            h.dialog().file().set_title("Import a design (model.json, or design text .nnb)").add_filter("Design", &["json", "nnb"]).pick_file(move |f| {
                let Some(path) = f.and_then(|f| f.into_path().ok()) else { return };
                let is_text = path.extension().is_some_and(|x| x.eq_ignore_ascii_case("nnb"));
                let res = std::fs::read_to_string(&path).map_err(e).and_then(|t| if is_text {
                    // the design text: built like the Text tab does (blocks with a known label keep their place)
                    nnb_core::dsl::apply(&t, &p.graph()).map_err(|d| {
                        d.first().map(|d| format!("{}: line {}: {}", path.display(), d.line, d.message)).unwrap_or_default()
                    })
                } else {
                    serde_json::from_str::<Graph>(&t).map_err(|x| format!("Not an NN Builder design: {x}"))
                })
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
            h.dialog().file().set_title("Export the design (.json with positions, or .nnb design text)")
                .set_file_name(format!("{}_design.json", p.name().replace(['/', '\\'], "_"))).add_filter("Design", &["json"]).add_filter("Design text", &["nnb"])
                .save_file(move |f| {
                    let Some(path) = f.and_then(|f| f.into_path().ok()) else { return };
                    let is_text = path.extension().is_some_and(|x| x.eq_ignore_ascii_case("nnb"));
                    let g = nnb_core::dsl::sync(&p.graph());
                    let out = if is_text { Ok(g.text) } else { serde_json::to_string_pretty(&g).map_err(e) };
                    match out.and_then(|t| std::fs::write(&path, t).map_err(e)) {
                        Ok(()) => toast(&h2, format!("Design saved to {}", path.display())),
                        Err(err) => toast(&h2, err),
                    }
                });
        }
        "close-project" => {
            // back to the Projects page with no project open; an unsaved demo is discarded (as when another project is opened)
            let Some(p) = st.project.lock().unwrap().take() else { return toast(h, "No project is open.") };
            if p.is_draft() {
                let _ = store::discard_draft();
            }
            let _ = store::save_setting("last_project", Value::Null);
            let _ = h.emit("project-changed", ());
            toast(h, if p.is_draft() { "Closed the demo (it was not saved).".to_string() } else { format!("Closed {}.", p.name()) });
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
    // an unsaved demo does not outlive the app (also after a crash)
    let _ = store::discard_draft();
    nnb_core::dsl::set_max_blocks(design_limit().0);
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
            app_state, templates_list, template_apply, project_from_template, project_save_as, project_new, project_open, set_projects_dir, project_delete, set_max_blocks, project_load, save_graph, design_from_text, design_format, save_dataset, save_train, code, export_code,
            set_kaggle, set_hf, set_aws, clear_secret, test_kaggle, test_hf, test_aws, tools_install,
            machine_save, machine_delete, machine_test, machine_connect, machine_prepare, hf_info, hf_preview, kaggle_files, local_scan, pick_folder,
            update_settings, set_auto_update, update_check, update_install, update_restart, targets, run_start, run_start_seeds, runs_summary, runs_export, run_resume_on, runs_list, run_status, run_stop, run_fetch, run_results, run_reveal, job_status, open_url
        ])
        .build(tauri::generate_context!())
        .expect("error while running NN Builder")
        .run(|_, ev| {
            if let tauri::RunEvent::Exit = ev {
                let _ = store::discard_draft();
            }
        });
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(super::base64(b""), "");
        assert_eq!(super::base64(b"M"), "TQ==");
        assert_eq!(super::base64(b"Ma"), "TWE=");
        assert_eq!(super::base64(b"Man"), "TWFu");
        assert_eq!(super::base64(&[0xff, 0xfe, 0x00, 0x10]), "//4AEA==");
    }
}

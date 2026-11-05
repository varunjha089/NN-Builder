//! The Mac-side cloud helpers: a small Python environment (settings folder/tools) with the official `kaggle` CLI and `boto3`. No PyTorch:
//! the Mac never trains. Credentials are passed to each call in the environment (from the Keychain), never in files or arguments.

use crate::jobs::{capture, stream, Log};
use crate::store;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const AWS_HELPER: &str = include_str!("../../../python/helpers/aws_ec2.py");

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AwsCreds {
    pub access_key_id: String,
    pub secret_access_key: String,
}

pub fn tools_dir() -> PathBuf {
    store::home().join("tools")
}

fn bin(name: &str) -> PathBuf {
    tools_dir().join(if cfg!(windows) { "Scripts" } else { "bin" }).join(if cfg!(windows) { format!("{name}.exe") } else { name.into() })
}

pub fn tools_ready() -> bool {
    bin("kaggle").exists() && bin("python").exists()
}

/// make the tools environment and install kaggle + boto3 (about 60 MB)
pub fn install_tools(log: &Log) -> Result<String, String> {
    if !bin("python").exists() {
        let base = ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"]
            .iter()
            .flat_map(|n| ["/opt/homebrew/bin/", "/usr/local/bin/", "/usr/bin/", ""].iter().map(move |d| format!("{d}{n}")))
            .find(|exe| capture(Command::new(exe).args(["-c", "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,14) else 1)"]), None).0 == 0)
            .ok_or("No Python 3.9-3.14 found. Install Python from python.org.")?;
        log(format!("Making the tools environment with {base} ..."));
        let (code, _) = stream(Command::new(&base).args(["-m", "venv"]).arg(tools_dir()), None, log);
        if code != 0 {
            return Err("Could not make the tools environment.".into());
        }
    }
    log("Installing kaggle and boto3 ...".into());
    let (code, _) = stream(Command::new(bin("python")).args(["-m", "pip", "install", "--disable-pip-version-check", "-q", "kaggle", "boto3"]), None, log);
    if code != 0 {
        return Err("The installation failed. Is there an internet connection?".into());
    }
    Ok("Tools ready: kaggle and boto3 (used only to talk to Kaggle and AWS; nothing trains on this Mac).".into())
}

// ---------------------------------------------------------------- Kaggle
fn kaggle(c: &KaggleCreds) -> Result<Command> {
    if !tools_ready() {
        bail!("Install the tools first (Connection tab).");
    }
    if c.username.is_empty() || c.key.is_empty() {
        bail!("Add your Kaggle API token first (Connection tab).");
    }
    let mut cmd = Command::new(bin("kaggle"));
    // a private config folder, so the CLI never reads or writes ~/.kaggle on this Mac
    let cfg = tools_dir().join("kaggle-config");
    let _ = std::fs::create_dir_all(&cfg);
    cmd.env("KAGGLE_USERNAME", &c.username).env("KAGGLE_KEY", &c.key).env("KAGGLE_CONFIG_DIR", cfg);
    Ok(cmd)
}

fn last_line(out: &str) -> String {
    out.trim().lines().last().unwrap_or("no answer").to_string()
}

pub fn kaggle_test(c: &KaggleCreds) -> Result<String> {
    let (code, out) = capture(kaggle(c)?.args(["kernels", "list", "--mine", "--page-size", "1", "--csv"]), None);
    if code != 0 {
        bail!("Kaggle refused: {}", last_line(&out));
    }
    Ok(format!("Kaggle works for {}.", c.username))
}

/// the dataset's files (path, size) through the Kaggle API: nothing is downloaded
pub fn kaggle_files(c: &KaggleCreds, slug: &str) -> Result<Vec<(String, String)>> {
    if !slug.chars().all(|ch| ch.is_ascii_alphanumeric() || "-_./".contains(ch)) || slug.matches('/').count() != 1 {
        bail!("Kaggle dataset: owner/name");
    }
    let (mut code, mut out) = capture(kaggle(c)?.args(["datasets", "files", slug, "--csv", "--page-size", "1000"]), None);
    if code != 0 {
        // older / newer CLIs differ in the paging option
        (code, out) = capture(kaggle(c)?.args(["datasets", "files", slug, "--csv"]), None);
    }
    if code != 0 {
        bail!("Kaggle: {}", last_line(&out));
    }
    let mut files = vec![];
    for (i, line) in out.lines().enumerate() {
        if i == 0 || line.trim().is_empty() || line.starts_with("Next Page") {
            continue;
        }
        let mut parts = line.splitn(3, ',');
        if let (Some(name), Some(size)) = (parts.next(), parts.next()) {
            files.push((name.trim_matches('"').to_string(), size.to_string()));
        }
    }
    Ok(files)
}

/// folders whose subfolders hold images: candidates for "one subfolder per class" (folder, class names)
pub fn class_folders(files: &[(String, String)]) -> Vec<(String, Vec<String>)> {
    use std::collections::{BTreeMap, BTreeSet};
    let img = |f: &str| {
        let l = f.to_ascii_lowercase();
        [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".npy"].iter().any(|e| l.ends_with(e))
    };
    let mut by_parent: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (f, _) in files.iter().filter(|(f, _)| img(f)) {
        let parts: Vec<&str> = f.split('/').collect();
        if parts.len() >= 2 {
            let class = parts[parts.len() - 2].to_string();
            let parent = parts[..parts.len() - 2].join("/");
            by_parent.entry(parent).or_default().insert(class);
        }
    }
    by_parent.into_iter().filter(|(_, c)| c.len() >= 2).map(|(p, c)| (p, c.into_iter().collect())).collect()
}

pub fn kernel_slug(title: &str) -> String {
    let mut s: String = title.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').chars().take(50).collect::<String>().trim_matches('-').to_string()
}

/// push dir/train.py as a private GPU notebook; returns the kernel reference "user/slug"
pub fn kaggle_push(c: &KaggleCreds, dir: &Path, title: &str, dataset_slug: Option<&str>) -> Result<String> {
    let slug = kernel_slug(title);
    let reference = format!("{}/{}", c.username, slug);
    let meta = serde_json::json!({
        "id": reference, "title": title, "code_file": "train.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": dataset_slug.map(|d| vec![d.to_string()]).unwrap_or_default(), "competition_sources": [], "kernel_sources": []
    });
    let push_dir = dir.join("kaggle_push");
    std::fs::create_dir_all(&push_dir)?;
    std::fs::copy(dir.join("train.py"), push_dir.join("train.py")).context("train.py missing")?;
    store::write_json(&push_dir.join("kernel-metadata.json"), &meta)?;
    let (code, out) = capture(kaggle(c)?.args(["kernels", "push", "-p"]).arg(&push_dir), None);
    if code != 0 || !out.to_lowercase().contains("successfully pushed") {
        bail!("Kaggle push failed: {}", last_line(&out));
    }
    Ok(reference)
}

/// the owner's own dataset folder as a private Kaggle dataset <user>/<key>, so a Kaggle notebook can attach it: created the first time,
/// a new version when the folder changed (signature kept in <settings>/uploads.json), nothing when it did not. Subfolders go up as zip
/// files, which Kaggle unpacks. The folder itself is not written to: the upload is made from a folder of links next to the tools.
pub fn kaggle_upload(c: &KaggleCreds, dir: &Path, key: &str, log: &Log) -> Result<String> {
    let slug = format!("{}/{}", c.username, key);
    let sig = crate::localdata::signature(dir)?;
    let uploads_path = store::home().join("uploads.json");
    let mut uploads: serde_json::Map<String, Value> = std::fs::read_to_string(&uploads_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    if uploads.get(&slug).and_then(|v| v.as_str()) == Some(sig.as_str()) {
        log(format!("Your files are already on Kaggle as {slug} (unchanged since the last upload)."));
        return Ok(slug);
    }
    let stage = tools_dir().join("kaggle-upload").join(key);
    if stage.exists() {
        std::fs::remove_dir_all(&stage)?; // only links and the metadata file: the owner's files are never inside
    }
    std::fs::create_dir_all(&stage)?;
    for e in std::fs::read_dir(dir)?.flatten() {
        let name = e.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        link(&e.path(), &stage.join(&name))?;
    }
    let meta = serde_json::json!({"title": key, "id": slug, "licenses": [{"name": "other"}]});
    store::write_json(&stage.join("dataset-metadata.json"), &meta)?;
    let exists = capture(kaggle(c)?.args(["datasets", "files", &slug, "--csv"]), None).0 == 0;
    log(format!("{} your files on Kaggle as the private dataset {slug} ({sig}) ...", if exists { "Updating" } else { "Uploading" }));
    let mut cmd = kaggle(c)?;
    if exists {
        cmd.args(["datasets", "version", "-p"]).arg(&stage).args(["-m", "NN Builder: the folder changed", "--dir-mode", "zip"]);
    } else {
        cmd.args(["datasets", "create", "-p"]).arg(&stage).args(["--dir-mode", "zip"]);
    }
    let (code, out) = stream(&mut cmd, None, log);
    if code != 0 {
        bail!("Kaggle upload failed: {}", out.last().map(|l| l.as_str()).unwrap_or("no answer"));
    }
    // Kaggle unpacks and checks the files before a notebook can attach them
    for _ in 0..120 {
        let (code, out) = capture(kaggle(c)?.args(["datasets", "status", &slug]), None);
        let l = out.to_lowercase();
        if code != 0 || l.contains("ready") {
            break; // ready, or a CLI without the status command: the notebook waits for the data itself
        }
        if code == 0 && l.contains("error") {
            bail!("Kaggle could not process the upload: {}", last_line(&out));
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    uploads.insert(slug.clone(), Value::String(sig));
    std::fs::write(&uploads_path, serde_json::to_string_pretty(&uploads)?)?;
    let _ = std::fs::remove_dir_all(&stage);
    Ok(slug)
}

#[cfg(unix)]
fn link(from: &Path, to: &Path) -> Result<()> {
    std::os::unix::fs::symlink(from, to).context("making the upload folder")
}

#[cfg(windows)]
fn link(from: &Path, to: &Path) -> Result<()> {
    let r = if from.is_dir() { std::os::windows::fs::symlink_dir(from, to) } else { std::os::windows::fs::symlink_file(from, to) };
    r.context("making the upload folder (Windows needs Developer Mode for links)")
}

/// "running" | "done" | "error" | "stopped"
pub fn kaggle_status(c: &KaggleCreds, reference: &str) -> Result<(String, String)> {
    let (code, out) = capture(kaggle(c)?.args(["kernels", "status", reference]), None);
    if code != 0 {
        bail!("Kaggle: {}", last_line(&out));
    }
    let l = out.to_lowercase();
    let s = if l.contains("complete") {
        "done"
    } else if l.contains("error") {
        "error"
    } else if l.contains("cancel") {
        "stopped"
    } else {
        "running"
    };
    Ok((s.into(), out.trim().to_string()))
}

/// the notebook's log so far: Kaggle also gives it while the notebook is still running (the text of its stream records)
pub fn kaggle_logs(c: &KaggleCreds, reference: &str) -> Result<String> {
    let (code, out) = capture(kaggle(c)?.args(["kernels", "logs", reference]), None);
    if code != 0 {
        bail!("Kaggle: {}", last_line(&out));
    }
    Ok(stream_text(&out))
}

/// a Kaggle log (a JSON list of {"stream_name", "time", "data"} records, possibly after other output) as plain text
pub fn stream_text(raw: &str) -> String {
    let json = match (raw.find('['), raw.rfind(']')) {
        (Some(a), Some(b)) if b > a => &raw[a..=b],
        _ => return raw.to_string(),
    };
    serde_json::from_str::<Vec<serde_json::Value>>(json)
        .map(|v| v.iter().filter_map(|r| r["data"].as_str()).collect::<String>())
        .unwrap_or_else(|_| raw.to_string())
}

pub fn kaggle_output(c: &KaggleCreds, reference: &str, into: &Path) -> Result<()> {
    std::fs::create_dir_all(into)?;
    let (code, out) = capture(kaggle(c)?.args(["kernels", "output", reference, "-p"]).arg(into).arg("-o"), None);
    if code != 0 {
        bail!("Kaggle: {}", last_line(&out));
    }
    Ok(())
}

// ---------------------------------------------------------------- AWS EC2
pub fn aws(c: &AwsCreds, action: &str, region: &str, instance: &str) -> Result<Value> {
    if !tools_ready() {
        bail!("Install the tools first (Connection tab).");
    }
    if c.access_key_id.is_empty() || c.secret_access_key.is_empty() {
        bail!("Add your AWS access key first (Connection tab).");
    }
    let helper = tools_dir().join("aws_ec2.py");
    std::fs::write(&helper, AWS_HELPER)?;
    let mut cmd = Command::new(bin("python"));
    cmd.arg(&helper).args([action, region, instance]).env("AWS_ACCESS_KEY_ID", &c.access_key_id).env("AWS_SECRET_ACCESS_KEY", &c.secret_access_key)
        .env_remove("AWS_PROFILE");
    let (_, out) = capture(&mut cmd, None);
    let v: Value = out.lines().rev().find_map(|l| serde_json::from_str(l).ok()).ok_or_else(|| anyhow!("AWS: {}", last_line(&out)))?;
    if v["ok"].as_bool() != Some(true) {
        bail!("AWS: {}", v["error"].as_str().unwrap_or("unknown error"));
    }
    Ok(v)
}

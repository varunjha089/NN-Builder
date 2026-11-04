//! Files: the settings folder (NN_BUILDER_HOME, else ~/Library/Application Support/NN Builder), settings.json, and project folders.
//! A project = project.json, model.json (the graph), dataset.json, train.json and runs/. Projects live wherever the owner chooses.

use crate::codegen::TrainSettings;
use crate::dataset::Dataset;
use crate::model::{self, Graph};
use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub fn home() -> PathBuf {
    let h = if let Ok(v) = std::env::var("NN_BUILDER_HOME") {
        PathBuf::from(v)
    } else {
        let base = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
        if cfg!(target_os = "macos") {
            base.join("Library/Application Support/NN Builder")
        } else if cfg!(windows) {
            PathBuf::from(std::env::var("APPDATA").unwrap_or_else(|_| ".".into())).join("NN Builder")
        } else {
            base.join(".local/share/nn-builder")
        }
    };
    let _ = std::fs::create_dir_all(&h);
    h
}

pub fn read_json<T: DeserializeOwned>(p: &Path) -> Option<T> {
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok())
}

/// atomic write (a crash never leaves half a file)
pub fn write_json<T: Serialize>(p: &Path, v: &T) -> Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(v)?)?;
    std::fs::rename(&tmp, p)?;
    Ok(())
}

pub fn settings() -> Value {
    read_json(&home().join("settings.json")).unwrap_or_else(|| json!({}))
}

pub fn save_setting(key: &str, value: Value) -> Result<()> {
    let mut s = settings();
    s.as_object_mut().context("settings.json is not an object")?.insert(key.into(), value);
    write_json(&home().join("settings.json"), &s)
}

pub fn default_projects_dir() -> PathBuf {
    settings()["projects_dir"].as_str().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("NN_Builder_Projects"))
}

#[derive(Debug, Clone)]
pub struct Project {
    pub dir: PathBuf,
}

impl Project {
    pub fn name(&self) -> String {
        read_json::<Value>(&self.dir.join("project.json")).and_then(|v| v["name"].as_str().map(String::from)).unwrap_or_else(|| self.dir.file_name().unwrap_or_default().to_string_lossy().into())
    }
    pub fn graph(&self) -> Graph {
        read_json(&self.dir.join("model.json")).unwrap_or_else(model::starter)
    }
    pub fn save_graph(&self, g: &Graph) -> Result<()> {
        write_json(&self.dir.join("model.json"), g)
    }
    pub fn dataset(&self) -> Dataset {
        read_json(&self.dir.join("dataset.json")).unwrap_or_default()
    }
    pub fn save_dataset(&self, d: &Dataset) -> Result<()> {
        write_json(&self.dir.join("dataset.json"), d)
    }
    pub fn train(&self) -> TrainSettings {
        read_json::<TrainSettings>(&self.dir.join("train.json")).unwrap_or_default().clean()
    }
    pub fn save_train(&self, t: &TrainSettings) -> Result<()> {
        write_json(&self.dir.join("train.json"), t)
    }
    pub fn runs_dir(&self) -> PathBuf {
        self.dir.join("runs")
    }
}

pub fn is_project(p: &Path) -> bool {
    p.join("project.json").is_file()
}

fn remember(p: &Path) -> Result<()> {
    let p = p.to_string_lossy().to_string();
    let mut list: Vec<String> = settings()["projects"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    list.retain(|x| *x != p);
    list.insert(0, p.clone());
    list.truncate(50);
    save_setting("projects", json!(list))?;
    save_setting("last_project", json!(p))
}

pub fn create(name: &str, location: Option<&Path>) -> Result<Project> {
    let name = name.trim();
    if name.is_empty() {
        bail!("Give the project a name.");
    }
    let folder: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect::<String>().trim_matches(|c| c == ' ' || c == '.').to_string();
    let dir = location.map(PathBuf::from).unwrap_or_else(default_projects_dir).join(if folder.is_empty() { "project".into() } else { folder });
    if dir.exists() {
        bail!("A folder {} already exists. Choose another name.", dir.display());
    }
    std::fs::create_dir_all(dir.join("runs"))?;
    write_json(&dir.join("project.json"), &json!({"name": name, "created": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(), "app": "NN Builder"}))?;
    let pr = Project { dir: dir.clone() };
    pr.save_graph(&model::starter())?;
    pr.save_dataset(&Dataset::default())?;
    pr.save_train(&TrainSettings::default())?;
    remember(&dir)?;
    Ok(pr)
}

pub fn open(path: &Path) -> Result<Project> {
    if !is_project(path) {
        bail!("That folder is not an NN Builder project (it has no project.json).");
    }
    let dir = path.canonicalize()?;
    remember(&dir)?;
    Ok(Project { dir })
}

pub fn listing() -> Vec<Value> {
    settings()["projects"].as_array().cloned().unwrap_or_default().iter().filter_map(|v| v.as_str()).map(|p| {
        let pp = Path::new(p);
        json!({"path": p, "exists": is_project(pp), "name": if is_project(pp) { Project { dir: pp.into() }.name() } else { pp.file_name().unwrap_or_default().to_string_lossy().into() }})
    }).collect()
}

pub fn last_project() -> Option<Project> {
    let p = PathBuf::from(settings()["last_project"].as_str()?);
    is_project(&p).then_some(Project { dir: p })
}

//! Training runs. A run is a folder <project>/runs/<id>/ with model.json, dataset.json, train.json, train.py (standalone), run.json (where
//! it runs and its state), and, as results arrive, metrics.jsonl, log.txt, results.json and model.pt.
//!
//! Kaggle: the script is pushed as a private GPU notebook; the app checks its status and downloads the outputs when it ends.
//! SSH (college) and AWS: the run folder is copied with rsync and started detached (SLURM / systemd-run / tmux / nohup+setsid), so it does not
//! depend on the ssh connection; the app copies new metrics lines while it is open and the results when the run ends. AWS: the app starts
//! the instance first, and the run stops it at the end.

use crate::cloud::{self, AwsCreds, KaggleCreds};
use crate::codegen;
use crate::jobs::{stream, Log};
use crate::remote::{self, sh_quote, Machine, Prepared};
use crate::shapes;
use crate::store::{self, Project};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const KAGGLE: &str = "Kaggle";
const KAGGLE_POLL: f64 = 30.0;
const SSH_POLL: f64 = 3.0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunInfo {
    pub id: String,
    pub target: String,
    /// "kaggle" | "ssh" | "aws"
    pub kind: String,
    pub created: String,
    pub status: String,
    pub epochs: usize,
    pub dataset: String,
    #[serde(default)]
    pub kernel: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub remote_dir: String,
    #[serde(default)]
    pub launch: String,
    #[serde(default)]
    pub job: String,
    #[serde(default)]
    pub offset: u64,
    #[serde(default)]
    pub fetched: bool,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub last_poll: f64,
    #[serde(default)]
    pub reachable: bool,
    #[serde(default)]
    pub note: String,
}

/// what a run may need from the Keychain (the app passes them in; core never stores them)
#[derive(Debug, Clone, Default)]
pub struct Creds {
    pub kaggle: Option<KaggleCreds>,
    pub aws: Option<AwsCreds>,
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub id: String,
    pub state: String,
    pub info: RunInfo,
    pub events: Vec<Value>,
    pub next: usize,
    pub log: Vec<String>,
    pub has_results: bool,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn run_dir(pr: &Project, id: &str) -> Result<PathBuf> {
    if id.is_empty() || id.contains('/') || id.contains("..") {
        bail!("bad run id");
    }
    let d = pr.runs_dir().join(id);
    if !d.join("run.json").exists() {
        bail!("no such run");
    }
    Ok(d)
}

pub fn info(d: &Path) -> RunInfo {
    store::read_json(&d.join("run.json")).unwrap_or_default()
}

fn save(d: &Path, i: &RunInfo) -> Result<()> {
    store::write_json(&d.join("run.json"), i)
}

pub fn read_events(d: &Path, since: usize) -> Vec<Value> {
    std::fs::read_to_string(d.join("metrics.jsonl"))
        .unwrap_or_default()
        .lines()
        .skip(since)
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

fn log_tail(d: &Path) -> Vec<String> {
    let t = std::fs::read_to_string(d.join("log.txt")).unwrap_or_default();
    let lines: Vec<String> = t.lines().filter(|l| !l.starts_with("NNB_EVENT ")).map(String::from).collect();
    lines[lines.len().saturating_sub(60)..].to_vec()
}

fn end_event(d: &Path) -> Option<String> {
    read_events(d, 0).iter().rev().find_map(|e| e["event"].as_str().filter(|s| ["done", "stopped", "error"].contains(s)).map(String::from))
}

// ---------------------------------------------------------------- start
/// make the run folder and start it on the target; returns the run id. Long (AWS start, prepare): call it from a background job.
pub fn start(pr: &Project, target: &str, gpu: &str, creds: &Creds, copy_tokens: bool, log: &Log) -> Result<String> {
    let (g, ds, t) = (pr.graph(), pr.dataset(), pr.train());
    let a = shapes::analyze(&g, &ds);
    if !a.ok {
        bail!("The design has problems (red blocks or the Dataset tab). Fix them before training.");
    }
    if !gpu.is_empty() && !gpu.chars().all(|c| c.is_ascii_digit() || c == ',') {
        bail!("GPU: a number like 0, or 0,1 (empty = any).");
    }
    let id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let d = pr.runs_dir().join(&id);
    if d.exists() {
        bail!("A run was started this second; try again.");
    }
    let title = format!("nnb {} {}", pr.name(), id);
    let script = codegen::train_script(&g, &ds, &a, &t, &title)?;
    std::fs::create_dir_all(&d)?;
    std::fs::write(d.join("train.py"), &script)?;
    store::write_json(&d.join("model.json"), &g)?;
    store::write_json(&d.join("dataset.json"), &ds)?;
    store::write_json(&d.join("train.json"), &t)?;
    let ds_label = match ds.source.as_str() {
        "builtin" => ds.builtin.clone(),
        "huggingface" => format!("hf:{}", ds.hf_repo),
        _ => format!("kaggle:{}", ds.kaggle_slug),
    };
    let mut info = RunInfo { id: id.clone(), target: target.into(), created: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(), status: "starting".into(),
        epochs: t.epochs, dataset: ds_label, ..Default::default() };
    if target == KAGGLE {
        info.kind = "kaggle".into();
        save(&d, &info)?;
        let c = creds.kaggle.as_ref().ok_or_else(|| anyhow!("Add your Kaggle token first (Connection tab)."))?;
        log("Pushing the notebook to Kaggle (private, GPU and internet on) ...".into());
        let res = cloud::kaggle_push(c, &d, &title, (ds.source == "kaggle").then_some(ds.kaggle_slug.as_str()));
        match res {
            Ok(r) => {
                info.url = format!("https://www.kaggle.com/code/{r}");
                info.kernel = r;
                info.status = "running".into();
                info.note = "Running on Kaggle's servers. Kaggle shows progress only when the notebook ends; the results are then downloaded here.".into();
                save(&d, &info)?;
                log(format!("Pushed: {}", info.url));
                return Ok(id);
            }
            Err(e) => {
                info.status = "failed".into();
                info.error = e.to_string();
                save(&d, &info)?;
                return Err(e);
            }
        }
    }
    let mut m = remote::machine(target).ok_or_else(|| anyhow!("Unknown machine {target:?} (Connection tab)."))?;
    info.kind = m.kind.clone();
    save(&d, &info)?;
    let res = start_remote(pr, &d, &mut info, &mut m, &t, gpu, creds, copy_tokens, ds.source.as_str(), log);
    if let Err(e) = &res {
        info.status = "failed".into();
        info.error = e.to_string();
        save(&d, &info)?;
    }
    res.map(|_| id)
}

/// AWS: start the instance and wait until ssh answers; sets m.host
pub fn aws_up(m: &mut Machine, creds: &Creds, log: &Log) -> Result<()> {
    let c = creds.aws.as_ref().ok_or_else(|| anyhow!("Add your AWS access key first (Connection tab)."))?;
    log(format!("Starting EC2 instance {} in {} ...", m.instance_id, m.region));
    let v = cloud::aws(c, "start", &m.region, &m.instance_id)?;
    m.host = v["host"].as_str().unwrap_or("").to_string();
    if m.host.is_empty() {
        bail!("The instance is running but has no public address (give it a public IP or use an Elastic IP).");
    }
    log(format!("Running at {}; waiting for ssh ...", m.host));
    for _ in 0..36 {
        if m.probe().is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    bail!("ssh did not answer within 3 minutes (security group allows port 22? the right key file?)")
}

pub fn prepare(m: &Machine, log: &Log) -> Result<Prepared> {
    if let Some(p) = m.key_problem() {
        bail!("{p}");
    }
    log(format!("Preparing {} ({}) ...", m.name, m.dest()));
    let (code, lines) = stream(&mut m.script_command(), Some(&remote::prepare_script(m)), log);
    let p = remote::parse_prepare(&lines);
    if code != 0 || !p.fail.is_empty() || p.torch.is_empty() {
        let why = if !p.fail.is_empty() {
            p.fail.clone()
        } else if m.auth == "password" && !m.connected() {
            "not logged in: press Connect first".into()
        } else {
            "see the log above".into()
        };
        bail!("Preparing failed: {why}");
    }
    remote::save_prepared(&m.name, &p)?;
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
fn start_remote(pr: &Project, d: &Path, info: &mut RunInfo, m: &mut Machine, t: &codegen::TrainSettings, gpu: &str, creds: &Creds, copy_tokens: bool,
    source: &str, log: &Log) -> Result<()> {
    if m.kind == "aws" {
        aws_up(m, creds, log)?;
        info.host = m.host.clone();
    } else if let Err(e) = m.probe() {
        bail!("{e}");
    }
    let p = match remote::prepared(&m.name) {
        Some(p) if !p.python.is_empty() => p,
        _ => prepare(m, log)?,
    };
    if !p.cuda {
        log("Note: no CUDA GPU was found on this machine; training will use its CPU.".into());
    }
    if copy_tokens {
        if source == "kaggle" {
            if let Some(k) = &creds.kaggle {
                let json = serde_json::json!({"username": k.username, "key": k.key}).to_string();
                let (_, out) = m.run_script(&remote::put_secret_script(".kaggle/kaggle.json", &json));
                if !out.contains("NNB_OK") {
                    bail!("Could not copy the Kaggle token to the machine.");
                }
                log("Kaggle token copied to ~/.kaggle/kaggle.json on the machine (readable only by you).".into());
            }
        }
        if source == "huggingface" {
            if let Some(tok) = &creds.hf_token {
                let (_, out) = m.run_script(&remote::put_secret_script(".cache/huggingface/token", tok));
                if !out.contains("NNB_OK") {
                    bail!("Could not copy the Hugging Face token to the machine.");
                }
                log("Hugging Face token copied to ~/.cache/huggingface/token on the machine (readable only by you).".into());
            }
        }
    }
    let method = remote::choose_launch(m, &p);
    let rdir = format!("{}/runs/{}", m.workdir.trim_end_matches('/'), info.id);
    let secs = (t.max_hours * 3600.0).round() as u64;
    let stop = m.kind == "aws" && t.stop_instance;
    let run_sh = format!(
        "#!/bin/bash\n# written by NN Builder: runs train.py, ends it after {h} h, records the exit code{stop_note}\ncd \"$(dirname \"$0\")\"\n\
         export NNB_DATA={data} PYTHONUNBUFFERED=1\n{gpu}echo $$ > pid.txt\n{timeout}{py} train.py > log.txt 2>&1\necho $? > exit_code.txt\n{stop}",
        h = t.max_hours,
        stop_note = if stop { "; then stops the instance" } else { "" },
        data = remote::remote_path(&format!("{}/data", m.workdir.trim_end_matches('/'))),
        gpu = if gpu.is_empty() { String::new() } else { format!("export CUDA_VISIBLE_DEVICES={gpu}\n") },
        timeout = if p.timeout { format!("timeout --signal=TERM --kill-after=180 {secs}s ") } else { String::new() },
        py = sh_quote(&p.python),
        stop = if stop { "sudo -n shutdown -h +2 >> log.txt 2>&1\n" } else { "" },
    );
    std::fs::write(d.join("run.sh"), run_sh)?;
    let rpath = remote::remote_path(&rdir);
    let (code, out) = m.run_script(&format!("mkdir -p {rpath} && echo NNB_OK"));
    if code != 0 || !out.contains("NNB_OK") {
        bail!("Could not make the run folder on the machine: {}", out.trim().lines().last().unwrap_or(""));
    }
    log(format!("Copying the run to {}:{rdir} ...", m.dest()));
    let rel = rdir.strip_prefix("~/").unwrap_or(&rdir);
    let (code, out) = crate::jobs::capture(&mut m.rsync_command(&format!("{}/", d.display()), &format!("REMOTE:{rel}/")), None);
    if code != 0 {
        bail!("Copying failed: {}", out.trim().lines().last().unwrap_or(""));
    }
    let hours = t.max_hours.ceil() as u64;
    let launch = match method.as_str() {
        "slurm" => format!("cd {rpath} && J=$(sbatch --parsable --job-name=nnb-{id} --output=slurm.out --gres=gpu:1 --time={hours}:00:00 run.sh) && echo \"NNB_JOB $J\"", id = info.id),
        "systemd" => format!("systemd-run --user --unit=nnb-{id} --collect --working-directory={rpath} bash {rpath}/run.sh >/dev/null 2>&1 && echo NNB_JOB nnb-{id}", id = info.id),
        "tmux" => format!("tmux new-session -d -s nnb-{id} \"bash {rpath}/run.sh\" && echo NNB_JOB nnb-{id}", id = info.id),
        _ => format!("cd {rpath} && S=''; command -v setsid >/dev/null 2>&1 && S=setsid; (nohup $S bash run.sh >/dev/null 2>&1 < /dev/null &); \
                      for i in 1 2 3 4 5 6 7 8 9 10; do [ -s pid.txt ] && break; sleep 0.5; done; echo \"NNB_JOB $(cat pid.txt)\""),
    };
    let (_, out) = m.run_script(&launch);
    let job = out.lines().find_map(|l| l.strip_prefix("NNB_JOB ")).map(|s| s.trim().split(';').next().unwrap_or("").to_string()).unwrap_or_default();
    if job.is_empty() {
        bail!("Could not start the run ({method}): {}", out.trim().lines().last().unwrap_or("no answer"));
    }
    log(format!("Started ({method}, job {job}). {}", remote::launch_note(&method, &p)));
    info.remote_dir = rdir;
    info.launch = method.clone();
    info.job = job;
    info.status = "running".into();
    info.reachable = true;
    info.note = remote::launch_note(&method, &p).into();
    info.last_poll = 0.0;
    save(d, info)?;
    let _ = pr;
    Ok(())
}

// ---------------------------------------------------------------- follow
fn alive_check(info: &RunInfo) -> String {
    let j = sh_quote(&info.job);
    match info.launch.as_str() {
        "slurm" => format!("squeue -h -j {j} 2>/dev/null | grep -q . && A=1"),
        "systemd" => format!("systemctl --user is-active --quiet {j} && A=1"),
        "tmux" => format!("tmux has-session -t {j} 2>/dev/null && A=1"),
        _ => format!("kill -0 {j} 2>/dev/null && A=1"),
    }
}

/// copy new metrics lines and the log tail from the machine; Some(alive) or None when it does not answer
fn poll_remote(m: &Machine, d: &Path, info: &mut RunInfo) -> Option<bool> {
    let script = format!(
        "cd {rd} 2>/dev/null || {{ echo NNB_NODIR; exit 0; }}\nA=0\n{alive}\n[ -f exit_code.txt ] && {{ A=0; echo \"NNB_EXIT $(cat exit_code.txt)\"; }}\n\
         echo NNB_ALIVE $A\necho NNB_LOG_BEGIN\ntail -n 60 log.txt 2>/dev/null | grep -v '^NNB_EVENT '\necho NNB_LOG_END\necho NNB_METRICS_BEGIN\ntail -c +{off} metrics.jsonl 2>/dev/null\n",
        rd = remote::remote_path(&info.remote_dir),
        alive = alive_check(info),
        off = info.offset + 1
    );
    let (code, out) = m.run_script(&script);
    if code != 0 || !out.contains("NNB_ALIVE") {
        return None;
    }
    let log = out.split("NNB_LOG_BEGIN\n").nth(1).and_then(|s| s.split("NNB_LOG_END").next()).unwrap_or("");
    let _ = std::fs::write(d.join("log.txt"), log);
    let new = out.split("NNB_METRICS_BEGIN\n").nth(1).unwrap_or("");
    let complete = &new[..new.rfind('\n').map(|i| i + 1).unwrap_or(0)];
    if !complete.is_empty() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(d.join("metrics.jsonl")) {
            let _ = f.write_all(complete.as_bytes());
        }
        info.offset += complete.len() as u64;
    }
    Some(out.contains("NNB_ALIVE 1"))
}

/// copy the run's outputs back (ssh / aws). AWS: if the instance is stopped, start it, copy, stop it again.
pub fn fetch(pr: &Project, id: &str, creds: &Creds, log: &Log) -> Result<()> {
    let d = run_dir(pr, id)?;
    let mut info = info(&d);
    if info.kind == "kaggle" {
        let c = creds.kaggle.as_ref().ok_or_else(|| anyhow!("Add your Kaggle token first (Connection tab)."))?;
        return kaggle_fetch(c, &d, &mut info);
    }
    let mut m = remote::machine(&info.target).ok_or_else(|| anyhow!("The machine of this run was deleted."))?;
    let mut started = false;
    if m.kind == "aws" {
        let c = creds.aws.as_ref().ok_or_else(|| anyhow!("Add your AWS access key first (Connection tab)."))?;
        let v = cloud::aws(c, "describe", &m.region, &m.instance_id)?;
        if v["state"] == "running" {
            m.host = v["host"].as_str().unwrap_or("").into();
        } else {
            aws_up(&mut m, creds, log)?;
            started = true;
        }
    }
    let rel = info.remote_dir.strip_prefix("~/").unwrap_or(&info.remote_dir).to_string();
    for name in ["metrics.jsonl", "log.txt", "results.json", "model.pt", "train.py"] {
        let (code, out) = crate::jobs::capture(&mut m.rsync_command(&format!("REMOTE:{rel}/{name}"), &d.join(name).to_string_lossy()), None);
        if code != 0 && (name == "metrics.jsonl" || name == "results.json") && end_event(&d).as_deref() != Some("error") {
            log(format!("{name}: {}", out.trim().lines().last().unwrap_or("")));
        }
    }
    info.offset = std::fs::metadata(d.join("metrics.jsonl")).map(|m| m.len()).unwrap_or(0);
    info.fetched = end_event(&d).is_some();
    save(&d, &info)?;
    if started {
        log("Stopping the instance again ...".into());
        cloud::aws(creds.aws.as_ref().unwrap(), "stop", &m.region, &m.instance_id)?;
    }
    Ok(())
}

fn kaggle_fetch(c: &KaggleCreds, d: &Path, info: &mut RunInfo) -> Result<()> {
    let out = d.join("kaggle_output");
    cloud::kaggle_output(c, &info.kernel, &out)?;
    for name in ["metrics.jsonl", "results.json", "model.pt"] {
        if out.join(name).exists() {
            std::fs::copy(out.join(name), d.join(name))?;
        }
    }
    // the notebook's own log (a JSON list of stream records) -> log.txt
    if let Some(logf) = std::fs::read_dir(&out)?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.extension().map(|x| x == "log").unwrap_or(false)) {
        let raw = std::fs::read_to_string(&logf).unwrap_or_default();
        let text = serde_json::from_str::<Vec<Value>>(&raw)
            .map(|v| v.iter().filter_map(|r| r["data"].as_str()).collect::<String>())
            .unwrap_or(raw);
        std::fs::write(d.join("log.txt"), text)?;
    }
    info.fetched = true;
    save(d, info)
}

pub fn status(pr: &Project, id: &str, since: usize, creds: &Creds) -> Result<Status> {
    let d = run_dir(pr, id)?;
    let mut info = info(&d);
    let mut state = info.status.clone();
    if !["failed"].contains(&state.as_str()) && !info.fetched {
        let due = now() - info.last_poll >= if info.kind == "kaggle" { KAGGLE_POLL } else { SSH_POLL };
        if due && info.status != "starting" {
            info.last_poll = now();
            if info.kind == "kaggle" {
                if let Some(c) = &creds.kaggle {
                    match cloud::kaggle_status(c, &info.kernel) {
                        Ok((s, _)) => {
                            info.reachable = true;
                            if s != "running" {
                                if kaggle_fetch(c, &d, &mut info).is_err() && s == "error" {
                                    info.error = format!("The notebook failed on Kaggle: see {}", info.url);
                                }
                                info.fetched = true;
                                state = end_event(&d).unwrap_or(s);
                            } else {
                                state = "running".into();
                            }
                        }
                        Err(e) => {
                            info.reachable = false;
                            info.error = e.to_string();
                        }
                    }
                }
            } else if let Some(mut m) = remote::machine(&info.target) {
                if m.kind == "aws" {
                    m.host = info.host.clone();
                }
                match poll_remote(&m, &d, &mut info) {
                    Some(alive) => {
                        info.reachable = true;
                        state = match end_event(&d) {
                            Some(e) => e,
                            None if alive => if read_events(&d, 0).iter().any(|e| e["event"] == "start") { "running".into() } else { "starting".into() },
                            None => "crashed".into(),
                        };
                        if state != "running" && state != "starting" {
                            let noop: Log = std::sync::Arc::new(|_| {});
                            let _ = fetch(pr, id, creds, &noop);
                            info = self::info(&d);
                            info.last_poll = now();
                        }
                    }
                    None => info.reachable = false,
                }
            }
            info.status = state.clone();
            save(&d, &info)?;
        }
    }
    let events = read_events(&d, since);
    let next = since + events.len();
    Ok(Status { id: id.into(), state: info.status.clone(), has_results: d.join("results.json").exists(), log: log_tail(&d), info, events, next })
}

pub fn stop(pr: &Project, id: &str) -> Result<String> {
    let d = run_dir(pr, id)?;
    let info = info(&d);
    if info.kind == "kaggle" {
        bail!("Kaggle notebooks cannot be stopped through the API: open {} and press Cancel run.", info.url);
    }
    let mut m = remote::machine(&info.target).ok_or_else(|| anyhow!("The machine of this run was deleted."))?;
    if m.kind == "aws" {
        m.host = info.host.clone();
    }
    let j = sh_quote(&info.job);
    let cmd = match info.launch.as_str() {
        "slurm" => format!("scancel --signal=TERM {j} && echo NNB_OK"),
        "systemd" => format!("systemctl --user kill --signal=TERM {j} && echo NNB_OK"),
        "tmux" => format!("tmux send-keys -t {j} C-c && echo NNB_OK"),
        _ => format!("(kill -TERM -- -{j} 2>/dev/null || kill -TERM {j}) && echo NNB_OK"),
    };
    let (_, out) = m.run_script(&cmd);
    if !out.contains("NNB_OK") {
        bail!("Could not stop it: {}", out.trim().lines().last().unwrap_or("no answer"));
    }
    Ok("Stopping: the current epoch ends early, the model is saved, then the run ends.".into())
}

pub fn listing(pr: &Project) -> Vec<Value> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(pr.runs_dir()) else { return out };
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("run.json").exists()).collect();
    dirs.sort();
    dirs.reverse();
    for d in dirs {
        let i = info(&d);
        let acc = read_events(&d, 0).iter().rev().find_map(|e| e["test_acc"].as_f64());
        out.push(serde_json::json!({"id": i.id, "target": i.target, "kind": i.kind, "created": i.created, "dataset": i.dataset, "status": i.status, "test_acc": acc}));
    }
    out
}

pub fn results(pr: &Project, id: &str) -> Result<Value> {
    let d = run_dir(pr, id)?;
    store::read_json(&d.join("results.json")).ok_or_else(|| anyhow!("No results for this run yet."))
}

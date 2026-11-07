//! Training runs. A run is a folder <project>/runs/<id>/ with model.json, dataset.json, train.json, train.py (standalone), run.json (where
//! it runs and its state), and, as results arrive, metrics.jsonl, log.txt, results.json and model.pt.
//!
//! Kaggle: the script is pushed as a private GPU notebook; the app checks its status and downloads the outputs when it ends.
//! SSH (college) and AWS: the run folder is copied with rsync and started detached (SLURM / systemd-run / tmux / nohup+setsid), so it does not
//! depend on the ssh connection; the app copies new metrics lines while it is open and the results when the run ends. AWS: the app starts
//! the instance first, and the run stops it at the end.
//!
//! EC2 spot instances (AWS may take one back with a 2-minute notice): run.sh watches the instance metadata for the notice (and for the
//! earlier rebalance recommendation) and tells train.py to save a checkpoint at once (SIGUSR1; SIGUSR2 = checkpoint and go on); train.py
//! then exits with 75 and the run keeps resume_wanted. train.py also checkpoints every few minutes (a hard interruption loses at most that).
//! When the instance comes back (a persistent spot request with interruption behaviour "stop" is restarted by AWS), resume.sh, started by
//! crontab @reboot, starts the run again: independent of this Mac. While the app is open it also follows the instance to its new address,
//! restarts a cut-off run itself, and (machine setting "mirror") keeps a copy of each new checkpoint here, so a run whose instance was
//! terminated can continue on another machine (resume_on). Guards: a lock (one copy of a run at a time), a limit on automatic resumes,
//! a time budget summed over all resumes, and a run continued elsewhere is stopped on its old machine when that one answers again.

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
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const KAGGLE: &str = "Kaggle";
const KAGGLE_POLL: f64 = 30.0;
const SSH_POLL: f64 = 3.0;
/// asking AWS about an instance that does not answer: at most this often
const AWS_DESCRIBE: f64 = 30.0;
/// train.py's exit code for "interrupted, start me again"; resume.sh writes 76 when it gives up (too many resumes)
pub const RESUME_EXIT: i32 = 75;
pub const GAVE_UP_EXIT: i32 = 76;
/// runs whose checkpoint is being copied to this Mac right now
static MIRRORING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

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
    /// a Kaggle run waiting for a free GPU (status "queued"): the dataset its notebook attaches, for the push that follows
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kaggle_attach: Option<String>,
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
    /// AWS: the instance is a spot instance (can be interrupted; the run resumes)
    #[serde(default)]
    pub spot: bool,
    #[serde(default)]
    pub last_describe: f64,
    #[serde(default)]
    pub last_relaunch: f64,
    /// this run continues another one from its checkpoint (resume_on) / was continued as another run
    #[serde(default)]
    pub resumed_from: String,
    #[serde(default)]
    pub superseded_by: String,
    /// Stop was pressed while the machine did not answer: stop it (and do not resume it) as soon as it answers
    #[serde(default)]
    pub cancel: bool,
    /// a Check data run (data report and one-batch check, no training)
    #[serde(default)]
    pub check: bool,
    /// the random seed of this run, and the group of runs started together with different seeds ("" = a single run)
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub group: String,
}

/// what a run may need from the Keychain (the app passes them in; core never stores them)
#[derive(Debug, Clone, Default)]
pub struct Creds {
    pub kaggle: Option<KaggleCreds>,
    pub aws: Option<AwsCreds>,
    pub hf_token: Option<String>,
    /// a Google Cloud service-account key (JSON) for private Cloud Storage buckets
    pub gcs_key: Option<String>,
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
    /// the last checkpoint train.py reported (its "checkpoint" event), and the copy on this Mac (mirror.json), if any
    pub checkpoint: Option<Value>,
    pub mirror: Option<Value>,
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
    start_seeded(pr, target, gpu, creds, copy_tokens, log, None, "")
}

/// Check data: a short run on the target that loads the data, reports it (files per class, bands, empty pixels, what looks wrong) and
/// runs the one-batch shape check, without training
pub fn start_check(pr: &Project, target: &str, gpu: &str, creds: &Creds, copy_tokens: bool, log: &Log) -> Result<String> {
    start_seeded(pr, target, gpu, creds, copy_tokens, log, None, CHECK)
}

/// the group name that marks a Check data run inside start_seeded
const CHECK: &str = "\u{0}check";

/// start runs of the same design and settings with seeds base, base+1, ... (one group), to report a mean and its spread; returns the ids
pub fn start_seeds(pr: &Project, target: &str, gpu: &str, creds: &Creds, copy_tokens: bool, count: usize, log: &Log) -> Result<Vec<String>> {
    let base = pr.train().seed;
    let group = format!("g{}", chrono::Local::now().format("%Y%m%d-%H%M%S"));
    let mut ids = vec![];
    for i in 0..count.clamp(1, 10) as u64 {
        log(format!("--- seed {} ({} of {count}) ---", base + i, i + 1));
        ids.push(start_seeded(pr, target, gpu, creds, copy_tokens, log, Some(base + i), &group)?);
    }
    Ok(ids)
}

#[allow(clippy::too_many_arguments)]
fn start_seeded(pr: &Project, target: &str, gpu: &str, creds: &Creds, copy_tokens: bool, log: &Log, seed: Option<u64>, group: &str) -> Result<String> {
    let (g, ds, mut t) = (pr.graph(), pr.dataset(), pr.train());
    if let Some(s) = seed {
        t.seed = s;
    }
    let check = group == CHECK;
    let group = if check { "" } else { group };
    t.check_only = check;
    let a = shapes::analyze(&g, &ds);
    if !a.ok {
        bail!("The design has problems (red blocks or the Dataset tab). Fix them before training.");
    }
    if !gpu.is_empty() && !gpu.chars().all(|c| c.is_ascii_digit() || c == ',') {
        bail!("GPU: a number like 0, or 0,1 (empty = any).");
    }
    preflight(target, creds)?;
    let mut id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    if !group.is_empty() {
        id = format!("{id}-s{}", t.seed);
    }
    if check {
        id = format!("{id}-check");
    }
    let d = pr.runs_dir().join(&id);
    if d.exists() {
        bail!("A run was started this second; try again.");
    }
    let title = cloud::kernel_title(&pr.name(), &id);
    if check {
        log("Check data: the run only loads the data, reports it and checks one batch (no training).".into());
    }
    let script = codegen::train_script(&g, &ds, &a, &t, &title)?;
    std::fs::create_dir_all(&d)?;
    std::fs::write(d.join("train.py"), &script)?;
    store::write_json(&d.join("model.json"), &g)?;
    store::write_json(&d.join("dataset.json"), &ds)?;
    store::write_json(&d.join("train.json"), &t)?;
    let ds_label = match ds.source.as_str() {
        "builtin" => ds.builtin.clone(),
        "huggingface" => format!("hf:{}", ds.hf_repo),
        "local" if !ds.cloud_url.is_empty() => format!("cloud:{}", ds.cloud_url),
        "local" => format!("own:{}", ds.local_dir.trim_end_matches('/').rsplit('/').next().unwrap_or("")),
        _ => format!("kaggle:{}", ds.kaggle_slug),
    };
    let mut info = RunInfo { id: id.clone(), target: target.into(), created: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(), status: "starting".into(),
        epochs: if check { 0 } else { t.epochs }, dataset: ds_label, seed: t.seed, group: group.into(), check, ..Default::default() };
    if target == KAGGLE {
        info.kind = "kaggle".into();
        save(&d, &info)?;
        let c = creds.kaggle.as_ref().ok_or_else(|| anyhow!("Add your Kaggle token first (Connection tab)."))?;
        let attach = match ds.source.as_str() {
            "kaggle" => Ok(Some(ds.kaggle_slug.clone())),
            "local" if !ds.cloud_url.is_empty() => Ok(None), // the notebook fetches it (public buckets / shared links)
            "local" if ds.local_on_machine => Err(anyhow!("Your files are on a training machine, not on this Mac: Kaggle cannot reach them. Train on that machine, or choose a folder on this Mac.")),
            "local" => cloud::kaggle_upload(c, &local_path(&ds.local_dir), &ds.local_key(), log).map(Some),
            _ => Ok(None),
        };
        log("Pushing the notebook to Kaggle (private, GPU and internet on) ...".into());
        let res = attach.and_then(|a| cloud::kaggle_push(c, &d, &title, a.as_deref()));
        match res {
            Ok(r) => {
                info.url = format!("https://www.kaggle.com/code/{r}");
                info.kernel = r;
                info.status = "running".into();
                info.note = "Running on Kaggle's servers. The curves come from the notebook's log, read every 30 s; the results are downloaded when it ends.".into();
                save(&d, &info)?;
                log(format!("Pushed: {}", info.url));
                return Ok(id);
            }
            Err(e) if is_session_limit(&e.to_string()) => {
                // Kaggle runs 2 GPU notebooks at a time (several seeds, or other runs): wait for a free one
                info.status = "queued".into();
                info.kaggle_attach = Some(kaggle_attach_of(&ds));
                info.note = QUEUED_NOTE.into();
                save(&d, &info)?;
                log("Kaggle runs 2 GPU notebooks at a time: this run waits and is pushed when one is free.".into());
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
    let res = start_remote(pr, &d, &mut info, &mut m, &t, gpu, creds, copy_tokens, &ds, log);
    if let Err(e) = &res {
        info.status = "failed".into();
        info.error = e.to_string();
        save(&d, &info)?;
    }
    res.map(|_| id)
}

/// what a target needs before a run is created at all (so that a missing tool or key does not leave a failed run behind)
pub fn preflight(target: &str, creds: &Creds) -> Result<()> {
    let aws = target != KAGGLE && remote::machine(target).map(|m| m.kind == "aws").unwrap_or(false);
    if (target == KAGGLE || aws) && !cloud::tools_ready() {
        bail!("Install the tools first: Connection → Install tools (once; the kaggle and boto3 helpers, about 60 MB).");
    }
    if target == KAGGLE && creds.kaggle.is_none() {
        bail!("Add your Kaggle API token first (Connection → Kaggle).");
    }
    if aws && creds.aws.is_none() {
        bail!("Add your AWS access key first (Connection → AWS).");
    }
    if target != KAGGLE && remote::machine(target).is_none() {
        bail!("Unknown machine {target:?} (Connection tab).");
    }
    Ok(())
}

/// AWS: start the instance and wait until ssh answers; sets m.host. Returns what AWS said about it (spot or not, the spot request).
pub fn aws_up(m: &mut Machine, creds: &Creds, log: &Log) -> Result<Value> {
    let c = creds.aws.as_ref().ok_or_else(|| anyhow!("Add your AWS access key first (Connection tab)."))?;
    // the state before (free): an on-demand instance whose shutdown behaviour is Terminate would be deleted, disk and all, when a run ends
    let before = cloud::aws(c, "describe", &m.region, &m.instance_id)?;
    let was = before["state"].as_str().unwrap_or("").to_string();
    match before["shutdown_behavior"].as_str() {
        Some("terminate") if before["lifecycle"] != "spot" => bail!(
            "This instance's shutdown behaviour is Terminate: when a run ends and the instance shuts down, AWS would delete it and its disk. \
             Change it to Stop first (EC2 console → the instance → Actions → Instance settings → Change shutdown behavior)."),
        None if before["lifecycle"] != "spot" => log("Note: the instance's shutdown behaviour could not be read (the key needs \
            ec2:DescribeInstanceAttribute); it must be Stop, not Terminate.".into()),
        _ => {}
    }
    log(format!("Starting EC2 instance {} in {} ...", m.instance_id, m.region));
    let started = was != "running";
    let mut v = match cloud::aws(c, "start", &m.region, &m.instance_id) {
        Ok(v) => v,
        Err(e) => {
            let t = e.to_string();
            if t.contains("spot-interrupted") {
                bail!("AWS has taken this spot instance's capacity back; it cannot be started by hand. AWS starts it again when capacity returns \
                       (persistent spot request), and an interrupted run then resumes by itself. Try later, or use another machine.");
            } else if t.contains("insufficient-capacity") {
                bail!("AWS has no spare GPU capacity in this zone right now (common for spot GPUs). Try again later, or use another machine.");
            }
            // e.g. the wait for "running" timed out: the instance may be starting all the same
            return Err(if started { anyhow!("{e}\n{}", aws_stop_again(m, creds, log)) } else { e });
        }
    };
    v["was"] = was.into();
    let res = (|| {
        m.host = v["host"].as_str().unwrap_or("").to_string();
        if m.host.is_empty() {
            bail!("The instance is running but has no public address (give it a public IP or use an Elastic IP).");
        }
        if v["lifecycle"] == "spot" {
            log(format!("Spot instance: {}", spot_summary(&v)));
        }
        log(format!("Running at {}; waiting for ssh ...", m.host));
        // (6 minutes: the first boot of an AMI such as the Deep Learning AMI loads its disk from the snapshot and can be slow)
        for _ in 0..72 {
            if m.probe().is_ok() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(5));
        }
        bail!("ssh did not answer within 6 minutes (security group allows port 22 from this Mac's address? the right key file and user?)")
    })();
    match res {
        Ok(()) => Ok(v),
        Err(e) if started => Err(anyhow!("{e}\n{}", aws_stop_again(m, creds, log))),
        Err(e) => Err(e),
    }
}

/// whether `aws_up`'s instance was started by that call (it was not running before), so that it is stopped again when the work fails
pub fn started_here(v: &Value) -> bool {
    v["was"] != "running"
}

/// stops an instance that the app started for work that then failed (it would cost money while it waits for nobody); the text for the
/// error. An instance that was already running is never stopped here: another run may be using it.
pub fn aws_stop_again(m: &Machine, creds: &Creds, log: &Log) -> String {
    log("Stopping the instance again ...".into());
    let res = match creds.aws.as_ref() {
        Some(c) => cloud::aws(c, "stop", &m.region, &m.instance_id).map(|_| ()),
        None => Err(anyhow!("no AWS key")),
    };
    match res {
        Ok(()) => "The instance was stopped again.".into(),
        Err(e) => format!("COULD NOT STOP THE INSTANCE ({e}): STOP IT IN THE EC2 CONSOLE ({} in {}); it costs money while it runs.",
            m.instance_id, m.region),
    }
}

/// one line about a spot instance's request and what an interruption will do to it
pub fn spot_summary(v: &Value) -> String {
    let sp = &v["spot"];
    if sp.get("error").is_some() || sp.is_null() {
        return "the spot request could not be read (allow ec2:DescribeSpotInstanceRequests to see its type and interruption behaviour).".into();
    }
    let ty = sp["type"].as_str().unwrap_or("?");
    let beh = sp["interruption"].as_str().unwrap_or("terminate");
    let what = match (ty, beh) {
        ("persistent", "stop") | ("persistent", "hibernate") => "after an interruption AWS restarts it when capacity returns, and the run resumes by itself.",
        (_, "terminate") => "an interruption DELETES the instance and its disk: only the checkpoint copy on this Mac survives (turn on \
                              'Keep a copy of checkpoints' for this machine). Better: a persistent request with interruption behaviour 'stop'.",
        _ => "after an interruption the instance stays stopped until it is started again.",
    };
    format!("{ty} request, interruption behaviour '{beh}': {what}")
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
    ds: &crate::dataset::Dataset, log: &Log) -> Result<()> {
    let mut started = false;
    if m.kind == "aws" {
        let v = aws_up(m, creds, log)?;
        info.host = m.host.clone();
        info.spot = v["lifecycle"] == "spot";
        started = started_here(&v);
    } else if let Err(e) = m.probe() {
        bail!("{e}");
    }
    let res = launch_remote(pr, d, info, m, t, gpu, creds, copy_tokens, ds, log);
    match res {
        // nothing runs on the instance: stop it again unless it was already running before
        Err(e) if started && info.job.is_empty() => Err(anyhow!("{e}\n{}", aws_stop_again(m, creds, log))),
        Err(e) if m.kind == "aws" && info.job.is_empty() => {
            log("The instance was already running before this run, so it is left running.".into());
            Err(e)
        }
        r => r,
    }
}

/// Prepare (if needed), tokens, data, run.sh, copy and launch on a machine that answers
#[allow(clippy::too_many_arguments)]
fn launch_remote(pr: &Project, d: &Path, info: &mut RunInfo, m: &mut Machine, t: &codegen::TrainSettings, gpu: &str, creds: &Creds, copy_tokens: bool,
    ds: &crate::dataset::Dataset, log: &Log) -> Result<()> {
    let source = ds.source.as_str();
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
    if source == "local" && !ds.local_on_machine && ds.cloud_url.is_empty() {
        upload_local(m, ds, log)?;
    }
    if source == "local" && ds.cloud_url.starts_with("gs://") && copy_tokens {
        if let Some(k) = &creds.gcs_key {
            let (_, out) = m.run_script(&remote::put_secret_script(".config/nnb/gcs-key.json", k));
            if !out.contains("NNB_OK") {
                bail!("Could not copy the Cloud Storage key to the machine.");
            }
            log("Cloud Storage key copied to ~/.config/nnb/gcs-key.json on the machine (readable only by you).".into());
        }
    }
    let method = remote::choose_launch(m, &p);
    let rdir = format!("{}/runs/{}", m.workdir.trim_end_matches('/'), info.id);
    std::fs::write(d.join("run.sh"), run_script(t, &p, m, gpu, info.spot))?;
    std::fs::write(d.join("max_resumes.txt"), format!("{}\n", t.max_resumes))?;
    if info.spot {
        let (_, out) = m.run_script(&install_resume_script(m));
        if out.contains("NNB_CRON") {
            log("Spot: interrupted runs are resumed when the instance boots again (crontab @reboot), also while this Mac is off.".into());
        } else {
            log("Spot: crontab is not available on the instance, so an interrupted run is resumed only while this app is open.".into());
        }
    }

    let rpath = remote::remote_path(&rdir);
    let (code, out) = m.run_script(&format!("mkdir -p {rpath} && echo NNB_OK"));
    if code != 0 || !out.contains("NNB_OK") {
        bail!("Could not make the run folder on the machine: {}", out.trim().lines().last().unwrap_or(""));
    }
    log(format!("Copying the run to {}:{rdir} ...", m.dest()));
    let rel = rdir.strip_prefix("~/").unwrap_or(&rdir);
    let (code, out) = m.upload_dir(d, rel);
    if code != 0 {
        bail!("Copying failed: {}", out.trim().lines().last().unwrap_or(""));
    }
    let hours = t.max_hours.ceil() as u64;
    let launch = match method.as_str() {
        "slurm" => format!("cd {rpath} && J=$(sbatch --parsable --job-name=nnb-{id} --output=slurm.out --gres=gpu:1 --time={hours}:00:00 run.sh) && echo \"NNB_JOB $J\"", id = info.id),
        "systemd" => format!("systemd-run --user --unit=nnb-{id} --collect --working-directory={rpath} bash {rpath}/run.sh >/dev/null 2>&1 && echo NNB_JOB nnb-{id}", id = info.id),
        "tmux" => format!("cd {rpath} && tmux new-session -d -s nnb-{id} -c \"$PWD\" 'bash run.sh' && echo NNB_JOB nnb-{id}", id = info.id),
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

/// a folder on this Mac: "~/" means the home folder
pub fn local_path(dir: &str) -> PathBuf {
    match dir.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(dir),
    }
}

/// the owner's own files to <workdir>/data/local/<key> on the machine, where train.py looks for them; skipped when the machine already has
/// this version (.nnb_signature there), and rsync copies only what changed
fn upload_local(m: &Machine, ds: &crate::dataset::Dataset, log: &Log) -> Result<()> {
    let local = local_path(&ds.local_dir);
    let sig = crate::localdata::signature(&local)?;
    let ddir = format!("{}/data/local/{}", m.workdir.trim_end_matches('/'), ds.local_key());
    let rpath = remote::remote_path(&ddir);
    let (_, out) = m.run_script(&format!("cat {rpath}/.nnb_signature 2>/dev/null; echo; echo NNB_OK"));
    if out.lines().any(|l| l.trim() == sig) {
        log(format!("Your files are already on the machine ({sig})."));
        return Ok(());
    }
    let (code, out) = m.run_script(&format!("mkdir -p {rpath} && echo NNB_OK"));
    if code != 0 || !out.contains("NNB_OK") {
        bail!("Could not make the data folder on the machine: {}", out.trim().lines().last().unwrap_or(""));
    }
    log(format!("Copying your files ({sig}) to {}:{ddir} (only the first time and after changes) ...", m.dest()));
    let (code, out) = m.upload_dir(&local, ddir.strip_prefix("~/").unwrap_or(&ddir));
    if code != 0 {
        bail!("Copying your files failed: {}", out.trim().lines().last().unwrap_or(""));
    }
    let (_, out) = m.run_script(&format!("printf '%s\\n' {} > {rpath}/.nnb_signature && echo NNB_OK", sh_quote(&sig)));
    if !out.contains("NNB_OK") {
        log("Note: could not record the copy on the machine; the files will be copied again next time.".into());
    }
    Ok(())
}

/// run.sh: runs train.py within a budget of running time summed over all resumes (used_seconds.txt), one copy at a time (flock on run.lock),
/// records the exit code. AWS: watches the instance metadata for a spot interruption notice (-> SIGUSR1: checkpoint and exit 75) and a
/// rebalance recommendation (-> SIGUSR2: checkpoint now). Spot: resume_wanted stays while the run is unfinished, for resume.sh.
pub fn run_script(t: &codegen::TrainSettings, p: &Prepared, m: &Machine, gpu: &str, spot: bool) -> String {
    let secs = (t.max_hours * 3600.0).round() as u64;
    let stop = if m.kind == "aws" && t.stop_instance { "sudo -n shutdown -h +2 >> log.txt 2>&1\n" } else { "" };
    let ck_env = if spot && t.checkpoint_minutes <= 0.0 { format!(" NNB_CHECKPOINT_MINUTES={}", codegen::SPOT_CHECKPOINT_MINUTES) } else { String::new() };
    let watch = m.kind == "aws";
    format!(
        "#!/bin/bash\n\
         # written by NN Builder: runs train.py within {h} h of running time (summed over resumes), records the exit code{stop_note}.\n\
         # Exit code {re} = interrupted (EC2 spot): resume_wanted stays, and resume.sh (at boot) or the app starts this script again.\n\
         cd \"$(dirname \"$0\")\"\n\
         if command -v flock >/dev/null 2>&1; then exec 9>run.lock; flock -n 9 || exit 0; fi\n\
         export NNB_DATA={data} PYTHONUNBUFFERED=1{ck_env}\n\
         A=\"$(dirname {py})/activate\"; if [ -f \"$A\" ]; then . \"$A\"; fi\n\
         {gpu}echo $$ > pid.txt\n\
         USED=$(cat used_seconds.txt 2>/dev/null || echo 0); LEFT=$(( {secs} - USED ))\n\
         if [ \"$LEFT\" -le 60 ]; then\n  echo \"NNB: the time budget of {h} h is used up\" >> log.txt; rm -f resume_wanted; echo 124 > exit_code.txt\n  {stop}  exit 0\nfi\n\
         rm -f exit_code.txt interrupted.txt py_pid.txt\n\
         {mark}T0=$(date +%s); STOPPED=0\n\
         {watch_fn}\
         {timeout}{py} train.py >> log.txt 2>&1 &\n\
         TP=$!\n\
         trap 'STOPPED=1; kill -TERM $TP 2>/dev/null' TERM INT HUP\n\
         {watch_start}wait $TP; CODE=$?\n\
         while kill -0 $TP 2>/dev/null; do wait $TP; CODE=$?; done\n\
         {watch_end}echo $(( USED + $(date +%s) - T0 )) > used_seconds.txt\n\
         echo $CODE > exit_code.txt\n\
         if [ \"$CODE\" = {re} ] && [ \"$STOPPED\" = 0 ]; then exit 0; fi\n\
         rm -f resume_wanted\n\
         {stop}",
        h = t.max_hours,
        re = RESUME_EXIT,
        stop_note = if stop.is_empty() { "" } else { "; then stops the instance" },
        data = remote::remote_path(&format!("{}/data", m.workdir.trim_end_matches('/'))),
        gpu = if gpu.is_empty() { String::new() } else { format!("export CUDA_VISIBLE_DEVICES={gpu}\n") },
        mark = if spot { "touch resume_wanted\n" } else { "" },
        watch_fn = if watch { SPOT_WATCH } else { "" },
        timeout = if p.timeout { "timeout --signal=TERM --kill-after=180 ${LEFT}s " } else { "" },
        py = sh_quote(&p.python),
        watch_start = if watch { "spot_watch & W=$!\n" } else { "" },
        watch_end = if watch { "kill $W 2>/dev/null\n" } else { "" },
    )
}

/// EC2 instance metadata (IMDSv2): the spot interruption notice (2 minutes ahead) and the rebalance recommendation (earlier, a risk)
const SPOT_WATCH: &str = r#"spot_watch() {
  local IM=${NNB_IMDS:-http://169.254.169.254} tok code rb=0
  while kill -0 "$TP" 2>/dev/null; do
    tok=$(curl -s -m 2 -X PUT "$IM/latest/api/token" -H 'X-aws-ec2-metadata-token-ttl-seconds: 600')
    code=$(curl -s -m 2 -o spot_action.txt -w '%{http_code}' -H "X-aws-ec2-metadata-token: $tok" "$IM/latest/meta-data/spot/instance-action")
    if [ "$code" = 200 ]; then
      echo "NNB: EC2 spot interruption notice ($(cat spot_action.txt)): saving a checkpoint" >> log.txt; touch interrupted.txt
      until kill -USR1 "$(cat py_pid.txt 2>/dev/null)" 2>/dev/null; do kill -0 "$TP" 2>/dev/null || return; sleep 1; done
      return
    fi
    if [ $rb = 0 ] && [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $tok" "$IM/latest/meta-data/events/recommendations/rebalance")" = 200 ]; then
      rb=1; echo "NNB: EC2 rebalance recommendation (interruption risk is high): saving a checkpoint now" >> log.txt
      kill -USR2 "$(cat py_pid.txt 2>/dev/null)" 2>/dev/null
    fi
    sleep 5
  done
}
"#;

/// resume.sh (in the machine's work folder): starts again every run that still wants it (resume_wanted, no final event, not running),
/// counting automatic resumes against max_resumes.txt. Run by crontab @reboot (after a pause for the GPU driver), or by the app with "now".
pub const RESUME_SH: &str = r#"#!/bin/bash
# written by NN Builder: starts again the runs that an EC2 spot interruption cut off (crontab @reboot; "now" = without the pause)
[ "$1" = now ] || sleep 45
cd "$(dirname "$0")/runs" 2>/dev/null || exit 0
for d in */; do
  d=${d%/}
  [ -f "$d/resume_wanted" ] || continue
  if grep -qE '"event": "(done|stopped|error)"' "$d/metrics.jsonl" 2>/dev/null; then rm -f "$d/resume_wanted"; continue; fi
  if command -v flock >/dev/null 2>&1 && ! flock -n "$d/run.lock" true; then continue; fi
  if [ -f "$d/py_pid.txt" ] && kill -0 "$(cat "$d/py_pid.txt")" 2>/dev/null; then continue; fi
  n=$(cat "$d/resumes.txt" 2>/dev/null || echo 0); max=$(cat "$d/max_resumes.txt" 2>/dev/null || echo 20)
  if [ "$n" -ge "$max" ]; then
    echo "NNB: not resumed again: $n automatic resumes already (limit $max)" >> "$d/log.txt"; rm -f "$d/resume_wanted"; echo 76 > "$d/exit_code.txt"
    continue
  fi
  echo $((n + 1)) > "$d/resumes.txt"
  echo "NNB: resuming after an interruption (automatic resume $((n + 1)) of at most $max)" >> "$d/log.txt"
  (cd "$d" && S=''; command -v setsid >/dev/null 2>&1 && S=setsid; nohup $S bash run.sh > /dev/null 2>&1 < /dev/null &)
  echo "NNB_RESUMED $d"
done
"#;

/// write resume.sh into the work folder and register it with crontab @reboot (once; NNB_CRON when crontab took it)
fn install_resume_script(m: &Machine) -> String {
    let w = remote::remote_path(m.workdir.trim_end_matches('/'));
    format!(
        "W={w}\nmkdir -p \"$W\" && cat > \"$W/resume.sh\" <<'NNB_EOF'\n{RESUME_SH}NNB_EOF\nchmod +x \"$W/resume.sh\"\n\
         if command -v crontab >/dev/null 2>&1; then (crontab -l 2>/dev/null | grep -v nn_builder_resume; echo \"@reboot bash $W/resume.sh # nn_builder_resume\") \
         | crontab - && echo NNB_CRON; fi\necho NNB_OK\n"
    )
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

/// what one look at the machine found
#[derive(Debug, Default)]
struct Poll {
    alive: bool,
    exit: Option<i32>,
    resume_wanted: bool,
    /// checkpoint.pt on the machine: "size mtime" (changes when train.py writes a new one)
    ck: Option<String>,
}

/// copy new metrics lines and the log tail from the machine; None when it does not answer
fn poll_remote(m: &Machine, d: &Path, info: &mut RunInfo) -> Option<Poll> {
    let lock = if info.kind == "aws" {
        // a run started again by resume.sh has a new process: the lock it holds says it is running
"if command -v flock >/dev/null 2>&1 && ! flock -n run.lock true 2>/dev/null; then A=1; fi\n\
         [ -f py_pid.txt ] && kill -0 \"$(cat py_pid.txt)\" 2>/dev/null && A=1\n"
    } else {
        ""
    };
    let script = format!(
        "cd {rd} 2>/dev/null || {{ echo NNB_NODIR; exit 0; }}\nA=0\n{alive}\n{lock}[ -f exit_code.txt ] && {{ A=0; echo \"NNB_EXIT $(cat exit_code.txt)\"; }}\n\
         [ -f resume_wanted ] && echo NNB_RESUME_WANTED\n\
         [ -f checkpoint.pt ] && echo \"NNB_CK $(stat -c '%s %Y' checkpoint.pt 2>/dev/null || stat -f '%z %m' checkpoint.pt)\"\n\
         echo NNB_ALIVE $A\necho NNB_LOG_BEGIN\ntail -n 60 log.txt 2>/dev/null | grep -v '^NNB_EVENT '\necho NNB_LOG_END\necho NNB_METRICS_BEGIN\ntail -c +{off} metrics.jsonl 2>/dev/null\n",
        rd = remote::remote_path(&info.remote_dir),
        alive = alive_check(info),
        off = info.offset + 1
    );
    let (code, out) = m.run_script(&script);
    if code != 0 || !out.contains("NNB_ALIVE") {
        return None;
    }
    let head = out.split("NNB_LOG_BEGIN").next().unwrap_or("");
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
    Some(Poll {
        alive: head.contains("NNB_ALIVE 1"),
        exit: head.lines().find_map(|l| l.strip_prefix("NNB_EXIT ")).and_then(|c| c.trim().parse().ok()),
        resume_wanted: head.contains("NNB_RESUME_WANTED"),
        ck: head.lines().find_map(|l| l.strip_prefix("NNB_CK ")).map(|c| c.trim().to_string()).filter(|c| !c.is_empty()),
    })
}

/// copy the machine's newest checkpoint to <run>/checkpoint.pt in the background (rsync writes a temporary file and renames it, so the
/// copy here is always a whole checkpoint); mirror.json records which one
fn mirror_checkpoint(m: &Machine, d: &Path, info: &RunInfo, stamp: &str) {
    let have: Option<Value> = store::read_json(&d.join("mirror.json"));
    if have.as_ref().and_then(|v| v["stamp"].as_str()) == Some(stamp) {
        return;
    }
    {
        let mut busy = MIRRORING.lock().unwrap_or_else(|e| e.into_inner());
        if busy.iter().any(|p| p == d) {
            return;
        }
        busy.push(d.to_path_buf());
    }
    let (m, d, stamp) = (m.clone(), d.to_path_buf(), stamp.to_string());
    let rel = info.remote_dir.strip_prefix("~/").unwrap_or(&info.remote_dir).to_string();
    std::thread::spawn(move || {
        let (code, out) = m.download_file(&format!("{rel}/checkpoint.pt"), &d.join("checkpoint.pt"));
        let last = read_events(&d, 0).into_iter().rev().find(|e| e["event"] == "checkpoint");
        let rec = if code == 0 {
            serde_json::json!({"stamp": stamp, "copied": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                "bytes": std::fs::metadata(d.join("checkpoint.pt")).map(|x| x.len()).unwrap_or(0), "checkpoint": last})
        } else {
            let mut v = have.unwrap_or_else(|| serde_json::json!({}));
            v["error"] = Value::String(format!("copying the checkpoint failed: {}", out.trim().lines().last().unwrap_or("")));
            v
        };
        let _ = store::write_json(&d.join("mirror.json"), &rec);
        MIRRORING.lock().unwrap_or_else(|e| e.into_inner()).retain(|p| p != &d);
    });
}

/// what AWS says about an instance that does not answer
enum Aws {
    /// not asked (asked less than AWS_DESCRIBE ago, no key, or AWS did not answer)
    NotAsked,
    /// running (the host is updated when it came back with a new address): try ssh
    Up,
    /// down; the run's state for the window ("interrupted", "terminated") or None when it is just stopped
    Down(Option<String>),
}

/// ask AWS what happened to an instance that does not answer
fn aws_explain(m: &Machine, creds: &Creds, info: &mut RunInfo) -> Aws {
    let Some(c) = creds.aws.as_ref() else { return Aws::NotAsked };
    if now() - info.last_describe < AWS_DESCRIBE {
        return Aws::NotAsked;
    }
    info.last_describe = now();
    let Ok(v) = cloud::aws(c, "describe", &m.region, &m.instance_id) else { return Aws::NotAsked };
    let state = v["state"].as_str().unwrap_or("");
    let reason = v["state_reason"].as_str().unwrap_or("");
    let spot_status = v["spot"]["status"].as_str().unwrap_or("");
    let host = v["host"].as_str().unwrap_or("");
    match state {
        "running" | "pending" => {
            if !host.is_empty() && host != info.host {
                info.host = host.to_string();
                info.note = "The instance came back with a new address; following it there.".into();
            }
            Aws::Up
        }
        "terminated" | "shutting-down" => {
            info.note = "The instance was terminated (a spot request with interruption behaviour 'terminate', or deleted): its disk and \
                         the checkpoints on it are gone. Continue from the copy on this Mac with Resume on another machine, if there is one."
                .into();
            Aws::Down(Some("terminated".into()))
        }
        "stopped" | "stopping" if info.spot && (reason.starts_with("Server.Spot") || spot_status.contains("stopped-by")
            || spot_status.contains("no-capacity") || spot_status.contains("marked-for-stop") || spot_status.contains("capacity")) =>
        {
            info.note = format!(
                "AWS took the spot capacity back ({}). A persistent spot request restarts the instance when capacity returns; the run then \
                 resumes by itself from its last checkpoint. Or continue now with Resume on another machine.",
                if spot_status.is_empty() { reason } else { spot_status }
            );
            Aws::Down(Some("interrupted".into()))
        }
        "stopped" | "stopping" => {
            info.note = "The instance is stopped, but the run had not finished. Fetch results starts it to copy what is there.".into();
            Aws::Down(None)
        }
        _ => Aws::NotAsked,
    }
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
            started = started_here(&aws_up(&mut m, creds, log)?);
        }
    }
    let rel = info.remote_dir.strip_prefix("~/").unwrap_or(&info.remote_dir).to_string();
    for name in ["metrics.jsonl", "log.txt", "results.json", "model.pt", "train.py", "model.ts", "model.onnx"] {
        let (code, out) = m.download_file(&format!("{rel}/{name}"), &d.join(name));
        if code != 0 && (name == "metrics.jsonl" || name == "results.json") && end_event(&d).as_deref() != Some("error") {
            log(format!("{name}: {}", out.trim().lines().last().unwrap_or("")));
        }
    }
    info.offset = std::fs::metadata(d.join("metrics.jsonl")).map(|m| m.len()).unwrap_or(0);
    info.fetched = end_event(&d).is_some();
    let saved = save(&d, &info);
    if started {
        log("Stopping the instance again ...".into());
        if let Err(e) = cloud::aws(creds.aws.as_ref().unwrap(), "stop", &m.region, &m.instance_id) {
            bail!("COULD NOT STOP THE INSTANCE ({e}): STOP IT IN THE EC2 CONSOLE ({} in {}); it costs money while it runs.", m.instance_id, m.region);
        }
    }
    saved
}

/// the log text without the NNB_EVENT lines (those are the metrics)
fn log_without_events(text: &str) -> String {
    text.lines().map(|l| l.rsplit('\r').next().unwrap_or(l)).filter(|l| !l.starts_with("NNB_EVENT ")).collect::<Vec<_>>().join("\n")
}

/// live progress of a running Kaggle notebook: its log so far gives the metrics (NNB_EVENT lines) and the log tail. The files are
/// rewritten whole each time (the log only grows); when the notebook ends, the downloaded outputs replace them.
fn kaggle_live(c: &KaggleCreds, d: &Path, info: &mut RunInfo) {
    let Ok(text) = cloud::kaggle_logs(c, &info.kernel) else { return };
    let events: Vec<&str> = text.lines().filter_map(|l| l.rsplit('\r').next().and_then(|l| l.strip_prefix("NNB_EVENT ")))
        .filter(|l| serde_json::from_str::<Value>(l).is_ok()).collect();
    if !events.is_empty() {
        let body = events.join("\n") + "\n";
        if std::fs::write(d.join("metrics.jsonl"), &body).is_ok() {
            info.offset = body.len() as u64;
        }
    }
    let _ = std::fs::write(d.join("log.txt"), log_without_events(&text));
}

fn kaggle_fetch(c: &KaggleCreds, d: &Path, info: &mut RunInfo) -> Result<()> {
    let out = d.join("kaggle_output");
    cloud::kaggle_output(c, &info.kernel, &out)?;
    for name in ["metrics.jsonl", "results.json", "model.pt", "model.ts", "model.onnx"] {
        if out.join(name).exists() {
            std::fs::copy(out.join(name), d.join(name))?;
        }
    }
    // the notebook's own log (a JSON list of stream records) -> log.txt
    if let Some(logf) = std::fs::read_dir(&out)?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.extension().map(|x| x == "log").unwrap_or(false)) {
        let text = cloud::stream_text(&std::fs::read_to_string(&logf).unwrap_or_default());
        std::fs::write(d.join("log.txt"), log_without_events(&text))?;
    }
    // the files that matter are copied into the run folder; the download itself is not kept
    let _ = std::fs::remove_dir_all(&out);
    info.fetched = true;
    save(d, info)
}

const QUEUED_NOTE: &str = "Waiting for a free Kaggle GPU (Kaggle runs 2 GPU notebooks at a time): pushed when one is free, while its \
    project's Train tab is open (or the next time it is opened).";

/// Kaggle refused a push because 2 GPU notebooks are already running
fn is_session_limit(msg: &str) -> bool {
    msg.contains("Maximum batch GPU session") || msg.contains("maximum number of concurrent")
}

/// the dataset a Kaggle notebook attaches ("" = none): a Kaggle dataset, or the private upload of the owner's own files
fn kaggle_attach_of(ds: &crate::dataset::Dataset) -> String {
    match ds.source.as_str() {
        "kaggle" => ds.kaggle_slug.clone(),
        "local" if ds.cloud_url.is_empty() && !ds.local_on_machine => ds.local_key(),
        _ => String::new(),
    }
}

/// the start of the note a run gets while Kaggle cannot be asked about it (cleared when it answers again)
const KAGGLE_DOWN: &str = "Kaggle could not be reached";

pub fn status(pr: &Project, id: &str, since: usize, creds: &Creds) -> Result<Status> {
    let d = run_dir(pr, id)?;
    let mut info = info(&d);
    let mut state = info.status.clone();
    // a spot run continued elsewhere is still looked at (when opened): if its old instance came back, it is stopped there
    let watch_old = state == "continued" && info.spot;
    if state == "queued" && info.kind == "kaggle" {
        if let Some(c) = creds.kaggle.as_ref().filter(|_| now() - info.last_poll >= KAGGLE_POLL) {
            info.last_poll = now();
            let attach = info.kaggle_attach.clone().unwrap_or_default();
            // (an own-files dataset was uploaded under the owner's name: <user>/<key>)
            let attach = if attach.is_empty() || attach.contains('/') { attach } else { format!("{}/{attach}", c.username) };
            match cloud::kaggle_push(c, &d, &cloud::kernel_title(&pr.name(), id), (!attach.is_empty()).then_some(attach.as_str())) {
                Ok(r) => {
                    info.url = format!("https://www.kaggle.com/code/{r}");
                    info.kernel = r;
                    info.status = "running".into();
                    info.note = "Running on Kaggle's servers. The curves come from the notebook's log, read every 30 s; the results are downloaded when it ends.".into();
                    state = "running".into();
                }
                Err(e) if is_session_limit(&e.to_string()) => {}
                Err(e) => {
                    info.status = "failed".into();
                    info.error = e.to_string();
                    state = "failed".into();
                }
            }
            save(&d, &info)?;
        }
    }
    if !["failed", "terminated", "continued"].contains(&state.as_str()) && !info.fetched || watch_old {
        let due = now() - info.last_poll >= if info.kind == "kaggle" { KAGGLE_POLL } else { SSH_POLL };
        if due && info.status != "starting" {
            info.last_poll = now();
            if info.kind == "kaggle" {
                if let Some(c) = &creds.kaggle {
                    let was_down = info.error.starts_with(KAGGLE_DOWN);
                    match cloud::kaggle_status(c, &info.kernel) {
                        Ok((s, _)) => {
                            info.reachable = true;
                            if was_down {
                                info.error.clear(); // Kaggle answers again: the earlier poll's error is over
                            }
                            if s != "running" {
                                if kaggle_fetch(c, &d, &mut info).is_err() && s == "error" {
                                    info.error = format!("The notebook failed on Kaggle: see {}", info.url);
                                }
                                info.fetched = true;
                                state = end_event(&d).unwrap_or(s);
                            } else {
                                state = "running".into();
                                kaggle_live(c, &d, &mut info);
                            }
                        }
                        Err(e) => {
                            // (no network for a moment, or Kaggle busy: the run on Kaggle goes on; the next poll tries again)
                            info.reachable = false;
                            let why = e.to_string();
                            info.error = format!("{KAGGLE_DOWN} ({}); the run itself is not affected and the app tries again.",
                                if why.contains("NameResolution") || why.contains("ConnectionError") { "no connection".to_string() } else { why.chars().take(200).collect() });
                        }
                    }
                }
            } else if let Some(mut m) = remote::machine(&info.target) {
                if m.kind == "aws" {
                    m.host = info.host.clone();
                }
                // an instance known to be down: ask AWS (rarely) instead of waiting for ssh every few seconds
                let mut look = true;
                if m.kind == "aws" && (!info.reachable || watch_old) && ["interrupted", "terminated", "continued"].contains(&info.status.as_str()) {
                    match aws_explain(&m, creds, &mut info) {
                        Aws::Up => m.host = info.host.clone(),
                        Aws::NotAsked => look = false,
                        Aws::Down(s) => {
                            look = false;
                            if let Some(s) = s {
                                state = if info.superseded_by.is_empty() { s } else { "continued".into() };
                            }
                        }
                    }
                }
                match if look { poll_remote(&m, &d, &mut info) } else { None } {
                    Some(p) => {
                        info.reachable = true;
                        // Stop was pressed while it did not answer, or the run continues elsewhere: end it here, without resuming
                        if (info.cancel || !info.superseded_by.is_empty()) && (p.alive || p.resume_wanted) {
                            let _ = cancel_remote(&m, &info);
                            info.note = if info.cancel { "Stopped as asked (it did not answer earlier).".into() }
                                        else { format!("Stopped here: the run continues as {}.", info.superseded_by) };
                        }
                        if m.mirror {
                            if let Some(stamp) = &p.ck {
                                mirror_checkpoint(&m, &d, &info, stamp);
                            }
                        }
                        state = match end_event(&d) {
                            Some(e) => e,
                            None if p.alive => if read_events(&d, 0).iter().any(|e| e["event"] == "start") { "running".into() } else { "starting".into() },
                            None if p.exit == Some(GAVE_UP_EXIT) => {
                                info.error = "Interrupted too many times: not resumed again (Train settings: automatic resumes).".into();
                                "interrupted".into()
                            }
                            None if p.resume_wanted && !info.cancel && info.superseded_by.is_empty() => {
                                // the instance is up but the run is not: start it again now (resume.sh counts it and keeps one copy)
                                if now() - info.last_relaunch > 60.0 {
                                    info.last_relaunch = now();
                                    let w = remote::remote_path(m.workdir.trim_end_matches('/'));
                                    let (_, out) = m.run_script(&format!("[ -f {w}/resume.sh ] || exit 0; bash {w}/resume.sh now"));
                                    if out.contains("NNB_RESUMED") {
                                        info.note = "Interrupted, and started again from its last checkpoint.".into();
                                    }
                                }
                                "interrupted".into()
                            }
                            None if !info.superseded_by.is_empty() => "continued".into(),
                            None => "crashed".into(),
                        };
                        if !info.superseded_by.is_empty() {
                            state = "continued".into();
                        }
                        if !["running", "starting", "interrupted", "continued"].contains(&state.as_str()) {
                            let noop: Log = std::sync::Arc::new(|_| {});
                            let _ = fetch(pr, id, creds, &noop);
                            info = self::info(&d);
                            info.last_poll = now();
                        }
                    }
                    None => {
                        info.reachable = false;
                        if look && m.kind == "aws" && end_event(&d).is_none() {
                            if let Aws::Down(Some(s)) = aws_explain(&m, creds, &mut info) {
                                state = if info.superseded_by.is_empty() { s } else { "continued".into() };
                            }
                        }
                    }
                }
            }
            info.status = state.clone();
            save(&d, &info)?;
        }
    }
    let events = read_events(&d, since);
    let next = since + events.len();
    let checkpoint = read_events(&d, 0).into_iter().rev().find(|e| e["event"] == "checkpoint");
    let mirror = store::read_json(&d.join("mirror.json"));
    Ok(Status { id: id.into(), state: info.status.clone(), has_results: d.join("results.json").exists(), log: log_tail(&d), info, events, next, checkpoint, mirror })
}

/// end the run on its machine and make sure it is not resumed (resume_wanted removed first)
fn cancel_remote(m: &Machine, info: &RunInfo) -> Result<()> {
    let j = sh_quote(&info.job);
    let kill = match info.launch.as_str() {
        "slurm" => format!("scancel --signal=TERM {j}"),
        "systemd" => format!("systemctl --user kill --signal=TERM {j}"),
        "tmux" => format!("tmux send-keys -t {j} C-c"),
        _ => format!("(kill -TERM -- -{j} 2>/dev/null || kill -TERM {j})"),
    };
    // a run started again by resume.sh has a new process: also signal the one in py_pid.txt
    let cmd = format!(
        "cd {rd} 2>/dev/null || {{ echo NNB_OK; exit 0; }}\nrm -f resume_wanted\n{kill} 2>/dev/null\n[ -f py_pid.txt ] && kill -TERM \"$(cat py_pid.txt)\" 2>/dev/null\necho NNB_OK\n",
        rd = remote::remote_path(&info.remote_dir)
    );
    let (_, out) = m.run_script(&cmd);
    if !out.contains("NNB_OK") {
        bail!("{}", out.trim().lines().last().unwrap_or("no answer"));
    }
    Ok(())
}

pub fn stop(pr: &Project, id: &str) -> Result<String> {
    let d = run_dir(pr, id)?;
    let mut info = info(&d);
    if info.kind == "kaggle" {
        bail!("Kaggle notebooks cannot be stopped through the API: open {} and press Cancel run.", info.url);
    }
    let mut m = remote::machine(&info.target).ok_or_else(|| anyhow!("The machine of this run was deleted."))?;
    if m.kind == "aws" {
        m.host = info.host.clone();
    }
    match cancel_remote(&m, &info) {
        Ok(()) => Ok("Stopping: the current epoch ends early, the model is saved, then the run ends (and is not resumed).".into()),
        Err(e) if info.spot => {
            info.cancel = true;
            save(&d, &info)?;
            Ok(format!("The instance does not answer ({e}). The run will be stopped as soon as it answers again, instead of resuming; \
                        while this app is closed AWS may still restart the instance, so also cancel the spot request in the AWS console \
                        if you do not want that."))
        }
        Err(e) => bail!("Could not stop it: {e}"),
    }
}

/// continue a run on another machine from the copy of its checkpoint on this Mac (the old instance was terminated, or its capacity does
/// not come back): a new run with the same train.py, the checkpoint and the metrics so far; the old run is stopped there if it answers again
pub fn resume_on(pr: &Project, from: &str, target: &str, gpu: &str, creds: &Creds, copy_tokens: bool, log: &Log) -> Result<String> {
    let src = run_dir(pr, from)?;
    if !src.join("checkpoint.pt").exists() {
        bail!("There is no copy of this run's checkpoint on this Mac. Turn on 'Keep a copy of checkpoints on this Mac' for the machine \
               (Connection tab) before training; a run whose machine still answers can also be fetched first.");
    }
    if target == KAGGLE {
        bail!("Resuming on Kaggle is not supported (the checkpoint would have to be uploaded as a Kaggle dataset). Choose an SSH or AWS machine.");
    }
    let mut m = remote::machine(target).ok_or_else(|| anyhow!("Unknown machine {target:?} (Connection tab)."))?;
    let mut old = info(&src);
    let id = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let d = pr.runs_dir().join(&id);
    if d.exists() {
        bail!("A run was started this second; try again.");
    }
    std::fs::create_dir_all(&d)?;
    for f in ["train.py", "model.json", "dataset.json", "train.json", "checkpoint.pt", "metrics.jsonl"] {
        if src.join(f).exists() {
            std::fs::copy(src.join(f), d.join(f))?;
        }
    }
    let t: codegen::TrainSettings = store::read_json::<codegen::TrainSettings>(&d.join("train.json")).unwrap_or_default().clean();
    let ck = store::read_json::<Value>(&src.join("mirror.json")).and_then(|v| v.get("checkpoint").cloned()).unwrap_or(Value::Null);
    let mut info = RunInfo { id: id.clone(), target: target.into(), kind: m.kind.clone(), created: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        status: "starting".into(), epochs: old.epochs, dataset: old.dataset.clone(), resumed_from: from.into(),
        offset: std::fs::metadata(d.join("metrics.jsonl")).map(|x| x.len()).unwrap_or(0), ..Default::default() };
    save(&d, &info)?;
    log(format!("Continuing run {from} on {target} from its checkpoint{} ...",
        if ck.is_null() { String::new() } else { format!(" (epoch {}, batch {}/{})", ck["epoch"], ck["step"], ck["steps"]) }));
    let ds = store::read_json::<crate::dataset::Dataset>(&d.join("dataset.json")).unwrap_or_default();
    let res = start_remote(pr, &d, &mut info, &mut m, &t, gpu, creds, copy_tokens, &ds, log);
    if let Err(e) = &res {
        info.status = "failed".into();
        info.error = e.to_string();
        save(&d, &info)?;
        return Err(anyhow!("{e}"));
    }
    old.superseded_by = id.clone();
    old.status = "continued".into();
    let stopped_old = old.reachable && remote::machine(&old.target).map(|mut om| {
        om.host = old.host.clone();
        cancel_remote(&om, &old).is_ok()
    }).unwrap_or(false);
    old.note = if stopped_old {
        format!("Continued as run {id} on {target}; stopped on its old machine.")
    } else {
        format!("Continued as run {id} on {target}. If the old instance comes back it would resume there too: when you open this run while it \
                 is up, the app stops it; otherwise cancel its spot request in the AWS console.")
    };
    save(&src, &old)?;
    Ok(id)
}

pub fn listing(pr: &Project) -> Vec<Value> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(pr.runs_dir()) else { return out };
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("run.json").exists()).collect();
    dirs.sort();
    dirs.reverse();
    for d in dirs {
        let i = info(&d);
        let ev = read_events(&d, 0);
        let acc = ev.iter().rev().find_map(|e| e["test_acc"].as_f64());
        // Check data: how many things looked wrong
        let warnings = ev.iter().rev().find(|e| e["event"] == "done").and_then(|e| e["warnings"].as_u64());
        // regression runs report R² in test_acc's place
        let task = ev.iter().find(|e| e["event"] == "start").and_then(|e| e["task"].as_str()).unwrap_or("classify").to_string();
        out.push(serde_json::json!({"id": i.id, "target": i.target, "kind": i.kind, "created": i.created, "dataset": i.dataset, "status": i.status,
            "test_acc": acc, "seed": i.seed, "group": i.group, "check": i.check, "warnings": warnings, "task": task}));
    }
    out
}

/// one row per run for comparing and exporting: settings, epochs, best epoch, test accuracy (and top-5, mIoU), time; and mean / sample
/// standard deviation per group (runs started together with different seeds; runs outside a group form their own)
pub fn summary(pr: &Project, ids: &[String]) -> Result<Value> {
    let mut rows = vec![];
    for id in ids {
        let d = run_dir(pr, id)?;
        let i = info(&d);
        let ev = read_events(&d, 0);
        let start = ev.iter().find(|e| e["event"] == "start").cloned().unwrap_or(Value::Null);
        let done = ev.iter().rev().find(|e| e["event"] == "done" || e["event"] == "stopped").cloned().unwrap_or(Value::Null);
        let epochs: Vec<&Value> = ev.iter().filter(|e| e["event"] == "epoch").collect();
        let t: codegen::TrainSettings = store::read_json(&d.join("train.json")).unwrap_or_default();
        let res: Value = store::read_json(&d.join("results.json")).unwrap_or(Value::Null);
        let best_val = epochs.iter().filter_map(|e| e["val_acc"].as_f64()).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
        rows.push(serde_json::json!({
            "id": i.id, "group": if i.group.is_empty() { i.id.clone() } else { i.group.clone() }, "seed": t.seed, "target": i.target,
            "dataset": i.dataset, "status": i.status, "params": start["params"], "gpu": start["gpu"],
            "optimizer": t.optimizer, "lr": t.lr, "batch_size": t.batch_size, "schedule": t.schedule, "early_stop": t.early_stop,
            "epochs_set": t.epochs, "epochs_done": epochs.len(), "best_epoch": done["best_epoch"], "best_val_acc": best_val,
            "test_acc": res["test_acc"].as_f64().or(done["test_acc"].as_f64()), "test_top5": res["test_top5"].as_f64().or(done["test_top5"].as_f64()),
            "test_miou": res["miou"].as_f64(),
            "seconds": epochs.iter().filter_map(|e| e["seconds"].as_f64()).sum::<f64>(),
        }));
    }
    let mut groups: Vec<Value> = vec![];
    let mut names: Vec<String> = rows.iter().map(|r| r["group"].as_str().unwrap_or("").to_string()).collect();
    names.dedup();
    let mut seen = std::collections::HashSet::new();
    for g in names.into_iter().filter(|g| seen.insert(g.clone())) {
        let members: Vec<&Value> = rows.iter().filter(|r| r["group"] == g.as_str()).collect();
        let stat = |key: &str| {
            let v: Vec<f64> = members.iter().filter_map(|r| r[key].as_f64()).collect();
            if v.is_empty() {
                return serde_json::json!(null);
            }
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let sd = if v.len() > 1 { (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt() } else { 0.0 };
            serde_json::json!({"mean": mean, "std": sd, "n": v.len()})
        };
        groups.push(serde_json::json!({"group": g, "runs": members.len(), "seeds": members.iter().map(|r| r["seed"].clone()).collect::<Vec<_>>(),
            "test_acc": stat("test_acc"), "test_top5": stat("test_top5"), "test_miou": stat("test_miou"), "epochs_done": stat("epochs_done")}));
    }
    Ok(serde_json::json!({"rows": rows, "groups": groups}))
}

/// write the summary as CSV and as a LaTeX table (booktabs) into <project>/exports/; returns their paths
pub fn export(pr: &Project, ids: &[String]) -> Result<(PathBuf, PathBuf)> {
    let s = summary(pr, ids)?;
    let dir = pr.dir.join("exports");
    std::fs::create_dir_all(&dir)?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let pct = |v: &Value| v.as_f64().map(|x| format!("{:.2}", x * 100.0)).unwrap_or_default();
    let mut csv = String::from("run,group,seed,target,dataset,optimizer,lr,batch_size,schedule,epochs_set,epochs_done,best_epoch,test_acc,test_top5,test_miou,seconds\n");
    for r in s["rows"].as_array().unwrap() {
        let f = |k: &str| match &r[k] { Value::Null => String::new(), Value::String(x) => x.replace(',', ";"), v => v.to_string() };
        csv += &[f("id"), f("group"), f("seed"), f("target"), f("dataset"), f("optimizer"), f("lr"), f("batch_size"), f("schedule"), f("epochs_set"),
            f("epochs_done"), f("best_epoch"), f("test_acc"), f("test_top5"), f("test_miou"), format!("{:.1}", r["seconds"].as_f64().unwrap_or(0.0))].join(",");
        csv.push('\n');
    }
    let has_top5 = s["rows"].as_array().unwrap().iter().any(|r| r["test_top5"].is_number());
    let tex_esc = |x: &str| x.replace('_', "\\_").replace('%', "\\%").replace('&', "\\&");
    let mut tex = format!("% NN Builder: results of {} run(s), {stamp}. Needs \\usepackage{{booktabs}}.\n\\begin{{tabular}}{{@{{}}lrrr{}@{{}}}}\n\\toprule\n\
        Run & Seed & Epochs & Top-1 (\\%){} \\\\\n\\midrule\n", ids.len(), if has_top5 { "r" } else { "" }, if has_top5 { " & Top-5 (\\%)" } else { "" });
    for r in s["rows"].as_array().unwrap() {
        tex += &format!("{} & {} & {} & {}{} \\\\\n", tex_esc(r["id"].as_str().unwrap_or("")), r["seed"], r["epochs_done"], pct(&r["test_acc"]),
            if has_top5 { format!(" & {}", pct(&r["test_top5"])) } else { String::new() });
    }
    let pm = |v: &Value| match (v["mean"].as_f64(), v["std"].as_f64()) {
        (Some(m), Some(sd)) => format!("{:.2} $\\pm$ {:.2}", m * 100.0, sd * 100.0),
        _ => String::new(),
    };
    for g in s["groups"].as_array().unwrap().iter().filter(|g| g["runs"].as_u64().unwrap_or(0) > 1) {
        tex += &format!("\\midrule\nMean $\\pm$ s.d. ({} seeds) & & & {}{} \\\\\n", g["runs"], pm(&g["test_acc"]),
            if has_top5 { format!(" & {}", pm(&g["test_top5"])) } else { String::new() });
    }
    tex += "\\bottomrule\n\\end{tabular}\n";
    let (c, t) = (dir.join(format!("runs-{stamp}.csv")), dir.join(format!("runs-{stamp}.tex")));
    std::fs::write(&c, csv)?;
    std::fs::write(&t, tex)?;
    Ok((c, t))
}

pub fn results(pr: &Project, id: &str) -> Result<Value> {
    let d = run_dir(pr, id)?;
    store::read_json(&d.join("results.json")).ok_or_else(|| anyhow!("No results for this run yet."))
}

#[cfg(test)]
mod queue_tests {
    use super::*;
    use crate::dataset::Dataset;

    #[test]
    fn a_full_kaggle_waits_instead_of_failing() {
        assert!(is_session_limit("Kaggle push failed: Kernel push error: Maximum batch GPU session count of 2 reached."));
        assert!(!is_session_limit("Kaggle push failed: 400 Client Error: Bad Request"));
        let own = Dataset { source: "local".into(), local_dir: "/data/x".into(), ..Default::default() };
        assert_eq!(kaggle_attach_of(&own), own.local_key(), "the private upload of the owner's files");
        assert_eq!(kaggle_attach_of(&Dataset { source: "kaggle".into(), kaggle_slug: "a/b".into(), ..Default::default() }), "a/b");
        assert_eq!(kaggle_attach_of(&Dataset { source: "huggingface".into(), ..Default::default() }), "");
    }
}

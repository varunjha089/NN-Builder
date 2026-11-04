//! Whole runs without network or GPU: stand-in `ssh`, `rsync`, `python` and `kaggle` scripts. The "remote machine" is a scratch folder,
//! and the stand-in python pretends to train (writes the events and results a real run writes). One test function, because it changes
//! environment variables (HOME, PATH, NN_BUILDER_HOME).
#![cfg(unix)]

use nnb_core::jobs::Log;
use nnb_core::remote::{self, Machine};
use nnb_core::runs::{self, Creds, KAGGLE};
use nnb_core::cloud::KaggleCreds;
use nnb_core::store;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const FAKE_SSH: &str = r#"#!/bin/bash
# stand-in ssh: skip options; "-O check" fails; otherwise run the command with HOME = the fake remote home
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-p|-i|-J|-S|-l) shift 2 ;;
    -O) exit 255 ;;
    -*) shift ;;
    *) break ;;
  esac
done
echo "$1" >> "$NNB_FAKE_LOG"; shift
# no login shell: on macOS it would put /usr/bin before the stand-ins (and a real crontab would be changed)
if [ "$1" = bash ] && [ "$2" = -l ]; then set -- bash "${@:3}"; fi
cd "$NNB_FAKE_REMOTE" && HOME="$NNB_FAKE_REMOTE" exec "$@"
"#;

const FAKE_RSYNC: &str = r#"#!/bin/bash
# stand-in rsync: drop "-e <ssh>", map host:path to the fake remote home, then copy with the real rsync
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -e) shift 2 ;;
    *@*:*|*.*:*) p="${1#*:}"; case "$p" in /*) ;; *) p="$NNB_FAKE_REMOTE/$p" ;; esac; args+=("$p"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done
exec /usr/bin/rsync "${args[@]}"
"#;

/// stand-in for the remote PyTorch python: answers Prepare's checks and "trains" train.py by writing the files a real run writes
const FAKE_PY: &str = r#"#!/bin/bash
if [ "$1" = "-" ]; then cat >/dev/null; echo "NNB_TORCH 2.9.0 cuda 3.12.3"; exit 0; fi
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$1" = "train.py" ]; then
  grep -q "class Net(nn.Module)" train.py || exit 9
  t=$(date +%s)
  echo '{"event": "start", "device": "cuda", "params": 10, "train_size": 100, "test_size": 10, "epochs": 2}' >> metrics.jsonl
  for e in 1 2; do sleep 1; echo "{\"event\": \"epoch\", \"epoch\": $e, \"train_loss\": 0.5, \"train_acc\": 0.8, \"val_loss\": 0.4, \"val_acc\": 0.9}" >> metrics.jsonl; done
  echo '{"test_acc": 0.9, "test_loss": 0.4, "classes": ["a","b"], "confusion": [[5,0],[1,4]], "samples": [], "feature_maps": [], "filters": null}' > results.json
  echo "weights" > model.pt
  echo '{"event": "done", "test_acc": 0.9}' >> metrics.jsonl
  echo "trained"
  exit 0
fi
exit 1
"#;

/// stand-in kaggle CLI: push records the kernel, status says running then complete, output writes the results
const FAKE_KAGGLE: &str = r#"#!/bin/bash
S="$NNB_FAKE_REMOTE/kaggle_state"
[ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ] || { echo "no credentials"; exit 1; }
case "$1 $2" in
  "kernels push") cp "$4/kernel-metadata.json" "$NNB_FAKE_REMOTE/pushed.json"; grep -q "class Net" "$4/train.py" || exit 3; echo 0 > "$S"; echo "Kernel version 1 successfully pushed." ;;
  "kernels status") n=$(cat "$S"); echo $((n+1)) > "$S"; if [ "$n" -ge 1 ]; then echo "$3 has status \"KernelWorkerStatus.COMPLETE\""; else echo "$3 has status \"KernelWorkerStatus.RUNNING\""; fi ;;
  "kernels output") mkdir -p "$5"; echo '{"event": "done", "test_acc": 0.8}' > "$5/metrics.jsonl"; echo '{"test_acc": 0.8, "classes": []}' > "$5/results.json"
     echo '[{"stream_name":"stdout","time":1,"data":"training...\n"}]' > "$5/nnb-x.log" ;;
  *) echo "unexpected $*"; exit 2 ;;
esac
"#;

/// stand-in train.py for a spot instance: the first time it trains until SIGUSR1 (the interruption notice), then saves a checkpoint and
/// exits 75; started again (a checkpoint exists) it resumes and finishes
const FAKE_SPOT_PY: &str = r#"#!/bin/bash
if [ "$1" = "-" ]; then cat >/dev/null; echo "NNB_TORCH 2.9.0 cuda 3.12.3"; exit 0; fi
if [ "$1" = "-c" ]; then exit 0; fi
[ "$1" = "train.py" ] || exit 1
echo $$ > py_pid.txt
if [ -f checkpoint.pt ]; then
  echo '{"event": "resume", "epoch": 2, "step": 3, "steps": 5, "resumes": 1}' >> metrics.jsonl
  echo '{"event": "start", "device": "cuda", "params": 10, "train_size": 100, "test_size": 10, "epochs": 2}' >> metrics.jsonl
  echo '{"event": "epoch", "epoch": 2, "train_loss": 0.4, "train_acc": 0.85, "val_loss": 0.3, "val_acc": 0.9}' >> metrics.jsonl
  echo '{"test_acc": 0.9, "classes": ["a","b"], "confusion": [[5,0],[1,4]], "samples": []}' > results.json
  echo "weights" > model.pt
  echo '{"event": "done", "test_acc": 0.9}' >> metrics.jsonl
  exit 0
fi
on_notice() {
  echo '{"event": "checkpoint", "epoch": 2, "step": 3, "steps": 5, "reason": "spot interruption", "time": 1}' >> metrics.jsonl
  echo ck-epoch2-step3 > checkpoint.pt
  echo '{"event": "interrupted", "epoch": 2, "step": 3, "steps": 5}' >> metrics.jsonl
  exit 75
}
trap on_notice USR1
echo '{"event": "start", "device": "cuda", "params": 10, "train_size": 100, "test_size": 10, "epochs": 2}' >> metrics.jsonl
echo '{"event": "epoch", "epoch": 1, "train_loss": 0.5, "train_acc": 0.8, "val_loss": 0.4, "val_acc": 0.85}' >> metrics.jsonl
echo ck-epoch1 > checkpoint.pt
echo '{"event": "checkpoint", "epoch": 1, "step": 0, "steps": 5, "reason": "epoch", "time": 1}' >> metrics.jsonl
for i in $(seq 1 120); do sleep 0.5; done
exit 1
"#;

/// stand-in curl for the EC2 metadata service: the spot interruption notice is served once, after the test creates <remote>/notice
const FAKE_CURL: &str = r#"#!/bin/bash
out=/dev/stdout; url=
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2 ;; -w|-H|-m|-X) shift 2 ;; http*) url=$1; shift ;; *) shift ;; esac; done
case "$url" in
  */api/token) echo tok ;;
  */spot/instance-action)
    if [ -f "$NNB_FAKE_REMOTE/notice" ]; then rm -f "$NNB_FAKE_REMOTE/notice"; echo '{"action": "stop"}' > "$out"; printf 200; else printf 404; fi ;;
  *) printf 404 ;;
esac
"#;

/// stand-in crontab: kept in a file in the fake remote home (the real crontab is never touched)
const FAKE_CRONTAB: &str = r#"#!/bin/bash
F="$NNB_FAKE_REMOTE/crontab.txt"
case "$1" in -l) cat "$F" 2>/dev/null; exit 0 ;; -) cat > "$F"; exit 0 ;; *) exit 1 ;; esac
"#;

/// stand-in for the tools python running aws_ec2.py: a spot instance (persistent request, interruption behaviour stop)
const FAKE_AWS_PY: &str = r#"#!/bin/bash
case "$2" in
  start|describe) echo '{"ok": true, "state": "running", "host": "ec2-1-2-3-4.example.com", "type": "g4dn.xlarge", "lifecycle": "spot", "state_reason": "", "shutdown_behavior": "stop", "spot": {"type": "persistent", "interruption": "stop", "state": "active", "status": "fulfilled"}}' ;;
  stop) echo '{"ok": true, "state": "stopping", "host": ""}' ;;
  *) exit 0 ;;
esac
"#;

fn script(dir: &Path, name: &str, text: &str) {
    std::fs::write(dir.join(name), text).unwrap();
    std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn remote_and_kaggle_runs() {
    let base = tempfile::tempdir().unwrap();
    let b = base.path();
    let (bin, home, remote_home, settings) = (b.join("bin"), b.join("home"), b.join("remote"), b.join("settings"));
    for d in [&bin, &home, &remote_home, &settings] {
        std::fs::create_dir_all(d).unwrap();
    }
    script(&bin, "ssh", FAKE_SSH);
    script(&bin, "rsync", FAKE_RSYNC);
    script(&remote_home, "fakepython", FAKE_PY);
    std::env::set_var("HOME", &home);
    std::env::set_var("NN_BUILDER_HOME", &settings);
    std::env::set_var("NNB_FAKE_REMOTE", &remote_home);
    std::env::set_var("NNB_FAKE_LOG", b.join("hosts.log"));
    std::env::set_var("PATH", format!("{}:{}", bin.display(), std::env::var("PATH").unwrap()));

    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let l2 = lines.clone();
    let log: Log = Arc::new(move |s| l2.lock().unwrap().push(s));
    let pr = store::create("Flow test", Some(&b.join("projects"))).unwrap();

    // ---- an SSH machine (college style), launched with nohup so the test does not depend on tmux/systemd being installed
    let m = remote::save_machine(Machine { name: "college".into(), host: "gpu01.example.edu".into(), user: "student".into(),
        python: format!("{}/fakepython", remote_home.display()), launch: "nohup".into(), ..Default::default() }).unwrap();
    assert!(m.probe().is_ok(), "{:?}", m.probe());
    let p = runs::prepare(&m, &log).unwrap();
    assert!(p.cuda && p.torch == "2.9.0", "{p:?}\n{:?}", lines.lock().unwrap());
    assert!(remote_home.join(".nn_builder/runs").is_dir());
    let id = runs::start(&pr, "college", "0", &Creds::default(), false, &log).unwrap_or_else(|e| panic!("{e}\n{:?}", lines.lock().unwrap()));
    let mut last = None;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(1100));
        let s = runs::status(&pr, &id, 0, &Creds::default()).unwrap();
        if !["starting", "running"].contains(&s.state.as_str()) {
            last = Some(s);
            break;
        }
    }
    let s = last.expect("the run did not end");
    assert_eq!(s.state, "done", "{s:?}");
    assert_eq!(s.events.iter().filter(|e| e["event"] == "epoch").count(), 2);
    let local = pr.runs_dir().join(&id);
    assert!(local.join("results.json").exists() && local.join("model.pt").exists(), "results fetched back");
    let run_sh = std::fs::read_to_string(remote_home.join(format!(".nn_builder/runs/{id}/run.sh"))).unwrap();
    assert!(run_sh.contains("CUDA_VISIBLE_DEVICES=0") && !run_sh.contains("shutdown"));
    assert!(std::fs::read_to_string(b.join("hosts.log")).unwrap().contains("student@gpu01.example.edu"));
    assert!(runs::listing(&pr)[0]["test_acc"].as_f64() == Some(0.9));

    // ---- the same machine launching with tmux (when tmux is installed here): the run path must be quoted right
    if std::process::Command::new("tmux").arg("-V").output().map(|o| o.status.success()).unwrap_or(false) {
        let mut m2 = m.clone();
        m2.launch = "tmux".into();
        remote::save_machine(m2).unwrap();
        std::thread::sleep(Duration::from_millis(1100));
        let id = runs::start(&pr, "college", "", &Creds::default(), false, &log).unwrap_or_else(|e| panic!("{e}\n{:?}", lines.lock().unwrap()));
        assert_eq!(runs::info(&pr.runs_dir().join(&id)).launch, "tmux");
        let mut state = String::new();
        for _ in 0..40 {
            std::thread::sleep(Duration::from_millis(1100));
            state = runs::status(&pr, &id, 0, &Creds::default()).unwrap().state;
            if !["starting", "running"].contains(&state.as_str()) {
                break;
            }
        }
        assert_eq!(state, "done", "tmux run");
        assert!(pr.runs_dir().join(&id).join("results.json").exists());
    }

    // ---- Kaggle with a stand-in kaggle CLI in the tools folder
    let tools_bin = settings.join("tools/bin");
    std::fs::create_dir_all(&tools_bin).unwrap();
    script(&tools_bin, "kaggle", FAKE_KAGGLE);
    script(&tools_bin, "python", "#!/bin/bash\nexit 0\n");
    let creds = Creds { kaggle: Some(KaggleCreds { username: "someone".into(), key: "k".into() }), ..Default::default() };
    assert!(runs::start(&pr, KAGGLE, "", &Creds::default(), false, &log).is_err(), "needs the token");
    std::thread::sleep(Duration::from_millis(1100));
    let id = runs::start(&pr, KAGGLE, "", &creds, false, &log).unwrap();
    let pushed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(remote_home.join("pushed.json")).unwrap()).unwrap();
    assert_eq!(pushed["enable_gpu"], "true");
    assert!(pushed["id"].as_str().unwrap().starts_with("someone/nnb-flow-test-"));
    let mut info = runs::info(&pr.runs_dir().join(&id));
    let mut state = String::new();
    for _ in 0..3 {
        info.last_poll = 0.0; // skip the 30 s wait between Kaggle polls
        store::write_json(&pr.runs_dir().join(&id).join("run.json"), &info).unwrap();
        state = runs::status(&pr, &id, 0, &creds).unwrap().state;
        info = runs::info(&pr.runs_dir().join(&id));
        if state == "done" {
            break;
        }
    }
    assert_eq!(state, "done");
    assert!(std::fs::read_to_string(pr.runs_dir().join(&id).join("log.txt")).unwrap().contains("training"));
    assert!(runs::stop(&pr, &id).unwrap_err().to_string().contains("Cancel run"));

    // ---- an EC2 spot instance: interruption notice -> checkpoint + exit 75 -> resumed -> done; checkpoint copied to the Mac;
    //      then the run continued on another machine from that copy
    script(&bin, "curl", FAKE_CURL);
    script(&bin, "crontab", FAKE_CRONTAB);
    script(&tools_bin, "python", FAKE_AWS_PY);
    script(&remote_home, "spotpython", FAKE_SPOT_PY);
    let key = b.join("spot.pem");
    std::fs::write(&key, "not a real key").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let spot = remote::save_machine(Machine { name: "spot".into(), kind: "aws".into(), user: "ubuntu".into(), region: "ap-south-1".into(),
        instance_id: "i-0123456789abcdef0".into(), key: key.to_string_lossy().into(), python: format!("{}/spotpython", remote_home.display()),
        launch: "nohup".into(), mirror: true, ..Default::default() }).unwrap();
    let mut probe = spot.clone();
    probe.host = "ec2-1-2-3-4.example.com".into();
    let (_, which) = probe.run_script("command -v crontab");
    assert!(which.trim() == bin.join("crontab").to_string_lossy(), "the remote shell must use the stand-in crontab, found {which:?}");
    let aws = Creds { aws: Some(nnb_core::cloud::AwsCreds { access_key_id: "AKIATEST".into(), secret_access_key: "x".into() }), ..Default::default() };
    // never "stop the instance" here: the stand-in instance is this Mac (run.sh would run sudo shutdown)
    pr.save_train(&nnb_core::codegen::TrainSettings { stop_instance: false, ..pr.train() }).unwrap();
    std::thread::sleep(Duration::from_millis(1100));
    let id = runs::start(&pr, "spot", "", &aws, false, &log).unwrap_or_else(|e| panic!("{e}\n{:?}", lines.lock().unwrap()));
    let rdir = remote_home.join(format!(".nn_builder/runs/{id}"));
    assert!(runs::info(&pr.runs_dir().join(&id)).spot);
    assert!(std::fs::read_to_string(remote_home.join("crontab.txt")).unwrap().contains("resume.sh # nn_builder_resume"), "resume at boot registered");
    let run_sh = std::fs::read_to_string(rdir.join("run.sh")).unwrap();
    assert!(!run_sh.contains("shutdown"), "no shutdown in a test");
    assert!(run_sh.contains("NNB_CHECKPOINT_MINUTES=10") && run_sh.contains("spot_watch") && run_sh.contains("touch resume_wanted"));
    // wait until it trains (checkpoint of epoch 1 written), then AWS gives notice
    for _ in 0..40 {
        if rdir.join("checkpoint.pt").exists() { break; }
        std::thread::sleep(Duration::from_millis(250));
    }
    std::fs::write(remote_home.join("notice"), "").unwrap();
    let (mut states, mut last) = (vec![], None);
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(1100));
        let s = runs::status(&pr, &id, 0, &aws).unwrap();
        if states.last() != Some(&s.state) { states.push(s.state.clone()); }
        if s.state == "done" { last = Some(s); break; }
    }
    let s = last.unwrap_or_else(|| panic!("the spot run did not finish: states {states:?}\nlog {:?}", std::fs::read_to_string(rdir.join("log.txt"))));
    let names: Vec<&str> = s.events.iter().filter_map(|e| e["event"].as_str()).collect();
    for want in ["checkpoint", "interrupted", "resume", "done"] {
        assert!(names.contains(&want), "{want} missing in {names:?}\nstates {states:?}\nremote log:\n{}\nmetrics:\n{}", std::fs::read_to_string(rdir.join("log.txt")).unwrap_or_default(),
            std::fs::read_to_string(rdir.join("metrics.jsonl")).unwrap_or_default());
    }
    assert!(std::fs::read_to_string(rdir.join("log.txt")).unwrap().contains("spot interruption notice"));
    assert_eq!(std::fs::read_to_string(rdir.join("resumes.txt")).unwrap().trim(), "1", "resumed exactly once");
    assert!(!rdir.join("resume_wanted").exists(), "a finished run is not resumed again");
    assert!(rdir.join("used_seconds.txt").exists(), "running time is counted over resumes");
    // the checkpoint copy on this Mac (written in the background)
    let local = pr.runs_dir().join(&id);
    for _ in 0..20 {
        if local.join("mirror.json").exists() { break; }
        std::thread::sleep(Duration::from_millis(250));
    }
    assert!(local.join("checkpoint.pt").exists(), "checkpoint copied to the Mac");

    // continue that run on the college machine from the Mac copy
    std::thread::sleep(Duration::from_millis(1100));
    let new = runs::resume_on(&pr, &id, "college", "", &Creds::default(), false, &log).unwrap_or_else(|e| panic!("{e}\n{:?}", lines.lock().unwrap()));
    assert!(remote_home.join(format!(".nn_builder/runs/{new}/checkpoint.pt")).exists(), "the checkpoint went to the new machine");
    assert_eq!(runs::info(&pr.runs_dir().join(&new)).resumed_from, id);
    assert_eq!(runs::info(&local).superseded_by, new);
    assert!(runs::resume_on(&pr, &id, runs::KAGGLE, "", &Creds::default(), false, &log).is_err(), "no resuming on Kaggle");
}

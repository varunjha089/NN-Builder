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
}

//! SSH machines: a college GPU server (password or key) or an AWS EC2 instance (key file; its address comes from the EC2 API at run time).
//! Uses the system's own ssh and rsync, so ~/.ssh/config, the agent, jump hosts and host-key checks work as in the Terminal.
//!
//! A machine stores how to reach it, never a password, key contents or passphrase. Every command shares one connection (ControlMaster):
//! for a password machine the user logs in once in a Terminal window (connect_command), and the app's commands use BatchMode (never prompting).

use crate::jobs::capture;
use crate::store;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Machine {
    pub name: String,
    /// "ssh" (college / any server) | "aws" (EC2 instance started and stopped by the app)
    #[serde(default)]
    pub kind: String,
    /// ssh: host or ~/.ssh/config alias. aws: filled in at run time from the instance's public DNS
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub user: String,
    #[serde(default)]
    pub port: String,
    /// path of a private key file (EC2 .pem); empty = ssh config / agent
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub jump: String,
    /// "key" | "password"
    #[serde(default)]
    pub auth: String,
    /// remote folder for environment, data and runs (relative to the remote home unless it starts with /)
    #[serde(default)]
    pub workdir: String,
    /// an existing Python with PyTorch on the machine; empty = the app makes <workdir>/venv
    #[serde(default)]
    pub python: String,
    /// "auto" | "slurm" | "systemd" | "tmux" | "nohup"
    #[serde(default)]
    pub launch: String,
    #[serde(default)]
    pub region: String,
    #[serde(default)]
    pub instance_id: String,
    /// copy every new checkpoint of a run back to this Mac while the app is open (for an EC2 spot instance that AWS may terminate, the Mac
    /// copy is what "Resume on another machine" continues from)
    #[serde(default)]
    pub mirror: bool,
}

/// what Prepare found on the machine
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Prepared {
    pub host_info: String,
    pub gpus: Vec<String>,
    pub python: String,
    pub torch: String,
    pub cuda: bool,
    pub slurm: bool,
    pub systemd: bool,
    pub linger: bool,
    pub tmux: bool,
    pub timeout: bool,
    pub fail: String,
}

fn safe(s: &str, extra: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || extra.contains(c))
}

impl Machine {
    /// validate what the window sends; Err with a plain-words reason
    pub fn clean(mut self) -> Result<Machine> {
        for f in [&mut self.name, &mut self.host, &mut self.user, &mut self.port, &mut self.key, &mut self.jump, &mut self.workdir, &mut self.python,
            &mut self.region, &mut self.instance_id] {
            *f = f.trim().to_string();
        }
        if self.name.is_empty() || self.name.len() > 40 || !safe(&self.name, " -_.") {
            bail!("Give the machine a short name (letters, digits, spaces, - _ .).");
        }
        if self.name.eq_ignore_ascii_case("kaggle") {
            bail!("The name Kaggle is used for Kaggle notebooks.");
        }
        self.kind = if self.kind == "aws" { "aws".into() } else { "ssh".into() };
        if self.kind == "ssh" && (self.host.is_empty() || !safe(&self.host, "-_.:[]")) {
            bail!("Host: a host name, an IP address, or an alias from ~/.ssh/config.");
        }
        if self.kind == "aws" {
            if !(self.instance_id.starts_with("i-") && safe(&self.instance_id, "-")) {
                bail!("Instance id: like i-0123456789abcdef0 (EC2 console, Instances).");
            }
            if self.region.is_empty() || !safe(&self.region, "-") {
                bail!("Region: like ap-south-1.");
            }
            if self.key.is_empty() {
                bail!("Key file: the .pem file of the instance's key pair.");
            }
            if self.user.is_empty() {
                self.user = "ubuntu".into();
            }
            self.host.clear();
        }
        if !safe(&self.user, "-_.") || !safe(&self.jump, "-_.@:[]") {
            bail!("User and jump host may contain letters, digits and - _ . @ : only.");
        }
        if !self.port.is_empty() && !self.port.parse::<u16>().map(|p| p > 0).unwrap_or(false) {
            bail!("Port must be a number (usually 22).");
        }
        self.auth = if self.auth == "password" && self.kind == "ssh" { "password".into() } else { "key".into() };
        if let Some(rest) = self.key.strip_prefix("~/") {
            self.key = format!("{}/{rest}", std::env::var("HOME").unwrap_or_default());
        }
        if self.workdir.is_empty() {
            self.workdir = ".nn_builder".into();
        }
        if !safe(&self.workdir, "/._-~") || self.workdir.contains("..") {
            bail!("Remote folder: letters, digits and / . _ - only.");
        }
        if !self.python.is_empty() && (!safe(&self.python, "/._-~ ") || self.python.contains("..")) {
            bail!("Remote Python: a path with letters, digits, spaces and / . _ - only.");
        }
        if !["auto", "slurm", "systemd", "tmux", "nohup"].contains(&self.launch.as_str()) {
            self.launch = "auto".into();
        }
        Ok(self)
    }

    /// None, or why ssh would refuse the key file
    pub fn key_problem(&self) -> Option<String> {
        if self.key.is_empty() {
            return None;
        }
        let p = PathBuf::from(&self.key);
        let meta = match std::fs::metadata(&p) {
            Ok(m) => m,
            Err(_) => return Some(format!("The key file {} does not exist.", p.display())),
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o077 != 0 {
                return Some(format!("Other users can read the key file; ssh will refuse it. Run in Terminal: chmod 600 '{}'", p.display()));
            }
        }
        let _ = meta;
        None
    }

    pub fn dest(&self) -> String {
        if self.user.is_empty() { self.host.clone() } else { format!("{}@{}", self.user, self.host) }
    }

    fn base_options(&self) -> Vec<String> {
        let mut o = vec![];
        if !self.port.is_empty() {
            o.extend(["-p".into(), self.port.clone()]);
        }
        if !self.key.is_empty() {
            o.extend(["-i".into(), self.key.clone(), "-o".into(), "IdentitiesOnly=yes".into()]);
        }
        if !self.jump.is_empty() {
            o.extend(["-J".into(), self.jump.clone()]);
        }
        if self.kind == "aws" {
            // the instance gets a new address (and host key entry) after every start
            o.extend(["-o".into(), "StrictHostKeyChecking=accept-new".into()]);
        }
        o
    }

    pub fn ssh_options(&self) -> Vec<String> {
        // connection sharing (ControlMaster) is not available in Windows' OpenSSH: every command opens its own connection there
        let shared: &[&str] = if cfg!(windows) { &[] } else { &["-o", "ControlMaster=auto", "-o", "ControlPersist=15m"] };
        let mut o: Vec<String> = shared.iter().map(|s| s.to_string()).collect();
        if !cfg!(windows) {
            o.extend(["-o".into(), format!("ControlPath={}", control_path())]);
        }
        o.extend(["-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes"].iter().map(|s| s.to_string()));
        o.extend(self.base_options());
        o
    }

    /// copy the contents of a local folder into a folder on the machine (relative to its home unless absolute): rsync when this
    /// computer has it, else a tar stream over ssh (Windows 10/11 have tar and OpenSSH but no rsync)
    pub fn upload_dir(&self, local: &std::path::Path, remote: &str) -> (i32, String) {
        if have_rsync() {
            return capture(&mut self.rsync_command(&format!("{}/", local.display()), &format!("REMOTE:{remote}/")), None);
        }
        let tar = match Command::new("tar").arg("-cf").arg("-").arg("-C").arg(local).arg(".").output() {
            Ok(o) if o.status.success() => o.stdout,
            Ok(o) => return (1, format!("tar: {}", String::from_utf8_lossy(&o.stderr))),
            Err(e) => return (1, format!("tar is not available: {e}")),
        };
        let rp = remote_path(remote);
        pipe(Command::new("ssh").args(self.ssh_options()).arg(self.dest()).arg(format!("mkdir -p {rp} && tar -xf - -C {rp}")), &tar).0
    }

    /// copy one file from the machine to this computer (written to a temporary file first, so the local copy is always whole)
    pub fn download_file(&self, remote: &str, local: &std::path::Path) -> (i32, String) {
        if have_rsync() {
            let existed = local.exists();
            let r = capture(&mut self.rsync_command(&format!("REMOTE:{remote}"), &local.to_string_lossy()), None);
            // macOS's openrsync leaves an empty folder named like the file when the file is missing (seen copying locally): remove it,
            // or the next copy would land inside it
            if r.0 != 0 && !existed && local.is_dir() {
                let _ = std::fs::remove_dir(local);
            }
            return r;
        }
        let ((code, err), bytes) = pipe(Command::new("ssh").args(self.ssh_options()).arg(self.dest()).arg(format!("cat {}", remote_path(remote))), &[]);
        if code != 0 {
            return (code, err);
        }
        let tmp = local.with_extension("part");
        match std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, local)) {
            Ok(()) => (0, String::new()),
            Err(e) => (1, e.to_string()),
        }
    }

    /// `ssh ... bash -l -s`: runs the script given on stdin in a login shell (the user's PATH, conda, modules)
    pub fn script_command(&self) -> Command {
        let mut c = Command::new("ssh");
        c.args(self.ssh_options()).arg(self.dest()).args(["bash", "-l", "-s"]);
        c
    }

    pub fn run_script(&self, script: &str) -> (i32, String) {
        capture(&mut self.script_command(), Some(script))
    }

    /// rsync over the same options; a remote path is written "REMOTE:path"
    pub fn rsync_command(&self, src: &str, dst: &str) -> Command {
        let e = std::iter::once("ssh".to_string()).chain(self.ssh_options()).map(|a| sh_quote(&a)).collect::<Vec<_>>().join(" ");
        let fix = |p: &str| p.strip_prefix("REMOTE:").map(|r| format!("{}:{r}", self.dest())).unwrap_or_else(|| p.to_string());
        let mut c = Command::new("rsync");
        c.args(["-az", "-e", &e, &fix(src), &fix(dst)]);
        c
    }

    /// the command the user runs once in a Terminal window to log in (password machines); it stays in the background (ControlPersist)
    pub fn connect_command(&self) -> Vec<String> {
        let mut c: Vec<String> = ["ssh", "-M", "-f", "-N", "-o", &format!("ControlPath={}", control_path()), "-o", "ControlPersist=6h", "-o", "ServerAliveInterval=30"]
            .iter().map(|s| s.to_string()).collect();
        c.extend(self.base_options());
        c.push(self.dest());
        c
    }

    pub fn connected(&self) -> bool {
        let mut c = Command::new("ssh");
        c.args(["-o", &format!("ControlPath={}", control_path()), "-O", "check"]).args(self.base_options()).arg(self.dest());
        capture(&mut c, None).0 == 0
    }

    pub fn probe(&self) -> Result<String, String> {
        if let Some(p) = self.key_problem() {
            return Err(p);
        }
        let (code, out) = self.run_script("echo NNB_OK; uname -sm");
        if code == 0 && out.contains("NNB_OK") {
            return Ok(format!("Connected: {}", out.split("NNB_OK").nth(1).unwrap_or("").trim()));
        }
        if self.auth == "password" && !self.connected() {
            return Err("Not logged in yet: press Connect and log in once in the Terminal window.".into());
        }
        Err(format!("Could not connect: {}", out.trim().lines().last().unwrap_or("no answer")))
    }

    /// the remote folder as a shell expression ("$HOME"/... for relative paths)
    pub fn workdir_sh(&self) -> String {
        remote_path(&self.workdir)
    }
}

/// a remote path for the shell: relative -> "$HOME"/'rel', ~/x -> "$HOME"/'x', absolute -> quoted
pub fn remote_path(p: &str) -> String {
    let rel = p.strip_prefix("~/").unwrap_or(p);
    if rel.starts_with('/') { sh_quote(rel) } else { format!("\"$HOME\"/{}", sh_quote(rel)) }
}

/// rsync on this computer (macOS and Linux have it; Windows does not). NNB_NO_RSYNC=1 forces the tar / cat way (tests).
pub fn have_rsync() -> bool {
    std::env::var_os("NNB_NO_RSYNC").is_none()
        && Command::new("rsync").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// run a command with bytes on its stdin; ((exit code, stderr), stdout bytes)
fn pipe(cmd: &mut Command, input: &[u8]) -> ((i32, String), Vec<u8>) {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = match cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return ((1, e.to_string()), vec![]),
    };
    let mut stdin = child.stdin.take().unwrap();
    let data = input.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let out = child.wait_with_output();
    let _ = writer.join();
    match out {
        Ok(o) => ((o.status.code().unwrap_or(1), String::from_utf8_lossy(&o.stderr).into_owned()), o.stdout),
        Err(e) => ((1, e.to_string()), vec![]),
    }
}

pub fn sh_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:@,%+".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// short socket path (Unix sockets are limited to ~104 characters)
pub fn control_path() -> String {
    let d = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".ssh");
    if !d.exists() {
        // only a folder the app makes gets its permissions set; an existing ~/.ssh is left as it is
        let _ = std::fs::create_dir_all(&d);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700));
        }
    }
    d.join("nnb-%C").to_string_lossy().into()
}

/// checks the machine, installs PyTorch (no sudo) and tests what can keep a job running after logout. NNB_ lines are read by parse_prepare.
pub fn prepare_script(m: &Machine) -> String {
    let py = if m.python.is_empty() { "PY=\"\"".to_string() } else { format!("PY={}", remote_path(&m.python)) };
    format!(
        r#"set -u
W={w}
mkdir -p "$W/runs" "$W/data" || {{ echo "NNB_FAIL cannot make $W"; exit 1; }}
echo "NNB_HOST $(hostname) $(uname -sm)"
DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/NNB_GPU /'
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
fi
command -v sbatch >/dev/null 2>&1 && echo "NNB_SLURM 1"
command -v tmux >/dev/null 2>&1 && echo "NNB_TMUX 1"
command -v timeout >/dev/null 2>&1 && echo "NNB_TIMEOUT 1"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  echo "NNB_SYSTEMD 1"
  [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ] && echo "NNB_LINGER 1"
fi
{py}
if [ -z "$PY" ] && [ -x /opt/pytorch/bin/python ]; then
  PY=/opt/pytorch/bin/python
  echo "Using the PyTorch of the AWS Deep Learning AMI (/opt/pytorch)"
fi
if [ -z "$PY" ]; then
  if [ ! -x "$W/venv/bin/python" ]; then
    BASE=""
    for c in python3.12 python3.11 python3.13 python3.10 python3; do
      if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,13) else 1)'; then BASE=$c; break; fi
    done
    [ -z "$BASE" ] && {{ echo "NNB_FAIL no Python 3.9-3.13 on this machine: give the path of an existing Python with PyTorch (machine settings)"; exit 1; }}
    echo "Making $W/venv with $BASE ..."
    $BASE -m venv "$W/venv" || {{ echo "NNB_FAIL could not make a venv (python3-venv missing?): give the path of an existing Python with PyTorch"; exit 1; }}
  fi
  PY="$W/venv/bin/python"
fi
# a venv's activate script may set what its PyTorch needs (the Deep Learning AMI keeps CUDA inside /opt/pytorch); run.sh does the same
if [ -f "$(dirname "$PY")/activate" ]; then set +u; . "$(dirname "$PY")/activate"; set -u; fi
if ! "$PY" -c 'import torch, torchvision, datasets, kaggle' >/dev/null 2>&1; then
  EXTRA=""
  if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 525 ] 2>/dev/null; then EXTRA="--index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple"; echo "Older NVIDIA driver ($DRIVER): CUDA 11.8 build of PyTorch"; fi
  echo "Installing PyTorch, torchvision, datasets, kaggle (a few minutes) ..."
  # (an environment this user may not write to, e.g. the Deep Learning AMI's /opt/pytorch: with sudo when it needs no password)
  "$PY" -m pip install --disable-pip-version-check -q $EXTRA torch torchvision datasets pillow kaggle \
    || sudo -n "$PY" -m pip install --disable-pip-version-check -q $EXTRA torch torchvision datasets pillow kaggle \
    || {{ echo "NNB_FAIL pip could not install the packages (no internet on this machine? a proxy? no right to write into $PY's environment?)"; exit 1; }}
fi
echo "NNB_PYTHON $PY"
"$PY" - <<'PYEOF'
import sys, torch
d = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(64, 32, device=d, requires_grad=True)
(x @ x.T).sum().backward()
print("NNB_TORCH", torch.__version__, d, sys.version.split()[0])
PYEOF
"#,
        w = m.workdir_sh(),
        py = py
    )
}

pub fn parse_prepare(lines: &[String]) -> Prepared {
    let mut p = Prepared::default();
    for l in lines {
        let rest = |k: &str| l.strip_prefix(k).map(|r| r.trim().to_string());
        if let Some(v) = rest("NNB_GPU ") {
            p.gpus.push(v);
        } else if let Some(v) = rest("NNB_PYTHON ") {
            p.python = v;
        } else if let Some(v) = rest("NNB_TORCH ") {
            let parts: Vec<&str> = v.split_whitespace().collect();
            p.torch = parts.first().unwrap_or(&"").to_string();
            p.cuda = parts.get(1) == Some(&"cuda");
        } else if let Some(v) = rest("NNB_FAIL ") {
            p.fail = v;
        } else if let Some(v) = rest("NNB_HOST ") {
            p.host_info = v;
        } else if l.starts_with("NNB_SLURM") {
            p.slurm = true;
        } else if l.starts_with("NNB_TMUX") {
            p.tmux = true;
        } else if l.starts_with("NNB_SYSTEMD") {
            p.systemd = true;
        } else if l.starts_with("NNB_LINGER") {
            p.linger = true;
        } else if l.starts_with("NNB_TIMEOUT") {
            p.timeout = true;
        }
    }
    p
}

/// how to keep the job running without the ssh connection: the machine's setting, or the best the machine allows
pub fn choose_launch(m: &Machine, p: &Prepared) -> String {
    match m.launch.as_str() {
        "slurm" | "systemd" | "tmux" | "nohup" => m.launch.clone(),
        _ if p.slurm => "slurm".into(),
        _ if p.systemd && p.linger => "systemd".into(),
        _ if p.tmux => "tmux".into(),
        _ => "nohup".into(),
    }
}

pub fn launch_note(method: &str, p: &Prepared) -> &'static str {
    match method {
        "slurm" => "SLURM job: runs in the cluster's queue, independent of your login.",
        "systemd" => "systemd user service with lingering: keeps running after logout.",
        "tmux" if !p.linger && p.systemd => "tmux session. This server may stop all your processes when you log out (systemd without lingering): if runs vanish, ask the administrator to enable lingering (loginctl enable-linger) or use SLURM.",
        "tmux" => "tmux session: keeps running after the ssh connection closes.",
        _ => "nohup + setsid: keeps running after the ssh connection closes (unless the server stops all processes at logout).",
    }
}

// ---------------------------------------------------------------- machines in settings.json
pub fn machines() -> Vec<Machine> {
    serde_json::from_value(store::settings()["machines"].clone()).unwrap_or_default()
}

pub fn machine(name: &str) -> Option<Machine> {
    machines().into_iter().find(|m| m.name == name)
}

pub fn save_machine(m: Machine) -> Result<Machine> {
    let m = m.clean()?;
    let mut all: Vec<Machine> = machines().into_iter().filter(|x| x.name != m.name).collect();
    all.push(m.clone());
    store::save_setting("machines", serde_json::to_value(&all)?)?;
    Ok(m)
}

pub fn delete_machine(name: &str) -> Result<()> {
    let all: Vec<Machine> = machines().into_iter().filter(|x| x.name != name).collect();
    store::save_setting("machines", serde_json::to_value(&all)?)?;
    let mut prep = store::settings()["prepared"].clone();
    if let Some(o) = prep.as_object_mut() {
        o.remove(name);
    }
    store::save_setting("prepared", prep)
}

pub fn prepared(name: &str) -> Option<Prepared> {
    serde_json::from_value(store::settings()["prepared"][name].clone()).ok()
}

pub fn save_prepared(name: &str, p: &Prepared) -> Result<()> {
    let mut all = store::settings()["prepared"].clone();
    if !all.is_object() {
        all = json!({});
    }
    all[name] = serde_json::to_value(p)?;
    store::save_setting("prepared", all)
}

/// a secret file written on the machine through ssh stdin (never on a command line), readable only by the user
pub fn put_secret_script(path_under_home: &str, content: &str) -> String {
    format!(
        "umask 077\nF=\"$HOME\"/{p}\nmkdir -p \"$(dirname \"$F\")\"\ncat > \"$F\" <<'NNB_SECRET_END'\n{content}\nNNB_SECRET_END\nchmod 600 \"$F\"\necho NNB_OK\n",
        p = sh_quote(path_under_home)
    )
}

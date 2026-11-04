//! One background job at a time (install the tools, prepare a machine, start a run on AWS) with a log the window polls, and the helper
//! that runs an external command and streams its output.

use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Default)]
struct State {
    name: String,
    state: String,
    lines: Vec<String>,
    message: String,
    started: Option<Instant>,
}

#[derive(Clone, Default)]
pub struct Jobs(Arc<Mutex<State>>);

#[derive(Serialize)]
pub struct Snapshot {
    pub name: String,
    pub state: String,
    pub message: String,
    pub log: Vec<String>,
    pub next: usize,
    pub seconds: u64,
}

pub type Log = Arc<dyn Fn(String) + Send + Sync>;

impl Jobs {
    pub fn snapshot(&self, since: usize) -> Snapshot {
        let s = self.0.lock().unwrap();
        Snapshot {
            name: s.name.clone(),
            state: if s.state.is_empty() { "idle".into() } else { s.state.clone() },
            message: s.message.clone(),
            log: s.lines.iter().skip(since).cloned().collect(),
            next: s.lines.len(),
            seconds: s.started.map(|t| t.elapsed().as_secs()).unwrap_or(0),
        }
    }

    /// run f(log) in a thread; returns false if a job is already running
    pub fn start<F>(&self, name: &str, f: F) -> bool
    where
        F: FnOnce(Log) -> Result<String, String> + Send + 'static,
    {
        {
            let mut s = self.0.lock().unwrap();
            if s.state == "running" {
                return false;
            }
            *s = State { name: name.into(), state: "running".into(), started: Some(Instant::now()), ..Default::default() };
        }
        let me = self.0.clone();
        let log_to = self.0.clone();
        let log: Log = Arc::new(move |l: String| log_to.lock().unwrap().lines.push(l));
        std::thread::spawn(move || {
            let r = f(log);
            let mut s = me.lock().unwrap();
            match r {
                Ok(m) => {
                    s.state = "done".into();
                    s.message = m;
                }
                Err(m) => {
                    s.state = "failed".into();
                    s.message = m;
                }
            }
        });
        true
    }
}

/// run a command, send every output line to log, write `stdin` to its input; returns (exit code, all output lines)
pub fn stream(cmd: &mut Command, stdin: Option<&str>, log: &Log) -> (i32, Vec<String>) {
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            log(format!("cannot start {:?}: {e}", cmd.get_program()));
            return (127, vec![]);
        }
    };
    if let (Some(text), Some(mut w)) = (stdin, child.stdin.take()) {
        let _ = w.write_all(text.as_bytes());
    }
    let lines = Arc::new(Mutex::new(vec![]));
    let err = child.stderr.take().map(|e| {
        let (log, lines) = (log.clone(), lines.clone());
        std::thread::spawn(move || {
            for l in BufReader::new(e).lines().map_while(Result::ok) {
                log(l.clone());
                lines.lock().unwrap().push(l);
            }
        })
    });
    if let Some(o) = child.stdout.take() {
        for l in BufReader::new(o).lines().map_while(Result::ok) {
            log(l.clone());
            lines.lock().unwrap().push(l);
        }
    }
    if let Some(t) = err {
        let _ = t.join();
    }
    let code = child.wait().map(|s| s.code().unwrap_or(1)).unwrap_or(1);
    let out = lines.lock().unwrap().clone();
    (code, out)
}

/// run a command quietly with optional stdin; (exit code, stdout + stderr)
pub fn capture(cmd: &mut Command, stdin: Option<&str>) -> (i32, String) {
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (127, format!("cannot start {:?}: {e}", cmd.get_program())),
    };
    if let (Some(text), Some(mut w)) = (stdin, child.stdin.take()) {
        let _ = w.write_all(text.as_bytes());
    }
    match child.wait_with_output() {
        Ok(o) => (o.status.code().unwrap_or(1), String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr)),
        Err(e) => (1, e.to_string()),
    }
}

// The window: top bar with the tabs (Connection set apart on the right), the project, background tasks and messages.
import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { toggleTheme, useTheme } from "./theme";
import { api, errText, type Analysis, type AppState, type Dataset, type Graph, type JobSnap, type Template, type TrainSettings, type UpdateCheck } from "./api";
import { Design, arrange } from "./Design";
import { DatasetTab } from "./DatasetTab";
import { TrainTab } from "./TrainTab";
import { ResultsTab } from "./ResultsTab";
import { ConnectionTab } from "./ConnectionTab";

type Tab = "project" | "design" | "dataset" | "code" | "train" | "results" | "connection";

export default function App() {
  const [st, setSt] = useState<AppState | null>(null);
  const [tab, setTab] = useState<Tab>("design");
  const theme = useTheme();
  const [saving, setSaving] = useState(false);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [ds, setDsState] = useState<Dataset | null>(null);
  const [dsProblems, setDsProblems] = useState<string[]>([]);
  const [train, setTrainState] = useState<TrainSettings | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const [job, setJob] = useState<JobSnap | null>(null);
  const [jobLog, setJobLog] = useState<string[]>([]);
  const [update, setUpdate] = useState<UpdateCheck["release"] | null>(null);
  const timers = useRef<Record<string, number>>({});
  // Undo / Redo of the design: snapshots before each change; changes in quick succession (typing a number) count as one step
  const past = useRef<Graph[]>([]), future = useRef<Graph[]>([]), lastPush = useRef(0);
  const graphRef = useRef<Graph | null>(null);
  graphRef.current = graph;
  const remember = useCallback((prev: Graph) => {
    const now = Date.now();
    if (now - lastPush.current > 600 || past.current.length === 0) {
      past.current.push(prev);
      if (past.current.length > 100) past.current.shift();
    }
    lastPush.current = now;
    future.current = [];
  }, []);
  const applyGraph = useCallback((g: Graph) => {
    setGraph(g);
    api.saveGraph(g).then(setAnalysis).catch(() => {});
  }, []);

  const toast = useCallback((m: string) => {
    setMsg(m);
    window.clearTimeout(timers.current.toast);
    timers.current.toast = window.setTimeout(() => setMsg(""), Math.min(14000, 3500 + m.length * 40));
  }, []);

  const reload = useCallback(async () => {
    const s = await api.state();
    setSt(s);
    return s;
  }, []);

  const loadProject = useCallback(async (keepHistory = false) => {
    if (keepHistory && graphRef.current) { lastPush.current = 0; remember(graphRef.current); }
    else { past.current = []; future.current = []; }
    const p = await api.load();
    setGraph(p.graph); setAnalysis(p.analysis); setDsState(p.dataset); setDsProblems(p.dataset_problems); setTrainState(p.train);
    const runs = await api.runs();
    setRunId(runs[0]?.id ?? null);
  }, [remember]);

  useEffect(() => {
    reload().then((s) => (s.project ? loadProject() : setTab("project"))).catch((e) => toast(errText(e)));
    const un = listen<string>("open-tab", (e) => setTab(e.payload as Tab));
    // automatic update check: at most once a day, when switched on (the Rust side decides)
    api.updateCheck(false).then((u) => { if (u?.release?.newer) setUpdate(u.release); }).catch(() => {});
    const un2 = listen("check-update", () => {
      api.updateCheck(true).then((u) => {
        if (u?.release?.newer) setUpdate(u.release);
        else toast(`You're up to date: NN Builder ${u?.current ?? ""} is the newest version.`);
      }).catch((e) => toast(errText(e)));
    });
    const un3 = listen<string>("menu", (e) => menuRef.current(e.payload));
    const un4 = listen<string>("toast", (e) => toast(e.payload));
    const un5 = listen("project-changed", () => { reload().then(() => loadProject()).then(() => setTab("design")).catch((err) => toast(errText(err))); });
    return () => { [un, un2, un3, un4, un5].forEach((u) => u.then((f) => f())); };
  }, [reload, loadProject, toast]);

  const setDs = (d: Dataset) => {
    setDsState(d);
    window.clearTimeout(timers.current.ds);
    timers.current.ds = window.setTimeout(() => api.saveDataset(d).then((r) => { setAnalysis(r.analysis); setDsProblems(r.problems); }).catch((e) => toast(errText(e))), 250);
  };
  const setTrain = (t: TrainSettings) => {
    setTrainState(t);
    window.clearTimeout(timers.current.tr);
    timers.current.tr = window.setTimeout(() => api.saveTrain(t).catch((e) => toast(errText(e))), 400);
  };

  /** follow the background task (install tools, prepare a machine, start a run, fetch results) */
  const startJob = useCallback(() => {
    setJobLog([]);
    let since = 0;
    const tick = async () => {
      const j = await api.job(since);
      since = j.next;
      setJob(j);
      if (j.log.length) setJobLog((l) => [...l, ...j.log].slice(-400));
      if (j.state === "running") { window.setTimeout(tick, 700); return; }
      if (j.state === "done" && j.message === "UPDATE_READY") { toast("Update installed: restarting …"); window.setTimeout(() => api.updateRestart(), 800); return; }
      if (j.state === "done" && j.message.startsWith("RUN ")) { setRunId(j.message.slice(4)); setTab("train"); }
      else if (j.message) toast(j.message);
      reload();
    };
    tick();
  }, [reload, toast]);

  const typing = () => { const a = document.activeElement; return !!a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT"); };
  /** menu items handled in the window (File / Edit) */
  const menuRef = useRef<(id: string) => void>(() => {});
  menuRef.current = (id: string) => {
    const g = graphRef.current;
    if (id === "undo" || id === "redo") {
      if (typing()) { document.execCommand(id); return; }
      const [from, to] = id === "undo" ? [past.current, future.current] : [future.current, past.current];
      const target = from.pop();
      if (!target || !g) { toast(id === "undo" ? "Nothing to undo." : "Nothing to redo."); return; }
      to.push(g);
      lastPush.current = 0;
      applyGraph(target);
      if (tab !== "design") setTab("design");
    } else if (id === "arrange" && g) {
      remember(g); applyGraph(arrange(g)); setTab("design");
    } else if (id === "new-project") {
      setTab("project");
      window.setTimeout(() => document.getElementById("newProjectName")?.focus(), 50);
    } else if (id === "save-project") {
      if (st?.project?.draft) setSaving(true);
      else toast(st?.project ? "Already saved: every change is written as you make it." : "Open or create a project first.");
    } else if (id === "night") {
      toggleTheme();
    } else if (id === "export-code") {
      api.exportCode().then((p) => toast(`Saved ${p}`)).catch((err) => toast(errText(err)));
    }
  };

  if (!st) return <div className="page muted">Loading…</div>;
  const ok = !!analysis?.ok && dsProblems.length === 0;
  const tabs: [Tab, string][] = [["design", "Design"], ["dataset", "Dataset"], ["code", "Code"], ["train", "Train"], ["results", "Results"]];
  const missing = (!st.tools_ready ? 1 : 0) + (!st.secrets.kaggle && !st.machines.length ? 1 : 0);

  return (
    <div className="app">
      <header>
        <strong className="brand">NN Builder</strong>
        <button className={`tab proj ${tab === "project" ? "on" : ""}`} onClick={() => setTab("project")}>{st.project ? st.project.name : "No project"} ▾</button>
        {st.project?.draft && <><span className="pill stopped" title="A demo: nothing is in your projects folder until you save it">not saved</span>
          <button className="small primary" onClick={() => setSaving(true)}>Save…</button></>}
        <nav>{st.project && tabs.map(([t, label]) => <button key={t} className={`tab ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>{label}</button>)}</nav>
        <span className="grow" />
        {analysis && st.project && <span className={`summary ${ok ? "st-ok" : "st-error"}`}>{ok ? `${analysis.total_params.toLocaleString()} params · ready` : analysis.ok ? "dataset incomplete" : "design has problems"}</span>}
        <button className="tab theme-btn" title={theme === "night" ? "Day mode (⇧⌘N)" : "Night mode (⇧⌘N)"} onClick={toggleTheme}>{theme === "night" ? "☀" : "☾"}</button>
        <button className={`tab conn-tab ${tab === "connection" ? "on" : ""}`} onClick={() => setTab("connection")}>Connection{missing ? <span className="dot" /> : null}</button>
      </header>
      {update && <div className="updatebar">
        <strong>NN Builder {update.version} is available</strong> <span className="muted">(you have {st.version})</span>
        {update.asset ? <button className="primary" onClick={() => api.updateInstall().then(startJob).catch((e) => toast(errText(e)))}>
          Install and restart ({(update.asset.size / 1e6).toFixed(0)} MB)</button>
          : <span className="muted">no installer for this system in the release</span>}
        <button onClick={() => api.openUrl(update.page)}>What's new</button>
        <button onClick={() => setUpdate(null)}>Later</button>
      </div>}
      <main>
        {tab === "project" && <ProjectTab st={st} toast={toast} opened={async () => { await reload(); await loadProject(); setTab("design"); }} reload={reload} />}
        {tab === "design" && graph && <Design st={st} graph={graph} setGraph={setGraph} analysis={analysis} setAnalysis={setAnalysis} toast={toast}
          reloadProject={() => { reload(); loadProject(true); }} remember={remember} />}
        {tab === "dataset" && ds && <DatasetTab st={st} ds={ds} setDs={setDs} problems={dsProblems} toast={toast} />}
        {tab === "code" && <CodeTab toast={toast} />}
        {tab === "train" && train && <TrainTab st={st} ok={ok} train={train} setTrain={setTrain} runId={runId} openRun={setRunId} toast={toast} startJob={startJob} datasetSource={ds?.source ?? "builtin"} datasetTask={ds?.task ?? "classify"} />}
        {tab === "results" && <ResultsTab runId={runId} />}
        {tab === "connection" && <ConnectionTab st={st} reload={reload} toast={toast} startJob={startJob} />}
      </main>
      {job && job.name && (job.state === "running" || jobLog.length > 0) && <div className="job">
        <div className="jobhead"><strong>{job.name}</strong><span className="muted">{job.state === "running" ? `running… ${job.seconds} s` : job.state}</span>
          <button className="small" onClick={() => { setJob(null); setJobLog([]); }}>×</button></div>
        <pre>{jobLog.join("\n")}{job.state !== "running" && job.message ? `\n${job.message}` : ""}</pre>
      </div>}
      {saving && st.project?.draft && <SaveDraft st={st} close={() => setSaving(false)} toast={toast}
        saved={async () => { setSaving(false); await reload(); await loadProject(); }} />}
      {msg && <div className="toast" onClick={() => setMsg("")}>{msg}</div>}
    </div>
  );
}

/** the unsaved demo becomes a project in the projects folder */
function SaveDraft({ st, close, saved, toast }: { st: AppState; close: () => void; saved: () => void; toast: (m: string) => void }) {
  const [name, setName] = useState(st.project?.name ?? "");
  const save = () => api.projectSaveAs(name).then((m) => { toast(m); saved(); }).catch((e) => toast(errText(e)));
  return (
    <div className="modal" onClick={close}>
      <div className="card" onClick={(e) => e.stopPropagation()}>
        <h3>Save project</h3>
        <p className="row"><input autoFocus className="grow" value={name} onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") close(); }} /></p>
        <p className="muted small">Saved in {st.projects_dir}</p>
        <p className="row"><span className="grow" /><button onClick={close}>Cancel</button><button className="primary" onClick={save}>Save</button></p>
      </div>
    </div>
  );
}

function CodeTab({ toast }: { toast: (m: string) => void }) {
  const [c, setC] = useState<{ ok: boolean; code?: string; error?: string } | null>(null);
  useEffect(() => { api.code().then(setC).catch((e) => toast(errText(e))); }, [toast]);
  return (
    <div className="page">
      <h2>Code</h2>
      <p className="row"><button onClick={() => c?.code && navigator.clipboard.writeText(c.code).then(() => toast("Copied."))} disabled={!c?.ok}>Copy</button>
        <button onClick={() => api.exportCode().then((p) => toast(`Saved ${p}`)).catch((e) => toast(errText(e)))} disabled={!c?.ok}>Export train.py</button></p>
      <p className="muted">The whole training script the GPU machine runs: the model as PyTorch code, the dataset loading, the training loop with checkpoints and resume. It runs anywhere with PyTorch: <code>python train.py</code>.</p>
      {c && !c.ok && <div className="msg warn">{c.error}</div>}
      {c?.code && <pre className="code">{c.code}</pre>}
    </div>
  );
}

function ProjectTab({ st, toast, opened, reload }: { st: AppState; toast: (m: string) => void; opened: () => void; reload: () => void }) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [demos, setDemos] = useState<Template[]>([]);
  useEffect(() => { api.templates().then(setDemos).catch(() => {}); }, []);
  const fromDemo = (t: Template) => api.projectFromTemplate(t.key).then((m) => { toast(m); opened(); }).catch((e) => toast(errText(e)));
  return (
    <div className="page">
      <h2>Projects</h2>
      <h3>New project</h3>
      <p className="row"><input id="newProjectName" value={name} onChange={(e) => setName(e.target.value)} placeholder="name" />
        <button className="primary" onClick={() => api.projectNew(name).then(() => { setName(""); opened(); }).catch((e) => toast(errText(e)))}>Create</button></p>
      <p className="muted small">Made in {st.projects_dir} <button className="small" onClick={() => { const d = prompt("Folder for new projects:", st.projects_dir); if (d) api.setProjectsDir(d).then(reload); }}>change</button></p>
      <h3>Open</h3>
      <table className="grid"><tbody>
        {st.projects.map((p) => <tr key={p.path}><td>{p.name}</td><td className="muted small">{p.path}</td>
          <td>{p.exists ? <button onClick={() => api.projectOpen(p.path).then(opened).catch((e) => toast(errText(e)))}>Open</button> : "missing"}</td></tr>)}
        {st.draft && <tr><td>{st.draft.name} <span className="pill stopped">not saved{st.draft.edited ? ", changed" : ""}</span></td>
          <td className="muted small">opened demo: kept only until the next demo, or until you save it (⌘S)</td>
          <td><button onClick={() => api.projectOpen(st.draft!.path).then(opened).catch((e) => toast(errText(e)))}>Open</button></td></tr>}
        {!st.projects.length && !st.draft && <tr><td className="muted">No projects yet.</td></tr>}
      </tbody></table>
      <p className="row"><input className="grow" value={path} onChange={(e) => setPath(e.target.value)} placeholder="folder of a project (contains project.json)" />
        <button onClick={() => api.projectOpen(path).then(opened).catch((e) => toast(errText(e)))}>Open folder</button></p>
      <h3>Start from a demo architecture</h3>
      <p className="muted small">Opens the network on the canvas as an unsaved project: nothing is written to your projects folder until you save it
        (Save… or ⌘S), and the next demo replaces it. Every demo is also in the Design tab under "Famous architectures".</p>
      <div className="demos">
        {demos.map((t) => (
          <div key={t.key} className="demo">
            <div className="row"><strong>{t.title}</strong><span className="muted small">{t.year}</span>
              {t.task === "segment" && <span className="pill">segmentation</span>}</div>
            <p className="small muted">{(() => { const parts = t.about.split(". "); return (/et al$| & /.test(parts[0]) ? parts[1] : parts[0]).replace(/\.$/, ""); })()}.</p>
            <div className="row"><span className="muted small">input {t.input[1]}×{t.input[2]}×{t.input[0]}</span><span className="grow" />
              <button onClick={() => fromDemo(t)}>Open demo</button></div>
          </div>
        ))}
      </div>
      <p className="muted small">NN Builder {st.version} · settings in {st.home}. This Mac only designs; training runs on Kaggle, a college GPU or AWS.</p>
      <AutoUpdate />
    </div>
  );
}

function AutoUpdate() {
  const [on, setOn] = useState<boolean | null>(null);
  useEffect(() => { api.updateSettings().then((u) => setOn(u.auto)).catch(() => {}); }, []);
  if (on === null) return null;
  return <p className="row small"><label className="chk"><input type="checkbox" checked={on} onChange={(e) => { setOn(e.target.checked); api.setAutoUpdate(e.target.checked); }} />
    Check for updates automatically (once a day, from the GitHub releases)</label><span className="muted">· or Connection menu → Check for Updates…</span></p>;
}

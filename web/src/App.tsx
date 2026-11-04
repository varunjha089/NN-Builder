// The window: top bar with the tabs (Connection set apart on the right), the project, background tasks and messages.
import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api, errText, type Analysis, type AppState, type Dataset, type Graph, type JobSnap, type TrainSettings } from "./api";
import { Design } from "./Design";
import { DatasetTab } from "./DatasetTab";
import { TrainTab } from "./TrainTab";
import { ResultsTab } from "./ResultsTab";
import { ConnectionTab } from "./ConnectionTab";

type Tab = "project" | "design" | "dataset" | "code" | "train" | "results" | "connection";

export default function App() {
  const [st, setSt] = useState<AppState | null>(null);
  const [tab, setTab] = useState<Tab>("design");
  const [graph, setGraph] = useState<Graph | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [ds, setDsState] = useState<Dataset | null>(null);
  const [dsProblems, setDsProblems] = useState<string[]>([]);
  const [train, setTrainState] = useState<TrainSettings | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const [job, setJob] = useState<JobSnap | null>(null);
  const [jobLog, setJobLog] = useState<string[]>([]);
  const timers = useRef<Record<string, number>>({});

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

  const loadProject = useCallback(async () => {
    const p = await api.load();
    setGraph(p.graph); setAnalysis(p.analysis); setDsState(p.dataset); setDsProblems(p.dataset_problems); setTrainState(p.train);
    const runs = await api.runs();
    setRunId(runs[0]?.id ?? null);
  }, []);

  useEffect(() => {
    reload().then((s) => (s.project ? loadProject() : setTab("project"))).catch((e) => toast(errText(e)));
    const un = listen<string>("open-tab", (e) => setTab(e.payload as Tab));
    return () => { un.then((f) => f()); };
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
      if (j.state === "done" && j.message.startsWith("RUN ")) { setRunId(j.message.slice(4)); setTab("train"); }
      else if (j.message) toast(j.message);
      reload();
    };
    tick();
  }, [reload, toast]);

  if (!st) return <div className="page muted">Loading…</div>;
  const ok = !!analysis?.ok && dsProblems.length === 0;
  const tabs: [Tab, string][] = [["design", "Design"], ["dataset", "Dataset"], ["code", "Code"], ["train", "Train"], ["results", "Results"]];
  const missing = (!st.tools_ready ? 1 : 0) + (!st.secrets.kaggle && !st.machines.length ? 1 : 0);

  return (
    <div className="app">
      <header>
        <strong className="brand">NN Builder</strong>
        <button className={`tab proj ${tab === "project" ? "on" : ""}`} onClick={() => setTab("project")}>{st.project ? st.project.name : "No project"} ▾</button>
        <nav>{st.project && tabs.map(([t, label]) => <button key={t} className={`tab ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>{label}</button>)}</nav>
        <span className="grow" />
        {analysis && st.project && <span className={`summary ${ok ? "st-ok" : "st-error"}`}>{ok ? `${analysis.total_params.toLocaleString()} params · ready` : analysis.ok ? "dataset incomplete" : "design has problems"}</span>}
        <button className={`tab conn-tab ${tab === "connection" ? "on" : ""}`} onClick={() => setTab("connection")}>Connection{missing ? <span className="dot" /> : null}</button>
      </header>
      <main>
        {tab === "project" && <ProjectTab st={st} toast={toast} opened={async () => { await reload(); await loadProject(); setTab("design"); }} reload={reload} />}
        {tab === "design" && graph && <Design st={st} graph={graph} setGraph={setGraph} analysis={analysis} setAnalysis={setAnalysis} toast={toast} />}
        {tab === "dataset" && ds && <DatasetTab st={st} ds={ds} setDs={setDs} problems={dsProblems} toast={toast} />}
        {tab === "code" && <CodeTab toast={toast} />}
        {tab === "train" && train && <TrainTab st={st} ok={ok} train={train} setTrain={setTrain} runId={runId} openRun={setRunId} toast={toast} startJob={startJob} datasetSource={ds?.source ?? "builtin"} />}
        {tab === "results" && <ResultsTab runId={runId} />}
        {tab === "connection" && <ConnectionTab st={st} reload={reload} toast={toast} startJob={startJob} />}
      </main>
      {job && job.name && (job.state === "running" || jobLog.length > 0) && <div className="job">
        <div className="jobhead"><strong>{job.name}</strong><span className="muted">{job.state === "running" ? `running… ${job.seconds} s` : job.state}</span>
          <button className="small" onClick={() => { setJob(null); setJobLog([]); }}>×</button></div>
        <pre>{jobLog.join("\n")}{job.state !== "running" && job.message ? `\n${job.message}` : ""}</pre>
      </div>}
      {msg && <div className="toast" onClick={() => setMsg("")}>{msg}</div>}
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
  return (
    <div className="page">
      <h2>Projects</h2>
      <h3>New project</h3>
      <p className="row"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="name" />
        <button className="primary" onClick={() => api.projectNew(name).then(() => { setName(""); opened(); }).catch((e) => toast(errText(e)))}>Create</button></p>
      <p className="muted small">Made in {st.projects_dir} <button className="small" onClick={() => { const d = prompt("Folder for new projects:", st.projects_dir); if (d) api.setProjectsDir(d).then(reload); }}>change</button></p>
      <h3>Open</h3>
      <table className="grid"><tbody>
        {st.projects.map((p) => <tr key={p.path}><td>{p.name}</td><td className="muted small">{p.path}</td>
          <td>{p.exists ? <button onClick={() => api.projectOpen(p.path).then(opened).catch((e) => toast(errText(e)))}>Open</button> : "missing"}</td></tr>)}
        {!st.projects.length && <tr><td className="muted">No projects yet.</td></tr>}
      </tbody></table>
      <p className="row"><input className="grow" value={path} onChange={(e) => setPath(e.target.value)} placeholder="folder of a project (contains project.json)" />
        <button onClick={() => api.projectOpen(path).then(opened).catch((e) => toast(errText(e)))}>Open folder</button></p>
      <p className="muted small">NN Builder {st.version} · settings in {st.home}. This Mac only designs; training runs on Kaggle, a college GPU or AWS.</p>
    </div>
  );
}

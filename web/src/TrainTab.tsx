// Train on Kaggle, a college GPU or AWS: settings, start, follow (curves, log), stop, fetch; the list of runs.
import { useEffect, useRef, useState } from "react";
import { api, errText, type AppState, type RunStatus, type TrainSettings } from "./api";
import { Chart } from "./Preview";

export function TrainTab({ st, ok, train, setTrain, runId, openRun, toast, startJob, datasetSource }: {
  st: AppState; ok: boolean; train: TrainSettings; setTrain: (t: TrainSettings) => void; runId: string | null; openRun: (id: string | null) => void;
  toast: (m: string) => void; startJob: () => void; datasetSource: string;
}) {
  const [targets, setTargets] = useState<{ name: string; kind: string; prepared?: boolean }[]>([]);
  const [target, setTarget] = useState("Kaggle");
  const [gpu, setGpu] = useState("");
  const [copyTokens, setCopyTokens] = useState(false);
  const [runs, setRuns] = useState<Awaited<ReturnType<typeof api.runs>>>([]);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const events = useRef<Record<string, any>[]>([]);
  const since = useRef(0);

  useEffect(() => { api.targets().then(setTargets).catch(() => {}); api.runs().then(setRuns).catch(() => {}); }, [st]);

  // follow the open run: every 3 s while it is running (the Rust side limits how often it really asks the machine / Kaggle)
  useEffect(() => {
    events.current = []; since.current = 0; setStatus(null);
    if (!runId) return;
    let stop = false, timer = 0;
    const tick = async () => {
      try {
        const s = await api.runStatus(runId, since.current);
        events.current = [...events.current, ...s.events];
        since.current = s.next;
        if (stop) return;
        setStatus({ ...s, events: events.current });
        if (["starting", "running"].includes(s.state)) timer = window.setTimeout(tick, 3000);
        else api.runs().then(setRuns).catch(() => {});
      } catch (e) { if (!stop) toast(errText(e)); }
    };
    tick();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [runId, toast]);

  const kind = targets.find((t) => t.name === target)?.kind ?? "kaggle";
  const set = (p: Partial<TrainSettings>) => setTrain({ ...train, ...p });
  const start = async () => {
    try { await api.runStart(target, gpu, copyTokens); startJob(); toast(`Starting on ${target} …`); } catch (e) { toast(errText(e)); }
  };
  const ep = (status?.events ?? []).filter((e) => e.event === "epoch"), bt = (status?.events ?? []).filter((e) => e.event === "batch");
  const startEv = status?.events.find((e) => e.event === "start");
  const errEv = status?.events.find((e) => e.event === "error");
  const check = status?.events.find((e) => e.event === "check");
  const running = status && ["starting", "running"].includes(status.state);
  const needsToken = (datasetSource === "kaggle" && kind !== "kaggle") || datasetSource === "huggingface";

  return (
    <div className="page">
      <h2>Train</h2>
      <p className="muted">Training never runs on this Mac. Choose where: Kaggle (free GPU notebooks), or a machine from the Connection tab (college server or AWS).</p>
      <div className="form">
        <label>Train on<select value={target} onChange={(e) => setTarget(e.target.value)}>
          {targets.map((t) => <option key={t.name} value={t.name}>{t.name}{t.kind === "aws" ? " (AWS EC2)" : t.kind === "ssh" ? (t.prepared ? " (SSH)" : " (SSH, prepared on first run)") : " (Kaggle GPU)"}</option>)}
        </select></label>
        {kind !== "kaggle" && <label>GPU on the machine<input value={gpu} onChange={(e) => setGpu(e.target.value.trim())} placeholder="any (or 0, or 0,1)" /></label>}
        <label>Epochs<input type="number" min={1} max={1000} value={train.epochs} onChange={(e) => set({ epochs: Number(e.target.value) })} /></label>
        <label>Batch size<input type="number" min={1} max={8192} value={train.batch_size} onChange={(e) => set({ batch_size: Number(e.target.value) })} /></label>
        <label>Learning rate<input type="number" step={0.0001} min={0.0000001} max={1} value={train.lr} onChange={(e) => set({ lr: Number(e.target.value) })} /></label>
        <label>Optimizer<select value={train.optimizer} onChange={(e) => set({ optimizer: e.target.value })}><option value="adam">Adam</option><option value="sgd">SGD (momentum 0.9)</option></select></label>
        <label>Training images (0 = all)<input type="number" min={0} value={train.limit_train} onChange={(e) => set({ limit_train: Number(e.target.value) })} /></label>
        {kind !== "kaggle" && <label>End the run after (hours)<input type="number" min={0.1} max={72} step={0.5} value={train.max_hours} onChange={(e) => set({ max_hours: Number(e.target.value) })} /></label>}
        <label className="chk wide"><input type="checkbox" checked={train.amp} onChange={(e) => set({ amp: e.target.checked })} />Mixed precision on the GPU (faster on a T4)</label>
        {kind === "aws" && <label className="chk wide"><input type="checkbox" checked={train.stop_instance} onChange={(e) => set({ stop_instance: e.target.checked })} />Stop the EC2 instance when the run ends (saves money)</label>}
        {kind !== "kaggle" && needsToken && <label className="chk wide"><input type="checkbox" checked={copyTokens} onChange={(e) => setCopyTokens(e.target.checked)} />
          Copy my {datasetSource === "kaggle" ? "Kaggle" : "Hugging Face"} token to the machine (a file only you can read; needed for Kaggle data, and for gated Hugging Face data)</label>}
      </div>
      <p className="row"><button className="primary" onClick={start} disabled={!ok}>Train on {target} ▶</button>
        {!ok && <span className="st-error">Fix the design and the dataset first.</span>}</p>

      {status && <div className="card">
        <div className="row"><strong>Run {status.id}</strong><span className={`pill ${status.state}`}>{status.state === "running" && ep.length ? `epoch ${ep[ep.length - 1].epoch}/${status.info.epochs}` : status.state}</span>
          <span className="muted">on {status.info.target}{startEv ? ` · ${startEv.gpu ?? startEv.device}` : ""}{startEv?.train_size ? ` · ${startEv.train_size.toLocaleString()} training images` : ""}</span>
          <span className="grow" />
          {status.info.url && <button onClick={() => api.openUrl(status.info.url)}>Open on Kaggle</button>}
          {running && status.info.kind !== "kaggle" && <button onClick={() => api.runStop(status.id).then(toast).catch((e) => toast(errText(e)))}>Stop</button>}
          {status.info.kind !== "kaggle" && !running && <button onClick={() => api.runFetch(status.id).then(startJob).catch((e) => toast(errText(e)))}>Fetch results</button>}
          <button onClick={() => api.runReveal(status.id)}>Show files</button>
        </div>
        {status.info.note && <p className="muted small">{status.info.note}</p>}
        {!status.info.reachable && running && <div className="msg warn">The machine does not answer right now; the run goes on there and the app will catch up.</div>}
        {check && <div className="msg ok">{check.message}</div>}
        {errEv && <div className="msg error">{errEv.message}</div>}
        {status.info.error && <div className="msg error">{status.info.error}</div>}
        {status.state === "crashed" && <div className="msg error">The run ended without a result: see the log.</div>}
        <Chart title="Loss" series={[
          { name: "batch loss", color: "#c7d2e0", thin: true, pts: bt.map((e) => [e.epoch - 1 + e.step / e.steps, e.loss]) },
          { name: "train", color: "#2f6fde", pts: ep.map((e) => [e.epoch, e.train_loss]) },
          { name: "validation", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_loss]) }]} />
        <Chart title="Accuracy" percent series={[
          { name: "train", color: "#2f6fde", pts: ep.map((e) => [e.epoch, e.train_acc]) },
          { name: "validation", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_acc]) }]} />
        <details><summary>Log</summary><pre className="log">{status.log.join("\n")}</pre></details>
      </div>}

      <h3>Runs</h3>
      <table className="grid"><tbody>
        <tr><th>Run</th><th>Where</th><th>Data</th><th>State</th><th className="num">Test acc.</th></tr>
        {runs.map((r) => <tr key={r.id} className={`click ${r.id === runId ? "sel" : ""}`} onClick={() => openRun(r.id)}><td>{r.id}</td><td>{r.target}</td>
          <td>{r.dataset}</td><td>{r.status}</td><td className="num">{r.test_acc != null ? `${(r.test_acc * 100).toFixed(2)}%` : ""}</td></tr>)}
        {!runs.length && <tr><td className="muted">No runs yet.</td></tr>}
      </tbody></table>
    </div>
  );
}

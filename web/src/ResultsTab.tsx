// The results of a finished run: accuracy, confusion matrix, sample predictions, first-layer filters, feature maps.
import { useEffect, useState } from "react";
import { api, errText } from "./api";
import { Img } from "./Preview";

export function ResultsTab({ runId }: { runId: string | null }) {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setR(null); setErr("");
    if (runId) api.runResults(runId).then(setR).catch((e) => setErr(errText(e)));
  }, [runId]);
  if (!runId) return <div className="page"><p className="muted">Open a run in the Train tab; when it has finished, its results are shown here.</p></div>;
  if (err || !r) return <div className="page"><p className="muted">Run {runId}: {err || "loading…"}</p></div>;
  const classes: string[] = r.classes ?? [];
  const conf: number[][] = r.confusion ?? [];
  const maxv = Math.max(1, ...conf.flat());
  return (
    <div className="page">
      <h2>Results of run {runId}</h2>
      <div className="row cards">
        <div className="card"><div className="muted">Test accuracy</div><div className="big">{(r.test_acc * 100).toFixed(2)}%</div></div>
        {r.test_loss != null && <div className="card"><div className="muted">Test loss</div><div className="big">{Number(r.test_loss).toFixed(4)}</div></div>}
        <button onClick={() => api.runReveal(runId)}>Show files (model.pt, train.py, results.json)</button>
      </div>
      {conf.length > 0 && <>
        <h3>Confusion matrix (rows: true class, columns: predicted)</h3>
        <table className="grid cm"><tbody>
          <tr><th></th>{classes.map((c, j) => <th key={j} title={c}>{j}</th>)}<th>class acc.</th></tr>
          {conf.map((row, i) => {
            const sum = row.reduce((a, b) => a + b, 0);
            return <tr key={i}><th title={classes[i]}>{i} {String(classes[i] ?? "").slice(0, 14)}</th>
              {row.map((v, j) => <td key={j} style={{ background: `rgba(${i === j ? "46,125,50" : "198,40,40"},${(v / maxv) * 0.8})` }}>{v}</td>)}
              <td className="num">{sum ? `${((row[i] / sum) * 100).toFixed(1)}%` : ""}</td></tr>;
          })}
        </tbody></table>
      </>}
      {r.samples?.length > 0 && <><h3>Sample predictions</h3><div className="thumbs">
        {r.samples.map((s: any, i: number) => <div key={i} className={`thumb ${s.pred !== s.label ? "wrong" : ""}`} title={`true: ${classes[s.label]}`}>
          <Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} /><div>{String(classes[s.pred] ?? s.pred).slice(0, 12)} {(s.prob * 100).toFixed(0)}%</div></div>)}
      </div></>}
      {r.filters && <><h3>Learned filters of the first convolution ({r.filters.kernel.join("×")})</h3><div className="thumbs">
        {r.filters.images.map((f: number[][][], i: number) => <Img key={i} img={f} scale={Math.max(3, Math.round(28 / f[0].length))} />)}
      </div></>}
      {r.feature_maps?.length > 0 && <><h3>Feature maps of the first test image (first 8 channels)</h3>
        {r.feature_maps.map((fm: any, i: number) => <div key={i}><p className="muted small">{fm.layer} ({fm.channels} channels)</p><div className="thumbs">
          {fm.maps.map((m: number[][], j: number) => <Img key={j} img={[m]} scale={Math.max(1, Math.round(48 / m.length))} />)}</div></div>)}
      </>}
    </div>
  );
}

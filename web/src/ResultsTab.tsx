// The results of a finished run: accuracy, confusion matrix, sample predictions, first-layer filters, feature maps.
import { useEffect, useRef, useState } from "react";
import { api, errText } from "./api";
import { Img } from "./Preview";

export function ResultsTab({ runId }: { runId: string | null }) {
  const [r, setR] = useState<any>(null);
  const [cam, setCam] = useState(false);
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
      {r.task === "segment" && <p className="muted small">"Test accuracy" is the share of pixels classified correctly; mean IoU (overlap of predicted and true areas per class) is the usual segmentation measure.</p>}
      <div className="row cards">
        <div className="card"><div className="muted">Test accuracy</div><div className="big">{(r.test_acc * 100).toFixed(2)}%</div></div>
        {r.test_top5 != null && <div className="card"><div className="muted">Top-5 accuracy</div><div className="big">{(r.test_top5 * 100).toFixed(2)}%</div></div>}
        {r.miou != null && <div className="card"><div className="muted">Mean IoU</div><div className="big">{(r.miou * 100).toFixed(2)}%</div></div>}
        {r.test_loss != null && <div className="card"><div className="muted">Test loss</div><div className="big">{Number(r.test_loss).toFixed(4)}</div></div>}
        <button onClick={() => api.runReveal(runId)}>Show files (model.pt, train.py, results.json)</button>
      </div>
      {r.iou && <table className="grid"><tbody><tr><th>Class</th><th className="num">IoU</th></tr>
        {r.iou.map((v: number, i: number) => <tr key={i}><td>{classes[i]}</td><td className="num">{(v * 100).toFixed(2)}%</td></tr>)}</tbody></table>}
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
      {r.task !== "segment" && conf.length > 0 && <PerClass classes={classes} conf={conf} />}
      {r.task === "segment" && r.samples?.length > 0 && <><h3>Sample predictions (image · true mask · predicted mask)</h3><div className="thumbs">
        {r.samples.map((s: any, i: number) => <div key={i} className="thumb seg"><div className="row">
          <Img img={s.image} scale={Math.max(1, Math.round(96 / s.image[0].length))} /><Img img={s.truth} scale={Math.max(1, Math.round(96 / s.truth[0].length))} />
          <Img img={s.pred} scale={Math.max(1, Math.round(96 / s.pred[0].length))} /></div><div>{(s.pixel_acc * 100).toFixed(1)}% of pixels right</div></div>)}
      </div></>}
      {r.task !== "segment" && r.samples?.length > 0 && <><div className="row"><h3 style={{ margin: 0 }}>Sample predictions</h3>
        {r.cam_layer && <label className="chk small"><input type="checkbox" checked={cam} onChange={(e) => setCam(e.target.checked)} />
          show Grad-CAM (where the network looked, from layer “{r.cam_layer}”)</label>}</div><div className="thumbs">
        {r.samples.map((s: any, i: number) => <div key={i} className={`thumb ${s.pred !== s.label ? "wrong" : ""}`} title={`true: ${classes[s.label]}`}>
          {cam && s.cam ? <CamImg img={s.image} cam={s.cam} size={64} /> : <Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} />}
          <div>{String(classes[s.pred] ?? s.pred).slice(0, 12)} {(s.prob * 100).toFixed(0)}%</div></div>)}
      </div>
      {cam && <p className="muted small">Red = the regions that raised the predicted class most (Grad-CAM, Selvaraju et al. 2017), blue = none.</p>}</>}
      {r.mistakes?.length > 0 && <><h3>Mistakes ({r.mistakes.length} misclassified test images)</h3><div className="thumbs">
        {r.mistakes.map((s: any, i: number) => <div key={i} className="thumb wrong" title={`test image ${s.index}`}>
          <Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} />
          <div>true: {String(classes[s.label] ?? s.label).slice(0, 10)}</div><div>said: {String(classes[s.pred] ?? s.pred).slice(0, 10)} {(s.prob * 100).toFixed(0)}%</div></div>)}
      </div><p className="muted small">The first misclassified images of the test set: they show what the network confuses (and sometimes a wrong label).</p></>}
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

/** precision (of the images predicted as a class, how many are it), recall (of the images of a class, how many were found) and F1 per
 *  class, from the confusion matrix; with their macro averages */
function PerClass({ classes, conf }: { classes: string[]; conf: number[][] }) {
  const k = conf.length;
  const rows = conf.map((row, i) => {
    const tp = row[i], actual = row.reduce((a, b) => a + b, 0), predicted = conf.reduce((a, r) => a + r[i], 0);
    const p = predicted ? tp / predicted : 0, rc = actual ? tp / actual : 0;
    return { name: classes[i] ?? String(i), p, r: rc, f1: p + rc ? (2 * p * rc) / (p + rc) : 0, n: actual };
  });
  const avg = (f: (x: (typeof rows)[number]) => number) => rows.reduce((a, x) => a + f(x), 0) / Math.max(1, k);
  const worst = [...rows].sort((a, b) => a.f1 - b.f1).slice(0, 3).map((x) => x.name);
  return <>
    <h3>Per class</h3>
    <p className="muted small">Macro average: precision {(avg((x) => x.p) * 100).toFixed(2)}%, recall {(avg((x) => x.r) * 100).toFixed(2)}%,
      F1 {(avg((x) => x.f1) * 100).toFixed(2)}%. Hardest classes (lowest F1): {worst.join(", ")}.</p>
    <details><summary>Precision, recall and F1 of every class</summary>
      <table className="grid small"><tbody><tr><th>Class</th><th className="num">Test images</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th></tr>
        {rows.map((x, i) => <tr key={i}><td>{x.name}</td><td className="num">{x.n}</td><td className="num">{(x.p * 100).toFixed(1)}%</td>
          <td className="num">{(x.r * 100).toFixed(1)}%</td><td className="num">{(x.f1 * 100).toFixed(1)}%</td></tr>)}</tbody></table></details>
  </>;
}

/** an image with its Grad-CAM heat map on top (blue → red, half transparent) */
function CamImg({ img, cam, size }: { img: number[][][]; cam: number[][]; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !img.length || !cam.length) return;
    const C = img.length, H = img[0].length, W = img[0][0].length;
    c.width = W;
    c.height = H;
    const g = c.getContext("2d")!;
    const d = g.createImageData(W, H);
    const ch = cam.length, cw = cam[0].length;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const o = (i * W + j) * 4;
      const v = (k: number) => 255 * img[Math.min(k, C - 1)][i][j];
      const t = cam[Math.min(ch - 1, Math.floor((i * ch) / H))][Math.min(cw - 1, Math.floor((j * cw) / W))];
      // a simple blue-cyan-yellow-red ramp
      const hr = Math.min(1, Math.max(0, 1.5 * t - 0.25)) * 255, hg = Math.max(0, 1 - Math.abs(2 * t - 1) * 1.2) * 255, hb = Math.min(1, Math.max(0, 1.25 - 1.5 * t)) * 255;
      d.data[o] = 0.5 * v(0) + 0.5 * hr; d.data[o + 1] = 0.5 * v(1) + 0.5 * hg; d.data[o + 2] = 0.5 * v(2) + 0.5 * hb; d.data[o + 3] = 255;
    }
    g.putImageData(d, 0, 0);
  }, [img, cam]);
  const H = img[0]?.length ?? 1;
  return <canvas ref={ref} className="pix" style={{ width: size, height: size * (H / (img[0]?.[0]?.length ?? 1)) }} />;
}

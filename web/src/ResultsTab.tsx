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
  if (r.check_only) return <DataReport runId={runId} r={r} />;
  const maxv = Math.max(1, ...conf.flat());
  return (
    <div className="page">
      <h2>Results of run {runId}</h2>
      {r.loss && <p className="muted small">Trained with {({ focal: "focal loss", dice: "Dice loss", dice_ce: "Dice + cross-entropy loss", mse: "mean squared error", l1: "L1 (mean absolute error) loss",
        huber: "Huber loss" } as Record<string, string>)[r.loss.kind] ?? r.loss.kind}
        {r.loss.kind === "focal" ? ` (γ ${r.loss.gamma})` : ""}.</p>}
      {r.class_weights && <p className="muted small">Trained with class weights{r.task === "multilabel" ? " (positive weight per class)" : ""}:
        {" "}{r.class_weights.map((w: number, i: number) => `${classes[i]} ${w}`).join(", ")}.</p>}
      {r.task === "multilabel" && <p className="muted small">Several classes per image: each class is a yes/no (score ≥ 50%). Micro F1 counts every
        (image, class) decision together, macro F1 averages the classes (rare classes count as much as common ones), mAP ranks the images
        by each class's score (independent of the 50% threshold), and exact match is the share of images with every class right.</p>}
      {r.task === "detect" && <p className="muted small">mAP@0.5: for every class, the average precision of the boxes found (sorted by
        score) when a box counts as right if it overlaps a true one by at least half (IoU ≥ 0.5); then the mean over the classes. mAP@0.5:0.95
        averages it over stricter overlaps (0.5, 0.55 … 0.95), as COCO does.</p>}
      {r.task === "regress" && <p className="muted small">R² is the share of the targets' variation the network explains (1 = perfect, 0 = no
        better than always predicting the mean, below 0 = worse); MAE is the mean absolute error and RMSE the root of the mean squared error,
        both in the target's own units. With several targets the cards show their averages.</p>}
      {r.task === "segment" && <p className="muted small">"Test accuracy" is the share of pixels classified correctly; mean IoU (overlap of predicted and true areas per class) is the usual segmentation measure.</p>}
      <div className="row cards">
        {r.task === "detect" ? <>
          <div className="card"><div className="muted">mAP@0.5</div><div className="big">{(r.map50 * 100).toFixed(2)}%</div></div>
          <div className="card"><div className="muted">mAP@0.5:0.95</div><div className="big">{(r.map * 100).toFixed(2)}%</div></div>
          <div className="card"><div className="muted">Test boxes</div><div className="big">{Number(r.boxes ?? 0).toLocaleString()}</div></div></>
        : r.task === "regress" ? <>
          <div className="card"><div className="muted">Test R²</div><div className="big">{Number(r.r2).toFixed(3)}</div></div>
          <div className="card"><div className="muted">MAE</div><div className="big">{fmt(r.mae)}</div></div>
          <div className="card"><div className="muted">RMSE</div><div className="big">{fmt(r.rmse)}</div></div></>
        : <div className="card"><div className="muted">{r.task === "multilabel" ? "Micro F1" : "Test accuracy"}</div><div className="big">{(r.test_acc * 100).toFixed(2)}%</div></div>}
        {r.task === "multilabel" && <>
          <div className="card"><div className="muted">Macro F1</div><div className="big">{(r.macro_f1 * 100).toFixed(2)}%</div></div>
          <div className="card"><div className="muted">mAP</div><div className="big">{(r.map * 100).toFixed(2)}%</div></div>
          <div className="card"><div className="muted">Exact match</div><div className="big">{(r.exact * 100).toFixed(2)}%</div></div>
        </>}
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
      {r.exits && <><h3>Early exits</h3><table className="grid"><tbody>
        <tr><th>Exit</th><th className="num">Share of the loss</th><th className="num">Test accuracy</th><th className="num">Images leaving here</th></tr>
        {r.exits.exits.map((e: any, i: number) => <tr key={i}><td>exit {e.exit}</td><td className="num">{e.weight}</td><td className="num">{(e.accuracy * 100).toFixed(2)}%</td>
          <td className="num">{(r.exits.early.share[i] * 100).toFixed(1)}%</td></tr>)}
        <tr><td>the end of the network</td><td></td><td className="num">{(r.test_acc * 100).toFixed(2)}%</td><td className="num">{(r.exits.early.share[r.exits.early.share.length - 1] * 100).toFixed(1)}%</td></tr>
      </tbody></table><p className="muted small">Stopping each image at the first exit that is at least {r.exits.early.threshold * 100}% sure gives {(r.exits.early.accuracy * 100).toFixed(2)}%
        test accuracy, with the share of images leaving at each exit shown above (the earlier they leave, the less computation).</p></>}
      {r.heads?.length > 0 && <><h3>Extra outputs</h3><table className="grid"><tbody>
        <tr><th>Output</th><th>Predicts</th><th className="num">Score</th><th>Details</th></tr>
        {r.heads.map((h: any) => <tr key={h.name}><td>{h.name}</td><td>{({ classify: "one class", multilabel: "several classes", regress: "numbers" } as Record<string, string>)[h.task]}</td>
          <td className="num">{h.metric} {h.task === "regress" ? h.score.toFixed(3) : `${(h.score * 100).toFixed(2)}%`}</td>
          <td className="small">{h.task === "regress" ? h.per_target.map((t: any) => `${t.target}: R² ${t.r2.toFixed(3)}, MAE ${fmt(t.mae)}, RMSE ${fmt(t.rmse)}`).join("; ")
            : h.task === "multilabel" ? `mAP ${(h.map * 100).toFixed(1)}%, macro F1 ${(h.macro_f1 * 100).toFixed(1)}%` : ""}</td></tr>)}
      </tbody></table><p className="muted small">The cards above are the main task's; these are the network's other outputs, on the same test images.</p></>}
      {r.task === "detect" && r.per_class && <>
        <table className="grid"><tbody><tr><th>Class</th><th className="num">Test boxes</th><th className="num">AP@0.5</th><th className="num">AP@0.5:0.95</th></tr>
          {r.per_class.map((c: any) => <tr key={c.class}><td>{c.class}</td><td className="num">{c.boxes}</td>
            <td className="num">{c.ap50 == null ? "–" : `${(c.ap50 * 100).toFixed(1)}%`}</td><td className="num">{c.ap == null ? "–" : `${(c.ap * 100).toFixed(1)}%`}</td></tr>)}
        </tbody></table>
        {r.samples?.length > 0 && <><h3>Sample detections (green: found, score ≥ 30%; white dashed: the true boxes)</h3><div className="thumbs">
          {r.samples.map((s: any, i: number) => <BoxImage key={i} s={s} classes={classes} />)}</div></>}
      </>}
      {r.task === "regress" && r.per_target && <>
        <table className="grid"><tbody>
          <tr><th>Target</th><th className="num">mean</th><th className="num">R²</th><th className="num">MAE</th><th className="num">RMSE</th></tr>
          {r.per_target.map((t: any) => <tr key={t.target}><td>{t.target}</td><td className="num">{fmt(t.mean)}</td><td className="num">{t.r2.toFixed(3)}</td>
            <td className="num">{fmt(t.mae)}</td><td className="num">{fmt(t.rmse)}</td></tr>)}</tbody></table>
        <h3>Predicted against true (test images{r.scatter?.truth?.[0]?.length >= 1000 ? ", 1000 of them" : ""}; on the diagonal = right)</h3>
        <div className="thumbs">{r.per_target.map((t: any, c: number) => <Scatter key={t.target} name={t.target} truth={r.scatter.truth[c]} pred={r.scatter.pred[c]} />)}</div>
        {r.samples?.length > 0 && <><h3>Sample predictions</h3><div className="thumbs">
          {r.samples.map((s: any, i: number) => <div key={i} className="thumb" style={{ width: 120 }}>
            <Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} />
            {s.target.map((v: number, k: number) => <div key={k} className="small">{classes.length > 1 ? `${String(classes[k]).slice(0, 10)}: ` : ""}{fmt(s.value[k])}
              <span className="muted"> (true {fmt(v)})</span></div>)}</div>)}</div></>}
        <p className="muted small">model.pt outputs standardised numbers: value = output × std + mean, with each target's mean and std in
          target_stats.json next to it.</p>
      </>}
      {r.task === "multilabel" && r.per_class && <table className="grid"><tbody>
        <tr><th>Class</th><th className="num">Images</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">AP</th>{r.thresholds && <th className="num">threshold</th>}</tr>
        {r.per_class.map((c: any, i: number) => <tr key={i}><td>{classes[i]}</td><td className="num">{c.support}</td><td className="num">{(c.precision * 100).toFixed(1)}%</td>
          <td className="num">{(c.recall * 100).toFixed(1)}%</td><td className="num">{(c.f1 * 100).toFixed(1)}%</td><td className="num">{c.ap == null ? "–" : `${(c.ap * 100).toFixed(1)}%`}</td>
          {r.thresholds && <td className="num">{(r.thresholds[i] * 100).toFixed(0)}%</td>}</tr>)}
      </tbody></table>}
      {r.task === "multilabel" && r.samples?.length > 0 && <><h3>Sample predictions (✓ right, ✗ wrong, ○ missed)</h3><div className="thumbs">
        {r.samples.map((s: any, i: number) => {
          const all = [...new Set([...s.labels, ...s.preds])].sort((a, b) => a - b);
          return <div key={i} className={`thumb ${all.some((k) => s.labels.includes(k) !== s.preds.includes(k)) ? "wrong" : ""}`} style={{ width: 120 }}>
            <Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} />
            {all.map((k) => <div key={k} className="small">{s.labels.includes(k) ? (s.preds.includes(k) ? "✓" : "○") : "✗"} {String(classes[k] ?? k).slice(0, 14)} {(s.probs[k] * 100).toFixed(0)}%</div>)}
            {all.length === 0 && <div className="small muted">no class</div>}</div>;
        })}</div></>}
      {r.task !== "segment" && r.task !== "multilabel" && r.task !== "regress" && r.samples?.length > 0 && <><div className="row"><h3 style={{ margin: 0 }}>Sample predictions</h3>
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

/** an image with its true boxes (dashed) and the boxes found (with class and score) */
function BoxImage({ s, classes }: { s: any; classes: string[] }) {
  const size = 192;
  const scale = Math.max(1, Math.round(size / s.image[0].length));
  const px = s.image[0].length * scale;
  const r = (b: number[]) => ({ x: b[0] * px, y: b[1] * px, width: Math.max(1, (b[2] - b[0]) * px), height: Math.max(1, (b[3] - b[1]) * px) });
  return <div className="thumb" style={{ width: px, position: "relative" }}>
    <Img img={s.image} scale={scale} />
    <svg width={px} height={px} style={{ position: "absolute", left: 0, top: 0 }}>
      {s.truth.map((b: number[], k: number) => <rect key={`t${k}`} {...r(b.slice(1))} fill="none" stroke="white" strokeDasharray="3 2" strokeWidth={1} />)}
      {s.boxes.map((b: number[], k: number) => <g key={`b${k}`}><rect {...r(b.slice(2))} fill="none" stroke="#39ff88" strokeWidth={1.5} />
        <text x={b[2] * px + 2} y={Math.max(9, b[3] * px - 2)} fontSize={9} fill="#39ff88">{String(classes[b[0]] ?? b[0]).slice(0, 10)} {Math.round(b[1] * 100)}%</text></g>)}
    </svg>
    <div className="small">{s.boxes.length} found · {s.truth.length} true</div>
  </div>;
}

/** a number with about 4 significant digits */
function fmt(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v)) return "–";
  return Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
}

/** regression: predicted against true value of one target, with the diagonal (a perfect prediction) */
function Scatter({ name, truth, pred }: { name: string; truth: number[]; pred: number[] }) {
  const S = 220, P = 30;
  const all = [...truth, ...pred];
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!all.length) return null;
  if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
  const X = (v: number) => P + ((v - lo) / (hi - lo)) * (S - P - 8), Y = (v: number) => S - P - ((v - lo) / (hi - lo)) * (S - P - 8);
  return <svg width={S} height={S} role="img" aria-label={`${name}: predicted against true`} style={{ background: "var(--panel)", border: "1px solid var(--line)" }}>
    <line x1={X(lo)} y1={Y(lo)} x2={X(hi)} y2={Y(hi)} stroke="var(--muted)" strokeDasharray="4 3" />
    {truth.map((t, i) => <circle key={i} cx={X(t)} cy={Y(pred[i])} r={2} fill="var(--accent)" fillOpacity={0.45} />)}
    <text x={P} y={12} fontSize={11} fill="var(--ink)">{name}</text>
    <text x={S - 8} y={S - 8} fontSize={10} fill="var(--muted)" textAnchor="end">true →</text>
    <text x={4} y={P} fontSize={10} fill="var(--muted)">↑ predicted</text>
    <text x={P} y={S - 8} fontSize={10} fill="var(--muted)">{fmt(lo)}</text>
    <text x={S - 8} y={S - 20} fontSize={10} fill="var(--muted)" textAnchor="end">{fmt(hi)}</text>
  </svg>;
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

/** a Check data run: what the run would train on, and what looks wrong */
function DataReport({ runId, r }: { runId: string; r: any }) {
  const rep = r.data_report ?? {};
  const classes: string[] = r.classes ?? [];
  const counts: number[] = rep.counts ?? [];
  const most = Math.max(1, ...counts);
  const num = (v: number | null) => (v == null ? "–" : Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(2) : v.toFixed(2));
  return <div className="page">
    <h2>Data check {runId}</h2>
    <p className="muted">The data as the run on the training machine loads it, before any training. One batch went through the network and every
      layer had the shape the design shows.</p>
    {rep.warnings?.length ? <>{rep.warnings.map((w: string) => <div key={w} className="msg warn">{w}</div>)}</>
      : <div className="msg info">Nothing looks wrong.</div>}
    <div className="row cards">
      {Object.entries(rep.sizes ?? {}).map(([k, v]) => <div key={k} className="card"><div className="muted">{k}</div><div className="big">{Number(v).toLocaleString()}</div></div>)}
    </div>
    {rep.targets ? <><h3>Targets of the training images</h3>
      <table className="grid"><tbody>
        <tr><th>Target</th><th className="num">min</th><th className="num">max</th><th className="num">mean</th><th className="num">spread</th></tr>
        {rep.targets.map((t: any) => <tr key={t.target}><td>{t.target}</td><td className="num">{num(t.min)}</td><td className="num">{num(t.max)}</td>
          <td className="num">{num(t.mean)}</td><td className="num">{num(t.std)}</td></tr>)}</tbody></table></>
    : <><h3>Training {rep.count_unit ?? "images"} per class{rep.counts_exact === false ? " (from a sample)" : ""}</h3>
    <table className="grid"><tbody>
      {counts.map((n, i) => <tr key={i}><td>{classes[i]}</td><td className="num">{n.toLocaleString()}</td>
        <td><span style={{ display: "inline-block", height: 8, width: `${(160 * n) / most}px`, background: n ? "var(--accent)" : "var(--err, #c53030)" }} /></td></tr>)}
    </tbody></table></>}
    {rep.bands?.length > 0 && <><h3>Bands (raw values in the files, from a sample of training images)</h3>
      <table className="grid"><tbody>
        <tr><th>Band</th><th className="num">min</th><th className="num">max</th><th className="num">mean</th><th className="num">spread</th><th className="num">empty</th><th className="num">zero</th></tr>
        {rep.bands.map((b: any) => <tr key={b.band}><td>{b.band}</td><td className="num">{num(b.min)}</td><td className="num">{num(b.max)}</td><td className="num">{num(b.mean)}</td>
          <td className="num">{num(b.std)}</td><td className="num">{(b.empty * 100).toFixed(1)}%</td><td className="num">{(b.zero * 100).toFixed(1)}%</td></tr>)}
      </tbody></table></>}
    {r.samples?.length > 0 && <><h3>Test images as the network gets them{r.data_report?.bands?.length ? " (shown as composites)" : ""}</h3><div className="thumbs">
      {r.samples.map((s: any, i: number) => <div key={i} className="thumb"><Img img={s.image} scale={Math.max(1, Math.round(64 / s.image[0].length))} />
        <div className="small">{r.task === "regress" ? s.label.map((v: number, k: number) => `${classes.length > 1 ? `${classes[k]} ` : ""}${num(v)}`).join(", ")
          : Array.isArray(s.label) ? s.label.map((k: number) => classes[k]).join(", ") || "no class" : s.label == null ? "" : classes[s.label]}</div></div>)}
    </div></>}
  </div>;
}

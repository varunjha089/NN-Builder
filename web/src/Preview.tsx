// Drawings: the sliding-window animation (Conv/Pool), line charts (training curves), image thumbnails (results).
import { useEffect, useRef } from "react";
import { cssVar, neon, useTheme } from "./theme";

/** the kernel window sliding over the (padded) input with the stride; each position fills one output cell */
export function SlidingPreview({ kind, n, k, s, p, d }: { kind: string; n: number; k: number; s: number; p: number; d: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const small = n + 2 * p > 14;
  const size = small ? 8 : n;
  const theme = useTheme();
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth;
    c.width = W * dpr;
    c.height = 200 * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const keff = d * (k - 1) + 1, N = size + 2 * p, out = Math.floor((N - keff) / s) + 1;
    if (out < 1) return;
    const cell = Math.max(5, Math.min(14, Math.floor((W - 30) / (N + out + 3))));
    const ox = 6, oy = 22, ox2 = ox + (N + 2) * cell;
    const color = theme === "night" ? (kind === "pool2d" ? "#00ffd5" : "#ff9a3c") : kind === "pool2d" ? "#4fb3a9" : "#f4a261";
    const [muted, cell0, padc, done, grid] = ["--muted", "--cell", "--cell-pad", "--cell-done", "--node-line"].map(cssVar);
    let step = 0;
    const draw = () => {
      g.clearRect(0, 0, W, 200);
      g.font = "11px -apple-system, sans-serif";
      g.fillStyle = muted;
      g.fillText(`input ${size}×${size}${p ? ` + padding ${p}` : ""}`, ox, 12);
      g.fillText(`output ${out}×${out}`, ox2, 12);
      const oi = Math.floor(step / out) % out, oj = step % out;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const pad = i < p || j < p || i >= N - p || j >= N - p;
        const di = i - oi * s, dj = j - oj * s;
        const inWin = di >= 0 && di < keff && dj >= 0 && dj < keff && di % d === 0 && dj % d === 0;
        g.fillStyle = inWin ? color : pad ? padc : cell0;
        g.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
        g.strokeStyle = grid;
        g.strokeRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
      }
      for (let i = 0; i < out; i++) for (let j = 0; j < out; j++) {
        g.fillStyle = i === oi && j === oj ? color : i * out + j < oi * out + oj ? done : cell0;
        g.fillRect(ox2 + j * cell, oy + i * cell, cell - 1, cell - 1);
        g.strokeStyle = grid;
        g.strokeRect(ox2 + j * cell, oy + i * cell, cell - 1, cell - 1);
      }
      step = (step + 1) % (out * out);
    };
    draw();
    const t = window.setInterval(draw, 550);
    return () => window.clearInterval(t);
  }, [kind, n, k, s, p, d, size, theme]);
  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height: 200 }} />
      <p className="muted small">{small ? `Illustration with an 8×8 input (the real one is ${n}×${n}); kernel, stride and padding as set.` : "The window slides with the stride; each position gives one output value."}</p>
    </div>
  );
}

export interface Series { name: string; color: string; pts: [number, number][]; thin?: boolean }

export function Chart({ title, series, percent }: { title: string; series: Series[]; percent?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const night = useTheme() === "night";
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const [ink, muted, line, sub] = ["--ink", "--muted", "--line", "--node-sub"].map(cssVar);
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = 190;
    c.width = W * dpr;
    c.height = H * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.font = "11px -apple-system, sans-serif";
    g.fillStyle = ink;
    g.fillText(title, 8, 14);
    const all = series.flatMap((s) => s.pts);
    if (!all.length) { g.fillStyle = muted; g.fillText("no data yet", W / 2 - 25, H / 2); return; }
    const xmax = Math.max(1, ...all.map((p) => p[0]));
    let [ymin, ymax] = percent ? [0, 1] : [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
    if (ymax - ymin < 1e-9) { ymax += 0.5; ymin -= 0.5; }
    const L = 44, R = 10, T = 22, B = 22;
    const X = (x: number) => L + (x / xmax) * (W - L - R), Y = (y: number) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
    g.strokeStyle = line;
    g.fillStyle = muted;
    for (let i = 0; i <= 4; i++) {
      const y = ymin + (i / 4) * (ymax - ymin);
      g.beginPath(); g.moveTo(L, Y(y)); g.lineTo(W - R, Y(y)); g.stroke();
      g.fillText(percent ? `${(y * 100).toFixed(0)}%` : y.toPrecision(3), 4, Y(y) + 4);
    }
    const tick = Math.max(1, Math.ceil(xmax / 12));
    for (let e = tick; e <= xmax; e += tick) g.fillText(String(e), X(e) - 3, H - 6);
    let lx = L + 60;
    for (const s of series) {
      if (!s.pts.length) continue;
      const sc = night && !s.thin ? neon(s.color) : s.color;
      g.strokeStyle = sc;
      g.lineWidth = s.thin ? 1 : 2;
      g.shadowColor = sc;
      g.shadowBlur = night && !s.thin ? 8 : 0;
      g.beginPath();
      s.pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
      g.stroke();
      g.shadowBlur = 0;
      g.fillStyle = sc;
      g.fillRect(lx, 8, 10, 3);
      g.fillStyle = sub;
      g.fillText(s.name, lx + 14, 13);
      lx += 24 + g.measureText(s.name).width;
    }
    g.lineWidth = 1;
  }, [title, series, percent, night]);
  return <canvas ref={ref} style={{ width: "100%", height: 190 }} />;
}

/** an image given as [channel][row][col] values 0..1 */
export function Img({ img, scale, title }: { img: number[][][]; scale: number; title?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !img.length) return;
    const C = img.length, H = img[0].length, W = img[0][0].length;
    c.width = W;
    c.height = H;
    const g = c.getContext("2d")!;
    const d = g.createImageData(W, H);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const o = (i * W + j) * 4;
      const v = (ch: number) => Math.round(255 * img[Math.min(ch, C - 1)][i][j]);
      d.data[o] = v(0); d.data[o + 1] = v(1); d.data[o + 2] = v(2); d.data[o + 3] = 255;
    }
    g.putImageData(d, 0, 0);
  }, [img]);
  const H = img[0]?.length ?? 0, W = img[0]?.[0]?.length ?? 0;
  return <canvas ref={ref} title={title} className="pix" style={{ width: W * scale, height: H * scale }} />;
}

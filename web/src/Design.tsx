// The canvas: layer nodes joined by tensor edges (React Flow), a palette, and the inspector for the selected node.
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ReactFlow, Background, Controls, MiniMap, Handle, Position, addEdge, applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange, type NodeProps, MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, fmt, errText, type Analysis, type AppState, type Graph, type NodeResult, type Params, type Template } from "./api";
import { SlidingPreview } from "./Preview";
import { neon, useTheme } from "./theme";

export const COLORS: Record<string, string> = {
  input: "#3b6fd8", conv2d: "#d9822b", dwconv2d: "#c96d1a", convtranspose2d: "#b5651d", pool2d: "#2a9d8f", global_pool: "#21867a", upsample: "#3c8d7f",
  batchnorm: "#8e5bb5", layernorm: "#7d4fa3", dropout: "#9c6ac6", flatten: "#9a8c1c", dense: "#4b5bb5", activation: "#6c7a89", add: "#c0392b",
  concat: "#a93226", multiply: "#b03a5b", output: "#2e8b57", seg_output: "#1f7a4d",
  to_tokens: "#0f766e", cls_pos: "#0e7490", attention: "#7c3aed", token_pool: "#0369a1",
};
const GROUPS: [string, string[]][] = [
  ["Image", ["conv2d", "dwconv2d", "pool2d", "global_pool", "flatten"]],
  ["Up-sampling", ["convtranspose2d", "upsample"]],
  ["Dense and outputs", ["dense", "output", "seg_output"]],
  ["Normalise / regularise", ["batchnorm", "layernorm", "dropout", "activation"]],
  ["Transformer", ["to_tokens", "cls_pos", "attention", "token_pool"]],
  ["Branches", ["add", "concat", "multiply"]],
];

type LayerData = { kind: string; title: string; params: Params; result?: NodeResult; summary: string };

const n2 = (v: unknown, d = 3): [number, number] => (Array.isArray(v) ? [Number(v[0]), Number(v[1])] : [Number(v ?? d), Number(v ?? d)]);
const k2 = (v: [number, number]) => (v[0] === v[1] ? `${v[0]}` : `${v[0]}×${v[1]}`);

function summary(kind: string, p: Params): string {
  switch (kind) {
    case "conv2d": return `${p.filters} filters · ${k2(n2(p.kernel))} · stride ${k2(n2(p.stride, 1))} · pad ${p.padding}${Number(p.dilation ?? 1) > 1 ? ` · dil ${p.dilation}` : ""} · ${p.activation}`;
    case "dwconv2d": return `×${p.multiplier} · ${k2(n2(p.kernel))} · stride ${k2(n2(p.stride, 1))} · pad ${p.padding} · ${p.activation}`;
    case "convtranspose2d": return `${p.filters} filters · ${k2(n2(p.kernel, 2))} · stride ${k2(n2(p.stride, 2))}`;
    case "pool2d": return `${p.mode} ${k2(n2(p.size, 2))} · stride ${p.stride === "size" ? "= size" : k2(n2(p.stride, 2))}`;
    case "global_pool": return `${p.mode}`;
    case "upsample": return `×${p.scale} ${p.mode}`;
    case "dropout": return `rate ${p.rate}`;
    case "dense": return `${p.units} units · ${p.activation}`;
    case "activation": return `${p.fn}`;
    case "to_tokens": return p.mode === "rows" ? "rows as tokens" : "pixels/patches as tokens";
    case "cls_pos": return p.cls === false ? "position embedding" : "class token + position embedding";
    case "attention": return `${p.heads} heads${Number(p.dropout) > 0 ? ` · dropout ${p.dropout}` : ""}`;
    case "token_pool": return p.mode === "mean" ? "mean of tokens" : "class token";
    default: return "";
  }
}

/** a kind's colour: as defined by day, its neon version at night */
function useKindColor(): ((kind: string) => string) & { night: boolean } {
  const night = useTheme() === "night";
  // night: neon outlines from styles.css (red/yellow for problems), so no inline border
  return Object.assign((kind: string) => (night ? neon(COLORS[kind] ?? "#999999") : COLORS[kind] ?? "#999999"), { night });
}

function LayerNode({ data, selected }: NodeProps<Node<LayerData>>) {
  const color = useKindColor();
  const r = data.result;
  const st = r?.status ?? "ok";
  const border = st === "error" ? "#c62828" : st === "warn" ? "#d08a00" : st === "blocked" || st === "unconnected" ? "#9aa3ae" : "#d5dae1";
  return (
    <div className={`layer st-${st} ${selected ? "sel" : ""}`} style={{ borderColor: color.night ? undefined : border, opacity: st === "unconnected" ? 0.6 : 1, "--c": color(data.kind) } as CSSProperties}>
      {data.kind !== "input" && <Handle type="target" position={Position.Left} />}
      <div className="lh" style={{ background: COLORS[data.kind] }}>{data.title}{st === "error" ? " ✕" : st === "warn" ? " ⚠" : ""}</div>
      {data.summary && <div className="ls">{data.summary}</div>}
      <div className="lo">{r?.out ? `→ ${fmt(r.out)}` : st === "unconnected" ? "not connected" : st === "blocked" ? "fix the layer above" : r?.messages[0]?.slice(0, 70) ?? ""}
        {r && r.params > 0 ? <span className="lp"> · {r.params.toLocaleString()} params</span> : null}</div>
      {!data.kind.endsWith("output") && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
const nodeTypes = { layer: LayerNode };

const STEP_X = 300, STEP_Y = 150;

/** left-to-right layout: a column per depth (longest path from the Input), branches stacked in their column */
export function arrange(g: Graph): Graph {
  const depth: Record<string, number> = {};
  const indeg: Record<string, number> = Object.fromEntries(g.nodes.map((n) => [n.id, 0]));
  g.edges.forEach((e) => { if (e.target in indeg) indeg[e.target]++; });
  const queue = g.nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id);
  queue.forEach((id) => (depth[id] = 0));
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    for (const e of g.edges.filter((e) => e.source === id)) {
      depth[e.target] = Math.max(depth[e.target] ?? 0, depth[id] + 1);
      if (--indeg[e.target] === 0) queue.push(e.target);
    }
  }
  const cols: Record<number, string[]> = {};
  g.nodes.forEach((n) => { const d = depth[n.id] ?? 0; (cols[d] ??= []).push(n.id); });
  const widest = Math.max(1, ...Object.values(cols).map((c) => c.length));
  const pos: Record<string, [number, number]> = {};
  Object.entries(cols).forEach(([d, ids]) => ids.forEach((id, i) => {
    pos[id] = [60 + Number(d) * STEP_X, 60 + ((widest - ids.length) * STEP_Y) / 2 + i * STEP_Y];
  }));
  return { ...g, nodes: g.nodes.map((n) => ({ ...n, x: pos[n.id][0], y: pos[n.id][1] })) };
}

/** a design drawn top to bottom (from before the left-to-right view): most arrows point down */
export function isVertical(g: Graph): boolean {
  const at = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
  const v = g.edges.filter((e) => at[e.source] && at[e.target] && Math.abs(at[e.target].y - at[e.source].y) > Math.abs(at[e.target].x - at[e.source].x)).length;
  return g.edges.length > 0 && v > g.edges.length / 2;
}

export function Design({ st, graph, setGraph, analysis, setAnalysis, toast, reloadProject, remember }: {
  st: AppState; graph: Graph; setGraph: (g: Graph) => void; analysis: Analysis | null; setAnalysis: (a: Analysis) => void; toast: (m: string) => void;
  reloadProject: () => void;
  /** called with the design as it was, before every change (for Undo) */
  remember: (prev: Graph) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const color = useKindColor(), night = color.night;
  // designs saved in the earlier top-to-bottom view are turned left to right once
  useEffect(() => {
    if (isVertical(graph)) {
      const g = arrange(graph);
      setGraph(g);
      api.saveGraph(g).then(setAnalysis).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const saveTimer = useRef<number>(0);
  const titles = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.title])), [st.kinds]);
  const defaults = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.defaults])), [st.kinds]);

  const save = useCallback((g: Graph) => {
    remember(graph);
    setGraph(g);
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => api.saveGraph(g).then(setAnalysis).catch((e) => toast(errText(e))), 200);
  }, [setGraph, setAnalysis, toast, remember, graph]);

  const results = useMemo(() => Object.fromEntries((analysis?.nodes ?? []).map((r) => [r.id, r])), [analysis]);
  // React Flow keeps its own node state (it needs each node's measured size); the design (graph) is synced into it
  const [rfNodes, setRfNodes] = useState<Node<LayerData>[]>([]);
  useEffect(() => {
    setRfNodes((prev) => graph.nodes.map((n) => {
      const old = prev.find((o) => o.id === n.id);
      return {
        ...(old ?? {}), id: n.id, type: "layer", position: old?.dragging ? old.position : { x: n.x, y: n.y }, selected: n.id === selected,
        deletable: n.kind !== "input",
        data: { kind: n.kind, title: titles[n.kind] ?? n.kind, params: n.params, result: results[n.id], summary: summary(n.kind, n.params) },
      };
    }));
  }, [graph, results, selected, titles]);
  const edges: Edge[] = graph.edges.map((e) => ({
    id: e.id, source: e.source, target: e.target, label: analysis?.edges[e.id] ? fmt(analysis.edges[e.id]) : undefined,
    labelStyle: { fontSize: 10, fill: "var(--edge-label)" }, labelBgStyle: { fill: "var(--bg)" },
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: night ? "#00e5ff" : undefined },
  }));

  const onNodesChange = useCallback((changes: NodeChange<Node<LayerData>>[]) => {
    setRfNodes((nds) => applyNodeChanges(changes, nds));
    const sel = changes.find((c) => c.type === "select" && c.selected);
    if (sel && "id" in sel) setSelected(sel.id);
    const removed = changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id);
    const ends = changes.filter((c) => c.type === "position" && c.dragging === false && c.position) as { id: string; position: { x: number; y: number } }[];
    if (!removed.length && !ends.length) return;
    save({
      nodes: graph.nodes.filter((n) => !removed.includes(n.id)).map((n) => {
        const m = ends.find((x) => x.id === n.id);
        return m ? { ...n, x: Math.round(m.position.x), y: Math.round(m.position.y) } : n;
      }),
      edges: graph.edges.filter((e) => !removed.includes(e.source) && !removed.includes(e.target)),
    });
  }, [graph, save]);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    const kept = applyEdgeChanges(changes, edges);
    if (changes.some((c) => c.type === "remove")) save({ ...graph, edges: graph.edges.filter((e) => kept.some((k) => k.id === e.id)) });
  }, [edges, graph, save]);

  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target || c.source === c.target) return;
    if (graph.edges.some((e) => e.source === c.source && e.target === c.target)) return;
    const added = addEdge({ ...c, id: `e${Date.now().toString(36)}` }, []);
    save({ ...graph, edges: [...graph.edges, { id: added[0].id, source: c.source, target: c.target }] });
  }, [graph, save]);

  /** add a block: below the selected one (and connected to it), or at the bottom of the design */
  const addBlock = (kind: string) => {
    const base = graph.nodes.find((n) => n.id === selected) ?? [...graph.nodes].sort((a, b) => b.x - a.x)[0];
    const id = `n${Date.now().toString(36)}`;
    const right = [...graph.nodes].sort((a, b) => b.x - a.x)[0];
    const anchor = selected ? base : right;
    const node = { id, kind, params: structuredClone(defaults[kind] ?? {}) as Params, x: anchor ? anchor.x + STEP_X : 60, y: anchor ? anchor.y : 80 };
    const edges = [...graph.edges];
    if (base && base.kind !== "output") {
      // insert between the selected node and its (single) successor
      const outs = edges.filter((e) => e.source === base.id);
      if (selected && outs.length === 1) {
        const next = outs[0];
        edges.splice(edges.indexOf(next), 1, { id: `e${id}a`, source: base.id, target: id }, { id: `e${id}b`, source: id, target: next.target });
        const shifted = graph.nodes.map((n) => (n.x > base.x ? { ...n, x: n.x + STEP_X } : n));
        save({ nodes: [...shifted, node], edges });
        setSelected(id);
        return;
      }
      edges.push({ id: `e${id}`, source: base.id, target: id });
    }
    save({ nodes: [...graph.nodes, node], edges });
    setSelected(id);
  };

  const sel = graph.nodes.find((n) => n.id === selected) ?? null;

  /** a skip connection from `src` to the selected node: a join (Add or Concat) is inserted after the selected node, and when the shapes do not
   *  match a projection is added on the skip path (1×1 convolution with a stride for Add, pooling for Concat, Dense for vectors) */
  const addSkip = (src: string, mode: "add" | "concat") => {
    if (!sel || !analysis) return;
    const t = analysis.nodes.find((r) => r.id === sel.id)?.out, s = analysis.nodes.find((r) => r.id === src)?.out;
    if (!t || !s) return toast("Both blocks need a valid shape first.");
    const id = Date.now().toString(36);
    const nodes = graph.nodes.map((n) => (n.x > sel.x ? { ...n, x: n.x + STEP_X } : n));
    let edges = graph.edges.map((e) => (e.source === sel.id ? { ...e, source: `j${id}` } : e));
    const join = { id: `j${id}`, kind: mode, params: {}, x: sel.x + STEP_X, y: sel.y };
    let from = src;
    const extra: typeof graph.nodes = [];
    if (t.length !== s.length) return toast("A skip connection must join two images or two vectors (not an image and a vector).");
    if (t.length === 3) {
      const [r1, r2] = [s[1] / t[1], s[2] / t[2]];
      if (!Number.isInteger(r1) || !Number.isInteger(r2) || r1 < 1 || r2 < 1)
        return toast(`The skip source is ${s[1]}×${s[2]} and the target ${t[1]}×${t[2]}: the source must be the same size or a whole multiple larger.`);
      const needProj = mode === "add" ? s[0] !== t[0] || r1 > 1 || r2 > 1 : r1 > 1 || r2 > 1;
      if (needProj) {
        const pid = `p${id}`;
        extra.push(mode === "add"
          ? { id: pid, kind: "conv2d", params: { filters: t[0], kernel: [1, 1], stride: [r1, r2], padding: "valid", dilation: 1, activation: "none", bias: false }, x: sel.x, y: sel.y + 160 }
          : { id: pid, kind: "pool2d", params: { mode: "max", size: [r1, r2], stride: "size", padding: 0 }, x: sel.x, y: sel.y + 160 });
        edges = [...edges, { id: `e${id}p`, source: src, target: pid }];
        from = pid;
      }
    } else if (mode === "add" && s[0] !== t[0]) {
      const pid = `p${id}`;
      extra.push({ id: pid, kind: "dense", params: { units: t[0], activation: "none" }, x: sel.x, y: sel.y + 160 });
      edges = [...edges, { id: `e${id}p`, source: src, target: pid }];
      from = pid;
    }
    edges = [...edges, { id: `e${id}a`, source: sel.id, target: join.id }, { id: `e${id}b`, source: from, target: join.id }];
    save({ nodes: [...nodes, ...extra, join], edges });
    setSelected(join.id);
    toast(extra.length ? `Skip connection added, with a ${extra[0].kind === "conv2d" ? "1×1 convolution" : extra[0].kind === "pool2d" ? "pooling" : "Dense"} to match the shapes.`
      : "Skip connection added.");
  };
  // earlier blocks (in computation order) that can feed a skip connection into the selected one
  const order = analysis?.nodes.filter((r) => r.out).map((r) => r.id) ?? [];
  const candidates = sel && order.includes(sel.id) ? order.slice(0, order.indexOf(sel.id)).filter((id) => !graph.edges.some((e) => e.source === id && e.target === sel.id))
    .map((id) => ({ id, label: `${order.indexOf(id)}. ${analysis!.nodes.find((r) => r.id === id)!.title} ${fmt(analysis!.nodes.find((r) => r.id === id)!.out)}` })) : [];
  const setParam = (key: string, v: unknown) => {
    if (!sel) return;
    save({ ...graph, nodes: graph.nodes.map((n) => (n.id === sel.id ? { ...n, params: { ...n.params, [key]: v } } : n)) });
  };
  const setParams = (p: Params) => sel && save({ ...graph, nodes: graph.nodes.map((n) => (n.id === sel.id ? { ...n, params: { ...n.params, ...p } } : n)) });

  return (
    <div className="design">
      <div className="palette">
        <TemplatePicker toast={toast} onLoaded={reloadProject} />
        <button className="arrange" onClick={() => save(arrange(graph))}>Arrange left to right</button>
        <p className="muted small">Click a block to add it after the selected one. Drag from a block's right dot to another block's left dot to connect (several arrows into Add / Concat / Multiply). Select + Backspace deletes.</p>
        {GROUPS.map(([g, kinds]) => (
          <div key={g}><h4>{g}</h4>{kinds.map((k) => (
            <button key={k} className="pal" style={{ borderLeftColor: COLORS[k], "--c": color(k) } as CSSProperties} onClick={() => addBlock(k)}>{titles[k]}</button>
          ))}</div>
        ))}
      </div>
      <div className="canvas">
        {rfNodes.length > 0 && <ReactFlow nodes={rfNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
          onPaneClick={() => setSelected(null)} fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} minZoom={0.2} deleteKeyCode={["Backspace", "Delete"]}>
          <Background gap={24} />
          <Controls />
          <MiniMap pannable zoomable nodeColor={(n) => color((n.data as LayerData).kind)}
            maskColor={night ? "rgba(5,6,12,.7)" : undefined} />
        </ReactFlow>}
      </div>
      <Inspector node={sel} result={sel ? results[sel.id] : undefined} analysis={analysis} st={st} setParam={setParam} setParams={setParams}
        candidates={candidates} addSkip={addSkip} />
    </div>
  );
}

// ---------------------------------------------------------------- inspector
function Num({ label, value, onChange, min = 1, max = 4096, step = 1 }: { label: string; value: unknown; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  const [text, setText] = useState(String(value ?? ""));
  useEffect(() => setText(String(value ?? "")), [value]);
  return <label>{label}<input type="number" value={text} min={min} max={max} step={step}
    onChange={(e) => { setText(e.target.value); const v = Number(e.target.value); if (e.target.value !== "" && v >= min && v <= max) onChange(v); }} /></label>;
}

function Sel({ label, value, options, onChange }: { label: string; value: unknown; options: [string, string][]; onChange: (v: string) => void }) {
  return <label>{label}<select value={String(value)} onChange={(e) => onChange(e.target.value)}>{options.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>;
}

/** a height×width pair with a "square" link: while linked, editing one side sets both */
function Pair({ label, value, onChange, min = 1, max = 15, def = 3 }: { label: string; value: unknown; onChange: (v: [number, number]) => void; min?: number; max?: number; def?: number }) {
  const v = n2(value, def);
  const [linked, setLinked] = useState(v[0] === v[1]);
  return (
    <div className="pair"><span className="plabel">{label}</span>
      <Num label="" value={v[0]} min={min} max={max} onChange={(x) => onChange(linked ? [x, x] : [x, v[1]])} />
      <span>×</span>
      <Num label="" value={v[1]} min={min} max={max} onChange={(x) => onChange(linked ? [x, x] : [v[0], x])} />
      <label className="chk"><input type="checkbox" checked={linked} onChange={(e) => { setLinked(e.target.checked); if (e.target.checked) onChange([v[0], v[0]]); }} />square</label>
    </div>
  );
}

const PRESETS: [string, Params][] = [
  ["3×3, stride 1 (keep size)", { kernel: [3, 3], stride: [1, 1], padding: "same" }],
  ["3×3, stride 2 (halve size)", { kernel: [3, 3], stride: [2, 2], padding: "same" }],
  ["1×1 (mix channels)", { kernel: [1, 1], stride: [1, 1], padding: "valid" }],
  ["5×5, stride 1", { kernel: [5, 5], stride: [1, 1], padding: "same" }],
  ["7×7, stride 2 (ResNet stem)", { kernel: [7, 7], stride: [2, 2], padding: "same" }],
];

function TemplatePicker({ toast, onLoaded }: { toast: (m: string) => void; onLoaded: () => void }) {
  const [list, setList] = useState<Template[]>([]);
  const [key, setKey] = useState("");
  useEffect(() => { api.templates().then(setList).catch(() => {}); }, []);
  const t = list.find((x) => x.key === key);
  const load = () => {
    if (!t || !confirm(`Replace the current design with ${t.title}? (The current design is kept as model.previous.json in the project folder.)`)) return;
    api.templateApply(t.key).then((m) => { toast(m); onLoaded(); }).catch((e) => toast(errText(e)));
  };
  return (
    <div className="templates">
      <h4>Famous architectures</h4>
      <select value={key} onChange={(e) => setKey(e.target.value)}>
        <option value="">choose…</option>
        {list.map((x) => <option key={x.key} value={x.key}>{x.title} ({x.year}){x.task === "segment" ? " · segmentation" : ""}</option>)}
      </select>
      {t && <><p className="small muted">{t.about} Input {t.input[1]}×{t.input[2]}×{t.input[0]}.</p><button onClick={load}>Load {t.title}</button></>}
    </div>
  );
}

function SkipBox({ candidates, addSkip }: { candidates: { id: string; label: string }[]; addSkip: (src: string, mode: "add" | "concat") => void }) {
  const [src, setSrc] = useState("");
  const [mode, setMode] = useState<"add" | "concat">("add");
  if (!candidates.length) return null;
  return (
    <div className="skip">
      <h4>Skip connection</h4>
      <p className="muted small">Join an earlier block's output to this block's output: Add (residual, ResNet) or Concat (DenseNet, U-Net). A projection is added when the shapes differ.</p>
      <label>From<select value={src} onChange={(e) => setSrc(e.target.value)}><option value="">choose an earlier block…</option>
        {candidates.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label>
      <div className="row"><label className="chk"><input type="radio" checked={mode === "add"} onChange={() => setMode("add")} />Add</label>
        <label className="chk"><input type="radio" checked={mode === "concat"} onChange={() => setMode("concat")} />Concat</label>
        <button disabled={!src} onClick={() => { addSkip(src, mode); setSrc(""); }}>Add skip connection</button></div>
    </div>
  );
}

function Inspector({ node, result, analysis, st, setParam, setParams, candidates, addSkip }: {
  node: { id: string; kind: string; params: Params } | null; result?: NodeResult; analysis: Analysis | null; st: AppState;
  setParam: (k: string, v: unknown) => void; setParams: (p: Params) => void;
  candidates: { id: string; label: string }[]; addSkip: (src: string, mode: "add" | "concat") => void;
}) {
  const color = useKindColor();
  if (!node) {
    return (
      <div className="inspector">
        <h3>Design</h3>
        {analysis && <p><strong>{analysis.total_params.toLocaleString()}</strong> trainable parameters · input {fmt(analysis.input)} · {analysis.classes} classes</p>}
        {analysis?.problems.map((p) => <div key={p} className="msg error">{p}</div>)}
        {analysis?.ok && <div className="msg ok">The design is complete: every shape fits. Next: Dataset, then Train.</div>}
        <p className="muted">Click a block to set its filters, kernel, stride, padding or pooling, and to see how its output size is computed.</p>
        <table className="grid small"><tbody>
          <tr><th>Block</th><th>Out</th><th className="num">Params</th><th>RF</th></tr>
          {analysis?.nodes.filter((r) => r.out).map((r) => <tr key={r.id} className={`st-${r.status}`}><td>{r.title}</td><td>{fmt(r.out)}</td>
            <td className="num">{r.params.toLocaleString()}</td><td>{r.rf ? `${r.rf[0]}×${r.rf[1]}` : ""}</td></tr>)}
        </tbody></table>
      </div>
    );
  }
  const p = node.params;
  const acts: [string, string][] = st.activations.map((a) => [a, a]);
  const pads: [string, string][] = [["same", "same (keep size)"], ["valid", "valid (none)"], ["1", "1"], ["2", "2"], ["3", "3"]];
  const padVal = typeof p.padding === "number" ? String(p.padding) : String(p.padding ?? "same");
  const padSet = (v: string) => setParam("padding", v === "same" || v === "valid" ? v : Number(v));
  return (
    <div className="inspector">
      <h3 style={{ color: color(node.kind), "--c": color(node.kind) } as CSSProperties}>{result?.title ?? node.kind}</h3>
      <div className="fields">
        {node.kind === "conv2d" && <>
          <Num label="Filters (output channels)" value={p.filters} onChange={(v) => setParam("filters", v)} />
          <Pair label="Kernel" value={p.kernel} max={32} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={32} def={1} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={padVal} options={pads} onChange={padSet} />
          <Sel label="Dilation" value={p.dilation ?? 1} options={[["1", "1"], ["2", "2"], ["3", "3"]]} onChange={(v) => setParam("dilation", Number(v))} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
          <label className="chk"><input type="checkbox" checked={p.bias !== false} onChange={(e) => setParam("bias", e.target.checked)} />bias</label>
          <Sel label="Preset" value="" options={[["", "choose…"], ...PRESETS.map(([t], i): [string, string] => [String(i), t])]} onChange={(v) => v && setParams(PRESETS[Number(v)][1])} />
        </>}
        {node.kind === "dwconv2d" && <>
          <Num label="Channel multiplier" value={p.multiplier} max={8} onChange={(v) => setParam("multiplier", v)} />
          <Pair label="Kernel" value={p.kernel} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={32} def={1} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={padVal} options={pads} onChange={padSet} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
        </>}
        {node.kind === "convtranspose2d" && <>
          <Num label="Filters" value={p.filters} onChange={(v) => setParam("filters", v)} />
          <Pair label="Kernel" value={p.kernel} def={2} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={32} def={2} onChange={(v) => setParam("stride", v)} />
          <Num label="Padding" value={p.padding} min={0} max={5} onChange={(v) => setParam("padding", v)} />
          <Num label="Output padding" value={p.output_padding} min={0} max={3} onChange={(v) => setParam("output_padding", v)} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
        </>}
        {node.kind === "pool2d" && <>
          <Sel label="Mode" value={p.mode} options={[["max", "Max"], ["avg", "Average"]]} onChange={(v) => setParam("mode", v)} />
          <Pair label="Pool size" value={p.size} max={8} def={2} onChange={(v) => setParam("size", v)} />
          <label className="chk"><input type="checkbox" checked={p.stride === "size"} onChange={(e) => setParam("stride", e.target.checked ? "size" : n2(p.size, 2))} />stride = pool size</label>
          {p.stride !== "size" && <Pair label="Stride" value={p.stride} max={32} def={2} onChange={(v) => setParam("stride", v)} />}
          <Sel label="Padding" value={p.padding ?? 0} options={[["0", "0"], ["1", "1"], ["2", "2"]]} onChange={(v) => setParam("padding", Number(v))} />
        </>}
        {node.kind === "global_pool" && <Sel label="Mode" value={p.mode} options={[["avg", "Average"], ["max", "Max"]]} onChange={(v) => setParam("mode", v)} />}
        {node.kind === "upsample" && <>
          <Sel label="Scale" value={p.scale} options={[["2", "×2"], ["3", "×3"], ["4", "×4"]]} onChange={(v) => setParam("scale", Number(v))} />
          <Sel label="Mode" value={p.mode} options={[["nearest", "nearest"], ["bilinear", "bilinear"]]} onChange={(v) => setParam("mode", v)} />
        </>}
        {node.kind === "dropout" && <Num label="Rate" value={p.rate} min={0} max={0.95} step={0.05} onChange={(v) => setParam("rate", v)} />}
        {node.kind === "dense" && <>
          <Num label="Units" value={p.units} max={1 << 20} onChange={(v) => setParam("units", v)} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
        </>}
        {node.kind === "to_tokens" && <>
          <Sel label="Tokens" value={p.mode} options={[["patches", "every pixel (patch) is a token (ViT: after a patch convolution)"], ["rows", "every image row is a token"]]} onChange={(v) => setParam("mode", v)} />
          <p className="muted">(channels, height, width) → patches: (height×width tokens, channels); rows: (height tokens, channels×width).</p>
        </>}
        {node.kind === "cls_pos" && <label className="chk"><input type="checkbox" checked={p.cls !== false} onChange={(e) => setParam("cls", e.target.checked)} />add a learnable class token in front</label>}
        {node.kind === "attention" && <>
          <Num label="Heads" value={p.heads} max={64} onChange={(v) => setParam("heads", v)} />
          <Num label="Dropout" value={p.dropout} min={0} max={0.9} step={0.05} onChange={(v) => setParam("dropout", v)} />
          <p className="muted">Every token looks at every other token (multi-head self-attention). The token dimension must divide evenly by the number of heads. Parameters: 4·d² + 4·d.</p>
        </>}
        {node.kind === "token_pool" && <Sel label="Use" value={p.mode} options={[["cls", "the class token (ViT)"], ["mean", "the mean of all tokens"]]} onChange={(v) => setParam("mode", v)} />}
        {node.kind === "activation" && <Sel label="Function" value={p.fn} options={acts.filter(([a]) => a !== "none")} onChange={(v) => setParam("fn", v)} />}
        {node.kind === "layernorm" && <p className="muted">On an image it normalises over (channels, height, width) (GroupNorm with one group); on a vector, over its features.</p>}
        {node.kind === "add" && <p className="muted">Adds two or more branches element by element (a residual connection): their shapes must be equal.</p>}
        {node.kind === "concat" && <p className="muted">Joins branches along channels (images: equal height and width) or features (vectors). The order of the arrows is the order of the channels.</p>}
        {node.kind === "output" && <p className="muted">One score per class of the dataset ({analysis?.classes ?? "?"}); the loss applies softmax.</p>}
        {node.kind === "seg_output" && <p className="muted">A 1×1 convolution giving a score per class ({analysis?.classes ?? "?"}) for every pixel. Its size must equal the input (the masks).</p>}
        {node.kind === "multiply" && <p className="muted">Multiplies two branches element by element: equal shapes, a one-channel map times an image (an attention gate: the map, after a sigmoid, weights every pixel), or one weight per channel (squeeze-and-excitation).</p>}
        {node.kind === "input" && <p className="muted">The images of the dataset (Dataset tab): {fmt(analysis?.input)}.</p>}
      </div>
      {result && <>
        <h4>{result.inputs.map(fmt).join(" + ")}{result.inputs.length ? " → " : ""}{fmt(result.out)}{result.params ? ` · ${result.params.toLocaleString()} parameters` : ""}</h4>
        {result.messages.map((m) => <div key={m} className={`msg ${result.status === "error" ? "error" : result.status === "warn" ? "warn" : "info"}`}>{m}</div>)}
        {result.formula.length > 0 && <>
          <p className="muted small">{node.kind === "convtranspose2d" ? "out = (in − 1)×stride − 2×padding + (kernel − 1) + output_padding + 1"
            : "out = floor((in + 2×padding − dilation×(kernel − 1) − 1) / stride) + 1"}</p>
          <pre className="formula">{result.formula.join("\n")}</pre>
        </>}
        {result.rf && <p className="muted small">Receptive field: one output value here sees {result.rf[0]}×{result.rf[1]} input pixels.</p>}
        {node.kind !== "input" && !node.kind.endsWith("output") && <SkipBox candidates={candidates} addSkip={addSkip} />}
        {(node.kind === "conv2d" || node.kind === "pool2d" || node.kind === "dwconv2d") && result.padding && result.stride && result.inputs[0] &&
          <SlidingPreview kind={node.kind} n={result.inputs[0][1]} k={n2(node.kind === "pool2d" ? p.size : p.kernel, node.kind === "pool2d" ? 2 : 3)[0]}
            s={result.stride[0]} p={result.padding[0]} d={node.kind === "conv2d" ? Number(p.dilation ?? 1) : 1} />}
      </>}
    </div>
  );
}

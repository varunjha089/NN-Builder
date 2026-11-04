// The canvas: layer nodes joined by tensor edges (React Flow), a palette, and the inspector for the selected node.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap, Handle, Position, addEdge, applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange, type NodeProps, MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, fmt, errText, type Analysis, type AppState, type Graph, type NodeResult, type Params } from "./api";
import { SlidingPreview } from "./Preview";

export const COLORS: Record<string, string> = {
  input: "#3b6fd8", conv2d: "#d9822b", dwconv2d: "#c96d1a", convtranspose2d: "#b5651d", pool2d: "#2a9d8f", global_pool: "#21867a", upsample: "#3c8d7f",
  batchnorm: "#8e5bb5", layernorm: "#7d4fa3", dropout: "#9c6ac6", flatten: "#9a8c1c", dense: "#4b5bb5", activation: "#6c7a89", add: "#c0392b",
  concat: "#a93226", output: "#2e8b57",
};
const GROUPS: [string, string[]][] = [
  ["Image", ["conv2d", "dwconv2d", "pool2d", "global_pool", "flatten"]],
  ["Up-sampling", ["convtranspose2d", "upsample"]],
  ["Dense", ["dense", "output"]],
  ["Normalise / regularise", ["batchnorm", "layernorm", "dropout", "activation"]],
  ["Branches", ["add", "concat"]],
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
    default: return "";
  }
}

function LayerNode({ data, selected }: NodeProps<Node<LayerData>>) {
  const r = data.result;
  const st = r?.status ?? "ok";
  const border = st === "error" ? "#c62828" : st === "warn" ? "#d08a00" : st === "blocked" || st === "unconnected" ? "#9aa3ae" : "#d5dae1";
  return (
    <div className={`layer ${selected ? "sel" : ""}`} style={{ borderColor: border, opacity: st === "unconnected" ? 0.6 : 1 }}>
      {data.kind !== "input" && <Handle type="target" position={Position.Top} />}
      <div className="lh" style={{ background: COLORS[data.kind] }}>{data.title}{st === "error" ? " ✕" : st === "warn" ? " ⚠" : ""}</div>
      {data.summary && <div className="ls">{data.summary}</div>}
      <div className="lo">{r?.out ? `→ ${fmt(r.out)}` : st === "unconnected" ? "not connected" : st === "blocked" ? "fix the layer above" : r?.messages[0]?.slice(0, 70) ?? ""}
        {r && r.params > 0 ? <span className="lp"> · {r.params.toLocaleString()} params</span> : null}</div>
      {data.kind !== "output" && <Handle type="source" position={Position.Bottom} />}
    </div>
  );
}
const nodeTypes = { layer: LayerNode };

export function Design({ st, graph, setGraph, analysis, setAnalysis, toast }: {
  st: AppState; graph: Graph; setGraph: (g: Graph) => void; analysis: Analysis | null; setAnalysis: (a: Analysis) => void; toast: (m: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const saveTimer = useRef<number>(0);
  const titles = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.title])), [st.kinds]);
  const defaults = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.defaults])), [st.kinds]);

  const save = useCallback((g: Graph) => {
    setGraph(g);
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => api.saveGraph(g).then(setAnalysis).catch((e) => toast(errText(e))), 200);
  }, [setGraph, setAnalysis, toast]);

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
    labelStyle: { fontSize: 10, fill: "#4b5563" }, labelBgStyle: { fill: "#f6f7f9" }, markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
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
    const base = graph.nodes.find((n) => n.id === selected) ?? [...graph.nodes].sort((a, b) => b.y - a.y)[0];
    const id = `n${Date.now().toString(36)}`;
    const node = { id, kind, params: structuredClone(defaults[kind] ?? {}) as Params, x: base ? base.x + (selected ? 0 : 0) : 80, y: base ? base.y + 110 : 40 };
    const edges = [...graph.edges];
    if (base && base.kind !== "output") {
      // insert between the selected node and its (single) successor
      const outs = edges.filter((e) => e.source === base.id);
      if (selected && outs.length === 1) {
        const next = outs[0];
        edges.splice(edges.indexOf(next), 1, { id: `e${id}a`, source: base.id, target: id }, { id: `e${id}b`, source: id, target: next.target });
        const shifted = graph.nodes.map((n) => (n.y > base.y ? { ...n, y: n.y + 110 } : n));
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
  const setParam = (key: string, v: unknown) => {
    if (!sel) return;
    save({ ...graph, nodes: graph.nodes.map((n) => (n.id === sel.id ? { ...n, params: { ...n.params, [key]: v } } : n)) });
  };
  const setParams = (p: Params) => sel && save({ ...graph, nodes: graph.nodes.map((n) => (n.id === sel.id ? { ...n, params: { ...n.params, ...p } } : n)) });

  return (
    <div className="design">
      <div className="palette">
        <p className="muted small">Click a block to add it below the selected one. Drag from a node's bottom dot to another node's top dot to connect (several arrows into Add / Concat). Select + Backspace deletes.</p>
        {GROUPS.map(([g, kinds]) => (
          <div key={g}><h4>{g}</h4>{kinds.map((k) => (
            <button key={k} className="pal" style={{ borderLeftColor: COLORS[k] }} onClick={() => addBlock(k)}>{titles[k]}</button>
          ))}</div>
        ))}
      </div>
      <div className="canvas">
        {rfNodes.length > 0 && <ReactFlow nodes={rfNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
          onPaneClick={() => setSelected(null)} fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} minZoom={0.2} deleteKeyCode={["Backspace", "Delete"]}>
          <Background gap={24} />
          <Controls />
          <MiniMap pannable zoomable nodeColor={(n) => COLORS[(n.data as LayerData).kind] ?? "#999"} />
        </ReactFlow>}
      </div>
      <Inspector node={sel} result={sel ? results[sel.id] : undefined} analysis={analysis} st={st} setParam={setParam} setParams={setParams} />
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

function Inspector({ node, result, analysis, st, setParam, setParams }: {
  node: { id: string; kind: string; params: Params } | null; result?: NodeResult; analysis: Analysis | null; st: AppState;
  setParam: (k: string, v: unknown) => void; setParams: (p: Params) => void;
}) {
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
      <h3 style={{ color: COLORS[node.kind] }}>{result?.title ?? node.kind}</h3>
      <div className="fields">
        {node.kind === "conv2d" && <>
          <Num label="Filters (output channels)" value={p.filters} onChange={(v) => setParam("filters", v)} />
          <Pair label="Kernel" value={p.kernel} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={8} def={1} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={padVal} options={pads} onChange={padSet} />
          <Sel label="Dilation" value={p.dilation ?? 1} options={[["1", "1"], ["2", "2"], ["3", "3"]]} onChange={(v) => setParam("dilation", Number(v))} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
          <label className="chk"><input type="checkbox" checked={p.bias !== false} onChange={(e) => setParam("bias", e.target.checked)} />bias</label>
          <Sel label="Preset" value="" options={[["", "choose…"], ...PRESETS.map(([t], i): [string, string] => [String(i), t])]} onChange={(v) => v && setParams(PRESETS[Number(v)][1])} />
        </>}
        {node.kind === "dwconv2d" && <>
          <Num label="Channel multiplier" value={p.multiplier} max={8} onChange={(v) => setParam("multiplier", v)} />
          <Pair label="Kernel" value={p.kernel} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={8} def={1} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={padVal} options={pads} onChange={padSet} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
        </>}
        {node.kind === "convtranspose2d" && <>
          <Num label="Filters" value={p.filters} onChange={(v) => setParam("filters", v)} />
          <Pair label="Kernel" value={p.kernel} def={2} onChange={(v) => setParam("kernel", v)} />
          <Pair label="Stride" value={p.stride} max={8} def={2} onChange={(v) => setParam("stride", v)} />
          <Num label="Padding" value={p.padding} min={0} max={5} onChange={(v) => setParam("padding", v)} />
          <Num label="Output padding" value={p.output_padding} min={0} max={3} onChange={(v) => setParam("output_padding", v)} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
        </>}
        {node.kind === "pool2d" && <>
          <Sel label="Mode" value={p.mode} options={[["max", "Max"], ["avg", "Average"]]} onChange={(v) => setParam("mode", v)} />
          <Pair label="Pool size" value={p.size} max={8} def={2} onChange={(v) => setParam("size", v)} />
          <label className="chk"><input type="checkbox" checked={p.stride === "size"} onChange={(e) => setParam("stride", e.target.checked ? "size" : n2(p.size, 2))} />stride = pool size</label>
          {p.stride !== "size" && <Pair label="Stride" value={p.stride} max={8} def={2} onChange={(v) => setParam("stride", v)} />}
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
        {node.kind === "activation" && <Sel label="Function" value={p.fn} options={acts.filter(([a]) => a !== "none")} onChange={(v) => setParam("fn", v)} />}
        {node.kind === "layernorm" && <p className="muted">On an image it normalises over (channels, height, width) (GroupNorm with one group); on a vector, over its features.</p>}
        {node.kind === "add" && <p className="muted">Adds two or more branches element by element (a residual connection): their shapes must be equal.</p>}
        {node.kind === "concat" && <p className="muted">Joins branches along channels (images: equal height and width) or features (vectors). The order of the arrows is the order of the channels.</p>}
        {node.kind === "output" && <p className="muted">One score per class of the dataset ({analysis?.classes ?? "?"}); the loss applies softmax.</p>}
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
        {(node.kind === "conv2d" || node.kind === "pool2d" || node.kind === "dwconv2d") && result.padding && result.stride && result.inputs[0] &&
          <SlidingPreview kind={node.kind} n={result.inputs[0][1]} k={n2(node.kind === "pool2d" ? p.size : p.kernel, node.kind === "pool2d" ? 2 : 3)[0]}
            s={result.stride[0]} p={result.padding[0]} d={node.kind === "conv2d" ? Number(p.dilation ?? 1) : 1} />}
      </>}
    </div>
  );
}

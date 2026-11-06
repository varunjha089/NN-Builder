// The canvas: layer nodes joined by tensor edges (React Flow), a palette, and the inspector for the selected node.
import { memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ReactFlow, Background, Controls, MiniMap, Handle, Position, addEdge, applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange, type NodeProps, MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, fmt, errText, type Analysis, type AppState, type GNode, type Graph, type NodeResult, type Params, type Template } from "./api";
import { SlidingPreview } from "./Preview";
import { neon, useTheme } from "./theme";

export const COLORS: Record<string, string> = {
  input: "#3b6fd8", conv2d: "#d9822b", dwconv2d: "#c96d1a", convtranspose2d: "#b5651d", pool2d: "#2a9d8f", global_pool: "#21867a", upsample: "#3c8d7f",
  batchnorm: "#8e5bb5", layernorm: "#7d4fa3", groupnorm: "#a066c9", repeat: "#0e7490", dropout: "#9c6ac6", flatten: "#9a8c1c", dense: "#4b5bb5", activation: "#6c7a89", add: "#c0392b",
  concat: "#a93226", multiply: "#b03a5b", output: "#2e8b57", seg_output: "#1f7a4d",
  to_tokens: "#0f766e", cls_pos: "#0e7490", attention: "#7c3aed", token_pool: "#0369a1", custom: "#475569", subtract: "#9b2c2c", shared: "#64748b",
  conv3d: "#b45309", temporal_pool: "#047857", to_sequence: "#0f766e", conv1d: "#c2410c", lstm: "#6d28d9", gru: "#7e22ce",
};
const GROUPS: [string, string[]][] = [
  ["Image", ["conv2d", "dwconv2d", "pool2d", "global_pool", "flatten"]],
  ["Up-sampling", ["convtranspose2d", "upsample"]],
  ["Dense and outputs", ["dense", "output", "seg_output"]],
  ["Normalise / regularise", ["batchnorm", "layernorm", "groupnorm", "dropout", "activation"]],
  ["Transformer", ["to_tokens", "cls_pos", "attention", "token_pool"]],
  ["Time series", ["conv3d", "temporal_pool", "to_sequence", "conv1d", "lstm", "gru"]],
  ["Branches", ["add", "concat", "multiply", "subtract", "shared"]],
  ["Your own", ["custom"]],
];

type LayerData = { kind: string; title: string; params: Params; result?: NodeResult; summary: string; label?: string; sig?: string };

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
    case "groupnorm": return `${p.groups} groups`;
    case "repeat": return `${p.count} identical blocks · double-click to expand`;
    case "dense": return `${p.units} units · ${p.activation}`;
    case "activation": return `${p.fn}`;
    case "to_tokens": return p.mode === "rows" ? "rows as tokens" : "pixels/patches as tokens";
    case "cls_pos": return p.cls === false ? "position embedding" : "class token + position embedding";
    case "attention": return `${p.heads} heads${Number(p.dropout) > 0 ? ` · dropout ${p.dropout}` : ""}`;
    case "token_pool": return p.mode === "mean" ? "mean of tokens" : p.mode === "last" ? "last step" : "class token";
    case "custom": return p.class ? `class ${p.class}` : "choose a class";
    case "output": return Number(p.head ?? 0) > 0 ? `output ${p.head}` : "";
    case "subtract": return p.abs ? "|a − b|" : "a − b";
    case "conv3d": return `${p.filters} filters · ${p.tkernel} dates × ${k2(n2(p.kernel))} · ${p.padding} · ${p.activation}`;
    case "temporal_pool": return `${p.mode} over the dates`;
    case "to_sequence": return "one vector per date";
    case "lstm": case "gru": return `${p.units} units${p.bidirectional ? " · both directions" : ""}`;
    case "conv1d": return `${p.filters} filters · kernel ${p.kernel} · stride ${p.stride ?? 1} · ${p.activation}`;
    case "shared": return p.of ? "the weights of another block" : "choose the block";
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
      <div className="lh" style={{ background: COLORS[data.kind] }}>{data.label && <span className="lnum" title="this block's label in the Text tab">{data.label}</span>}
        {data.title}{st === "error" ? " ✕" : st === "warn" ? " ⚠" : ""}</div>
      {data.summary && <div className="ls">{data.summary}</div>}
      <div className="lo">{r?.out ? `→ ${fmt(r.out)}` : st === "unconnected" ? "not connected" : st === "blocked" ? "fix the layer above" : r?.messages[0]?.slice(0, 70) ?? ""}
        {r && r.params > 0 ? <span className="lp"> · {r.params.toLocaleString()} params</span> : null}</div>
      {!data.kind.endsWith("output") && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
// memo: a block is drawn again only when its own data or selection changes (dragging one block in a 290-block design stays smooth)
const nodeTypes = { layer: memo(LayerNode) };

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

export function Design({ st, graph, setGraph, analysis, persist, focus, toast, reloadProject, remember }: {
  st: AppState; graph: Graph; setGraph: (g: Graph) => void; analysis: Analysis | null; toast: (m: string) => void;
  /** save the design (the text follows it) */
  persist: (g: Graph) => Promise<void>;
  /** a block to select when the tab opens */
  focus: string | null;
  reloadProject: () => void;
  /** called with the design as it was, before every change (for Undo) */
  remember: (prev: Graph) => void;
}) {
  const [selected, setSelected] = useState<string | null>(focus);
  const color = useKindColor(), night = color.night;
  // designs saved in the earlier top-to-bottom view are turned left to right once
  useEffect(() => {
    if (isVertical(graph)) {
      const g = arrange(graph);
      setGraph(g);
      persist(g).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const saveTimer = useRef<number>(0);
  const titles = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.title])), [st.kinds]);
  const defaults = useMemo(() => Object.fromEntries(st.kinds.map((k) => [k.kind, k.defaults])), [st.kinds]);

  const save = useCallback((changed: Graph) => {
    // a canvas action may build the design as { nodes, edges }: keep its text, so that the Rust side edits only the lines the change
    // concerns (functions, loops, names and comments stay) instead of printing a new text
    let g = changed.text === undefined && graph.text ? { ...changed, text: graph.text } : changed;
    // the design's PyTorch code (Custom blocks) stays with the design through every canvas action
    if (g.python === undefined && graph.python) g = { ...g, python: graph.python };
    remember(graph);
    setGraph(g);
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => persist(g).catch((e) => toast(errText(e))), 200);
  }, [setGraph, persist, toast, remember, graph]);

  const results = useMemo(() => Object.fromEntries((analysis?.nodes ?? []).map((r) => [r.id, r])), [analysis]);
  // React Flow keeps its own node state (it needs each node's measured size); the design (graph) is synced into it
  const [rfNodes, setRfNodes] = useState<Node<LayerData>[]>([]);
  // "×N" view: identical consecutive groups (e.g. 40 encoder blocks) drawn as one node; a read-only view, laid out compactly
  const [collapsed, setCollapsed] = useState(false);
  const foldable = useMemo(() => collapseRepeats(graph, analysis), [graph, analysis]);
  const folded = collapsed ? foldable : null;
  const canFold = !!foldable;
  // another design without repeats (or an edit that removed them) ends the ×N view, so the canvas is editable again
  useEffect(() => { if (!canFold) setCollapsed(false); }, [canFold]);
  useEffect(() => {
    if (folded) {
      setRfNodes(folded.nodes.map((n) => ({
        id: n.id, type: "layer", position: { x: n.x, y: n.y }, draggable: false, deletable: false, selectable: n.kind !== "repeat",
        data: n.kind === "repeat"
          ? { kind: "repeat", title: `${n.params.name} ×${n.params.count}`, params: n.params, summary: summary("repeat", n.params),
              result: { status: "ok", out: n.params.out, params: n.params.total, messages: [] } as unknown as NodeResult }
          : { kind: n.kind, title: titles[n.kind] ?? n.kind, params: n.params, result: results[n.id], summary: summary(n.kind, n.params) },
      })));
      return;
    }
    // a block whose look did not change keeps the same node object, so that React Flow and the memoised block skip it (a design of
    // thousands of blocks then redraws only what an edit touched)
    setRfNodes((prev) => {
      const byId = new Map(prev.map((o) => [o.id, o]));
      return graph.nodes.map((n) => {
        const old = byId.get(n.id), r = results[n.id];
        const sig = `${n.kind}|${JSON.stringify(n.params)}|${n.label ?? ""}|${r ? `${r.status}|${r.out}|${r.params}|${r.messages.join("~")}` : ""}`;
        const pos = old?.dragging ? old.position : { x: n.x, y: n.y };
        if (old && (old.data as LayerData & { sig?: string }).sig === sig && old.selected === (n.id === selected)
          && old.position.x === pos.x && old.position.y === pos.y) return old;
        return {
          ...(old ?? {}), id: n.id, type: "layer", position: pos, selected: n.id === selected, deletable: n.kind !== "input" || graph.nodes.filter((x) => x.kind === "input").length > 1,
          data: { kind: n.kind, title: titles[n.kind] ?? n.kind, params: n.params, result: r, summary: summary(n.kind, n.params), label: n.label, sig },
        };
      });
    });
  }, [graph, results, selected, titles, folded]);
  const shownEdges = (folded ?? graph).edges, edgeShapes = analysis?.edges;
  const edges: Edge[] = useMemo(() => shownEdges.map((e) => ({
    id: e.id, source: e.source, target: e.target, label: edgeShapes?.[e.id] ? fmt(edgeShapes[e.id]) : undefined,
    labelStyle: { fontSize: 10, fill: "var(--edge-label)" }, labelBgStyle: { fill: "var(--bg)" },
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: night ? "#00e5ff" : undefined },
  })), [shownEdges, edgeShapes, night]);

  const onNodesChange = useCallback((changes: NodeChange<Node<LayerData>>[]) => {
    setRfNodes((nds) => applyNodeChanges(changes, nds));
    if (collapsed) return; // the "×N" view is read-only
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
  }, [graph, save, collapsed]);

  // copy / paste of blocks (Edit menu or ⌘C ⌘V, outside text fields): the selected blocks and the arrows between them
  useEffect(() => {
    const typing = () => { const a = document.activeElement; return !!a && ["INPUT", "TEXTAREA", "SELECT"].includes(a.tagName); };
    const onCopy = (e: ClipboardEvent) => {
      if (typing() || collapsed) return;
      const ids = new Set(rfNodes.filter((n) => n.selected).map((n) => n.id).concat(selected ? [selected] : []));
      const nodes = graph.nodes.filter((n) => ids.has(n.id) && n.kind !== "input");
      if (!nodes.length) return;
      const keep = new Set(nodes.map((n) => n.id));
      e.clipboardData?.setData("text/plain", JSON.stringify({ nnb_blocks: 1, nodes, edges: graph.edges.filter((x) => keep.has(x.source) && keep.has(x.target)) }));
      e.preventDefault();
      toast(`Copied ${nodes.length} block${nodes.length > 1 ? "s" : ""}.`);
    };
    const onPaste = (e: ClipboardEvent) => {
      if (typing() || collapsed) return;
      let data: { nnb_blocks?: number; nodes: Graph["nodes"]; edges: Graph["edges"] };
      try { data = JSON.parse(e.clipboardData?.getData("text/plain") ?? ""); } catch { return; }
      if (!data?.nnb_blocks || !Array.isArray(data.nodes)) return;
      e.preventDefault();
      const tag = Date.now().toString(36), id = (old: string) => `${old}c${tag}`;
      const nodes = data.nodes.map((n) => ({ ...n, id: id(n.id), x: n.x + 60, y: n.y + 160 }));
      const edges = data.edges.map((x) => ({ id: id(x.id), source: id(x.source), target: id(x.target) }));
      save({ nodes: [...graph.nodes, ...nodes], edges: [...graph.edges, ...edges] });
      setSelected(nodes[0]?.id ?? null);
      toast(`Pasted ${nodes.length} block${nodes.length > 1 ? "s" : ""} below the originals: connect them with arrows.`);
    };
    document.addEventListener("copy", onCopy);
    document.addEventListener("paste", onPaste);
    return () => { document.removeEventListener("copy", onCopy); document.removeEventListener("paste", onPaste); };
  }, [rfNodes, selected, graph, save, toast, collapsed]);

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
  /** a residual block after the selected block, keeping its size and channels */
  const addResidual = () => {
    if (!sel || !analysis) return;
    const out = analysis.nodes.find((r) => r.id === sel.id)?.out;
    if (!out || out.length !== 3) return toast("A residual block goes after a block whose output is an image (channels, height, width).");
    const c = out[0], tag = Date.now().toString(36);
    const conv = { filters: c, kernel: [3, 3], stride: [1, 1], padding: "same", dilation: 1, activation: "none", bias: false };
    const chain: [string, string, Params][] = [["c1", "conv2d", conv], ["b1", "batchnorm", {}], ["r1", "activation", { fn: "relu" }],
      ["c2", "conv2d", conv], ["b2", "batchnorm", {}], ["add", "add", {}], ["r2", "activation", { fn: "relu" }]];
    const nodes = graph.nodes.map((n) => (n.x > sel.x ? { ...n, x: n.x + STEP_X * chain.length } : n));
    const made = chain.map(([k, kind, params], i) => ({ id: `${k}${tag}`, kind, params: { ...(defaults[kind] ?? {}), ...params } as Params,
      x: sel.x + STEP_X * (i + 1), y: sel.y + (k === "add" || k === "r2" ? 0 : 0) }));
    // what followed the selected block now follows the block's last ReLU
    let edges = graph.edges.map((e) => (e.source === sel.id ? { ...e, source: made[made.length - 1].id } : e));
    for (let i = 0; i < made.length - 1; i++) edges.push({ id: `e${tag}${i}`, source: made[i].id, target: made[i + 1].id });
    edges = [...edges, { id: `e${tag}in`, source: sel.id, target: made[0].id }, { id: `e${tag}skip`, source: sel.id, target: made[5].id }];
    save({ nodes: [...nodes, ...made], edges });
    setSelected(made[made.length - 1].id);
    toast(`Residual block added (${c} channels; 2 × ${(9 * c * c).toLocaleString()} convolution weights).`);
  };
  /** Siamese / twin branch: the straight branch from an Input to the selected block, again from the other band group's Input, with
   *  the same weights (Shared-weights blocks for every layer with weights; layers without weights copied) */
  const addTwin = () => {
    if (!sel) return;
    const chain: GNode[] = [];
    let cur: GNode | undefined = graph.nodes.find((n) => n.id === sel.id);
    while (cur && cur.kind !== "input") {
      const ins = graph.edges.filter((e) => e.target === cur!.id);
      if (ins.length !== 1) return toast("Twin branch copies a straight branch (one input per block) from an Input to the selected block.");
      chain.unshift(cur);
      cur = graph.nodes.find((n) => n.id === ins[0].source);
    }
    if (!cur || !chain.length) return toast("Select the last block of a branch that starts at an Input.");
    const tag = Date.now().toString(36);
    const inputs = graph.nodes.filter((n) => n.kind === "input");
    let nodes = graph.nodes;
    let other = inputs.find((n) => n.id !== cur!.id && Number(n.params.group ?? 0) === 2);
    if (!other) {
      other = { id: `in${tag}`, kind: "input", params: { group: 2 }, x: cur.x, y: cur.y + 260 };
      nodes = [...nodes.map((n) => (n.id === cur!.id ? { ...n, params: { ...n.params, group: 1 } } : n)), other];
    }
    const SHAREABLE = ["conv2d", "dwconv2d", "convtranspose2d", "dense", "batchnorm", "layernorm", "groupnorm", "attention", "cls_pos", "custom"];
    const made = chain.map((b, i) => ({ id: `tw${tag}${i}`, kind: SHAREABLE.includes(b.kind) ? "shared" : b.kind,
      params: SHAREABLE.includes(b.kind) ? { of: b.id } : { ...b.params }, x: b.x, y: b.y + 260 }));
    const edges = [...graph.edges, ...made.map((m, i) => ({ id: `e${tag}${i}`, source: i === 0 ? other!.id : made[i - 1].id, target: m.id }))];
    save({ ...graph, nodes: [...nodes, ...made], edges });
    setSelected(made[made.length - 1].id);
    toast(`Twin branch added: ${made.filter((m) => m.kind === "shared").length} layers share their weights. Join the two branches (e.g. Subtract |a − b| or Concat).`);
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
        <button className="arrange" onClick={() => save(arrange(graph))} disabled={collapsed}>Arrange left to right</button>
        {(canFold || collapsed) && <button className="arrange" onClick={() => setCollapsed(!collapsed)}
          title="Draw identical repeated blocks (e.g. Transformer encoder blocks) as one ×N node; a read-only view">
          {collapsed ? "Show every block" : "Collapse repeated blocks (×N)"}</button>}
        <button className="arrange" onClick={() => addResidual()} disabled={collapsed || !sel}
          title="After the selected block: Conv 3×3 → BN → ReLU → Conv 3×3 → BN → Add (skip from the selected block) → ReLU">Residual block after selected</button>
        <button className="arrange" onClick={() => addTwin()} disabled={collapsed || !sel}
          title="Siamese network: the branch from the Input to the selected block again, from the second band group's Input (e.g. the 'after' image), with the same weights">Twin branch (shared weights)</button>
        <p className="muted small">Click a block to add it after the selected one. Drag from a block's right dot to another block's left dot to connect (several arrows into Add / Concat / Multiply). Select + Backspace deletes.</p>
        {GROUPS.map(([g, kinds]) => (
          <div key={g}><h4>{g}</h4>{kinds.map((k) => (
            <button key={k} className="pal" style={{ borderLeftColor: COLORS[k], "--c": color(k) } as CSSProperties} onClick={() => addBlock(k)}>{titles[k]}</button>
          ))}</div>
        ))}
      </div>
      <div className="canvas">
        {/* measured: the canvas needs about 0.3 MB per block drawn (1,000 blocks ≈ 300 MB, 3,500 ≈ 1 GB); the parameter count does not matter */}
        {!collapsed && graph.nodes.length > 500 && <div className="msg warn big-hint">Large design: {graph.nodes.length.toLocaleString()} blocks drawn
          (about {Math.round(graph.nodes.length * 0.3)} MB for the canvas). {canFold ? <>
          <button className="small" onClick={() => setCollapsed(true)}>Collapse repeated blocks (×N)</button> draws each repeated group once and
          is much lighter;</> : null} the Text tab edits it without drawing.</div>}
        {rfNodes.length > 0 && <ReactFlow nodes={rfNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
          onPaneClick={() => setSelected(null)} fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} minZoom={0.05}
          deleteKeyCode={collapsed ? null : ["Backspace", "Delete"]} nodesConnectable={!collapsed}
          onNodeDoubleClick={(_, n) => { if (n.id.startsWith("repeat")) setCollapsed(false); }} key={collapsed ? "folded" : "all"}>
          <Background gap={24} />
          <Controls />
          <MiniMap pannable zoomable nodeColor={(n) => color((n.data as LayerData).kind)}
            maskColor={night ? "rgba(5,6,12,.7)" : undefined} />
        </ReactFlow>}
      </div>
      <Inspector node={sel} result={sel ? results[sel.id] : undefined} analysis={analysis} graph={graph} st={st} setParam={setParam} setParams={setParams}
        setPython={(code) => save({ ...graph, python: code })}
        addInput={() => {
          // mid fusion: this Input takes band group 1, a new Input below it group 2 (give it its own branch, then join them with Concat)
          if (!sel) return;
          const id = `n${Date.now().toString(36)}`;
          save({ nodes: [...graph.nodes.map((n) => (n.id === sel.id ? { ...n, params: { ...n.params, group: 1 } } : n)),
            { id, kind: "input", params: { group: 2 }, x: sel.x, y: sel.y + 180 }], edges: graph.edges });
          setSelected(id);
        }}
        candidates={candidates} addSkip={addSkip} />
    </div>
  );
}

// ---------------------------------------------------------------- inspector
const SHARE_KINDS = ["conv2d", "dwconv2d", "convtranspose2d", "dense", "batchnorm", "layernorm", "groupnorm", "attention", "cls_pos", "custom", "output", "seg_output"];
const EXAMPLE_CLASS = `class SE(nn.Module):
    """squeeze-and-excitation (Hu et al. 2018): one weight per channel, from the channel means"""
    # out: c, h, w
    def __init__(self, c, h, w):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(1, c // 4)), nn.ReLU(), nn.Linear(max(1, c // 4), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]
`;

/** the classes defined at the top level of the design's code (as the Rust side reads them: custom.rs) */
const pyClasses = (code: string) => [...code.matchAll(/^class ([A-Za-z_][A-Za-z0-9_]*)\s*[(:]/gm)].map((m) => m[1]);

/** Custom block: which class of the design's PyTorch code it runs, and that code (shared by every Custom block of the design) */
function CustomFields({ cls, code, setClass, setPython }: { cls: string; code: string; setClass: (c: string) => void; setPython: (c: string) => void }) {
  const [text, setText] = useState(code);
  const timer = useRef<number>(0);
  useEffect(() => { setText(code); }, [code]);
  const change = (t: string) => {
    setText(t);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setPython(t), 600);
  };
  const names = pyClasses(text);
  return <>
    <Sel label="Class" value={cls} options={[["", names.length ? "choose…" : "(no class in the code yet)"], ...names.map((n): [string, string] => [n, n])]}
      onChange={(v) => setClass(v)} />
    <label className="code-label">The design's PyTorch code (all its Custom blocks share it)
      <textarea className="code" spellCheck={false} rows={16} value={text} onChange={(e) => change(e.target.value)}
        onBlur={() => { window.clearTimeout(timer.current); if (text !== code) setPython(text); }}
        onKeyDown={(e) => {
          if (e.key !== "Tab") return;
          e.preventDefault();
          const t = e.currentTarget, a = t.selectionStart, b = t.selectionEnd;
          change(text.slice(0, a) + "    " + text.slice(b));
          requestAnimationFrame(() => { t.selectionStart = t.selectionEnd = a + 4; });
        }} /></label>
    <p className="row small"><button className="small" onClick={() => { const t = (text.trim() ? text.trimEnd() + "\n\n\n" : "") + EXAMPLE_CLASS; change(t); if (!cls) setClass("SE"); }}
      disabled={names.includes("SE")}>Insert an example (squeeze-and-excitation)</button></p>
    <p className="muted small">Each class is an <code>nn.Module</code> at the top level. Its <code>__init__(self, c, h, w)</code> receives the input's
      shape (<code>t, d</code> for tokens, <code>n</code> for a vector). A line <code># out: c * 2, h // 2, w // 2</code> inside the class gives
      the output's shape in those names (without it: the same as the input), so the rest of the design can be checked here. The code runs only
      on the training machine, where the one-batch check compares the real shape with the declared one before training. <code>torch</code> and
      <code>nn</code> are imported; import anything else at the top of the code.</p>
  </>;
}

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

// ---------------------------------------------------------------- layer table, grouped into stages
type Group = { name: string; rows: NodeResult[] };
const rank = (r: NodeResult) => r.out?.length ?? 0;
const hw = (s: number[]) => `${s[s.length - 2]}×${s[s.length - 1]}`;
const WORST = ["ok", "warn", "blocked", "unconnected", "error"];

/** Input; image stages by resolution (a pooling layer, a Dropout after it, and vector layers inside a stage such as a squeeze-and-
 *  excitation gate stay with their stage); for tokens an Embedding
 *  (with the patch convolution before it) and one group per encoder block (a block starts at a LayerNorm after the MLP's Add); the
 *  classifier (rank 1) */
export function groupLayers(nodes: NodeResult[]): Group[] {
  const groups: (Group & { key: string })[] = [];
  let stage = 0, block = 0;
  const rows = nodes.filter((r) => r.out);
  // vectors before the last image/token layer belong to their stage (a squeeze-and-excitation gate), not to the head
  let lastWide = -1;
  rows.forEach((r, i) => { if (rank(r) >= 2) lastWide = i; });
  for (const [i, r] of rows.entries()) {
    const out = r.out!, cur = groups[groups.length - 1], prev = cur?.rows[cur.rows.length - 1];
    let key: string;
    if (r.kind === "input") key = "input";
    else if (rank(r) >= 3) {
      const inp = r.inputs[0];
      key = r.kind === "dropout" && prev?.kind === "pool2d" && cur.key.startsWith("img") ? cur.key : `img ${hw(r.kind === "pool2d" && inp?.length >= 3 ? inp : out)}`;
    } else if (rank(r) === 2) {
      const starts = r.kind === "layernorm" && (prev?.kind === "cls_pos" || (prev?.kind === "add" && cur.rows.some((x) => x.kind === "dense")));
      key = starts || !cur?.key.startsWith("tok") ? `tok ${groups.length}` : cur.key;
    } else key = i < lastWide && cur ? cur.key : "head";
    if (cur && cur.key === key) { cur.rows.push(r); continue; }
    let name = "Input";
    if (key.startsWith("img")) {
      const up = prev && rank(prev) >= 3 && out[out.length - 1] > prev.out![prev.out!.length - 1];
      name = `Stage ${++stage} · ${key.slice(4)}${up ? " ↑" : ""}`;
    } else if (key.startsWith("tok")) name = "Tokens";
    else if (key === "head") name = r.kind === "token_pool" || nodes.some((n) => n.kind === "output") ? "Classifier" : "Dense";
    groups.push({ key, name, rows: [r] });
  }
  for (const g of groups) if (g.key.startsWith("tok")) {
    g.name = g.rows.some((r) => r.kind === "attention") ? `Encoder block ${++block}` : g.rows.some((r) => r.kind === "to_tokens") ? "Embedding" : "Final norm";
  }
  const e = groups.findIndex((g) => g.name === "Embedding");
  if (e > 1 && groups[e - 1].key.startsWith("img") && groups[e - 1].rows.length === 1) { groups[e].rows.unshift(...groups[e - 1].rows); groups.splice(e - 1, 1); }
  return groups;
}

/** the line of the design text that starts with `label` at the top level (not inside braces), without its label, braces and comment */
function textLine(text: string, label: string): string {
  let depth = 0;
  for (const raw of text.split("\n")) {
    const line = raw.replace(/#.*/, "");
    if (depth === 0) {
      const m = new RegExp(`^\\s*${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s+(.*)$`).exec(line);
      if (m) return m[1].replace(/\{\s*$/, "").trim();
    }
    depth += (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length;
  }
  return "";
}

/** for a design written as text with repeat / for / function calls: the blocks of one such line form one group, named after the line
 *  ("5 · for c, s in (w, 1), (w * 2, 2)"); the other blocks keep the groups found from their shapes. Unchanged without such lines. */
export function groupByText(nodes: NodeResult[], g: Graph): Group[] {
  const label = new Map(g.nodes.map((n) => [n.id, n.label ?? ""]));
  if (!g.text || ![...label.values()].some((l) => l.includes("."))) return groupLayers(nodes);
  // the table in the order of the text (the engine's order may interleave branches): by the line of the block's top-level label
  const lineOf = new Map<string, number>();
  let depth = 0;
  g.text.split("\n").forEach((raw, i) => {
    const line = raw.replace(/#.*/, "");
    const m = depth === 0 ? /^\s*([A-Za-z0-9_]+)\s+\S/.exec(line) : null;
    if (m) lineOf.set(m[1], i);
    depth += (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length;
  });
  const index = new Map(g.nodes.map((n, i) => [n.id, i]));
  const key = (r: NodeResult) => [lineOf.get((label.get(r.id) ?? "").split(".")[0]) ?? 1e9, index.get(r.id) ?? 1e9];
  const sorted = [...nodes].sort((a, b) => { const [x, y] = [key(a), key(b)]; return x[0] - y[0] || x[1] - y[1]; });
  const groups = groupLayers(sorted);
  const out: (Group & { key: string })[] = [];
  groups.forEach((gr, gi) => gr.rows.forEach((r) => {
    const l = label.get(r.id) ?? "";
    const top = l.split(".")[0];
    const key = l.includes(".") ? `line ${top}` : `shape ${gi}`;
    const cur = out[out.length - 1];
    if (cur && cur.key === key) { cur.rows.push(r); return; }
    const line = l.includes(".") ? textLine(g.text!, top) : "";
    const name = l.includes(".") ? `${top} · ${line.length > 42 ? line.slice(0, 40) + "…" : line}` : gr.name;
    out.push({ key, name, rows: [r] });
  }));
  // groups of plain lines: named by their lines (the shape name only where it still fits the blocks left in the group)
  for (const gr of out) if (gr.key.startsWith("shape")) {
    const ls = gr.rows.map((r) => label.get(r.id) ?? "").filter((l) => l);
    const range = ls.length > 1 ? `${ls[0]}–${ls[ls.length - 1]}` : ls[0] ?? "";
    const fits = !gr.name.startsWith("Encoder block") || gr.rows.some((r) => r.kind === "attention");
    gr.name = range ? `${range} · ${fits ? gr.name : "Tokens"}` : gr.name;
  }
  return out;
}

/** "Conv2D ×3, Pool": runs of the same block, each with its kind */
function contents(rows: NodeResult[]): { title: string; kind: string; n: number }[] {
  const parts: { title: string; kind: string; n: number }[] = [];
  for (const r of rows) {
    const last = parts[parts.length - 1];
    if (last && last.title === r.title) last.n++; else parts.push({ title: r.title, kind: r.kind, n: 1 });
  }
  return parts;
}
const contentsText = (rows: NodeResult[]) => contents(rows).map((p) => (p.n > 1 ? `${p.title} ×${p.n}` : p.title)).join(", ");

/** the design with each run of identical consecutive groups (same blocks, parameters and output) replaced by one "repeat" node, laid out
 *  left to right; null when nothing repeats */
export function collapseRepeats(g: Graph, a: Analysis | null): Graph | null {
  if (!a) return null;
  const groups = groupLayers(a.nodes);
  const sig = (gr: Group) => `${contentsText(gr.rows)}|${gr.rows.reduce((t, r) => t + r.params, 0)}|${JSON.stringify(gr.rows[gr.rows.length - 1].out)}`;
  const owner = new Map<string, string>();
  const synth: Graph["nodes"] = [];
  for (let i = 0; i < groups.length;) {
    let j = i;
    while (j + 1 < groups.length && sig(groups[j + 1]) === sig(groups[i])) j++;
    if (j > i) {
      const id = `repeat${synth.length}`;
      const members = groups.slice(i, j + 1);
      members.forEach((gr) => gr.rows.forEach((r) => owner.set(r.id, id)));
      synth.push({ id, kind: "repeat", x: 0, y: 0, params: { count: j - i + 1, name: groups[i].name.replace(/\s*\d+$/, ""),
        total: members.reduce((t, gr) => t + gr.rows.reduce((u, r) => u + r.params, 0), 0), out: groups[j].rows[groups[j].rows.length - 1].out } as Params });
    }
    i = j + 1;
  }
  if (!synth.length) return null;
  const seen = new Set<string>();
  const edges = g.edges.map((e) => ({ ...e, source: owner.get(e.source) ?? e.source, target: owner.get(e.target) ?? e.target }))
    .filter((e) => e.source !== e.target && !seen.has(`${e.source}>${e.target}`) && (seen.add(`${e.source}>${e.target}`), true));
  return arrange({ nodes: [...g.nodes.filter((n) => !owner.has(n.id)), ...synth], edges });
}

function LayerTable({ nodes, graph }: { nodes: NodeResult[]; graph: Graph }) {
  const groups = useMemo(() => groupByText(nodes, graph), [nodes, graph]);
  const color = useKindColor();
  /** a block's name in its kind's colour (as on the canvas), with ✕ / ⚠ when it has a problem */
  const kc = (kind: string, text: string, status = "ok") => <span className="kc" style={{ "--c": color(kind) } as CSSProperties}>{text}{status === "error" ? " ✕" : status === "warn" ? " ⚠" : ""}</span>;
  const mark = (st: string) => (st === "error" || st === "warn" ? `st-${st}` : "");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const all = open.size === groups.length;
  const flip = (name: string) => setOpen((o) => { const n = new Set(o); if (n.has(name)) n.delete(name); else n.add(name); return n; });
  const rf = (r?: NodeResult) => (r?.rf ? `${r.rf[0]}×${r.rf[1]}` : "");
  return (
    <>
      <div className="row"><span className="muted small grow">{groups.length} groups · {nodes.filter((r) => r.out).length} layers</span>
        <button className="small" onClick={() => setOpen(all ? new Set() : new Set(groups.map((g) => g.name)))}>{all ? "Collapse all" : "Expand all"}</button></div>
      <table className="grid small layers"><tbody>
        <tr><th>Block</th><th>Out</th><th className="num">Params</th><th>RF</th></tr>
        {groups.map((g) => {
          const last = g.rows[g.rows.length - 1], single = g.rows.length === 1;
          const worst = g.rows.reduce((w, r) => (WORST.indexOf(r.status) > WORST.indexOf(w) ? r.status : w), "ok");
          const isOpen = open.has(g.name) && !single;
          return [
            <tr key={g.name} className={`ghead ${mark(worst)} ${single ? "" : "click"}`} onClick={() => !single && flip(g.name)}>
              <td><span className="caret">{single ? "" : isOpen ? "▾" : "▸"}</span>
                {single ? <strong>{kc(last.kind, g.name, worst)}</strong> : <strong>{g.name}{worst === "error" ? " ✕" : worst === "warn" ? " ⚠" : ""}</strong>}
                {!single && <div className="gsub" title={contentsText(g.rows)}>{contents(g.rows).map((p, i) =>
                  <span key={i}>{i ? ", " : ""}{kc(p.kind, p.n > 1 ? `${p.title} ×${p.n}` : p.title)}</span>)}</div>}</td>
              <td>{fmt(last.out)}</td><td className="num">{g.rows.reduce((t, r) => t + r.params, 0).toLocaleString()}</td><td>{rf(last)}</td></tr>,
            ...(isOpen ? g.rows.map((r) => <tr key={r.id} className={`gitem ${mark(r.status)}`}><td>{kc(r.kind, r.title, r.status)}</td><td>{fmt(r.out)}</td>
              <td className="num">{r.params.toLocaleString()}</td><td>{rf(r)}</td></tr>) : []),
          ];
        })}
      </tbody></table>
    </>
  );
}

function Inspector({ node, result, analysis, graph, st, setParam, setParams, candidates, addSkip, addInput, setPython }: {
  node: { id: string; kind: string; params: Params } | null; result?: NodeResult; analysis: Analysis | null; graph: Graph; st: AppState;
  setParam: (k: string, v: unknown) => void; setParams: (p: Params) => void; addInput: () => void; setPython: (code: string) => void;
  candidates: { id: string; label: string }[]; addSkip: (src: string, mode: "add" | "concat") => void;
}) {
  const color = useKindColor();
  if (!node) {
    return (
      <div className="inspector">
        <h3>Design</h3>
        {analysis && <p><strong>{analysis.total_params.toLocaleString()}</strong> trainable parameters · input {fmt(analysis.input)} · {analysis.classes} classes
          {analysis.total_macs > 0 && <> · <span title="multiply-accumulate operations for one image (convolutions, dense layers, attention)">
            {analysis.total_macs >= 1e9 ? `${(analysis.total_macs / 1e9).toFixed(2)} G` : `${(analysis.total_macs / 1e6).toFixed(1)} M`} MACs per image</span></>}</p>}
        {analysis?.problems.map((p) => <div key={p} className="msg error">{p}</div>)}
        {analysis?.ok && <div className="msg ok">The design is complete: every shape fits. Next: Dataset, then Train.</div>}
        {graph.nodes.some((n) => n.kind === "custom") && <p className="muted small">The parameters of Custom blocks (your PyTorch classes) are not
          in this count: the training machine counts them (shown in the run's log).</p>}
        <p className="muted">Click a block to set its filters, kernel, stride, padding or pooling, and to see how its output size is computed.</p>
        {analysis && <LayerTable nodes={analysis.nodes} graph={graph} />}
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
          <Num label="Filters (output channels)" value={p.filters} max={65536} onChange={(v) => setParam("filters", v)} />
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
          <Num label="Filters" value={p.filters} max={65536} onChange={(v) => setParam("filters", v)} />
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
        {node.kind === "custom" && <CustomFields cls={String(p.class ?? "")} code={graph.python ?? ""} setClass={(c) => setParam("class", c)} setPython={setPython} />}
        {node.kind === "groupnorm" && <><Num label="Groups" value={p.groups} min={1} max={65536} onChange={(v) => setParam("groups", v)} />
          <p className="muted">Normalises each image over groups of channels (Wu &amp; He 2018): unlike BatchNorm it does not depend on the batch,
            so it works with small batches. The channels must divide evenly by the number of groups (32 is common; 1 group = LayerNorm over the image).</p></>}
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
          <Num label="Heads" value={p.heads} max={512} onChange={(v) => setParam("heads", v)} />
          <Num label="Dropout" value={p.dropout} min={0} max={0.9} step={0.05} onChange={(v) => setParam("dropout", v)} />
          <p className="muted">Every token looks at every other token (multi-head self-attention). The token dimension must divide evenly by the number of heads. Parameters: 4·d² + 4·d.</p>
        </>}
        {node.kind === "token_pool" && <Sel label="Use" value={p.mode} options={[["cls", "the class token (ViT)"], ["mean", "the mean of all tokens"], ["last", "the last step (after an LSTM / GRU)"]]} onChange={(v) => setParam("mode", v)} />}
        {node.kind === "activation" && <Sel label="Function" value={p.fn} options={acts.filter(([a]) => a !== "none")} onChange={(v) => setParam("fn", v)} />}
        {node.kind === "layernorm" && <p className="muted">On an image it normalises over (channels, height, width) (GroupNorm with one group); on a vector, over its features.</p>}
        {node.kind === "conv3d" && <>
          <Num label="Filters" value={p.filters} max={65536} onChange={(v) => setParam("filters", v)} />
          <Pair label="Kernel (pixels)" value={p.kernel} max={15} onChange={(v) => setParam("kernel", v)} />
          <Num label="Dates per window" value={p.tkernel} min={1} max={15} onChange={(v) => setParam("tkernel", v)} />
          <Pair label="Stride (pixels)" value={p.stride} max={8} def={1} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={String(p.padding ?? "same")} options={[["same", "same (keep dates and size; odd kernels)"], ["valid", "valid (none)"]]} onChange={(v) => setParam("padding", v)} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} />
          <p className="muted">A convolution over dates and pixels at once: each filter sees a few neighbouring dates of a small window.</p></>}
        {node.kind === "temporal_pool" && <>
          <Sel label="Combine the dates by" value={String(p.mode ?? "attention")} options={[["attention", "attention (learnt weights per pixel, U-TAE)"], ["mean", "mean"], ["max", "maximum"]]} onChange={(v) => setParam("mode", v)} />
          <p className="muted">(channels, dates, height, width) → (channels, height, width): after it every image block works, e.g. a decoder to a crop-type map.</p></>}
        {node.kind === "to_sequence" && <p className="muted">(channels, dates, height, width) → (dates, channels): the mean of every date's pixels. Then
          Conv1D, LSTM / GRU or the transformer blocks work along the dates (a time series of the whole patch).</p>}
        {(node.kind === "lstm" || node.kind === "gru") && <>
          <Num label="Units" value={p.units} max={65536} onChange={(v) => setParam("units", v)} />
          <label className="chk"><input type="checkbox" checked={!!p.bidirectional} onChange={(e) => setParam("bidirectional", e.target.checked)} />both directions (twice the units out)</label>
          <p className="muted">A recurrent layer along the steps; it gives its output at every step (Token pool “last” or “mean” makes one vector).</p></>}
        {node.kind === "conv1d" && <>
          <Num label="Filters" value={p.filters} max={65536} onChange={(v) => setParam("filters", v)} />
          <Num label="Kernel (steps)" value={p.kernel} min={1} max={31} onChange={(v) => setParam("kernel", v)} />
          <Num label="Stride" value={p.stride ?? 1} min={1} max={16} onChange={(v) => setParam("stride", v)} />
          <Sel label="Padding" value={String(p.padding ?? "same")} options={[["same", "same (odd kernels)"], ["valid", "valid"]]} onChange={(v) => setParam("padding", v)} />
          <Sel label="Activation" value={p.activation} options={acts} onChange={(v) => setParam("activation", v)} /></>}
        {node.kind === "subtract" && <><label className="chk"><input type="checkbox" checked={!!p.abs} onChange={(e) => setParam("abs", e.target.checked)} />
          absolute difference |a − b|</label>
          <p className="muted">The first input minus the second (the order of the arrows); equal shapes. In a Siamese change-detection network the
            two branches see the before and after images, and |a − b| keeps what changed.</p></>}
        {node.kind === "shared" && <>
          <Sel label="Uses the weights of" value={String(p.of ?? "")} options={[["", "choose…"], ...(analysis?.nodes ?? [])
            .filter((r) => SHARE_KINDS.includes(r.kind) && r.id !== node.id).map((r): [string, string] =>
              [r.id, `${graph.nodes.find((n) => n.id === r.id)?.label || "·"} ${r.title} ${fmt(r.out)}`])]} onChange={(v) => setParam("of", v)} />
          <p className="muted">Runs that block's layer again, with the same weights, on this block's input (which must have the shape that block's
            input has). The weights are trained by both uses and counted once: a Siamese (twin) network. The Twin branch button makes a whole branch of these.</p></>}
        {node.kind === "add" && <p className="muted">Adds two or more branches element by element (a residual connection): their shapes must be equal.</p>}
        {node.kind === "concat" && <p className="muted">Joins branches along channels (images: equal height and width) or features (vectors). The order of the arrows is the order of the channels.</p>}
        {node.kind === "output" && <>
          {(analysis?.head_names?.length ?? 0) > 0 && <Sel label="Output for" value={String(p.head ?? 0)}
            options={[["0", "the main task"], ...(analysis?.head_names ?? []).map((n, i): [string, string] => [String(i + 1), `${i + 1}: ${n}`])]}
            onChange={(v) => setParam("head", Number(v))} />}
          <p className="muted">{Number(p.head ?? 0) === 0 ? `One score per class of the dataset (${analysis?.classes ?? "?"}); the loss applies softmax.`
            : `The extra output “${analysis?.head_names?.[Number(p.head) - 1] ?? "?"}” (Dataset tab): its classes or numbers, with its own loss.`}
            {(analysis?.head_names?.length ?? 0) > 0 && " A multi-task network has one Output per output, usually each after its own small branch (a Dense or two) from a shared body."}</p>
        </>}
        {node.kind === "seg_output" && <p className="muted">A 1×1 convolution giving a score per class ({analysis?.classes ?? "?"}) for every pixel. Its size must equal the input (the masks).</p>}
        {node.kind === "multiply" && <p className="muted">Multiplies two branches element by element: equal shapes, a one-channel map times an image (an attention gate: the map, after a sigmoid, weights every pixel), or one weight per channel (squeeze-and-excitation).</p>}
        {node.kind === "input" && <>
          <p className="muted">The images of the dataset (Dataset tab): {fmt(analysis?.input)}.</p>
          <Sel label="Bands" value={String(p.group ?? 0)} options={[["0", "all (both band groups stacked: early fusion)"], ["1", "band group 1 (e.g. Sentinel-2)"],
            ["2", "band group 2 (e.g. Sentinel-1)"]]} onChange={(v) => setParam("group", Number(v))} />
          {graph.nodes.filter((n) => n.kind === "input").length === 1 && <p className="row small"><button className="small" onClick={addInput}>Add an Input for band group 2</button><span className="muted">two branches, one per sensor, joined by Concat (mid fusion)</span></p>}
        </>}
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

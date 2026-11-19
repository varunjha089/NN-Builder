// Canvas layouts for large designs: blocks grouped by the text's function calls and loops (one box per call), laid out by image size
// (a U-Net draws as a U), and the lanes the right-angle arrows take so that they do not lie on top of each other.
import type { Analysis, Graph, GNode, NodeResult, Params } from "./api";

export const STEP_X = 300, STEP_Y = 150;

// ---------------------------------------------------------------- the text's structure
type Stmt = { rest: string; body: Map<string, Stmt> | null };
type Def = { params: [string, string][]; body: Map<string, Stmt> };
type Parsed = { top: Map<string, Stmt>; defs: Map<string, Def> };

/** split at the commas outside brackets: "c * 2, f(a, b)" -> ["c * 2", "f(a, b)"] */
function args(text: string): string[] {
  const out: string[] = [];
  let depth = 0, cur = "";
  for (const ch of text) {
    if (ch === "," && depth === 0) { out.push(cur.trim()); cur = ""; continue; }
    if ("([".includes(ch)) depth++;
    if (")]".includes(ch)) depth--;
    cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** statements of a list of lines: name -> what follows it, with the lines inside { } parsed the same way; defs go to `defs` */
function statements(lines: string[], defs: Map<string, Def>): Map<string, Stmt> {
  const out = new Map<string, Stmt>();
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].replace(/#.*/, "").trim();
    const m = /^(\S+)\s+(.*)$/.exec(line);
    if (!m) continue;
    let body: string[] | null = null;
    if (/\{\s*$/.test(line)) {
      body = [];
      let depth = 1;
      for (i++; i < lines.length; i++) {
        const l = lines[i].replace(/#.*/, "");
        depth += (l.match(/\{/g) ?? []).length - (l.match(/\}/g) ?? []).length;
        if (depth <= 0) break;
        body.push(l);
      }
    }
    const rest = m[2].replace(/\{\s*$/, "").trim();
    if (m[1] === "def") {
      const d = /^([A-Za-z_]\w*)\s*\((.*)\)/.exec(rest);
      if (d) defs.set(d[1], { params: args(d[2]).map((a) => { const [k, v] = a.split("="); return [k.trim(), (v ?? "").trim()]; }),
        body: statements(body ?? [], defs) });
    } else if (m[1] !== "net" && m[1] !== "let") {
      out.set(m[1], { rest, body: body ? statements(body, defs) : null });
    }
  }
  return out;
}

export function parseText(text: string): Parsed {
  const defs = new Map<string, Def>();
  return { top: statements(text.split("\n"), defs), defs };
}

const isLoop = (s: Stmt) => /^(for|repeat)\b/.test(s.rest);
const callee = (s: Stmt, p: Parsed) => { const f = /^([A-Za-z_]\w*)\s*\(/.exec(s.rest)?.[1]; return f && p.defs.has(f) ? f : null; };

/** the loop's value in pass k ("for c in 128, 256, 512": pass 2 -> c = 256); null for tuples, ranges and repeat */
function loopValue(rest: string, k: number): [string, string] | null {
  const m = /^for\s+([A-Za-z_]\w*)\s+in\s+([^()]+)$/.exec(rest);
  const vals = m?.[2].split(",").map((v) => v.trim());
  return m && vals && vals[k - 1] ? [m[1], vals[k - 1]] : null;
}

/** a label prefix in words, with the values of loop variables and function arguments filled in: "8" -> "up(256, 256)",
 *  "7.2" -> "c = 256: down(256), basic(256)", "7.2.x" -> "down(256)", "8.g" -> "gate(256)" */
export function describe(p: Parsed, prefix: string): string {
  const segs = prefix.split(".");
  let stmt = p.top.get(segs[0]);
  if (!stmt) return prefix;
  let vars = new Map<string, string>();
  const fill = (t: string) => t.replace(/\b[A-Za-z_]\w*\b/g, (w) => vars.get(w) ?? w);
  const show = (s: Stmt) => fill(s.rest.replace(/\s*<-.*$/, ""));
  let text = show(stmt);
  for (let i = 1; i < segs.length && stmt; i++) {
    if (isLoop(stmt) && /^\d+$/.test(segs[i])) {
      const v = loopValue(stmt.rest, Number(segs[i]));
      if (v) vars.set(v[0], fill(v[1]));
      const n = (stmt.rest.match(/^repeat\s+(\d+)/) ?? [])[1];
      const inner = [...(stmt.body?.values() ?? [])].map(show);
      const what = inner.length && inner.length <= 3 ? inner.join(", ") : `pass ${segs[i]}`;
      text = v ? `${v[0]} = ${vars.get(v[0])}: ${what}` : `${what}${n ? ` (${segs[i]} of ${n})` : ""}`;
      continue; // the next segment names a statement of the loop's body
    }
    let body: Map<string, Stmt> | null | undefined = stmt.body;
    const f = callee(stmt, p);
    if (f) {
      // into the function: its parameters take the call's arguments (or their defaults)
      const def = p.defs.get(f)!;
      const given = args(show(stmt).replace(/^[A-Za-z_]\w*\s*\(/, "").replace(/\)[^)]*$/, ""));
      const next = new Map(vars);
      const named = new Map(given.filter((g) => /^\w+\s*=[^=]/.test(g)).map((g) => [g.split("=")[0].trim(), g.slice(g.indexOf("=") + 1).trim()]));
      const pos = given.filter((g) => !/^\w+\s*=[^=]/.test(g));
      def.params.forEach(([k, dflt], j) => next.set(k, named.get(k) ?? pos[j] ?? fill(dflt)));
      vars = next;
      body = def.body;
    }
    stmt = body?.get(segs[i]);
    if (stmt) text = show(stmt);
  }
  return text;
}

/** the prefixes opened at first: loops (their passes are shown), so a "for" over three stages draws as three boxes */
export function defaultOpen(g: Graph): Set<string> {
  const open = new Set<string>();
  for (const n of g.nodes) {
    const s = (n.label ?? "").split(".");
    for (let i = 1; i < s.length; i++) if (/^\d+$/.test(s[i])) open.add(s.slice(0, i).join("."));
  }
  return open;
}

/** whether the design has groups to fold (blocks made by functions or loops in the text) */
export const groupable = (g: Graph) => g.nodes.some((n) => (n.label ?? "").includes("."));

// ---------------------------------------------------------------- one box per call
const WORST = ["ok", "warn", "blocked", "unconnected", "error"];

/** the design with every closed call / loop pass drawn as one "group" node (params: name, title, count, total, out, status, members);
 *  arrows inside a group are left out, arrows between groups kept once (with the id of the first original arrow, for its shape) */
export function groupByCalls(g: Graph, a: Analysis | null, open: Set<string>): Graph {
  const parsed = parseText(g.text ?? "");
  const owner = new Map<string, string>(), members = new Map<string, string[]>();
  for (const n of g.nodes) {
    const s = (n.label ?? "").split(".");
    for (let k = 1; k < s.length; k++) {
      const p = s.slice(0, k).join(".");
      if (!open.has(p)) { owner.set(n.id, p); break; }
    }
    const p = owner.get(n.id);
    if (p) { if (!members.has(p)) members.set(p, []); members.get(p)!.push(n.id); }
  }
  for (const [p, ids] of members) if (ids.length === 1) { owner.delete(ids[0]); members.delete(p); }
  const res = new Map((a?.nodes ?? []).map((r) => [r.id, r]));
  const order = new Map((a?.nodes ?? []).map((r, i) => [r.id, i]));
  const gid = (p: string) => `grp:${p}`;
  const outs = new Map<string, string[]>();
  g.edges.forEach((e) => { if (!outs.has(e.source)) outs.set(e.source, []); outs.get(e.source)!.push(e.target); });
  const synth: GNode[] = [...members].map(([p, ids]) => {
    const rows = ids.map((id) => res.get(id)).filter((r): r is NodeResult => !!r);
    const set = new Set(ids);
    // the group's output: the last block (in computation order) whose arrow leaves the group
    const leaving = ids.filter((id) => (outs.get(id) ?? []).some((t) => !set.has(t)));
    const exit = (leaving.length ? leaving : ids).reduce((x, y) => ((order.get(y) ?? -1) > (order.get(x) ?? -1) ? y : x));
    const status = rows.reduce((w, r) => (WORST.indexOf(r.status) > WORST.indexOf(w) ? r.status : w), "ok");
    const kinds = new Map<string, number>();
    rows.forEach((r) => kinds.set(r.title, (kinds.get(r.title) ?? 0) + 1));
    return { id: gid(p), kind: "group", x: 0, y: 0, label: p, params: {
      name: p, title: describe(parsed, p), count: ids.length, total: rows.reduce((t, r) => t + r.params, 0), out: res.get(exit)?.out ?? null,
      status, exit, contents: [...kinds].map(([t, n]) => (n > 1 ? `${t} ×${n}` : t)).join(", "),
    } as Params };
  });
  const seen = new Set<string>();
  const edges = g.edges.map((e) => ({ ...e, source: owner.has(e.source) ? gid(owner.get(e.source)!) : e.source,
    target: owner.has(e.target) ? gid(owner.get(e.target)!) : e.target }))
    .filter((e) => e.source !== e.target && !seen.has(`${e.source}>${e.target}`) && (seen.add(`${e.source}>${e.target}`), true));
  return { ...g, nodes: [...g.nodes.filter((n) => !owner.has(n.id)), ...synth], edges };
}

// ---------------------------------------------------------------- the U layout
/** a block's level: how many times its image is smaller than the input (log2), so halving steps one level down and doubling one up;
 *  vectors and tokens keep the level of their input */
export function levels(g: Graph, a: Analysis | null): Map<string, number> {
  const res = new Map((a?.nodes ?? []).map((r) => [r.id, r]));
  const outOf = (n: GNode) => (n.kind === "group" ? (n.params.out as number[] | null) : res.get(n.id)?.out) ?? null;
  const h0 = a && a.input.length >= 3 ? a.input[a.input.length - 2] : 0;
  const lv = new Map<string, number>();
  const ins = new Map<string, string[]>();
  g.edges.forEach((e) => { if (!ins.has(e.target)) ins.set(e.target, []); ins.get(e.target)!.push(e.source); });
  for (const id of topoOrder(g)) {
    const n = g.nodes[index(g).get(id)!], out = outOf(n);
    lv.set(id, out && out.length >= 3 && h0 > 0 ? Math.round(Math.log2(h0 / out[out.length - 2]))
      : Math.max(0, ...(ins.get(id) ?? []).map((s) => lv.get(s) ?? 0)));
  }
  const low = Math.min(0, ...lv.values());
  for (const [k, v] of lv) lv.set(k, v - low);
  return lv;
}

const indexCache = new WeakMap<Graph, Map<string, number>>();
const index = (g: Graph) => { let m = indexCache.get(g); if (!m) { m = new Map(g.nodes.map((n, i) => [n.id, i])); indexCache.set(g, m); } return m; };

/** the blocks in computation order (Kahn), and each block's depth: the longest path from a block without inputs; blocks on a loop last */
function depths(g: Graph): { order: string[]; depth: Map<string, number> } {
  const indeg = new Map(g.nodes.map((n) => [n.id, 0]));
  const outs = new Map<string, string[]>();
  for (const e of g.edges) {
    if (!indeg.has(e.source) || !indeg.has(e.target)) continue;
    indeg.set(e.target, indeg.get(e.target)! + 1);
    if (!outs.has(e.source)) outs.set(e.source, []); outs.get(e.source)!.push(e.target);
  }
  const order = g.nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  const depth = new Map(order.map((id) => [id, 0]));
  for (let i = 0; i < order.length; i++) {
    const id = order[i];
    for (const t of outs.get(id) ?? []) {
      depth.set(t, Math.max(depth.get(t) ?? 0, depth.get(id)! + 1));
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) order.push(t);
    }
  }
  const placed = new Set(order);
  g.nodes.forEach((n) => { if (!placed.has(n.id)) { order.push(n.id); depth.set(n.id, depth.get(n.id) ?? 0); } });
  return { order, depth };
}
const topoOrder = (g: Graph) => depths(g).order;

/** left to right in computation order (a column per depth, the longest path from the Input), top to bottom by level; blocks of the same
 *  column and level are stacked, and each level is as tall as its tallest stack, so skip connections between equal sizes run flat */
export function arrangeByLevel(g: Graph, lv: Map<string, number>): Graph {
  const { depth } = depths(g);
  const cells = new Map<string, string[]>();
  for (const n of g.nodes) {
    const k = `${depth.get(n.id) ?? 0}|${lv.get(n.id) ?? 0}`;
    if (!cells.has(k)) cells.set(k, []); cells.get(k)!.push(n.id);
  }
  const tall = new Map<number, number>();
  for (const [k, ids] of cells) { const l = Number(k.split("|")[1]); tall.set(l, Math.max(tall.get(l) ?? 1, ids.length)); }
  const top = new Map<number, number>();
  let y = 60;
  for (const l of [...tall.keys()].sort((a, b) => a - b)) { top.set(l, y); y += tall.get(l)! * STEP_Y + 40; }
  const pos = new Map<string, [number, number]>();
  for (const [k, ids] of cells) {
    const [d, l] = k.split("|").map(Number);
    ids.forEach((id, i) => pos.set(id, [60 + d * STEP_X, top.get(l)! + i * STEP_Y]));
  }
  return { ...g, nodes: g.nodes.map((n) => ({ ...n, x: pos.get(n.id)![0], y: pos.get(n.id)![1] })) };
}

// ---------------------------------------------------------------- arrows
/** how an arrow is drawn (offsets in canvas pixels, from its ends):
 *  - "stub": it goes back to the left (a design wrapped into rows): two short labelled stubs instead of a line across the canvas;
 *  - "step": flat, or it turns once, in the gap just before its target (`turn`: pixels left of the target), when nothing is in its way;
 *  - "detour": a block stands on its row between its ends: it leaves into the gap between rows (`gap`: from the source's height),
 *    runs there, and turns down / up just before its target. Arrows sharing a gap or a turning column get lanes of their own. */
export type Route = { kind: "stub" | "step" | "detour"; gap: number; turn: number; from: string; to: string };

const LANE = 12;

export function lanes(g: Graph, name: (n: GNode) => string): Map<string, Route> {
  const at = new Map(g.nodes.map((n) => [n.id, n]));
  const out = new Map<string, Route>();
  // blocks by row (top y), to see what stands between an arrow's ends
  const rows = new Map<number, number[]>();
  for (const n of g.nodes) { const k = Math.round(n.y); if (!rows.has(k)) rows.set(k, []); rows.get(k)!.push(n.x); }
  const between = (y: number, x1: number, x2: number) => [...rows].some(([ry, xs]) => Math.abs(ry - y) < STEP_Y / 2 && xs.some((x) => x > x1 + 1 && x < x2 - 1));
  const turning = new Map<number, { id: string; key: number }[]>();   // target column -> arrows turning there
  const gaps = new Map<number, { id: string; x1: number; x2: number }[]>();   // gap height -> arrows running in it
  for (const e of g.edges) {
    const s = at.get(e.source), t = at.get(e.target);
    if (!s || !t) continue;
    const r: Route = { kind: "step", gap: 0, turn: 0, from: name(s), to: name(t) };
    out.set(e.id, r);
    if (t.x < s.x - 20) { r.kind = "stub"; continue; }
    if (between(s.y, s.x, t.x)) {
      r.kind = "detour";
      // the gap on the side of the target (flat arrows: above)
      r.gap = (t.y > s.y + 4 ? 1 : -1) * STEP_Y / 2;
      const k = Math.round(s.y + r.gap);
      if (!gaps.has(k)) gaps.set(k, []);
      gaps.get(k)!.push({ id: e.id, x1: s.x, x2: t.x });
    }
    if (r.kind === "detour" || Math.abs(t.y - s.y) > 4) {
      if (!turning.has(t.x)) turning.set(t.x, []);
      turning.get(t.x)!.push({ id: e.id, key: s.y + r.gap });
    }
  }
  // turning columns: one lane per arrow, the highest start nearest the target
  for (const list of turning.values()) list.sort((a, b) => a.key - b.key).forEach((x, i) => { out.get(x.id)!.turn = 14 + LANE * i; });
  // gaps: arrows whose stretches overlap get different heights (interval colouring, at most 5 lanes)
  for (const list of gaps.values()) {
    const ends: number[] = [];
    for (const x of list.sort((a, b) => a.x1 - b.x1)) {
      let lane = ends.findIndex((end) => end <= x.x1);
      if (lane < 0) { lane = ends.length; ends.push(0); }
      ends[lane] = x.x2;
      const r = out.get(x.id)!;
      r.gap += Math.sign(r.gap) * LANE * (lane % 5) * -1;
    }
  }
  return out;
}

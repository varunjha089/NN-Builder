// Auto numbering for the Text tab: Enter after a numbered line types the next label (a number at the top level, a letter inside braces) and
// moves the labels below it on by one when that label is taken; Renumber puts the numbers back in order. Labels and the references to them
// (after <- and shared) change together. The renames also go to the Rust side (dsl::apply_renamed), so the blocks keep their ids and places.

/** a label rename for the blocks: old label -> new, compared segment by segment from the start ("5.*.c": line 5, any pass, block c) */
export type Rule = [string, string];

type Line = { label: string | null; scope: number; python: boolean; opens: number[]; closes: number };
type Scope = { parent: number; depth: number; opener: number; owner: string | null };

/** a, b, …, z, a1, b1, … (as dsl.rs `letters`) */
export function letters(i: number): string {
  const c = String.fromCharCode(97 + ((i - 1) % 26));
  return i <= 26 ? c : `${c}${Math.floor((i - 1) / 26)}`;
}
/** the place of a label in its scope's sequence: 7 for "7" at the top level, 3 for "c" inside braces; null for a named label */
function value(label: string, depth: number): number | null {
  if (depth === 0) return /^\d+$/.test(label) ? Number(label) : null;
  const m = /^([a-z])(\d*)$/.exec(label);
  if (!m || m[2] === "0" || m[2].startsWith("0")) return null;
  return m[1].charCodeAt(0) - 96 + 26 * Number(m[2] || 0);
}
const show = (v: number, depth: number) => (depth === 0 ? String(v) : letters(v));
const split = (raw: string): [string, string] => {
  const h = raw.indexOf("#");
  return h >= 0 ? [raw.slice(0, h), raw.slice(h)] : [raw, ""];
};

/** every line's label and brace scope (scope 0 is the top level; the python """ section is skipped) */
function scan(lines: string[]) {
  const scopes: Scope[] = [{ parent: -1, depth: 0, opener: -1, owner: null }];
  const out: Line[] = [];
  let cur = 0, py = false;
  lines.forEach((raw, i) => {
    const t = raw.trim();
    if (py || t === "python \"\"\"") {
      if (py && t === "\"\"\"") py = false; else if (!py) py = true;
      out.push({ label: null, scope: cur, python: true, opens: [], closes: 0 });
      return;
    }
    const [code] = split(raw);
    const depth = scopes[cur].depth;
    // a label is the first word when another word follows it
    const m = /^\s*([A-Za-z0-9_]+)\s+[A-Za-z_]/.exec(code);
    const label = m && !["let", "def", "net", "python"].includes(m[1]) ? m[1] : null;
    const line: Line = { label, scope: cur, python: false, opens: [], closes: 0 };
    for (const ch of code.replace(/"[^"]*"/g, "")) {
      if (ch === "{") {
        // a group right under the top level belongs to its line: its blocks are "<line>.<pass>.<letter>"
        scopes.push({ parent: cur, depth: scopes[cur].depth + 1, opener: i, owner: scopes[cur].depth === 0 && depth === 0 ? label : null });
        cur = scopes.length - 1;
        line.opens.push(cur);
      } else if (ch === "}" && cur > 0) {
        cur = scopes[cur].parent;
        line.closes++;
      }
    }
    out.push(line);
  });
  return { scopes, info: out };
}

/** the labels' new names, per scope; the text with labels and references renamed, and the rules for the blocks */
function relabel(lines: string[], scan_: ReturnType<typeof scan>, maps: Map<number, Map<string, string>>): { lines: string[]; steps: Rule[][] } {
  const { scopes, info } = scan_;
  const top = maps.get(0) ?? new Map<string, string>();
  const locals = new Map<number, Set<string>>();
  info.forEach((l) => { if (l.label) { if (!locals.has(l.scope)) locals.set(l.scope, new Set()); locals.get(l.scope)!.add(l.label); } });
  const owned = new Map<string, number>();
  scopes.forEach((s, i) => { if (s.owner) owned.set(s.owner, i); });
  const ref = (r: string, scope: number) => {
    const segs = r.split(".");
    if (segs.length === 1) {
      const own = maps.get(scope);
      if (scope > 0 && locals.get(scope)?.has(r)) return own?.get(r) ?? r;
      return top.get(r) ?? r;
    }
    const inner = owned.has(segs[0]) ? maps.get(owned.get(segs[0])!) : undefined;
    if (segs.length >= 3 && inner?.has(segs[2])) segs[2] = inner.get(segs[2])!;
    segs[0] = top.get(segs[0]) ?? segs[0];
    return segs.join(".");
  };
  const out = lines.map((raw, i) => {
    const l = info[i];
    if (l.python) return raw;
    let [code, comment] = split(raw);
    const map = maps.get(l.scope);
    if (l.label && map?.has(l.label)) code = code.replace(/^(\s*)[A-Za-z0-9_]+/, (_, sp) => sp + map.get(l.label!)!);
    const arrow = code.indexOf("<-");
    if (arrow >= 0) code = code.slice(0, arrow) + code.slice(arrow).replace(/[A-Za-z0-9_.]+/g, (w) => ref(w, l.scope));
    code = code.replace(/\b(shared\s+)([A-Za-z0-9_.]+)/, (_, s, w) => s + ref(w, l.scope));
    return code + comment;
  });
  // the groups' letters first (their rules name the line by its old number), then the top-level numbers
  const steps: Rule[][] = [];
  const inner: Rule[] = [];
  maps.forEach((m, s) => {
    if (s > 0 && scopes[s].owner) m.forEach((n, o) => inner.push([`${scopes[s].owner}.*.${o}`, `${scopes[s].owner}.*.${n}`]));
  });
  if (inner.length) steps.push(inner);
  if (top.size) steps.push([...top].map(([o, n]) => [o, n] as Rule));
  return { lines: out, steps };
}

/** the labels of a scope with their values, in line order */
const labelled = (s: ReturnType<typeof scan>, scope: number) =>
  s.info.map((l, i) => ({ i, label: l.label, v: l.label && l.scope === scope ? value(l.label, s.scopes[scope].depth) : null }))
    .filter((x): x is { i: number; label: string; v: number } => x.v !== null);

/** move the run of labels starting at `from` (from, from + 1, … as long as they follow on) by `by` */
function shift(s: ReturnType<typeof scan>, scope: number, from: number, by: number): Map<number, Map<string, string>> {
  const depth = s.scopes[scope].depth;
  const have = new Set(labelled(s, scope).map((x) => x.v));
  const m = new Map<string, string>();
  for (let v = from; have.has(v); v++) m.set(show(v, depth), show(v + by, depth));
  return new Map([[scope, m]]);
}

/** Enter at `pos` (no selection): the new text, where the cursor goes and the renames; null for a plain Enter */
export function enter(text: string, pos: number): { text: string; cursor: number; steps: Rule[][] } | null {
  const lines = text.split("\n");
  const start = text.lastIndexOf("\n", pos - 1) + 1;
  const row = text.slice(0, start).split("\n").length - 1;
  if (text.slice(pos, text.indexOf("\n", pos) < 0 ? text.length : text.indexOf("\n", pos)).trim()) return null;
  const s = scan(lines);
  const l = s.info[row];
  if (l.python) return null;
  const [code] = split(lines[row]);
  const indent = /^\s*/.exec(lines[row])![0];
  const depth = s.scopes[l.scope].depth;
  const join = (ls: string[], at: number, col: number) => {
    const t = ls.join("\n");
    return t.split("\n").slice(0, at).reduce((n, x) => n + x.length + 1, 0) + col;
  };

  // Enter on a line holding only a label (one just typed for you): the label goes, and the labels below close the gap
  const only = /^\s*([A-Za-z0-9_]+)\s*$/.exec(code);
  if (only && value(only[1], depth) !== null && !split(lines[row])[1]) {
    const v = value(only[1], depth)!;
    const taken = labelled(s, l.scope).some((x) => x.v === v && x.i !== row);
    const r = relabel(lines, s, taken ? new Map() : shift(s, l.scope, v + 1, -1));
    r.lines[row] = "";
    return { text: r.lines.join("\n"), cursor: join(r.lines, row, 0), steps: r.steps };
  }

  let scope: number, next: number, ind: string;
  if (l.opens.length && code.trim().endsWith("{")) {
    // a line that opens a group: its first block, indented
    scope = l.opens[l.opens.length - 1];
    next = 1;
    ind = indent + "  ";
  } else if (l.label && value(l.label, depth) !== null && !l.closes) {
    scope = l.scope;
    next = value(l.label, depth)! + 1;
    ind = indent;
  } else if (code.trim().startsWith("}") && l.closes) {
    // after a group: the number after the group's line
    const after = scopeAfter(s, row);
    const opener = s.scopes.findIndex((x, i) => i > 0 && s.info[x.opener] && endsAt(s, i) === row);
    const ol = opener > 0 ? s.info[s.scopes[opener].opener] : null;
    const v = ol?.label ? value(ol.label, s.scopes[after].depth) : null;
    if (v === null) return plain(text, pos, indent);
    scope = after;
    next = v + 1;
    ind = indent;
  } else {
    return plain(text, pos, indent);
  }
  const d = s.scopes[scope].depth;
  const used = labelled(s, scope).some((x) => x.v === next);
  const r = relabel(lines, s, used ? shift(s, scope, next, 1) : new Map());
  const label = show(next, d);
  r.lines.splice(row + 1, 0, `${ind}${label} `);
  return { text: r.lines.join("\n"), cursor: join(r.lines, row + 1, ind.length + label.length + 1), steps: r.steps };
}

/** Enter keeping the line's indentation */
function plain(text: string, pos: number, indent: string) {
  return indent ? { text: text.slice(0, pos) + "\n" + indent + text.slice(pos), cursor: pos + 1 + indent.length, steps: [] } : null;
}
/** the scope a line is in after its closing braces */
function scopeAfter(s: ReturnType<typeof scan>, row: number): number {
  const next = s.info[row + 1];
  if (next) return next.scope;
  let sc = s.info[row].scope;
  for (let k = 0; k < s.info[row].closes && sc > 0; k++) sc = s.scopes[sc].parent;
  return sc;
}
/** the line where a scope's group ends */
function endsAt(s: ReturnType<typeof scan>, scope: number): number {
  for (let i = s.scopes[scope].opener + 1; i < s.info.length; i++) {
    let sc = s.info[i].scope;
    if (s.info[i].closes) {
      for (let k = 0; k < s.info[i].closes; k++) { if (sc === scope) return i; sc = s.scopes[sc].parent; }
    }
  }
  return -1;
}

/** Renumber: the top-level numbers 1, 2, 3, … and the letters a, b, c, … inside each group, in line order (named labels stay) */
export function renumber(text: string): { text: string; steps: Rule[][] } {
  const lines = text.split("\n");
  const s = scan(lines);
  const maps = new Map<number, Map<string, string>>();
  s.scopes.forEach((sc, i) => {
    if (i > 0 && !sc.owner) return; // a function's body or a group inside a group: its blocks are named from other lines
    const m = new Map<string, string>();
    labelled(s, i).forEach((x, k) => { if (x.label !== show(k + 1, sc.depth)) m.set(x.label, show(k + 1, sc.depth)); });
    if (m.size) maps.set(i, m);
  });
  const r = relabel(lines, s, maps);
  return { text: r.lines.join("\n"), steps: r.steps };
}

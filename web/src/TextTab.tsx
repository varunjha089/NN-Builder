// The Text tab: the design as numbered lines (crates/core/src/dsl.rs, PROGRAMMING_LANGUAGE.md), kept in step with the canvas. Typing
// applies the text after a short pause when it is valid (one undo step); a text with problems is kept here and the canvas stays as it was.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { errText, type Analysis, type Graph } from "./api";

type TextError = { line: number; col: number; message: string };

/** a text not applied yet (it has problems) when the tab is left: shown again on return, as long as the design has not changed meanwhile */
let draft: { base: string; text: string } | null = null;

/** the line of every top-level label (a block inside a repeat or a function call belongs to the line of its call: "5.3.b" -> line of 5) */
function labelLines(text: string): Map<string, number> {
  const out = new Map<string, number>();
  let depth = 0;
  text.split("\n").forEach((raw, i) => {
    const line = raw.replace(/#.*/, "");
    if (depth === 0) {
      const m = /^\s*([A-Za-z0-9_]+)\s+\S/.exec(line);
      if (m && !["let", "def", "net"].includes(m[1])) out.set(m[1], i + 1);
    }
    depth += (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length;
  });
  return out;
}

const HELP: [string, string][] = [
  ["1 input", "one block per line: a label (a number), the block and its settings"],
  ["2 conv 64 3x3 stride 2 relu", "filters, kernel, stride; also pad 1 / same / valid, dil 2, nobias, an activation (none = no activation)"],
  ["3 maxpool 2x2  ·  avgpool 3x3 stride 1 pad 1", "pools; global_pool (avg / max), upsample x2 bilinear, convT 64 2x2 stride 2"],
  ["4 bn · ln · gn 8 · dropout 0.3 · flatten · relu · gelu", "normalisation, dropout and activations"],
  ["5 dense 128 gelu  ·  output  ·  seg_output", "dense layers and the output (its classes come from the dataset)"],
  ["6 add <- 2, 5  ·  concat <- 3, 7", "inputs: the line above, unless named after <- (several for add / concat / mul)"],
  ["to_tokens · cls_pos · attention heads 8 · token_pool cls", "transformer blocks"],
  ["7 repeat 12 { a ln … g add <- c, f }", "a group repeated, each pass taking the previous one's output; inside, i = 1, 2, …, in = the group's input"],
  ["8 for c, s in (64, 1), (128, 2) { x block(c, s) }", "the group once per value, chained like repeat"],
  ["def block(c, s = 1) { a conv c 3x3 stride s … }", "a function: blocks with values; call it as a line: 9 block(256, 2) <- 8, 3 (in, in2 inside)"],
  ["let w = 64  ·  conv w * 2 3x3  ·  stride=2", "names and arithmetic (+ - * / // % ^); any setting as name=value"],
  ["<- 7.3.b", "a block inside a repeat or call: line 7, pass 3, block b"],
];

// ---------------------------------------------------------------- colours for the text (drawn behind a transparent textarea)
const BLOCKS = new Set(["input", "conv", "conv2d", "dwconv", "dwconv2d", "convt", "convtranspose", "convtranspose2d", "pool", "pool2d", "maxpool",
  "avgpool", "global_pool", "gap", "upsample", "bn", "batchnorm", "ln", "layernorm", "gn", "groupnorm", "dropout", "flatten", "dense", "linear",
  "activation", "act", "relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "add", "concat", "cat", "multiply", "mul", "to_tokens", "cls_pos",
  "attention", "attn", "token_pool", "output", "seg_output"]);
const KEYWORDS = new Set(["net", "let", "def", "repeat", "for", "in", "as", "none"]);
const SETTINGS = new Set(["stride", "pad", "dil", "outpad", "heads", "drop", "nobias", "bias", "same", "valid", "max", "avg", "nearest", "bilinear",
  "patches", "rows", "cls", "nocls", "mean", "size"]);
const esc = (t: string) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** the text as HTML with a class per kind of word: label, keyword, block, setting, number, function, reference, comment */
function highlight(text: string): string {
  const funcs = new Set([...text.matchAll(/\bdef\s+([A-Za-z_]\w*)/g)].map((m) => m[1]));
  return text.split("\n").map((line) => {
    const hash = line.indexOf("#");
    const code = hash >= 0 ? line.slice(0, hash) : line, comment = hash >= 0 ? line.slice(hash) : "";
    const toks = code.match(/\s+|<-|"[^"]*"|[A-Za-z0-9_.]+|./g) ?? [];
    const words = toks.filter((t) => /^[A-Za-z0-9_.]+$/.test(t));
    const w0 = words[0]?.toLowerCase(), w1 = words[1];
    // the first word is a label when a block, loop or call follows it (as the parser decides)
    const starts = (w?: string) => !!w && (BLOCKS.has(w.toLowerCase()) || w === "repeat" || w === "for" || funcs.has(w));
    const isLabel = !!w0 && !KEYWORDS.has(w0) && words.length > 1 && (starts(w1) || !starts(words[0]));
    let first = true, refs = false;
    const out = toks.map((t) => {
      if (!/^[A-Za-z0-9_.]+$/.test(t)) {
        if (t === "<-") { refs = true; return `<span class="hl-kw">&lt;-</span>`; }
        if (t.startsWith("\"")) return `<span class="hl-str">${esc(t)}</span>`;
        return esc(t);
      }
      const lw = t.toLowerCase();
      let c = "";
      if (first && isLabel) c = "hl-label";
      else if (refs) c = "hl-ref";
      else if (KEYWORDS.has(lw)) c = "hl-kw";
      else if (funcs.has(t)) c = "hl-fn";
      else if (BLOCKS.has(lw)) c = "hl-block";
      else if (SETTINGS.has(lw)) c = "hl-set";
      else if (/^(\d+(\.\d+)?(x\d+)?|x\d+(\.\d+)?)$/.test(lw)) c = "hl-num";
      first = false;
      return c ? `<span class="${c}">${esc(t)}</span>` : esc(t);
    }).join("");
    return out + (comment ? `<span class="hl-com">${esc(comment)}</span>` : "");
  }).join("\n") + "\n ";
}

export function TextTab({ graph, analysis, persist, applyText, format, showBlock, toast }: {
  graph: Graph; analysis: Analysis | null; toast: (m: string) => void;
  persist: (g: Graph) => Promise<void>;
  /** build the design from a text: [] when applied, else the problems */
  applyText: (text: string) => Promise<TextError[]>;
  format: () => Promise<void>;
  showBlock: (id: string) => void;
}) {
  const [text, setText] = useState(draft && draft.base === (graph.text ?? "") ? draft.text : graph.text ?? "");
  const [errors, setErrors] = useState<TextError[]>([]);
  const [busy, setBusy] = useState(false);
  const [help, setHelp] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const gutter = useRef<HTMLDivElement>(null);
  const shade = useRef<HTMLPreElement>(null);
  const timer = useRef(0);
  // the text this tab last saw on the design: a different one (canvas change, undo, Format) replaces what is shown, unless typing is pending
  const seen = useRef(graph.text ?? "");
  const typing = useRef(false);

  // an older design without its text: let the Rust side print it
  useEffect(() => { if (!graph.text) persist(graph).catch((e) => toast(errText(e))); }, [graph, persist, toast]);
  useEffect(() => {
    const t = graph.text ?? "";
    if (t !== seen.current && !typing.current) { setText(t); setErrors([]); }
    seen.current = t;
  }, [graph.text]);
  // the latest text, for keeping a draft when the tab is left
  const latest = useRef({ text, base: graph.text ?? "" });
  latest.current = { text, base: graph.text ?? "" };
  useEffect(() => {
    // a draft brought back: show its problems again
    if (draft && draft.text === latest.current.text && draft.text !== latest.current.base) apply(draft.text);
    return () => {
      window.clearTimeout(timer.current);
      draft = latest.current.text !== latest.current.base ? { base: latest.current.base, text: latest.current.text } : null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apply = useCallback((t: string) => {
    setBusy(true);
    applyText(t).then((errs) => {
      setErrors(errs);
      if (!errs.length) seen.current = t;
    }).catch((e) => toast(errText(e))).finally(() => { setBusy(false); typing.current = false; });
  }, [applyText, toast]);

  const change = (t: string) => {
    setText(t);
    typing.current = true;
    window.clearTimeout(timer.current);
    // the canvas follows the text a quarter of a second after typing stops (when the text is valid)
    timer.current = window.setTimeout(() => apply(t), 250);
  };

  const lines = useMemo(() => text.split("\n"), [text]);
  const where = useMemo(() => labelLines(text), [text]);
  // the design's own problems (shapes, missing inputs) on the lines of their blocks
  const problems = useMemo(() => {
    const byId = new Map(graph.nodes.map((n) => [n.id, n]));
    return (analysis?.nodes ?? []).filter((r) => r.status === "error" || r.status === "warn").map((r) => {
      const label = byId.get(r.id)?.label ?? "";
      return { id: r.id, label, line: where.get(label.split(".")[0]) ?? 0, status: r.status, message: r.messages[0] ?? r.status };
    });
  }, [analysis, graph.nodes, where]);
  const marks = useMemo(() => {
    const m = new Map<number, string>();
    problems.forEach((p) => { if (p.line) m.set(p.line, p.status === "error" ? "err" : m.get(p.line) ?? "warn"); });
    errors.forEach((e) => m.set(e.line, "err"));
    return m;
  }, [problems, errors]);

  const goto = (line: number, col = 1) => {
    const a = area.current;
    if (!a) return;
    const off = lines.slice(0, line - 1).reduce((t, l) => t + l.length + 1, 0) + col - 1;
    a.focus();
    a.setSelectionRange(off, off);
    a.scrollTop = Math.max(0, (line - 5) * 19);
  };
  const keys = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Tab") return;
    e.preventDefault();
    const a = e.currentTarget, s = a.selectionStart;
    change(text.slice(0, s) + "  " + text.slice(a.selectionEnd));
    requestAnimationFrame(() => a.setSelectionRange(s + 2, s + 2));
  };

  // ⌘-click on a label (or a path like 5.2.b): select that block on the canvas
  const pick = (e: React.MouseEvent<HTMLTextAreaElement>) => {
    if (!e.metaKey) return;
    const a = e.currentTarget, at = a.selectionStart;
    const left = /[A-Za-z0-9_.]*$/.exec(text.slice(0, at))?.[0] ?? "", right = /^[A-Za-z0-9_.]*/.exec(text.slice(at))?.[0] ?? "";
    const word = left + right;
    if (!word) return;
    // the top-level line this word is on (for a local label inside braces: a, b, …)
    let top = "", depth = 0;
    for (const raw of text.slice(0, at).split("\n")) {
      const line = raw.replace(/#.*/, "");
      const m = depth === 0 ? /^\s*([A-Za-z0-9_]+)\s+\S/.exec(line) : null;
      if (m) top = m[1];
      depth += (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length;
    }
    const tries = [word, `${top}.1.${word}`, `${top}.${word}`];
    const n = tries.map((l) => graph.nodes.find((x) => x.label === l)).find((x) => x)
      ?? graph.nodes.find((x) => (x.label ?? "").startsWith(`${word}.`));
    if (n) showBlock(n.id);
  };
  const pending = text !== (graph.text ?? "");
  const state = busy ? ["applying…", "running"] : errors.length ? [`not applied: ${errors.length === 1 ? "a problem" : `${errors.length} problems`} in the text`, "failed"]
    : pending ? ["typing…", "running"] : ["in step with the canvas", "done"];

  const editor = <div className="editor">
    <div className="gutter" ref={gutter}>
      {lines.map((_, i) => <div key={i} className={marks.get(i + 1) ?? ""}>{i + 1}</div>)}
    </div>
    <div className="code">
      <pre className="shade" ref={shade} aria-hidden dangerouslySetInnerHTML={{ __html: highlight(text) }} />
      <textarea ref={area} value={text} spellCheck={false} onChange={(e) => change(e.target.value)} onKeyDown={keys} onClick={pick}
        title="⌘-click a block's number to select it on the canvas"
        onScroll={(e) => {
          const t = e.currentTarget;
          if (gutter.current) gutter.current.scrollTop = t.scrollTop;
          if (shade.current) { shade.current.scrollTop = t.scrollTop; shade.current.scrollLeft = t.scrollLeft; }
        }} />
    </div>
  </div>;
  const problemList = (errors.length > 0 || problems.length > 0) && <div className="text-problems">
    {errors.map((e, i) => <div key={`t${i}`} className="msg error"><button className="link" onClick={() => goto(e.line, e.col)}>line {e.line}</button> {e.message}</div>)}
    {problems.map((p) => <div key={p.id} className={`msg ${p.status === "error" ? "error" : "warn"}`}>
      {p.line ? <button className="link" onClick={() => goto(p.line)}>line {p.line}</button> : null}{" "}
      <button className="link" title="show this block on the canvas" onClick={() => showBlock(p.id)}>block {p.label || p.id}</button>: {p.message}</div>)}
  </div>;

  return (
    <div className="page text-tab">
      <div className="row">
        <h2>Text</h2><span className={`pill ${state[1]}`}>{state[0]}</span>
        <span className="muted small">{graph.nodes.length} blocks{analysis ? ` · ${analysis.total_params.toLocaleString()} params` : ""}</span>
        <span className="grow" />
        <button onClick={() => setHelp(!help)}>{help ? "Hide help" : "Help"}</button>
        <button title="Print the text again from the blocks (hand-written functions and loops become lines and repeats)"
          onClick={() => format().then(() => setErrors([])).catch((e) => toast(errText(e)))}>Format</button>
        <button onClick={() => navigator.clipboard.writeText(text).then(() => toast("Copied the design text"))}>Copy</button>
      </div>
      <p className="muted small">The design as numbered lines, in step with the canvas both ways: typing changes the canvas a moment after the text
        is valid, and a change on the canvas edits only the lines it concerns (functions, loops, names and comments stay); only a change to a block
        made by a function or loop prints the text again. ⌘-click a number to select its block on the canvas. Undo (⌘Z) works for both. File → Export Design… saves it as a .nnb file.</p>
      {help && <table className="text-help"><tbody>{HELP.map(([code, what]) => <tr key={code}><td><code>{code}</code></td><td className="small">{what}</td></tr>)}</tbody></table>}
      {editor}
      {problemList}
    </div>
  );
}

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
    timer.current = window.setTimeout(() => apply(t), 500);
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

  const pending = text !== (graph.text ?? "");
  const state = busy ? ["applying…", "running"] : errors.length ? [`not applied: ${errors.length === 1 ? "a problem" : `${errors.length} problems`} in the text`, "failed"]
    : pending ? ["typing…", "running"] : ["in step with the canvas", "done"];

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
      <p className="muted small">The design as numbered lines. Edits here change the canvas (after a short pause, when the text is valid); a change on
        the canvas rewrites this text, unless it only moved blocks. Undo (⌘Z) works for both.</p>
      {help && <table className="text-help"><tbody>{HELP.map(([code, what]) => <tr key={code}><td><code>{code}</code></td><td className="small">{what}</td></tr>)}</tbody></table>}
      <div className="editor">
        <div className="gutter" ref={gutter}>
          {lines.map((_, i) => <div key={i} className={marks.get(i + 1) ?? ""}>{i + 1}</div>)}
        </div>
        <textarea ref={area} value={text} spellCheck={false} onChange={(e) => change(e.target.value)} onKeyDown={keys}
          onScroll={(e) => { if (gutter.current) gutter.current.scrollTop = e.currentTarget.scrollTop; }} />
      </div>
      {(errors.length > 0 || problems.length > 0) && <div className="text-problems">
        {errors.map((e, i) => <div key={`t${i}`} className="msg error"><button className="link" onClick={() => goto(e.line, e.col)}>line {e.line}</button> {e.message}</div>)}
        {problems.map((p) => <div key={p.id} className={`msg ${p.status === "error" ? "error" : "warn"}`}>
          {p.line ? <button className="link" onClick={() => goto(p.line)}>line {p.line}</button> : null}{" "}
          <button className="link" title="show this block on the canvas" onClick={() => showBlock(p.id)}>block {p.label || p.id}</button>: {p.message}</div>)}
      </div>}
    </div>
  );
}

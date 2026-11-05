# NNB text: a language for NN Builder designs (plan)

Status: built on 2026-11-05 (crates/core/src/dsl.rs, tests/text.rs, the Text tab in web/src/TextTab.tsx); section 9 says what differs
from this plan and what is left. FUTURE_PLAN.md points here.

## 1. Goal

Every design gets a text form that can be read, written and edited next to the canvas. Each block has a number, connections are written
by number, and repeated parts are written once. The text and the canvas are two views of the same `model.json` graph: editing one updates
the other.

What it is for:

- Writing a design quickly by keyboard instead of dragging blocks (a 30-block network is about 30 short lines).
- Reading a design exactly: every number of every block, which a screenshot of the canvas cannot show.
- Putting a design in a paper's appendix, a git diff, an e-mail or a chat with an AI assistant, and getting it back unchanged.
- Later: importing a torchvision model, comparing two designs line by line, and sweeps over one number ("width 32, 64, 128").

What it is not: a general programming language. There are no loops over data, no arithmetic on tensors and no Python. The text describes a
graph of blocks; the existing Rust engine checks shapes and generates the PyTorch code, so there is still one engine and one source of truth.

## 2. A first look

```
net "Small ResNet"                   # name shown in the window; the input shape comes from the Dataset tab
1  input
2  conv 64 3x3 relu
3  conv 64 3x3 relu
4  conv 64 3x3                       # no activation: Add comes first
5  add <- 2, 4                       # skip connection: the outputs of blocks 2 and 4
6  relu
7  conv 128 3x3 stride 2 relu
8  global_pool avg
9  output                            # as many classes as the dataset has
```

With a repeated part (a ViT encoder):

```
net "ViT-Tiny"
1  input
2  conv 192 16x16 stride 16 none     # patch embedding
3  to_tokens patches
4  cls_pos
5  repeat 12 as enc {                # 12 identical encoder blocks
     a  layernorm
     b  attention heads 3
     c  add <- in, b                 # "in" = the input of this repetition
     d  layernorm
     e  dense 768 gelu
     f  dense 192 none
     g  add <- c, f                  # the last line is the output of the repetition
   }
6  layernorm
7  token_pool cls
8  output
```

After "Format" the app adds what it computed as comments (never needed when writing):

```
2  conv 64 3x3 relu                  # → (64, 32, 32) · 1,792 params · RF 3×3
```

## 3. The rules

### 3.1 Lines and numbers

- One block per line: `<label> <kind> <parameters> [<- <inputs>]`.
- The label is a number (`1`, `2`, `17`) at the top level, or a short name (`a`, `b`, `proj`) inside a `repeat`. Labels must be unique in
  their scope. They are names, not positions: `1, 2, 5, 9` is allowed, and "Renumber" rewrites them as `1..N` in computation order.
- A line without `<-` takes its input from the line above it (the common case: a chain). The first block must be `input`.
- `<- 2, 4` gives the inputs explicitly. Order matters for `concat` (channels are joined in that order) and is kept.
- `#` starts a comment, to the end of the line. Blank lines are ignored. Indentation has no meaning (it is only for reading).
- Words are case-insensitive (`Conv`, `conv`); the formatter writes lower case.

### 3.2 Block kinds and their parameters

Parameters are written as plain words and numbers in any order; each kind knows which word means what. Anything left out takes the
canvas default (model.rs `defaults`). The inspector's names are also accepted as `key=value` (`filters=64`) for clarity or for values that
would be ambiguous.

| Kind (aliases) | Short form | Parameters (`key=value` names) | Default |
|---|---|---|---|
| `input` | `input` | none (shape from the dataset) | |
| `conv` (`conv2d`) | `conv 64 3x3 stride 2 pad valid dil 2 relu nobias` | filters, kernel `HxW` or `K`, stride, padding (`same`/`valid`/number), dilation, activation, bias | 32 3x3 stride 1 same relu |
| `dwconv` (`dwconv2d`) | `dwconv 3x3 x2 relu` | multiplier `xN`, kernel, stride, padding, activation, bias | x1 3x3 same relu |
| `convT` (`convtranspose2d`) | `convT 64 2x2 stride 2` | filters, kernel, stride, padding, output_padding `outpad 1`, activation | 32 2x2 stride 2 |
| `pool` (`pool2d`, `maxpool`, `avgpool`) | `maxpool 2x2`, `avgpool 3x3 stride 1 pad 1` | mode, size, stride (`stride size` = same as size), padding | max 2x2 |
| `global_pool` (`gap`) | `global_pool avg` | mode `avg`/`max` | avg |
| `upsample` | `upsample x2 bilinear` | scale, mode | x2 nearest |
| `bn` (`batchnorm`) | `bn` | none | |
| `ln` (`layernorm`) | `ln` | none | |
| `gn` (`groupnorm`) | `gn 8` | groups | 8 |
| `dropout` | `dropout 0.3` | rate | 0.25 |
| `flatten` | `flatten` | none | |
| `dense` (`linear`) | `dense 128 gelu` | units, activation | 128 relu |
| activation | `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `leaky_relu` alone on a line | fn | |
| `add` | `add <- 2, 4` | inputs (two or more) | |
| `concat` (`cat`) | `concat <- 3, 7` | inputs (two or more, in order) | |
| `multiply` (`mul`) | `mul <- 5, 9` | inputs (two or more) | |
| `to_tokens` | `to_tokens patches` / `to_tokens rows` | mode | patches |
| `cls_pos` | `cls_pos` / `cls_pos nocls` | cls | cls |
| `attention` (`attn`) | `attention heads 8 drop 0.1` | heads, dropout | heads 4 |
| `token_pool` | `token_pool cls` / `token_pool mean` | mode | cls |
| `output` | `output` | none (classes from the dataset) | |
| `seg_output` | `seg_output` | none (classes from the dataset) | |

Activation words (`relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `leaky_relu`, `none`) after a conv, dwconv, convT or dense block set its
activation; alone on a line they are an Activation block. New block kinds added to model.rs must get a row here and in the parser's
table (a test checks that every kind in KINDS can be written and read back).

### 3.3 Repeats

```
5  repeat 12 as enc {
     ...lines with local labels...
   }
```

- The body is a small network: `in` is its input (the block before the `repeat`, or the `<-` given on the `repeat` line), and its last line
  is its output, which the next top-level line receives.
- Each repetition gets its own copy of the blocks and its own weights (as in ViT, ResNet stages and MSG-Former).
- On the canvas the copies are ordinary blocks named `5.1a … 5.12g` (repetition 1–12, local label a–g); the existing ×N view shows them
  folded. Editing one copy on the canvas breaks the repeat (the printer then writes the copies out, or a shorter repeat plus the
  changed one).
- `as enc` is optional; it names the group in the layer table ("enc 1 … enc 12").
- A later line can refer to a block inside a repeat as `5.3c` (repetition 3, block c), for example for a U-Net skip; this is rare and the
  formatter writes it only when the graph needs it.
- Repeats may be nested one level (a stage of blocks of layers); deeper nesting is refused with a message.

### 3.4 The header

```
net "MSG-Former"                     # required name
uses dataset cifar10                 # optional: sets the Dataset tab when the text is applied as a new design
```

Training settings stay in the Train tab; they are not part of the design text (a design can be trained many ways).

### 3.5 Errors

Every error names the line and column and says what to do, in the same plain words the canvas uses:

- `line 5: add needs inputs of the same shape: block 2 gives (64, 32, 32), block 4 gives (128, 16, 16). Add a 1×1 conv with stride 2 on
  the shortcut, or use concat.`
- `line 7: "conv 64 3x3 stide 2": unknown word "stide" (did you mean "stride"?)`
- `line 9: block 12 does not exist (blocks so far: 1–8)`
- `line 3: a block may only use blocks above it (block 6 is below)` (the text is in computation order, so there are no cycles)
- Shape errors come from the existing shape engine (shapes.rs) and are mapped back to lines, so the text and the canvas never disagree.

The text is checked as it is typed (after a short pause). The canvas changes only when "Apply" is pressed or the text is valid and
"live" is on, so a half-typed line never destroys a design.

## 4. From design to text (the printer)

1. Order the blocks by the existing computation order (Kahn order in shapes.rs, ties broken by the canvas position from left to right).
2. Find repeats: maximal runs of identical consecutive groups, the same rule as `collapseRepeats` in Design.tsx, moved to Rust so the
   window and the printer agree. Identical = same kinds, same parameters and the same wiring inside the group, with one input and one
   output.
3. Number the top-level blocks `1..N`; label blocks inside a repeat `a, b, c, …` (or `a1 … ` after `z`).
4. For each block write the short form with only the parameters that differ from the default, and `<-` only when the input is not the
   line above (or there are several inputs).
5. Optionally add the computed comment (shape, parameters, receptive field, MACs).

Round-trip guarantee: text → design → text gives the same text (after Format), and design → text → design gives a graph with the same
blocks, parameters and connections (positions are recomputed by Arrange; block ids are kept when the text came from the design, see 5.2).

## 5. In the app

### 5.1 Window

- A "Text" switch on the Design tab: the canvas and the text side by side (or the text alone, full width).
- Editor: CodeMirror 6 (an npm package, pinned like the others) with a small grammar for colours, the block numbers as links (click a
  number: the block is selected on the canvas, and the other way round), errors underlined with the message on hover, and completion of
  kinds and parameter words.
- Buttons: Apply (one undo step), Format, Renumber, Copy, and "live" (apply automatically when valid).
- The canvas shows each block's number in a small badge, the same number as in the text and the layer table.

### 5.2 Keeping blocks the same

When the text came from the current design, each line remembers its block id, so applying an edited text keeps ids (and therefore the
canvas positions, the selection and anything a run refers to) for the lines that still exist. New lines get new ids and are placed by
Arrange next to their neighbours.

### 5.3 Files and menus

- File → Export Design as Text… (`.nnb`) and Import Design… accepts `.nnb` as well as `model.json`.
- Copy / paste of blocks on the canvas also puts the text form on the clipboard (as plain text next to the existing JSON), so selected
  blocks can be pasted into a document or a chat.
- The Code tab can show the design text above the generated PyTorch code.
- The manual gets a chapter with the grammar and the 15 demos in text form.

### 5.4 Engine (Rust, `crates/core/src/dsl.rs`)

- `parse(text) -> Result<(Graph, Header, LineMap), Vec<Diagnostic>>` (Diagnostic: line, column, message, optional fix).
- `print(graph, options) -> String` (options: comments on/off, repeats on/off).
- A hand-written lexer and recursive-descent parser (the grammar is small; no parser generator dependency).
- Tauri commands `design_to_text`, `text_to_design` (check only) and `apply_text` (save as the project's design).

### 5.5 Grammar (EBNF)

```
file      = header { line } ;
header    = "net" STRING [ "uses" "dataset" WORD ] NL ;
line      = ( block | repeat ) NL | NL ;
block     = LABEL kind { param } [ "<-" ref { "," ref } ] ;
repeat    = LABEL "repeat" INT [ "as" WORD ] [ "<-" ref ] "{" NL { block NL } "}" ;
ref       = LABEL | "in" | INT "." INT LABEL ;           (* 2, in, 5.3c *)
param     = WORD | NUMBER | SIZE | WORD "=" value | "x" NUMBER ;
SIZE      = INT "x" INT ;                               (* 3x3, 16x16 *)
LABEL     = INT | WORD ;
```

## 6. Tests

- Every kind in KINDS can be written and read back (a test fails when a new kind lacks a row in the table).
- The 15 demos: design → text → design gives the same blocks, parameters and connections, and the same parameter count; text → design →
  text is stable after Format. ViT-g/14 prints as about 12 lines with `repeat 40`.
- The 300 random designs from `dump_cases` round-trip (CI and locally: no PyTorch involved).
- Error messages: a table of wrong texts and the exact message and line expected.
- Hand-written examples (the two in section 2, a U-Net with skips into a repeat) parse to the expected graphs.
- The generated train.py of a design and of its text round-trip is byte-for-byte identical.

## 7. Steps (each usable on its own)

1. **Printer only:** design → text, shown read-only on the Code tab and copyable; block numbers on the canvas. (Useful at once for
   papers and for checking designs.)
2. **Parser for chains and `<-`:** text → design with Apply, errors with lines; without repeats.
3. **Repeats** in both directions; collapseRepeats moved to Rust.
4. **Editor comfort:** CodeMirror grammar, links between numbers and blocks, completion, live mode, ids kept on Apply.
5. **Files and clipboard:** `.nnb` import/export, text on copy, the manual chapter.
6. **Later:** import from torchvision (a script on the GPU machine or in CI prints a torchvision model as NNB text); parameters as
   variables (`let w = 64`, `conv w*2`) for sweeps; a `diff` of two designs by block number.

## 8. Open questions for the owner

1. The numbered-line style above, or a Python-like style (`x5 = add(x2, x4)`)? The plan assumes numbered lines (closest to "numbering
   each block and attaching the connection").
2. Should the text be stored in the project as well (`model.nnb` next to `model.json`), or only produced on demand? The plan assumes
   on demand, so there is one source of truth.
3. Name and file extension: "NNB text" / `.nnb`?

## 9. As built (2026-11-05)

Beyond the plan, at the owner's request:

- **Functions:** `def name(c, s = 1) { … }` with default values, called as a line `9 name(256, 2) <- 8, 3`; inside, `in`, `in2`, …
  are the call's inputs; functions may call functions (at most 16 deep, so a function calling itself is reported).
- **Loops with values:** `for c, s in (64, 1), (128, 2) { … }`, chained like `repeat`; `i` (the pass number) in both.
- **Names and arithmetic:** `let w = 64`, `conv w * 2 3x3`, `+ - * / // % ^`.
- Labels may be left out; `<- none` for a block without input.

How the two views stay in step: the graph is the truth and carries the text (`Graph.text`) and each block's label (`Node.label`).
`save_graph` calls `dsl::sync`: the text is kept when it still describes the blocks exactly (moves only), else printed again. The Text
tab calls `design_from_text` (`dsl::apply`: blocks keep ids and positions by label; new blocks go next to their first input; a text into
an empty design is arranged). `design_format` prints the text again. The window applies the text 0.5 s after typing stops.

Left for later:

1. **Keeping hand-written functions after a canvas change.** Today any change of blocks on the canvas prints the text again, so `def`
   and `for` become plain lines and repeats. Next step: when the changed block came from a plain top-level line, change only that line.
2. Editor comfort (step 4): CodeMirror with colours, completion and clickable numbers; ids are already kept on Apply.
3. Files and clipboard (step 5): `.nnb` import/export, the text next to copied blocks.
4. Section 7 step 6 (torchvision import, variables for sweeps, diff by block number).

//! NNB text: the design as numbered lines (PROGRAMMING_LANGUAGE.md). Every block has a label, its inputs are the line above unless
//! `<- 2, 4` names them, and repeated parts are written once with `repeat`, `for` or a function (`def`). The canvas and the text are
//! kept in step: `apply` builds the blocks from a text, `sync` keeps (or re-prints) the text after the canvas changed.
//!
//! ```text
//! net "Small ResNet"
//! let w = 64
//! def basic(c, s) {                  # a function: a group of blocks with parameters
//!   a conv c 3x3 stride s
//!   b conv c 3x3 none
//!   c add <- a, b
//! }
//! 1 input
//! 2 conv w 3x3
//! 3 for c, s in (64, 1), (128, 2) {  # once per value, chained
//!     x basic(c, s)
//!   }
//! 4 repeat 2 { a basic(128, 1) }     # i = 1, 2 inside
//! 5 global_pool
//! 6 output
//! ```
//!
//! Expanded, a block inside a repeat / for / function call gets the label `<line>.<pass>.<label>` (`3.2.x.a`), or `<line>.<label>` for a
//! call; that label is shown on the canvas and can be used in `<-` from later lines.

use crate::model::{self, Edge, Graph, Node};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

/// a problem in the text, at a line (1-based) and column
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Diagnostic {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

fn diag(line: usize, col: usize, message: impl Into<String>) -> Diagnostic {
    Diagnostic { line, col, message: message.into() }
}

const MAX_BLOCKS: usize = 20_000;
const MAX_DEPTH: usize = 16;

// ---------------------------------------------------------------- tokens
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Str(String),
    Sym(&'static str),
    Nl,
    End,
}

#[derive(Debug, Clone)]
struct T {
    tok: Tok,
    line: usize,
    col: usize,
}

const SYMS: &[&str] = &["<-", "//", "(", ")", ",", "{", "}", "=", "+", "-", "*", "/", "^", "%"];

fn lex(text: &str) -> Result<Vec<T>, Diagnostic> {
    let mut out = vec![];
    for (li, line) in text.lines().enumerate() {
        let ln = li + 1;
        let cs: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c == '#' {
                break;
            }
            if c.is_whitespace() || c == ';' {
                i += 1;
                continue;
            }
            if c == '"' {
                let start = i;
                i += 1;
                let mut s = String::new();
                while i < cs.len() && cs[i] != '"' {
                    s.push(cs[i]);
                    i += 1;
                }
                if i >= cs.len() {
                    return Err(diag(ln, start + 1, "a text in quotes is not closed"));
                }
                i += 1;
                out.push(T { tok: Tok::Str(s), line: ln, col: start + 1 });
                continue;
            }
            if c.is_alphanumeric() || c == '_' || c == '.' {
                let start = i;
                while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_' || cs[i] == '.') {
                    i += 1;
                }
                out.push(T { tok: Tok::Word(cs[start..i].iter().collect()), line: ln, col: start + 1 });
                continue;
            }
            let rest: String = cs[i..].iter().take(2).collect();
            match SYMS.iter().find(|s| rest.starts_with(**s)) {
                Some(s) => {
                    out.push(T { tok: Tok::Sym(s), line: ln, col: i + 1 });
                    i += s.len();
                }
                None => return Err(diag(ln, i + 1, format!("unexpected character '{c}'"))),
            }
        }
        out.push(T { tok: Tok::Nl, line: ln, col: cs.len() + 1 });
    }
    let last = text.lines().count().max(1);
    out.push(T { tok: Tok::End, line: last, col: 1 });
    Ok(out)
}

// ---------------------------------------------------------------- syntax tree
#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Word(String),
    Neg(Box<Expr>),
    Bin(Box<Expr>, &'static str, Box<Expr>),
}

#[derive(Debug, Clone)]
enum Arg {
    Pos(Expr),
    Key(String, Expr),
}

#[derive(Debug, Clone)]
enum Item {
    Block { word: String, args: Vec<Arg> },
    Call { name: String, args: Vec<Arg> },
    Repeat { count: Expr, body: Vec<Stmt> },
    For { vars: Vec<String>, values: Vec<Vec<Expr>>, body: Vec<Stmt> },
}

/// a reference after `<-`: the label, and its line and column
type Ref = (String, usize, usize);

#[derive(Debug, Clone)]
enum Stmt {
    Let { name: String, value: Expr, line: usize, col: usize },
    Node { label: Option<String>, item: Item, inputs: Option<Vec<Ref>>, line: usize, col: usize },
}

#[derive(Debug, Clone)]
struct Def {
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Stmt>,
}

#[derive(Debug, Default)]
struct Program {
    name: Option<String>,
    defs: HashMap<String, Def>,
    stmts: Vec<Stmt>,
}

const KEYWORDS: &[&str] = &["net", "let", "def", "repeat", "for", "in", "as", "none"];

struct Parser {
    t: Vec<T>,
    p: usize,
    funcs: HashSet<String>,
}

impl Parser {
    fn peek(&self) -> &T {
        &self.t[self.p]
    }
    fn peek2(&self) -> &T {
        &self.t[(self.p + 1).min(self.t.len() - 1)]
    }
    fn next(&mut self) -> T {
        let t = self.t[self.p].clone();
        if self.p < self.t.len() - 1 {
            self.p += 1;
        }
        t
    }
    fn is_sym(&self, s: &str) -> bool {
        matches!(&self.peek().tok, Tok::Sym(x) if *x == s)
    }
    fn expect_sym(&mut self, s: &str, what: &str) -> Result<(), Diagnostic> {
        if self.is_sym(s) {
            self.next();
            Ok(())
        } else {
            let t = self.peek();
            Err(diag(t.line, t.col, format!("expected '{s}' {what}")))
        }
    }
    fn word(&mut self, what: &str) -> Result<(String, usize, usize), Diagnostic> {
        let t = self.next();
        match t.tok {
            Tok::Word(w) => Ok((w, t.line, t.col)),
            _ => Err(diag(t.line, t.col, format!("expected {what}"))),
        }
    }
    fn skip_nl(&mut self) {
        while matches!(self.peek().tok, Tok::Nl) {
            self.next();
        }
    }
    fn end_of_stmt(&mut self) -> Result<(), Diagnostic> {
        match self.peek().tok {
            Tok::Nl => {
                self.next();
                Ok(())
            }
            Tok::End | Tok::Sym("}") => Ok(()),
            _ => {
                let t = self.peek();
                Err(diag(t.line, t.col, format!("unexpected '{}' (one block per line)", show(&t.tok))))
            }
        }
    }

    // expressions: + - * / // % ^ and parentheses over numbers and names
    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        let mut a = self.term()?;
        while let Tok::Sym(s @ ("+" | "-")) = self.peek().tok {
            self.next();
            a = Expr::Bin(Box::new(a), s, Box::new(self.term()?));
        }
        Ok(a)
    }
    fn term(&mut self) -> Result<Expr, Diagnostic> {
        let mut a = self.power()?;
        while let Tok::Sym(s @ ("*" | "/" | "//" | "%")) = self.peek().tok {
            self.next();
            a = Expr::Bin(Box::new(a), s, Box::new(self.power()?));
        }
        Ok(a)
    }
    fn power(&mut self) -> Result<Expr, Diagnostic> {
        let a = self.atom()?;
        if self.is_sym("^") {
            self.next();
            return Ok(Expr::Bin(Box::new(a), "^", Box::new(self.power()?)));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        let t = self.next();
        match t.tok {
            Tok::Sym("-") => Ok(Expr::Neg(Box::new(self.atom()?))),
            Tok::Sym("(") => {
                let e = self.expr()?;
                self.expect_sym(")", "to close the parenthesis")?;
                Ok(e)
            }
            Tok::Word(w) => Ok(match w.parse::<f64>() {
                Ok(v) if !w.ends_with('.') => Expr::Num(v),
                _ => Expr::Word(w),
            }),
            other => Err(diag(t.line, t.col, format!("expected a value, found '{}'", show(&other)))),
        }
    }

    fn args(&mut self, until_paren: bool) -> Result<Vec<Arg>, Diagnostic> {
        let mut out = vec![];
        loop {
            match &self.peek().tok {
                Tok::Nl | Tok::End | Tok::Sym("<-") | Tok::Sym("{") | Tok::Sym("}") => break,
                Tok::Sym(")") if until_paren => break,
                Tok::Sym(",") if until_paren => {
                    self.next();
                    continue;
                }
                _ => {}
            }
            if let (Tok::Word(k), Tok::Sym("=")) = (&self.peek().tok, &self.peek2().tok) {
                let k = k.clone();
                self.next();
                self.next();
                out.push(Arg::Key(k, self.expr()?));
            } else {
                out.push(Arg::Pos(self.expr()?));
            }
        }
        Ok(out)
    }

    fn inputs(&mut self) -> Result<Option<Vec<Ref>>, Diagnostic> {
        if !self.is_sym("<-") {
            return Ok(None);
        }
        self.next();
        let mut refs = vec![];
        loop {
            let (w, l, c) = self.word("a block label after '<-'")?;
            if w != "none" {
                refs.push((w, l, c));
            }
            if self.is_sym(",") {
                self.next();
            } else {
                break;
            }
        }
        Ok(Some(refs))
    }

    fn body(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
        self.expect_sym("{", "to start the repeated blocks")?;
        let stmts = self.block(true)?;
        self.expect_sym("}", "to end the repeated blocks")?;
        Ok(stmts)
    }

    /// statements until `}` (inside braces) or the end of the text
    fn block(&mut self, inner: bool) -> Result<Vec<Stmt>, Diagnostic> {
        let mut out = vec![];
        loop {
            self.skip_nl();
            match self.peek().tok.clone() {
                Tok::End => {
                    if inner {
                        let t = self.peek();
                        return Err(diag(t.line, t.col, "a '{' is not closed with '}'"));
                    }
                    return Ok(out);
                }
                Tok::Sym("}") => {
                    if inner {
                        return Ok(out);
                    }
                    let t = self.peek();
                    return Err(diag(t.line, t.col, "a '}' without its '{'"));
                }
                _ => {}
            }
            let first = self.peek().clone();
            let w = match &first.tok {
                Tok::Word(w) => w.clone(),
                other => return Err(diag(first.line, first.col, format!("a line starts with a label or a block, not '{}'", show(other)))),
            };
            match w.as_str() {
                "let" => {
                    self.next();
                    let (name, _, _) = self.word("a name after 'let'")?;
                    self.expect_sym("=", "after the name")?;
                    let value = self.expr()?;
                    out.push(Stmt::Let { name, value, line: first.line, col: first.col });
                    self.end_of_stmt()?;
                    continue;
                }
                "net" | "def" => {
                    return Err(diag(first.line, first.col, format!("'{w}' is only allowed at the top level, not inside braces")));
                }
                _ => {}
            }
            out.push(self.node_stmt()?);
            self.end_of_stmt()?;
        }
    }

    fn starts_item(&self, w: &str) -> bool {
        let lw = w.to_ascii_lowercase();
        lw == "repeat" || lw == "for" || kind_of(&lw).is_some() || self.funcs.contains(w)
    }

    fn node_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let first = self.peek().clone();
        let w = match &first.tok {
            Tok::Word(w) => w.clone(),
            _ => unreachable!(),
        };
        // a label first, unless the line starts with a block, repeat, for or a function (then a label is made up)
        let label = if self.starts_item(&w) && !matches!(self.peek2().tok, Tok::Word(_) if !self.starts_item(&w)) {
            None
        } else {
            self.next();
            if KEYWORDS.contains(&w.as_str()) {
                return Err(diag(first.line, first.col, format!("'{w}' cannot be a label")));
            }
            Some(w.clone())
        };
        let t = self.peek().clone();
        let word = match &t.tok {
            Tok::Word(w) => w.clone(),
            Tok::Nl | Tok::End => return Err(diag(t.line, t.col, "a label must be followed by a block, e.g. '5 conv 64 3x3'")),
            other => return Err(diag(t.line, t.col, format!("expected a block after the label, found '{}'", show(other)))),
        };
        self.next();
        let lw = word.to_ascii_lowercase();
        let item = if lw == "repeat" {
            let count = self.expr()?;
            if matches!(&self.peek().tok, Tok::Word(a) if a == "as") {
                self.next();
                self.word("a name after 'as'")?;
            }
            let inputs = self.inputs()?;
            let body = self.body()?;
            return Ok(Stmt::Node { label, item: Item::Repeat { count, body }, inputs, line: first.line, col: first.col });
        } else if lw == "for" {
            let mut vars = vec![self.word("a name after 'for'")?.0];
            while self.is_sym(",") {
                self.next();
                vars.push(self.word("a name")?.0);
            }
            let (kw, l, c) = self.word("'in'")?;
            if kw != "in" {
                return Err(diag(l, c, "expected 'in', as in 'for c in 64, 128, 256 { … }'"));
            }
            let mut values = vec![];
            loop {
                if self.is_sym("(") {
                    self.next();
                    let mut tuple = vec![self.expr()?];
                    while self.is_sym(",") {
                        self.next();
                        tuple.push(self.expr()?);
                    }
                    self.expect_sym(")", "to close the values")?;
                    values.push(tuple);
                } else {
                    values.push(vec![self.expr()?]);
                }
                if self.is_sym(",") {
                    self.next();
                } else {
                    break;
                }
            }
            for v in &values {
                if v.len() != vars.len() {
                    return Err(diag(t.line, t.col, format!("'for' has {} names but a value with {} parts", vars.len(), v.len())));
                }
            }
            let inputs = self.inputs()?;
            let body = self.body()?;
            return Ok(Stmt::Node { label, item: Item::For { vars, values, body }, inputs, line: first.line, col: first.col });
        } else if self.is_sym("(") {
            if !self.funcs.contains(&word) {
                return Err(diag(t.line, t.col, format!("'{word}(…)': there is no function '{word}' (define it with 'def {word}(…) {{ … }}')")));
            }
            self.next();
            let args = self.args(true)?;
            self.expect_sym(")", "to close the function's values")?;
            Item::Call { name: word, args }
        } else {
            if kind_of(&lw).is_none() {
                return Err(diag(t.line, t.col, unknown_block(&word)));
            }
            Item::Block { word: lw, args: self.args(false)? }
        };
        let inputs = self.inputs()?;
        Ok(Stmt::Node { label, item, inputs, line: first.line, col: first.col })
    }
}

fn show(t: &Tok) -> String {
    match t {
        Tok::Word(w) => w.clone(),
        Tok::Str(s) => format!("\"{s}\""),
        Tok::Sym(s) => s.to_string(),
        Tok::Nl => "end of line".into(),
        Tok::End => "end of text".into(),
    }
}

fn unknown_block(word: &str) -> String {
    let near = ALIASES.iter().map(|(a, _)| *a).min_by_key(|a| edit_distance(a, &word.to_ascii_lowercase())).unwrap_or("conv");
    if edit_distance(near, &word.to_ascii_lowercase()) <= 2 {
        format!("unknown block '{word}' (did you mean '{near}'?)")
    } else {
        format!("unknown block '{word}'")
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        prev = cur;
    }
    prev[b.len()]
}

fn parse(text: &str) -> Result<Program, Diagnostic> {
    let t = lex(text)?;
    // function names first, so that a function can be used above its definition
    let funcs: HashSet<String> = t.windows(2).filter_map(|w| match (&w[0].tok, &w[1].tok) {
        (Tok::Word(d), Tok::Word(n)) if d == "def" => Some(n.clone()),
        _ => None,
    }).collect();
    let mut p = Parser { t, p: 0, funcs };
    let mut prog = Program::default();
    loop {
        p.skip_nl();
        let first = p.peek().clone();
        match &first.tok {
            Tok::End => break,
            Tok::Word(w) if w == "net" => {
                p.next();
                match p.next().tok {
                    Tok::Str(s) => prog.name = Some(s),
                    _ => return Err(diag(first.line, first.col, "write the name in quotes: net \"My network\"")),
                }
                p.end_of_stmt()?;
            }
            Tok::Word(w) if w == "def" => {
                p.next();
                let (name, l, c) = p.word("a function name after 'def'")?;
                if kind_of(&name.to_ascii_lowercase()).is_some() || KEYWORDS.contains(&name.as_str()) {
                    return Err(diag(l, c, format!("'{name}' is a block or a keyword; choose another function name")));
                }
                if prog.defs.contains_key(&name) {
                    return Err(diag(l, c, format!("the function '{name}' is defined twice")));
                }
                p.expect_sym("(", "after the function name")?;
                let mut params = vec![];
                while !p.is_sym(")") {
                    let (pn, _, _) = p.word("a parameter name")?;
                    let default = if p.is_sym("=") {
                        p.next();
                        Some(p.expr()?)
                    } else {
                        None
                    };
                    params.push((pn, default));
                    if p.is_sym(",") {
                        p.next();
                    } else {
                        break;
                    }
                }
                p.expect_sym(")", "to close the parameters")?;
                let body = p.body()?;
                prog.defs.insert(name, Def { params, body });
                p.end_of_stmt()?;
            }
            _ => {
                // one top-level statement (the parser's block() handles let and node lines)
                let mut one = p.block_one()?;
                prog.stmts.append(&mut one);
            }
        }
    }
    Ok(prog)
}

impl Parser {
    fn block_one(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
        let first = self.peek().clone();
        if let Tok::Word(w) = &first.tok {
            if w == "let" {
                self.next();
                let (name, _, _) = self.word("a name after 'let'")?;
                self.expect_sym("=", "after the name")?;
                let value = self.expr()?;
                self.end_of_stmt()?;
                return Ok(vec![Stmt::Let { name, value, line: first.line, col: first.col }]);
            }
        } else if let Tok::Sym("}") = first.tok {
            return Err(diag(first.line, first.col, "a '}' without its '{'"));
        } else {
            return Err(diag(first.line, first.col, format!("a line starts with a label or a block, not '{}'", show(&first.tok))));
        }
        let s = self.node_stmt()?;
        self.end_of_stmt()?;
        Ok(vec![s])
    }
}

// ---------------------------------------------------------------- block kinds and their short forms
/// words for the block kinds (canonical kind, and a preset: the pool mode or the activation function)
const ALIASES: &[(&str, (&str, &str))] = &[
    ("input", ("input", "")), ("conv", ("conv2d", "")), ("conv2d", ("conv2d", "")), ("dwconv", ("dwconv2d", "")), ("dwconv2d", ("dwconv2d", "")),
    ("convt", ("convtranspose2d", "")), ("convtranspose", ("convtranspose2d", "")), ("convtranspose2d", ("convtranspose2d", "")),
    ("pool", ("pool2d", "")), ("pool2d", ("pool2d", "")), ("maxpool", ("pool2d", "max")), ("avgpool", ("pool2d", "avg")),
    ("global_pool", ("global_pool", "")), ("gap", ("global_pool", "")), ("upsample", ("upsample", "")),
    ("bn", ("batchnorm", "")), ("batchnorm", ("batchnorm", "")), ("ln", ("layernorm", "")), ("layernorm", ("layernorm", "")),
    ("gn", ("groupnorm", "")), ("groupnorm", ("groupnorm", "")), ("dropout", ("dropout", "")), ("flatten", ("flatten", "")),
    ("dense", ("dense", "")), ("linear", ("dense", "")), ("activation", ("activation", "")), ("act", ("activation", "")),
    ("relu", ("activation", "relu")), ("leaky_relu", ("activation", "leaky_relu")), ("gelu", ("activation", "gelu")),
    ("silu", ("activation", "silu")), ("sigmoid", ("activation", "sigmoid")), ("tanh", ("activation", "tanh")),
    ("add", ("add", "")), ("concat", ("concat", "")), ("cat", ("concat", "")), ("multiply", ("multiply", "")), ("mul", ("multiply", "")),
    ("to_tokens", ("to_tokens", "")), ("cls_pos", ("cls_pos", "")), ("attention", ("attention", "")), ("attn", ("attention", "")),
    ("token_pool", ("token_pool", "")), ("output", ("output", "")), ("seg_output", ("seg_output", "")),
];

fn kind_of(word: &str) -> Option<(&'static str, &'static str)> {
    ALIASES.iter().find(|(a, _)| *a == word).map(|(_, k)| *k)
}

/// the word a kind is printed with
fn kind_word(kind: &str) -> &str {
    match kind {
        "conv2d" => "conv",
        "dwconv2d" => "dwconv",
        "convtranspose2d" => "convT",
        "batchnorm" => "bn",
        "layernorm" => "ln",
        "groupnorm" => "gn",
        other => other,
    }
}

const PAIR_KEYS: &[&str] = &["kernel", "stride", "size"];

/// a number as JSON: whole numbers as integers
fn num(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        json!(v as i64)
    } else {
        json!(v)
    }
}

/// the parameters as the shape engine reads them: defaults filled in, sizes as [h, w], whole numbers as integers
pub fn effective(kind: &str, params: &Map<String, Value>) -> Map<String, Value> {
    let mut p = model::defaults(kind).as_object().cloned().unwrap_or_default();
    for (k, v) in params {
        p.insert(k.clone(), v.clone());
    }
    for (k, v) in p.iter_mut() {
        if let Some(f) = v.as_f64() {
            *v = if PAIR_KEYS.contains(&k.as_str()) { json!([num(f), num(f)]) } else { num(f) };
        } else if let Some(a) = v.as_array() {
            if a.iter().all(|x| x.is_number()) {
                *v = Value::Array(a.iter().map(|x| num(x.as_f64().unwrap_or(0.0))).collect());
            }
        }
    }
    p
}

#[derive(Debug, Clone)]
enum Val {
    Num(f64),
    Word(String),
}

fn size_word(w: &str) -> Option<[f64; 2]> {
    let (a, b) = w.split_once('x')?;
    Some([a.parse().ok()?, b.parse().ok()?])
}

fn to_json(key: &str, v: &Val) -> Value {
    match v {
        Val::Num(n) if PAIR_KEYS.contains(&key) => json!([num(*n), num(*n)]),
        Val::Num(n) => num(*n),
        Val::Word(w) => match (w.as_str(), size_word(w)) {
            ("true", _) => json!(true),
            ("false", _) => json!(false),
            (_, Some([a, b])) => json!([num(a), num(b)]),
            _ => json!(w),
        },
    }
}

const ACTS: &[&str] = &["relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "none"];

/// the block's parameters from the words and numbers after its kind
fn block_params(kind: &str, preset: &str, args: &[(Option<String>, Val)]) -> Result<Map<String, Value>, String> {
    let mut p = Map::new();
    match kind {
        "pool2d" if !preset.is_empty() => {
            p.insert("mode".into(), json!(preset));
        }
        "activation" if !preset.is_empty() => {
            p.insert("fn".into(), json!(preset));
        }
        _ => {}
    }
    let conv_like = matches!(kind, "conv2d" | "dwconv2d" | "convtranspose2d");
    let mut numbers = 0;
    let mut i = 0;
    let next_val = |i: &mut usize, what: &str| -> Result<Val, String> {
        *i += 1;
        match args.get(*i) {
            Some((None, v)) => Ok(v.clone()),
            _ => Err(format!("'{what}' needs a value after it")),
        }
    };
    while i < args.len() {
        let (key, v) = &args[i];
        if let Some(k) = key {
            let k = match (kind, k.as_str()) {
                (_, "pad") => "padding",
                (_, "dil") => "dilation",
                (_, "outpad") => "output_padding",
                ("attention", "drop") => "dropout",
                (_, other) => other,
            };
            p.insert(k.to_string(), to_json(k, v));
            i += 1;
            continue;
        }
        match v {
            Val::Num(n) => {
                numbers += 1;
                let key = match (kind, numbers) {
                    ("conv2d" | "convtranspose2d", 1) => "filters",
                    ("dense", 1) => "units",
                    ("dropout", 1) => "rate",
                    ("groupnorm", 1) => "groups",
                    ("pool2d", 1) => "size",
                    ("upsample", 1) => "scale",
                    ("attention", 1) => "heads",
                    _ => return Err(format!("what is the number {}? write it as name=value (e.g. filters={})", fmt_num(*n), fmt_num(*n))),
                };
                p.insert(key.into(), to_json(key, v));
            }
            Val::Word(w) => {
                let lw = w.to_ascii_lowercase();
                if let Some(sz) = size_word(&lw) {
                    let key = match kind {
                        "pool2d" => "size",
                        _ if conv_like => "kernel",
                        _ => return Err(format!("'{w}': a size is for a convolution's kernel or a pool's size")),
                    };
                    p.insert(key.into(), json!([num(sz[0]), num(sz[1])]));
                } else if let Some(m) = lw.strip_prefix('x').and_then(|m| m.parse::<f64>().ok()) {
                    match kind {
                        "dwconv2d" => p.insert("multiplier".into(), num(m)),
                        "upsample" => p.insert("scale".into(), num(m)),
                        _ => return Err(format!("'{w}': ×N is for a depthwise convolution (multiplier) or upsampling (scale)")),
                    };
                } else if ACTS.contains(&lw.as_str()) && (conv_like || kind == "dense") {
                    p.insert("activation".into(), json!(lw));
                } else if ACTS.contains(&lw.as_str()) && kind == "activation" && lw != "none" {
                    p.insert("fn".into(), json!(lw));
                } else {
                    match (kind, lw.as_str()) {
                        (_, "same" | "valid") if conv_like => {
                            p.insert("padding".into(), json!(lw));
                        }
                        (_, "stride") => {
                            let v = next_val(&mut i, "stride")?;
                            let j = if matches!(&v, Val::Word(s) if s == "size") { json!("size") } else { to_json("stride", &v) };
                            p.insert("stride".into(), j);
                        }
                        (_, "pad") => {
                            let v = next_val(&mut i, "pad")?;
                            p.insert("padding".into(), to_json("padding", &v));
                        }
                        (_, "dil") if conv_like => {
                            let v = next_val(&mut i, "dil")?;
                            p.insert("dilation".into(), to_json("dilation", &v));
                        }
                        ("convtranspose2d", "outpad") => {
                            let v = next_val(&mut i, "outpad")?;
                            p.insert("output_padding".into(), to_json("output_padding", &v));
                        }
                        (_, "nobias") if conv_like => {
                            p.insert("bias".into(), json!(false));
                        }
                        (_, "bias") if conv_like => {
                            p.insert("bias".into(), json!(true));
                        }
                        ("pool2d" | "global_pool", "max" | "avg") => {
                            p.insert("mode".into(), json!(lw));
                        }
                        ("upsample", "nearest" | "bilinear") => {
                            p.insert("mode".into(), json!(lw));
                        }
                        ("to_tokens", "patches" | "rows") => {
                            p.insert("mode".into(), json!(lw));
                        }
                        ("token_pool", "cls" | "mean") => {
                            p.insert("mode".into(), json!(lw));
                        }
                        ("cls_pos", "cls" | "nocls") => {
                            p.insert("cls".into(), json!(lw == "cls"));
                        }
                        ("attention", "heads") => {
                            let v = next_val(&mut i, "heads")?;
                            p.insert("heads".into(), to_json("heads", &v));
                        }
                        ("attention", "drop") => {
                            let v = next_val(&mut i, "drop")?;
                            p.insert("dropout".into(), to_json("dropout", &v));
                        }
                        _ => return Err(format!("'{w}' means nothing for {} (see the list of words in the help)", kind_word(kind))),
                    }
                }
            }
        }
        i += 1;
    }
    Ok(p)
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Number(n) => fmt_num(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(a) if a.len() == 2 && a.iter().all(|x| x.is_number()) => format!("{}x{}", fmt_value(&a[0]), fmt_value(&a[1])),
        other => other.to_string(),
    }
}

fn pair_text(v: &Value) -> String {
    match v.as_array() {
        Some(a) if a.len() == 2 && a[0] == a[1] => fmt_value(&a[0]),
        _ => fmt_value(v),
    }
}

/// the short form of a block: kind word and the parameters that differ from the defaults
fn block_text(kind: &str, params: &Map<String, Value>) -> String {
    let e = effective(kind, params);
    let d = effective(kind, &Map::new());
    let differs = |k: &str| e.get(k) != d.get(k);
    let mut words: Vec<String> = vec![];
    let mut done: HashSet<&str> = HashSet::new();
    let mut take = |k: &'static str, w: String, words: &mut Vec<String>| {
        done.insert(k);
        if !w.is_empty() {
            words.push(w);
        }
    };
    let get = |k: &str| e.get(k).cloned().unwrap_or(Value::Null);
    let head = match kind {
        "pool2d" => match get("mode").as_str() {
            Some("avg") => "avgpool".to_string(),
            _ => "maxpool".to_string(),
        },
        "activation" => get("fn").as_str().unwrap_or("relu").to_string(),
        k => kind_word(k).to_string(),
    };
    match kind {
        "conv2d" | "convtranspose2d" | "dwconv2d" => {
            if kind == "dwconv2d" {
                take("multiplier", if differs("multiplier") { format!("x{}", fmt_value(&get("multiplier"))) } else { String::new() }, &mut words);
            } else {
                take("filters", fmt_value(&get("filters")), &mut words);
            }
            take("kernel", fmt_value(&get("kernel")), &mut words);
            take("stride", if differs("stride") { format!("stride {}", pair_text(&get("stride"))) } else { String::new() }, &mut words);
            let pad = get("padding");
            take("padding", match (&pad, differs("padding")) {
                (_, false) => String::new(),
                (Value::String(s), true) if s == "same" || s == "valid" => s.clone(),
                (Value::Number(_), true) => format!("pad {}", fmt_value(&pad)),
                (_, true) => format!("pad={}", fmt_value(&pad)),
            }, &mut words);
            if kind == "conv2d" {
                take("dilation", if differs("dilation") { format!("dil {}", fmt_value(&get("dilation"))) } else { String::new() }, &mut words);
            }
            if kind == "convtranspose2d" {
                take("output_padding", if differs("output_padding") { format!("outpad {}", fmt_value(&get("output_padding"))) } else { String::new() }, &mut words);
            }
            take("activation", if differs("activation") { get("activation").as_str().unwrap_or("relu").to_string() } else { String::new() }, &mut words);
            if kind != "convtranspose2d" {
                take("bias", if get("bias") == json!(false) { "nobias".into() } else { String::new() }, &mut words);
            }
        }
        "pool2d" => {
            take("mode", String::new(), &mut words);
            take("size", fmt_value(&get("size")), &mut words);
            let st = get("stride");
            take("stride", if differs("stride") { format!("stride {}", if st == json!("size") { "size".into() } else { pair_text(&st) }) } else { String::new() }, &mut words);
            take("padding", if differs("padding") { format!("pad {}", fmt_value(&get("padding"))) } else { String::new() }, &mut words);
        }
        "global_pool" => take("mode", if differs("mode") { fmt_value(&get("mode")) } else { String::new() }, &mut words),
        "upsample" => {
            take("scale", format!("x{}", fmt_value(&get("scale"))), &mut words);
            take("mode", if differs("mode") { fmt_value(&get("mode")) } else { String::new() }, &mut words);
        }
        "dropout" => take("rate", fmt_value(&get("rate")), &mut words),
        "groupnorm" => take("groups", fmt_value(&get("groups")), &mut words),
        "dense" => {
            take("units", fmt_value(&get("units")), &mut words);
            take("activation", if differs("activation") { get("activation").as_str().unwrap_or("relu").to_string() } else { String::new() }, &mut words);
        }
        "activation" => take("fn", String::new(), &mut words),
        "to_tokens" => take("mode", if differs("mode") { fmt_value(&get("mode")) } else { String::new() }, &mut words),
        "token_pool" => take("mode", if differs("mode") { fmt_value(&get("mode")) } else { String::new() }, &mut words),
        "cls_pos" => take("cls", if get("cls") == json!(false) { "nocls".into() } else { String::new() }, &mut words),
        "attention" => {
            take("heads", format!("heads {}", fmt_value(&get("heads"))), &mut words);
            take("dropout", if differs("dropout") { format!("drop {}", fmt_value(&get("dropout"))) } else { String::new() }, &mut words);
        }
        _ => {}
    }
    // anything else (a parameter the short form does not know) as name=value, so that nothing is lost
    for (k, v) in &e {
        if !done.contains(k.as_str()) && d.get(k) != Some(v) {
            words.push(format!("{k}={}", fmt_value(v)));
        }
    }
    if words.is_empty() {
        head
    } else {
        format!("{head} {}", words.join(" "))
    }
}

// ---------------------------------------------------------------- from text to blocks
/// the result of applying a text: the blocks, and the line each top-level label is on
pub struct Expanded {
    pub graph: Graph,
    pub name: Option<String>,
    pub lines: BTreeMap<String, usize>,
}

struct Ctx<'a> {
    defs: &'a HashMap<String, Def>,
    nodes: Vec<Node>,
    edges: Vec<(String, String)>,
    /// every label made so far -> the label of the block that is its output (itself for a block)
    outputs: HashMap<String, String>,
    lines: BTreeMap<String, usize>,
    depth: usize,
}

struct Scope {
    prefix: String,
    vars: HashMap<String, Val>,
    inputs: Vec<String>,
    /// local labels -> output label
    local: HashMap<String, String>,
    auto: usize,
}

fn eval(e: &Expr, vars: &HashMap<String, Val>) -> Result<Val, String> {
    Ok(match e {
        Expr::Num(n) => Val::Num(*n),
        Expr::Word(w) => vars.get(w).cloned().unwrap_or_else(|| Val::Word(w.clone())),
        Expr::Neg(a) => match eval(a, vars)? {
            Val::Num(n) => Val::Num(-n),
            Val::Word(w) => return Err(format!("'-{w}': '{w}' is not a number (no such name)")),
        },
        Expr::Bin(a, op, b) => {
            let (x, y) = match (eval(a, vars)?, eval(b, vars)?) {
                (Val::Num(x), Val::Num(y)) => (x, y),
                (Val::Word(w), _) | (_, Val::Word(w)) => return Err(format!("'{w}' is not a number (no such name)")),
            };
            Val::Num(match *op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" if y != 0.0 => x / y,
                "//" if y != 0.0 => (x / y).floor(),
                "%" if y != 0.0 => x.rem_euclid(y),
                "^" => x.powf(y),
                _ => return Err("division by zero".into()),
            })
        }
    })
}

fn eval_args(args: &[Arg], vars: &HashMap<String, Val>) -> Result<Vec<(Option<String>, Val)>, String> {
    args.iter().map(|a| match a {
        Arg::Pos(e) => Ok((None, eval(e, vars)?)),
        Arg::Key(k, e) => Ok((Some(k.clone()), eval(e, vars)?)),
    }).collect()
}

fn count_of(e: &Expr, vars: &HashMap<String, Val>, what: &str) -> Result<usize, String> {
    match eval(e, vars)? {
        Val::Num(n) if n.fract() == 0.0 && (0.0..=10_000.0).contains(&n) => Ok(n as usize),
        Val::Num(n) => Err(format!("{what} must be a whole number, not {}", fmt_num(n))),
        Val::Word(w) => Err(format!("{what}: '{w}' is not a number")),
    }
}

impl Ctx<'_> {
    fn resolve(&self, sc: &Scope, r: &str) -> Option<String> {
        if r == "in" || r == "in1" {
            return sc.inputs.first().cloned();
        }
        if let Some(k) = r.strip_prefix("in").and_then(|k| k.parse::<usize>().ok()) {
            return sc.inputs.get(k.wrapping_sub(1)).cloned();
        }
        if let Some(o) = sc.local.get(r) {
            return Some(o.clone());
        }
        // a path into an earlier repeat or call: "5.3.b" (relative to this scope, or from the top)
        self.outputs.get(&format!("{}{r}", sc.prefix)).or_else(|| self.outputs.get(r)).cloned()
    }

    fn stmts(&mut self, stmts: &[Stmt], sc: &mut Scope) -> Result<Option<String>, Diagnostic> {
        let mut prev: Option<String> = sc.inputs.first().cloned();
        for s in stmts {
            match s {
                Stmt::Let { name, value, line, col } => {
                    let v = eval(value, &sc.vars).map_err(|m| diag(*line, *col, m))?;
                    sc.vars.insert(name.clone(), v);
                }
                Stmt::Node { label, item, inputs, line, col } => {
                    let label = match label {
                        Some(l) => l.clone(),
                        None => loop {
                            sc.auto += 1;
                            let l = if sc.prefix.is_empty() { sc.auto.to_string() } else { letters(sc.auto) };
                            if !sc.local.contains_key(&l) {
                                break l;
                            }
                        },
                    };
                    if sc.local.contains_key(&label) {
                        return Err(diag(*line, *col, format!("the label '{label}' is used twice")));
                    }
                    if let Ok(n) = label.parse::<usize>() {
                        sc.auto = sc.auto.max(n);
                    }
                    let full = format!("{}{label}", sc.prefix);
                    if sc.prefix.is_empty() {
                        self.lines.insert(label.clone(), *line);
                    }
                    let ins: Vec<String> = match inputs {
                        Some(refs) => {
                            let mut v = vec![];
                            for (r, l, c) in refs {
                                match self.resolve(sc, r) {
                                    Some(o) => v.push(o),
                                    None => {
                                        let known: Vec<&String> = sc.local.keys().collect();
                                        return Err(diag(*l, *c, if r.starts_with("in") && !sc.prefix.is_empty() {
                                            format!("'{r}': this group has only {} input(s)", sc.inputs.len())
                                        } else if known.is_empty() {
                                            format!("'{r}': there is no block '{r}' above this line")
                                        } else {
                                            format!("'{r}': there is no block '{r}' above this line (a block may only use blocks above it)")
                                        }));
                                    }
                                }
                            }
                            v
                        }
                        None => prev.iter().cloned().collect(),
                    };
                    let out = self.item(item, &full, &ins, sc, *line, *col)?;
                    sc.local.insert(label, out.clone());
                    self.outputs.insert(full, out.clone());
                    prev = Some(out);
                }
            }
        }
        Ok(prev)
    }

    fn item(&mut self, item: &Item, full: &str, ins: &[String], sc: &Scope, line: usize, col: usize) -> Result<String, Diagnostic> {
        let err = |m: String| diag(line, col, m);
        match item {
            Item::Block { word, args } => {
                let (kind, preset) = kind_of(word).ok_or_else(|| err(unknown_block(word)))?;
                let vals = eval_args(args, &sc.vars).map_err(err)?;
                let params = block_params(kind, preset, &vals).map_err(err)?;
                if self.nodes.len() >= MAX_BLOCKS {
                    return Err(err(format!("more than {MAX_BLOCKS} blocks: a repeat count is probably too large")));
                }
                let ins: &[String] = if kind == "input" { &[] } else { ins };
                for src in ins {
                    self.edges.push((src.clone(), full.to_string()));
                }
                self.nodes.push(Node { id: String::new(), kind: kind.into(), params: effective(kind, &params), label: full.to_string(), ..Default::default() });
                Ok(full.to_string())
            }
            Item::Call { name, args } => {
                let def = self.defs.get(name).ok_or_else(|| err(format!("there is no function '{name}'")))?;
                if self.depth >= MAX_DEPTH {
                    return Err(err(format!("functions call each other more than {MAX_DEPTH} deep (does '{name}' call itself?)")));
                }
                let vals = eval_args(args, &sc.vars).map_err(err)?;
                let mut vars = sc.vars.clone();
                let mut pos = 0;
                for (pn, default) in &def.params {
                    let v = vals.iter().find(|(k, _)| k.as_deref() == Some(pn)).map(|(_, v)| v.clone()).or_else(|| {
                        let p = vals.iter().filter(|(k, _)| k.is_none()).nth(pos).map(|(_, v)| v.clone());
                        if p.is_some() {
                            pos += 1;
                        }
                        p
                    });
                    let v = match (v, default) {
                        (Some(v), _) => v,
                        (None, Some(d)) => eval(d, &sc.vars).map_err(err)?,
                        (None, None) => return Err(err(format!("'{name}' needs a value for '{pn}'"))),
                    };
                    vars.insert(pn.clone(), v);
                }
                if vals.iter().filter(|(k, _)| k.is_none()).count() > pos {
                    return Err(err(format!("'{name}' takes {} value(s), more were given", def.params.len())));
                }
                let mut inner = Scope { prefix: format!("{full}."), vars, inputs: ins.to_vec(), local: HashMap::new(), auto: 0 };
                self.depth += 1;
                let out = self.stmts(&def.body.clone(), &mut inner)?;
                self.depth -= 1;
                out.ok_or_else(|| err(format!("the function '{name}' has no blocks")))
            }
            Item::Repeat { count, body } => {
                let n = count_of(count, &sc.vars, "repeat").map_err(err)?;
                if n == 0 {
                    return Err(err("repeat 0 times: write at least 1".into()));
                }
                let passes: Vec<HashMap<String, Val>> = (1..=n).map(|i| HashMap::from([("i".to_string(), Val::Num(i as f64))])).collect();
                self.passes(&passes, body, full, ins, sc, line, col)
            }
            Item::For { vars, values, body } => {
                let mut passes = vec![];
                for (i, tuple) in values.iter().enumerate() {
                    let mut m = HashMap::from([("i".to_string(), Val::Num((i + 1) as f64))]);
                    for (name, e) in vars.iter().zip(tuple) {
                        m.insert(name.clone(), eval(e, &sc.vars).map_err(err)?);
                    }
                    passes.push(m);
                }
                self.passes(&passes, body, full, ins, sc, line, col)
            }
        }
    }

    /// the body once per pass, each pass taking the previous one's output
    #[allow(clippy::too_many_arguments)]
    fn passes(&mut self, passes: &[HashMap<String, Val>], body: &[Stmt], full: &str, ins: &[String], sc: &Scope, line: usize, col: usize)
        -> Result<String, Diagnostic> {
        let mut cur: Vec<String> = ins.to_vec();
        for (r, extra) in passes.iter().enumerate() {
            let mut vars = sc.vars.clone();
            vars.extend(extra.clone());
            let mut inner = Scope { prefix: format!("{full}.{}.", r + 1), vars, inputs: cur.clone(), local: HashMap::new(), auto: 0 };
            let out = self.stmts(body, &mut inner)?.ok_or_else(|| diag(line, col, "the braces contain no blocks"))?;
            self.outputs.insert(format!("{full}.{}", r + 1), out.clone());
            // the next pass takes this one's output (and the same further inputs, if any)
            cur = std::iter::once(out).chain(ins.iter().skip(1).cloned()).collect();
        }
        Ok(cur[0].clone())
    }
}

/// labels inside braces: a, b, …, z, a1, b1, …
fn letters(i: usize) -> String {
    let i = i - 1;
    let c = (b'a' + (i % 26) as u8) as char;
    if i < 26 {
        c.to_string()
    } else {
        format!("{c}{}", i / 26)
    }
}

/// build the blocks of a text. Blocks keep the id and position of the block with the same label in `old`; new blocks are placed next
/// to their first input (or everything is arranged when nothing matches).
pub fn expand(text: &str, old: &Graph) -> Result<Expanded, Vec<Diagnostic>> {
    let prog = parse(text).map_err(|d| vec![d])?;
    let mut ctx = Ctx { defs: &prog.defs, nodes: vec![], edges: vec![], outputs: HashMap::new(), lines: BTreeMap::new(), depth: 0 };
    let mut top = Scope { prefix: String::new(), vars: HashMap::new(), inputs: vec![], local: HashMap::new(), auto: 0 };
    ctx.stmts(&prog.stmts, &mut top).map_err(|d| vec![d])?;
    let (mut nodes, edges, lines) = (ctx.nodes, ctx.edges, ctx.lines);
    // ids and positions: from the old block with the same label
    let by_label: HashMap<&str, &Node> = old.nodes.iter().filter(|n| !n.label.is_empty()).map(|n| (n.label.as_str(), n)).collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut kept = 0;
    for n in nodes.iter_mut() {
        if let Some(o) = by_label.get(n.label.as_str()).filter(|o| !used.contains(&o.id)) {
            n.id = o.id.clone();
            n.x = o.x;
            n.y = o.y;
            kept += 1;
            used.insert(n.id.clone());
        }
    }
    let taken: HashSet<String> = old.nodes.iter().map(|n| n.id.clone()).chain(used.iter().cloned()).collect();
    let mut fresh: HashSet<String> = HashSet::new();
    let mut new_ids: HashSet<String> = HashSet::new();
    for n in nodes.iter_mut().filter(|n| n.id.is_empty()) {
        let base = format!("t{}", n.label.replace('.', "_"));
        let mut id = base.clone();
        let mut k = 1;
        while taken.contains(&id) || fresh.contains(&id) {
            k += 1;
            id = format!("{base}_{k}");
        }
        fresh.insert(id.clone());
        new_ids.insert(id.clone());
        n.id = id;
    }
    let id_of: HashMap<String, String> = nodes.iter().map(|n| (n.label.clone(), n.id.clone())).collect();
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    let edges: Vec<Edge> = edges.into_iter().map(|(s, t)| {
        let (s, t) = (id_of[&s].clone(), id_of[&t].clone());
        let k = seen.entry((s.clone(), t.clone())).or_default();
        *k += 1;
        Edge { id: format!("e{s}-{t}-{k}"), source: s, target: t }
    }).collect();
    let mut g = Graph { nodes: std::mem::take(&mut nodes), edges, text: text.to_string() };
    if kept == 0 {
        model::layout(&mut g);
    } else {
        // a new block goes to the right of its first input, below anything already there
        for i in 0..g.nodes.len() {
            if !new_ids.contains(&g.nodes[i].id) {
                continue;
            }
            let id = g.nodes[i].id.clone();
            let src = g.edges.iter().find(|e| e.target == id).map(|e| e.source.clone());
            let (mut x, mut y) = match src.and_then(|s| g.nodes.iter().find(|n| n.id == s)) {
                Some(s) => (s.x + 300.0, s.y),
                None => (60.0, 60.0),
            };
            let busy = |g: &Graph, x: f64, y: f64| g.nodes.iter().any(|n| n.id != id && (n.x - x).abs() < 200.0 && (n.y - y).abs() < 100.0);
            let mut tries = 0;
            while busy(&g, x, y) && tries < 200 {
                y += 150.0;
                tries += 1;
            }
            if tries >= 200 {
                x += 300.0;
            }
            g.nodes[i].x = x;
            g.nodes[i].y = y;
        }
    }
    Ok(Expanded { graph: g, name: prog.name, lines })
}

// ---------------------------------------------------------------- from blocks to text
/// blocks in computation order: a block comes right after the inputs it needs, following one branch to its end before the next
/// (so a repeated group's blocks stay together); ties by position on the canvas
fn print_order(g: &Graph) -> Vec<usize> {
    let idx: HashMap<&str, usize> = g.nodes.iter().enumerate().map(|(i, n)| (n.id.as_str(), i)).collect();
    let n = g.nodes.len();
    let mut ins: Vec<Vec<usize>> = vec![vec![]; n];
    for e in &g.edges {
        if let (Some(&s), Some(&t)) = (idx.get(e.source.as_str()), idx.get(e.target.as_str())) {
            ins[t].push(s);
        }
    }
    let mut placed: Vec<Option<usize>> = vec![None; n];
    let mut order = vec![];
    while order.len() < n {
        // ready: all inputs placed; prefer the one whose newest input was placed last, then left / top on the canvas
        let pick = (0..n).filter(|&i| placed[i].is_none() && ins[i].iter().all(|&s| placed[s].is_some()))
            .max_by(|&a, &b| {
                let ka = ins[a].iter().map(|&s| placed[s].unwrap() as i64).max().unwrap_or(-1);
                let kb = ins[b].iter().map(|&s| placed[s].unwrap() as i64).max().unwrap_or(-1);
                ka.cmp(&kb).then(g.nodes[b].x.partial_cmp(&g.nodes[a].x).unwrap_or(std::cmp::Ordering::Equal))
                    .then(g.nodes[b].y.partial_cmp(&g.nodes[a].y).unwrap_or(std::cmp::Ordering::Equal))
            });
        // a loop on the canvas: place the rest in canvas order (their lines then use blocks below them, which the text reports)
        let i = pick.unwrap_or_else(|| (0..n).find(|&i| placed[i].is_none()).unwrap());
        placed[i] = Some(order.len());
        order.push(i);
    }
    order
}

/// a group of consecutive blocks repeated k times: start, length, times
struct Run {
    start: usize,
    len: usize,
    times: usize,
}

fn find_runs(g: &Graph, order: &[usize]) -> Vec<Run> {
    let n = order.len();
    let pos: HashMap<usize, usize> = order.iter().enumerate().map(|(p, &i)| (i, p)).collect();
    let idx: HashMap<&str, usize> = g.nodes.iter().enumerate().map(|(i, n)| (n.id.as_str(), i)).collect();
    // inputs and users of every block, by position in `order`
    let mut ins: Vec<Vec<usize>> = vec![vec![]; n];
    let mut users: Vec<Vec<usize>> = vec![vec![]; n];
    for e in &g.edges {
        if let (Some(&s), Some(&t)) = (idx.get(e.source.as_str()), idx.get(e.target.as_str())) {
            ins[pos[&t]].push(pos[&s]);
            users[pos[&s]].push(pos[&t]);
        }
    }
    let sig: Vec<String> = order.iter().map(|&i| {
        let nd = &g.nodes[i];
        format!("{}|{}", nd.kind, serde_json::to_string(&effective(&nd.kind, &nd.params)).unwrap_or_default())
    }).collect();
    // the shape of group [s, s+len): each block's signature and inputs (relative position, or "in" = the one block before the group)
    let group = |s: usize, len: usize| -> Option<(Vec<String>, usize)> {
        let mut entry: Option<usize> = None;
        let mut out = vec![];
        for p in s..s + len {
            if g.nodes[order[p]].kind == "input" {
                return None;
            }
            let mut refs = vec![];
            for &q in &ins[p] {
                if q >= s && q < s + len {
                    refs.push(format!("{}", q - s));
                } else {
                    if entry.is_some_and(|e| e != q) {
                        return None;
                    }
                    entry = Some(q);
                    refs.push("in".into());
                }
            }
            if refs.is_empty() {
                return None;
            }
            // only the group's last block may be used after the group
            if p != s + len - 1 && users[p].iter().any(|&u| u >= s + len || u < s) {
                return None;
            }
            out.push(format!("{}<{}", sig[p], refs.join(",")));
        }
        Some((out, entry?))
    };
    let mut runs = vec![];
    let mut s = 0;
    while s < n {
        let mut best: Option<Run> = None;
        for len in 2..=((n - s) / 2).min(64) {
            let Some((shape, _)) = group(s, len) else { continue };
            let mut times = 1;
            while s + (times + 1) * len <= n {
                match group(s + times * len, len) {
                    Some((sh, entry)) if sh == shape && entry == s + times * len - 1 => times += 1,
                    _ => break,
                }
            }
            if times >= 2 && best.as_ref().is_none_or(|b| times * len > b.times * b.len) {
                best = Some(Run { start: s, len, times });
            }
        }
        match best {
            Some(r) => {
                s = r.start + r.len * r.times;
                runs.push(r);
            }
            None => s += 1,
        }
    }
    runs
}

/// the text of a design (numbered lines; identical consecutive groups as `repeat`), and the label of every block (by node id)
pub fn print(g: &Graph) -> (String, HashMap<String, String>) {
    let order = print_order(g);
    let runs = find_runs(g, &order);
    let pos: HashMap<&str, usize> = order.iter().enumerate().map(|(p, &i)| (g.nodes[i].id.as_str(), p)).collect();
    let inputs = |p: usize| -> Vec<usize> {
        let id = &g.nodes[order[p]].id;
        g.edges.iter().filter(|e| &e.target == id).filter_map(|e| pos.get(e.source.as_str()).copied()).collect()
    };
    // labels: top-level numbers; inside a repeat "<n>.<pass>.<letter>"
    let mut label: Vec<String> = vec![String::new(); order.len()];
    // for references: a repeat's output is written as the repeat's own number
    let mut ref_name: Vec<String> = vec![String::new(); order.len()];
    let mut lines: Vec<String> = vec![];
    let mut num = 0;
    let mut p = 0;
    let mut prev_out: Option<usize> = None;
    let run_at: HashMap<usize, &Run> = runs.iter().map(|r| (r.start, r)).collect();
    // a block used before it is placed (only with a loop on the canvas) has no name yet: "?<id>", which the text then reports
    let refs_text = |ins: &[usize], ref_name: &[String]| {
        ins.iter().map(|&q| if ref_name[q].is_empty() { format!("?{}", g.nodes[order[q]].id) } else { ref_name[q].clone() }).collect::<Vec<_>>().join(", ")
    };
    while p < order.len() {
        num += 1;
        if let Some(r) = run_at.get(&p) {
            for t in 0..r.times {
                for j in 0..r.len {
                    let q = r.start + t * r.len + j;
                    label[q] = format!("{num}.{}.{}", t + 1, letters(j + 1));
                    ref_name[q] = label[q].clone();
                }
            }
            let last = r.start + r.times * r.len - 1;
            ref_name[last] = num.to_string();
            // the group's input: the block before it
            let first_ins = inputs(r.start).into_iter().find(|&q| q < r.start);
            let head = match first_ins {
                Some(e) if Some(e) == prev_out => format!("{num} repeat {} {{", r.times),
                Some(e) => format!("{num} repeat {} <- {} {{", r.times, ref_name[e]),
                None => format!("{num} repeat {} {{", r.times),
            };
            lines.push(head);
            for j in 0..r.len {
                let q = r.start + j;
                let nd = &g.nodes[order[q]];
                let ins: Vec<String> = inputs(q).iter().map(|&s| if s >= r.start && s < r.start + r.len { letters(s - r.start + 1) } else { "in".into() }).collect();
                let natural = if j == 0 { vec!["in".to_string()] } else { vec![letters(j)] };
                let tail = if ins == natural { String::new() } else { format!(" <- {}", ins.join(", ")) };
                lines.push(format!("  {} {}{tail}", letters(j + 1), block_text(&nd.kind, &nd.params)));
            }
            lines.push("}".into());
            prev_out = Some(last);
            p = last + 1;
            continue;
        }
        label[p] = num.to_string();
        ref_name[p] = num.to_string();
        let nd = &g.nodes[order[p]];
        let ins = inputs(p);
        let tail = if nd.kind == "input" && ins.is_empty() {
            String::new()
        } else if ins.is_empty() {
            " <- none".into()
        } else if ins.len() == 1 && Some(ins[0]) == prev_out {
            String::new()
        } else {
            // a block used before it is placed (a loop on the canvas) gets its label later: name it by its future number
            format!(" <- {}", refs_text(&ins, &ref_name))
        };
        lines.push(format!("{num} {}{tail}", block_text(&nd.kind, &nd.params)));
        prev_out = Some(p);
        p += 1;
    }
    let labels = order.iter().enumerate().map(|(p, &i)| (g.nodes[i].id.clone(), label[p].clone())).collect();
    (lines.join("\n") + "\n", labels)
}

// ---------------------------------------------------------------- keeping canvas and text in step
/// the same blocks (by id), kinds, parameters (as the engine reads them), labels and inputs (in order); positions do not count
pub fn same_structure(a: &Graph, b: &Graph) -> bool {
    if a.nodes.len() != b.nodes.len() {
        return false;
    }
    let bm: HashMap<&str, &Node> = b.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    a.nodes.iter().all(|n| match bm.get(n.id.as_str()) {
        Some(m) => n.kind == m.kind && n.label == m.label && effective(&n.kind, &n.params) == effective(&m.kind, &m.params)
            && a.inputs_of(&n.id) == b.inputs_of(&m.id),
        None => false,
    })
}

/// after a change on the canvas: keep the text when it still describes the blocks exactly (moves, or a text just applied), else print
/// the text again from the blocks (functions and loops written by hand then become plain lines and repeats) and label the blocks
pub fn sync(g: &Graph) -> Graph {
    if !g.text.is_empty() {
        if let Ok(e) = expand(&g.text, g) {
            if same_structure(&e.graph, g) {
                return g.clone();
            }
        }
    }
    let (text, labels) = print(g);
    let mut out = g.clone();
    out.text = text;
    for n in &mut out.nodes {
        n.label = labels.get(&n.id).cloned().unwrap_or_default();
    }
    out
}

/// after the text was edited: the blocks it describes (ids and positions kept by label), with the text
pub fn apply(text: &str, old: &Graph) -> Result<Graph, Vec<Diagnostic>> {
    Ok(expand(text, old)?.graph)
}

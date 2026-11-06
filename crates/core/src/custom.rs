//! The design's own PyTorch code (Custom blocks). A design may carry `nn.Module` classes (`Graph::python`); a Custom block names one of
//! them. Nothing here runs Python: the classes are read as text for their name, their `__init__` arguments (the input shape: `c, h, w`
//! for an image, `t, d` for tokens, `n` for a vector) and a `# out: …` line declaring the output shape in those names. The shape engine
//! uses the declaration; the one-batch check on the GPU machine compares it with the real shape.
//!
//! ```python
//! class SE(nn.Module):
//!     # out: c, h, w
//!     def __init__(self, c, h, w):
//!         super().__init__()
//!         self.fc = nn.Sequential(nn.Linear(c, c // 4), nn.ReLU(), nn.Linear(c // 4, c), nn.Sigmoid())
//!
//!     def forward(self, x):
//!         return x * self.fc(x.mean((2, 3)))[:, :, None, None]
//! ```
//!
//! In the text language the code is a section at the top: a line `python """`, the code, and a line `"""`.

use std::collections::HashMap;

/// one class of the design's code
#[derive(Debug, Clone, PartialEq)]
pub struct PyClass {
    pub name: String,
    /// the `__init__` arguments after `self` (the input shape, one name per dimension)
    pub args: Vec<String>,
    /// the declared output shape (`# out:` line), one expression per dimension; None = the input's shape
    pub out: Option<Vec<String>>,
    /// a problem with the class itself (no `__init__`, no `forward`, a bad `# out:` line)
    pub problem: Option<String>,
}

/// the most characters of code a design may carry
pub const MAX_CODE: usize = 200_000;

/// the classes defined at the top level of the code (`class Name(...):` at column 0), in order
pub fn classes(code: &str) -> Vec<PyClass> {
    let lines: Vec<&str> = code.lines().collect();
    let mut out: Vec<PyClass> = vec![];
    let mut i = 0;
    while i < lines.len() {
        let Some(name) = class_name(lines[i]) else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        // the class body: until the next line that starts at column 0 (comments and blank lines belong to the body)
        while j < lines.len() && (lines[j].trim().is_empty() || lines[j].starts_with([' ', '\t']) || lines[j].trim_start().starts_with('#') && !lines[j].starts_with('#')) {
            j += 1;
        }
        let body = &lines[i + 1..j];
        let mut c = PyClass { name, args: vec![], out: None, problem: None };
        let init = body.iter().find_map(|l| l.trim().strip_prefix("def __init__(").map(|r| r.to_string()));
        match init {
            Some(r) => {
                let inside = r.split(')').next().unwrap_or("");
                let names: Vec<String> = inside.split(',').map(|a| a.split([':', '=']).next().unwrap_or("").trim().to_string()).filter(|a| !a.is_empty()).collect();
                if names.first().map(|s| s.as_str()) != Some("self") {
                    c.problem = Some(format!("{}: __init__ must start with self", c.name));
                } else {
                    c.args = names[1..].to_vec();
                    if c.args.is_empty() || c.args.len() > 3 || c.args.iter().any(|a| a.starts_with('*')) {
                        c.problem = Some(format!("{}: __init__(self, …) takes the input shape: c, h, w for an image, t, d for tokens or n for a vector", c.name));
                    }
                }
            }
            None => c.problem = Some(format!("{}: no __init__(self, …) (it receives the input shape: c, h, w for an image)", c.name)),
        }
        if c.problem.is_none() && !body.iter().any(|l| l.trim().starts_with("def forward(")) {
            c.problem = Some(format!("{}: no forward(self, x)", c.name));
        }
        if let Some(decl) = body.iter().find_map(|l| l.trim().strip_prefix("# out:")) {
            let dims: Vec<String> = decl.split(',').map(|d| d.trim().to_string()).collect();
            if dims.is_empty() || dims.len() > 3 || dims.iter().any(|d| d.is_empty()) {
                c.problem.get_or_insert(format!("{}: '# out:' lists 1 to 3 sizes separated by commas, e.g. # out: c, h // 2, w // 2", c.name));
            } else {
                c.out = Some(dims);
            }
        }
        out.push(c);
        i = j;
    }
    out
}

fn class_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("class ")?;
    let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    Some(name)
}

/// a class name a Custom block may use (also what the text language accepts after `custom`)
pub fn ok_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.starts_with(|c: char| c.is_ascii_digit())
        && !RESERVED.contains(&name)
}

/// names the generated train.py already uses at the top level
const RESERVED: &[&str] = &["Net", "ToTokens", "ClsPos", "SelfAttention", "TokenPool", "OOM", "Interrupted", "DATASET", "TRAIN", "CLASSES", "LAYERS", "EXPECTED"];

/// the output shape a class declares for an input shape: Ok(shape) or the problem in plain words
pub fn out_shape(c: &PyClass, input: &[usize]) -> Result<Vec<usize>, String> {
    if let Some(p) = &c.problem {
        return Err(p.clone());
    }
    if c.args.len() != input.len() {
        let what = match input.len() {
            3 => "an image (c, h, w)",
            2 => "tokens (t, d)",
            _ => "a vector (n)",
        };
        return Err(format!("{}'s __init__ takes {} sizes ({}), but its input is {what} {:?}", c.name, c.args.len(), c.args.join(", "), input));
    }
    let Some(dims) = &c.out else { return Ok(input.to_vec()) };
    let vars: HashMap<&str, i64> = c.args.iter().map(|a| a.as_str()).zip(input.iter().map(|&v| v as i64)).collect();
    let mut shape = vec![];
    for d in dims {
        let v = calc(d, &vars).map_err(|e| format!("{}: # out: '{d}': {e}", c.name))?;
        if !(1..=1 << 31).contains(&v) {
            return Err(format!("{}: # out: '{d}' gives {v} for the input {:?}; every size must be at least 1", c.name, input));
        }
        shape.push(v as usize);
    }
    Ok(shape)
}

/// a whole-number expression: numbers, the shape names, + - * // % and parentheses (Python's `/` makes a float: write `//`)
fn calc(e: &str, vars: &HashMap<&str, i64>) -> Result<i64, String> {
    let toks: Vec<String> = {
        let mut t = vec![];
        let cs: Vec<char> = e.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c.is_whitespace() {
                i += 1;
            } else if c.is_ascii_alphanumeric() || c == '_' {
                let s = i;
                while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                    i += 1;
                }
                t.push(cs[s..i].iter().collect());
            } else if c == '/' && cs.get(i + 1) == Some(&'/') {
                t.push("//".into());
                i += 2;
            } else if "+-*%()".contains(c) {
                t.push(c.to_string());
                i += 1;
            } else if c == '/' {
                return Err("use // (whole-number division) for sizes".into());
            } else {
                return Err(format!("unexpected '{c}'"));
            }
        }
        t
    };
    let mut p = 0;
    let v = sum(&toks, &mut p, vars)?;
    if p != toks.len() {
        return Err(format!("unexpected '{}'", toks[p]));
    }
    Ok(v)
}

fn sum(t: &[String], p: &mut usize, vars: &HashMap<&str, i64>) -> Result<i64, String> {
    let mut a = product(t, p, vars)?;
    while *p < t.len() && (t[*p] == "+" || t[*p] == "-") {
        let op = t[*p].clone();
        *p += 1;
        let b = product(t, p, vars)?;
        a = if op == "+" { a.saturating_add(b) } else { a.saturating_sub(b) };
    }
    Ok(a)
}

fn product(t: &[String], p: &mut usize, vars: &HashMap<&str, i64>) -> Result<i64, String> {
    let mut a = atom(t, p, vars)?;
    while *p < t.len() && matches!(t[*p].as_str(), "*" | "//" | "%") {
        let op = t[*p].clone();
        *p += 1;
        let b = atom(t, p, vars)?;
        a = match op.as_str() {
            "*" => a.saturating_mul(b),
            _ if b == 0 => return Err("division by zero".into()),
            "//" => a.div_euclid(b),
            _ => a.rem_euclid(b),
        };
    }
    Ok(a)
}

fn atom(t: &[String], p: &mut usize, vars: &HashMap<&str, i64>) -> Result<i64, String> {
    let Some(tok) = t.get(*p).cloned() else { return Err("a size is missing".into()) };
    *p += 1;
    match tok.as_str() {
        "-" => Ok(-atom(t, p, vars)?),
        "(" => {
            let v = sum(t, p, vars)?;
            if t.get(*p).map(|s| s.as_str()) != Some(")") {
                return Err("a parenthesis is not closed".into());
            }
            *p += 1;
            Ok(v)
        }
        _ => match tok.parse::<i64>() {
            Ok(v) => Ok(v),
            Err(_) => vars.get(tok.as_str()).copied().ok_or_else(|| {
                let mut names: Vec<&str> = vars.keys().copied().collect();
                names.sort();
                format!("'{tok}' is not one of the input's sizes ({})", names.join(", "))
            }),
        },
    }
}

// ---------------------------------------------------------------- the section in the text language
const OPEN: &str = "python \"\"\"";
const CLOSE: &str = "\"\"\"";

/// the code of the text's python section, and the text with that section's lines blanked (so the line numbers of the blocks stay)
pub fn split_section(text: &str) -> (String, String) {
    let mut code = vec![];
    let mut rest = vec![];
    let mut inside = false;
    for line in text.lines() {
        if !inside && line.trim() == OPEN {
            inside = true;
            rest.push("");
        } else if inside && line.trim() == CLOSE {
            inside = false;
            rest.push("");
        } else if inside {
            code.push(line);
            rest.push("");
        } else {
            rest.push(line);
        }
    }
    let mut code = code.join("\n");
    if !code.is_empty() {
        code.push('\n');
    }
    let mut rest = rest.join("\n");
    if text.ends_with('\n') {
        rest.push('\n');
    }
    (code, rest)
}

/// the text with its python section holding `code`: replaced where it is, else put after the `net "…"` line (or at the top); an empty
/// `code` removes the section
pub fn set_section(text: &str, code: &str) -> String {
    let (old, _) = split_section(text);
    if old.trim_end() == code.trim_end() && (text.contains(OPEN) || code.trim().is_empty()) {
        return text.to_string();
    }
    let section = if code.trim().is_empty() { String::new() } else { format!("{OPEN}\n{}\n{CLOSE}\n", code.trim_end_matches('\n')) };
    let mut out = String::new();
    let mut inside = false;
    let mut placed = false;
    for line in text.lines() {
        if !inside && line.trim() == OPEN {
            inside = true;
            if !placed {
                out += &section;
                placed = true;
            }
            continue;
        }
        if inside {
            if line.trim() == CLOSE {
                inside = false;
            }
            continue;
        }
        out += line;
        out.push('\n');
    }
    if !placed && !section.is_empty() {
        // after the net "…" line when the text starts with one, else at the top
        match out.lines().next() {
            Some(first) if first.trim_start().starts_with("net ") || first.trim_start() == "net" => {
                let first = first.to_string();
                out = format!("{first}\n{section}{}", out[first.len()..].trim_start_matches('\n'));
            }
            _ => out = format!("{section}{out}"),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SE: &str = "import math\n\nclass SE(nn.Module):\n    # out: c, h, w\n    def __init__(self, c, h, w):\n        super().__init__()\n\n    def forward(self, x):\n        return x\n\nclass Down(nn.Module):\n    # out: c * 2, h // 2, w // 2\n    def __init__(self, c, h, w):\n        super().__init__()\n    def forward(self, x):\n        return x\n";

    #[test]
    fn reads_classes_and_shapes() {
        let cs = classes(SE);
        assert_eq!(cs.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["SE", "Down"]);
        assert_eq!(out_shape(&cs[0], &[32, 16, 16]), Ok(vec![32, 16, 16]));
        assert_eq!(out_shape(&cs[1], &[32, 15, 16]), Ok(vec![64, 7, 8]));
        assert!(out_shape(&cs[1], &[64, 32]).unwrap_err().contains("tokens"));
        let bad = classes("class X(nn.Module):\n    # out: c / 2\n    def __init__(self, c):\n        pass\n    def forward(self, x):\n        return x\n");
        assert!(out_shape(&bad[0], &[8]).unwrap_err().contains("//"));
        let none = classes("class Y(nn.Module):\n    def forward(self, x):\n        return x\n");
        assert!(none[0].problem.as_ref().unwrap().contains("__init__"));
    }

    #[test]
    fn section_round_trip() {
        let text = "net \"x\"\n1 input\n2 custom SE\n";
        let with = set_section(text, SE);
        assert!(with.starts_with("net \"x\"\npython \"\"\"\nimport math"), "{with}");
        let (code, rest) = split_section(&with);
        assert_eq!(code, SE);
        assert_eq!(rest.lines().count(), with.lines().count(), "blanked, not removed");
        assert_eq!(set_section(&with, SE), with, "unchanged code keeps the text");
        let changed = set_section(&with, "class Z(nn.Module):\n    pass\n");
        assert_eq!(split_section(&changed).0, "class Z(nn.Module):\n    pass\n");
        assert_eq!(set_section(&changed, ""), text);
    }
}

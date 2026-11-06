#!/usr/bin/env python3
"""Writes CHANGELOG.md from the git history: one entry per commit, newest first, headed by its time as dd_mm_yy_hr_min_sec (the
commit time). Commits whose subject starts with "CHANGELOG" (the updates of this file) are left out.  Run from the repository:
    python3 scripts/make_changelog.py
"""
import os
import subprocess

SEP, END = "\x1f", "\x1e"
root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
log = subprocess.run(["git", "log", "--date=format:%d_%m_%y_%H_%M_%S", f"--format=%cd{SEP}%h{SEP}%s{SEP}%b{END}"],
                     capture_output=True, text=True, check=True, cwd=root).stdout
out = ["# Changelog", "",
       "Every change to NN Builder, newest first. Each entry is headed by its time as `dd_mm_yy_hr_min_sec` (day, month, two-digit year,",
       "hour, minute, second; the commit time) and the commit. Regenerate with `python3 scripts/make_changelog.py`.", ""]
for rec in log.split(END):
    rec = rec.strip("\n")
    if not rec:
        continue
    when, short, subject, body = (rec.split(SEP) + ["", "", "", ""])[:4]
    if subject.startswith("CHANGELOG"):
        continue
    # attribution trailers (Co-Authored-By, Claude-Session links) are not part of what changed
    lines = [l.rstrip() for l in body.strip().splitlines() if not l.lower().startswith(("co-authored-by", "claude-session"))]
    while lines and not lines[-1]:
        lines.pop()
    out += [f"## {when}", "", f"**{subject}** (`{short}`)", ""]
    if lines:
        out += lines + [""]
with open(os.path.join(root, "CHANGELOG.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out).rstrip() + "\n")
print(f"CHANGELOG.md: {sum(1 for l in out if l.startswith('## '))} entries")

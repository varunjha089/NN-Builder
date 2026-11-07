"""CI only (GitHub Actions; never on the owner's Mac): an LZW-compressed GeoTIFF (as Earth Engine exports them) must be readable where
tifffile is installed but imagecodecs is not (Kaggle's case, found 2026-11-07). The harness's own tiff_read / pip_install are taken from
python/harness.py and run in two fresh processes: one where tifffile is not imported yet (imagecodecs is installed first) and one where
it already is (installed, then tifffile reloaded). imagecodecs is uninstalled before each.

usage: python ci_tiff_check.py <lzw.tif>
"""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def harness_functions():
    """the source of tiff_read and pip_install from harness.py (the rest needs PyTorch and a run folder)"""
    src = open(os.path.join(HERE, "harness.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("tiff_read", "pip_install")]
    assert len(keep) == 2, "tiff_read / pip_install not found in harness.py"
    return "import subprocess, sys\n" + "\n\n".join(ast.get_source_segment(src, n) for n in keep)


def run_case(path, preimport):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "--quiet", "imagecodecs"], check=False)
    code = harness_functions() + f"""
{"import tifffile" if preimport else ""}
a = tiff_read({path!r})
assert a.shape == (13, 8, 8) and int(a.sum()) == 13 * 64, a.shape
print("read", a.shape, "with tifffile imported before:", {preimport})
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout.strip())
    assert r.returncode == 0, r.stderr[-3000:]


def main(path):
    path = os.path.abspath(path)
    run_case(path, preimport=False)
    run_case(path, preimport=True)
    print("LZW GeoTIFF read without imagecodecs installed at the start: both ways")


if __name__ == "__main__":
    main(sys.argv[1])

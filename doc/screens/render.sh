#!/bin/bash
# Pictures of the Dataset tab for the manual (doc/figures/data_<case>.png), drawn by the app's own UI (web/dist) in headless Chrome with
# the answers the Mac side would give (examples/doc_screens.rs: the real dataset, design and shape analysis). No app window, nothing
# trained or downloaded. Run from the repository: (cd web && npm run build) && doc/screens/render.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
W=$(mktemp -d)
trap 'kill $SERVER 2>/dev/null; rm -rf "$W"' EXIT
cp -R web/dist/. "$W/"
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$W" >/dev/null 2>&1 &
SERVER=$!
sleep 1
# case  height  what to open after the tab is shown  [width, default 1440];  "render.sh designs": the Design tab with the worked examples of sec:build
SPECS=("builtin 900 -" "huggingface 1000 -" "folders 1150 -" "s2 1500 -" "fusion 1900 -" "scene 1650 gee" "segfiles 1450 -" "multilabel 1950 -")
TAB=Dataset PREFIX=data_ SCALE=1
if [ "${1:-}" = designs ]; then
  SPECS=("build_change 560 - 3900")
  TAB=Design PREFIX= SCALE=2
fi
for spec in "${SPECS[@]}"; do
  read -r case height open width <<< "$spec"
  cargo run -q -p nnb-core --example doc_screens -- "$case" > "$W/answers.json"
  sed -e "s/__OPEN__/$open/" -e "s/__TAB__/$TAB/" doc/screens/stub.js > "$W/stub.js"
  # the stub must run before the app's module script
  sed 's#<head>#<head><script src="/stub.js"></script>#' web/dist/index.html > "$W/index.html"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor="$SCALE" --window-size="${width:-1440}","$height" \
    --virtual-time-budget=8000 --screenshot="$PWD/doc/figures/$PREFIX$case.png" "http://127.0.0.1:8765/index.html" >/dev/null 2>&1
  # cut the empty page below the content (needs Pillow: PYTHON=<a python with Pillow>)
  "${PYTHON:-python3}" - "doc/figures/$PREFIX$case.png" "$TAB" <<'PY' 2>/dev/null || true
import sys
from PIL import Image, ImageChops
im = Image.open(sys.argv[1]).convert("RGB")
bg = Image.new("RGB", im.size, im.getpixel((im.width - 2, im.height - 2)))
box = ImageChops.difference(im, bg).getbbox()
if box:
    if sys.argv[2] == "Design":  # the canvas alone: cut to the blocks, found by their coloured headers (the body is below the header)
        sat = im.convert("HSV").split()[1].point(lambda v: 255 if v > 90 else 0)
        b = sat.getbbox()
        im = im.crop((max(0, b[0] - 24), max(0, b[1] - 24), min(im.width, b[2] + 24), min(im.height, b[3] + 100)))
        if im.width > 6 * im.height:  # too wide for a page: cut at the emptiest column near the middle, the right half under the left
            sat = im.convert("HSV").split()[1].point(lambda v: 1 if v > 90 else 0).load()
            cut = min(range(im.width * 2 // 5, im.width * 3 // 5), key=lambda x: sum(sat[x, y] for y in range(im.height)))
            left, right = im.crop((0, 0, cut, im.height)), im.crop((cut, 0, im.width, im.height))
            both = Image.new("RGB", (max(left.width, right.width), 2 * im.height + 40), bg.getpixel((0, 0)))
            both.paste(left, (0, 0))
            both.paste(right, (0, im.height + 40))
            im = both
        im.save(sys.argv[1], optimize=True)
    else:
        im.crop((0, 0, im.width, min(im.height, box[3] + 24))).save(sys.argv[1], optimize=True)
PY
  echo "doc/figures/$PREFIX$case.png"
done

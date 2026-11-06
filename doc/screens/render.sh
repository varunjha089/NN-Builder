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
# case  height  what to open after the Dataset tab is shown
for spec in "builtin 900 -" "huggingface 1000 -" "folders 1150 -" "s2 1500 -" "fusion 1900 -" "scene 1650 gee" "segfiles 1450 -" "multilabel 1950 -"; do
  read -r case height open <<< "$spec"
  cargo run -q -p nnb-core --example doc_screens -- "$case" > "$W/answers.json"
  sed "s/__OPEN__/$open/" doc/screens/stub.js > "$W/stub.js"
  # the stub must run before the app's module script
  sed 's#<head>#<head><script src="/stub.js"></script>#' web/dist/index.html > "$W/index.html"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=1440,"$height" \
    --virtual-time-budget=8000 --screenshot="$PWD/doc/figures/data_$case.png" "http://127.0.0.1:8765/index.html" >/dev/null 2>&1
  # cut the empty page below the content (needs Pillow: PYTHON=<a python with Pillow>)
  "${PYTHON:-python3}" - "doc/figures/data_$case.png" <<'PY' 2>/dev/null || true
import sys
from PIL import Image, ImageChops
im = Image.open(sys.argv[1]).convert("RGB")
bg = Image.new("RGB", im.size, im.getpixel((im.width - 2, im.height - 2)))
box = ImageChops.difference(im, bg).getbbox()
if box:
    im.crop((0, 0, im.width, min(im.height, box[3] + 24))).save(sys.argv[1], optimize=True)
PY
  echo "doc/figures/data_$case.png"
done

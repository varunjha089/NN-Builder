// Where the data comes from. Only metadata is read here (Hugging Face datasets-server, Kaggle file list): the data itself is downloaded
// on the GPU machine when a run starts.
import { useEffect, useState } from "react";
import { api, errText, type AppState, type Dataset, type Head, type LocalScan } from "./api";

export function DatasetTab({ st, ds, setDs, problems, toast }: { st: AppState; ds: Dataset; setDs: (d: Dataset) => void; problems: string[]; toast: (m: string) => void }) {
  const [hf, setHf] = useState<Awaited<ReturnType<typeof api.hfInfo>> | null>(null);
  const [kg, setKg] = useState<Awaited<ReturnType<typeof api.kaggleFiles>> | null>(null);
  const [busy, setBusy] = useState(false);
  const up = (p: Partial<Dataset>) => setDs({ ...ds, ...p });
  const conf = hf?.configs.find((c) => c.name === (ds.hf_config || hf.configs[0].name)) ?? hf?.configs[0];

  const readHf = async () => {
    setBusy(true);
    try {
      const info = await api.hfInfo(ds.hf_repo);
      setHf(info);
      const c = info.configs[0];
      // keep the columns already chosen when the dataset has them; otherwise the first image column, and the class column named
      // "label" or else the one with the most classes (the column list arrives sorted by name: cifar100 has coarse_label before fine_label)
      const img = c.columns.find((x) => x.type === "Image" && x.name === ds.hf_image_col)?.name ?? c.columns.find((x) => x.type === "Image")?.name ?? ds.hf_image_col;
      const labels = c.columns.filter((x) => x.type === (ds.task === "multilabel" ? "ClassLabelList" : "ClassLabel"));
      const lab = labels.find((x) => x.name === ds.hf_label_col) ?? labels.find((x) => x.name === "label")
        ?? [...labels].sort((a, b) => (b.names?.length ?? 0) - (a.names?.length ?? 0))[0];
      const splits = c.splits.map((s) => s.name);
      up({ hf_config: info.configs.length > 1 ? c.name : "", hf_image_col: img, hf_label_col: lab?.name ?? ds.hf_label_col,
        classes: ds.task === "regress" ? ds.classes.filter((t) => c.columns.some((x) => x.name === t)) : lab?.names ?? ds.classes,
        hf_train_split: splits.includes("train") ? "train" : splits[0] ?? "train", hf_test_split: splits.includes("test") ? "test" : splits.includes("validation") ? "validation" : "" });
    } catch (e) { toast(errText(e)); } finally { setBusy(false); }
  };
  const readKaggle = async () => {
    setBusy(true);
    try {
      const info = await api.kaggleFiles(ds.kaggle_slug);
      setKg(info);
      const f = info.folders[0];
      if (f) up({ kaggle_train_dir: f.folder, classes: f.classes });
      else toast("No folder with one subfolder per class was found in the first files; type the folder and the class names.");
    } catch (e) { toast(errText(e)); } finally { setBusy(false); }
  };

  return (
    <div className="page">
      <h2>Dataset</h2>
      <p className="muted">Choose the data here. Only its description is read on this Mac; the images are downloaded on the GPU machine when training starts.</p>
      <div className="row">
        <strong>Task:</strong>
        <label className="radio"><input type="radio" checked={ds.task === "classify"} onChange={() => up({ task: "classify" })} />Classification (one label per image)</label>
        <label className="radio"><input type="radio" checked={ds.task === "multilabel"} onChange={() => up({ task: "multilabel", ...(ds.source === "builtin" ? { source: "local" } : {}) })} />
          Multi-label (any number of classes per image, e.g. BigEarthNet)</label>
        <label className="radio"><input type="radio" checked={ds.task === "segment"} onChange={() => up({ task: "segment", ...(ds.source === "builtin" ? { source: "huggingface" } : {}) })} />
          Segmentation (a class for every pixel: images + masks)</label>
        <label className="radio"><input type="radio" checked={ds.task === "regress"} onChange={() => up({ task: "regress", ...(ds.source === "builtin" ? { source: "local" } : {}) })} />
          Regression (numbers per image, e.g. biomass or canopy height)</label>
      </div>
      <div className="row">
        <strong>Source:</strong>
        {(["builtin", "huggingface", "kaggle", "local"] as const).filter((s) => !(s === "builtin" && ds.task !== "classify")).map((s) => (
          <label key={s} className="radio"><input type="radio" checked={ds.source === s} onChange={() => up({ source: s })} />
            {{ builtin: "Built-in (torchvision)", huggingface: "Hugging Face", kaggle: "Kaggle", local: "My own files (e.g. Sentinel from Google Earth Engine)" }[s]}</label>
        ))}
      </div>

      {ds.source === "builtin" && <div className="form">
        <label>Dataset<select value={ds.builtin} onChange={(e) => up({ builtin: e.target.value })}>
          {st.builtins.map((b) => <option key={b.key} value={b.key}>{b.label} ({b.shape.join("×")}, {b.classes.length} classes)</option>)}</select></label>
        <label className="chk"><input type="checkbox" checked={ds.resize} onChange={(e) => up({ resize: e.target.checked })} />Resize the images (e.g. to 224×224 for AlexNet)</label>
      </div>}

      {ds.source === "huggingface" && <>
        <div className="row"><input className="grow" placeholder="dataset id, e.g. ylecun/mnist" value={ds.hf_repo} onChange={(e) => up({ hf_repo: e.target.value.trim() })} />
          <button onClick={readHf} disabled={busy || !ds.hf_repo}>Read info</button></div>
        {hf && <div className="form">
          {hf.configs.length > 1 && <label>Config<select value={ds.hf_config || hf.configs[0].name} onChange={(e) => up({ hf_config: e.target.value })}>
            {hf.configs.map((c) => <option key={c.name}>{c.name}</option>)}</select></label>}
          <label>Training split<select value={ds.hf_train_split} onChange={(e) => up({ hf_train_split: e.target.value })}>
            {conf?.splits.map((s) => <option key={s.name} value={s.name}>{s.name} ({s.n?.toLocaleString()})</option>)}</select></label>
          <label>Test split<select value={ds.hf_test_split} onChange={(e) => up({ hf_test_split: e.target.value })}>
            <option value="">10% of the training split</option>{conf?.splits.map((s) => <option key={s.name} value={s.name}>{s.name} ({s.n?.toLocaleString()})</option>)}</select></label>
          <label>Image column<select value={ds.hf_image_col} onChange={(e) => up({ hf_image_col: e.target.value })}>
            {conf?.columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}</select></label>
          {ds.task === "segment" && <label>Mask column<select value={ds.hf_mask_col} onChange={(e) => up({ hf_mask_col: e.target.value })}>
            {conf?.columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}</select></label>}
          {ds.task === "regress" && <div className="wide"><span>Target columns (the numbers to predict)</span>
            <div className="bandgrid">{conf?.columns.filter((c) => c.type !== "Image" && c.type !== "ClassLabel" && c.type !== "ClassLabelList").map((c) =>
              <label key={c.name} className="chk"><input type="checkbox" checked={ds.classes.includes(c.name)}
                onChange={() => up({ classes: ds.classes.includes(c.name) ? ds.classes.filter((x) => x !== c.name) : [...ds.classes, c.name] })} />{c.name} ({c.type})</label>)}</div></div>}
          {ds.task !== "segment" && ds.task !== "regress" && <label>Label column<select value={ds.hf_label_col} onChange={(e) => { const c = conf?.columns.find((x) => x.name === e.target.value); up({ hf_label_col: e.target.value, ...(c?.names ? { classes: c.names } : {}) }); }}>
            {conf?.columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}</select></label>}
        </div>}
        {hf && conf && !conf.columns.some((c) => c.type === "Image") && <div className="msg error">This dataset has no image column
          ({conf.columns.slice(0, 6).map((c) => `${c.name}: ${c.type}`).join(", ")}{conf.columns.length > 6 ? ", …" : ""}). NN Builder trains networks on
          images: choose an image dataset, for example uoft-cs/cifar10, ylecun/mnist or food101.</div>}
        {hf && conf && ds.task === "classify" && conf.columns.some((c) => c.type === "Image") && !conf.columns.some((c) => c.type === "ClassLabel") &&
          <div className="msg warn">No class column (ClassLabel) was found: type the class names below, in label order.</div>}
        {hf && conf && ds.task === "multilabel" && !conf.columns.some((c) => c.type === "ClassLabelList") &&
          <div className="msg warn">No column holds a list of class labels: choose the column whose rows list the classes of each image, and type the class names below.</div>}
        {hf && conf?.columns.some((c) => c.type === "Image") && <HfPreview repo={ds.hf_repo} config={ds.hf_config || conf.name} split={ds.hf_train_split}
          labelCol={ds.task === "segment" ? "" : ds.hf_label_col} names={ds.classes} />}
        {!st.secrets.huggingface && <p className="muted small">Gated or private datasets need your Hugging Face token (Connection tab).</p>}
        <BandsBox ds={ds} up={up} />
      </>}

      {ds.source === "kaggle" && <>
        <div className="row"><input className="grow" placeholder="owner/dataset, e.g. faizalkarim/flood-area-segmentation" value={ds.kaggle_slug} onChange={(e) => up({ kaggle_slug: e.target.value.trim() })} />
          <button onClick={readKaggle} disabled={busy || !ds.kaggle_slug || !st.secrets.kaggle || !st.tools_ready}>Read files</button></div>
        {(!st.secrets.kaggle || !st.tools_ready) && <p className="muted small">Reading the file list needs the tools and your Kaggle token (Connection tab).</p>}
        {kg && <p className="muted small">{kg.files.toLocaleString()} files listed{kg.folders.length ? `; folders with one subfolder per class: ${kg.folders.map((f) => f.folder || "(top)").join(", ")}` : ""}.</p>}
        {ds.task === "multilabel" || ds.task === "regress" ? <LabelsCsv ds={ds} up={up} /> : ds.task === "segment" ? <div className="form">
          <label>Images folder<input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="e.g. Image" /></label>
          <label>Masks folder (same file names)<input value={ds.kaggle_mask_dir} onChange={(e) => up({ kaggle_mask_dir: e.target.value.trim() })} placeholder="e.g. Mask" /></label>
          {kg && <p className="muted small wide">Folders seen: {[...new Set(kg.sample.map((f) => f.split("/").slice(0, -1).join("/")))].slice(0, 8).join(", ")}</p>}
        </div> : <div className="form">
          <label>Training folder (one subfolder per class)
            {kg?.folders.length ? <><select value={ds.kaggle_train_dir} onChange={(e) => { const f = kg.folders.find((x) => x.folder === e.target.value); up({ kaggle_train_dir: e.target.value, classes: f?.classes ?? ds.classes }); }}>
              {kg.folders.map((f) => <option key={f.folder} value={f.folder}>{f.folder || "(top level)"} — {f.classes.length} classes</option>)}</select>
              <FolderCounts f={kg.folders.find((x) => x.folder === ds.kaggle_train_dir)} /></>
              : <input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="e.g. data/train" />}</label>
          <label>Test folder (optional)<input value={ds.kaggle_test_dir} onChange={(e) => up({ kaggle_test_dir: e.target.value.trim() })} placeholder="empty = 10% of the training images" /></label>
        </div>}
        <BandsBox ds={ds} up={up} />
      </>}

      {ds.source === "local" && <LocalBox ds={ds} up={up} toast={toast} />}
      {ds.source !== "builtin" && ds.task !== "segment" && <HeadsBox ds={ds} up={up} />}

      {(ds.source !== "builtin" || ds.resize) && <div className="form">
        <label>Image height<input type="number" min={8} max={1024} value={ds.image_size[0]} onChange={(e) => up({ image_size: [Number(e.target.value), ds.image_size[1]] })} /></label>
        <label>Image width<input type="number" min={8} max={1024} value={ds.image_size[1]} onChange={(e) => up({ image_size: [ds.image_size[0], Number(e.target.value)] })} /></label>
        {ds.bands?.length ? <span className="muted small">Channels: {ds.bands.length + (ds.bands2?.length ?? 0)} (the bands chosen above)</span>
          : <label>Channels<select value={ds.channels} onChange={(e) => up({ channels: Number(e.target.value) })}><option value={1}>1 (grey)</option><option value={3}>3 (colour)</option></select></label>}
        {ds.source !== "builtin" && <label className="wide">{ds.task === "regress" ? "Target columns, one per line (numeric columns of the labels table; the Output gives one number per target)" : ds.task === "multilabel" ? "Class names, one per line (from the labels CSV with Read labels, or typed)" : ds.task === "segment" ? "Classes, one per line (mask pixel value 0 = first line; with two classes any non-zero pixel is the second)" : "Class names, one per line, in label order"} ({ds.classes.length})
          <ClassBox classes={ds.classes} onChange={(classes) => up({ classes })} /></label>}
      </div>}
      {problems.map((p) => <div key={p} className="msg warn">{p}</div>)}
    </div>
  );
}

/** class names, one per line; keeps what is typed (empty lines included) and reports the cleaned list */
function ClassBox({ classes, onChange }: { classes: string[]; onChange: (c: string[]) => void }) {
  const [text, setText] = useState(classes.join("\n"));
  useEffect(() => {
    const clean = text.split("\n").map((s) => s.trim()).filter(Boolean);
    if (clean.join("\n") !== classes.join("\n")) setText(classes.join("\n"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classes]);
  return <textarea rows={6} value={text} onChange={(e) => { setText(e.target.value); onChange(e.target.value.split("\n").map((s) => s.trim()).filter(Boolean)); }} />;
}

/** the class balance of the chosen split (counted by Hugging Face's dataset viewer) and, on request, 8 example images */
function HfPreview({ repo, config, split, labelCol, names }: { repo: string; config: string; split: string; labelCol: string; names: string[] }) {
  const [freq, setFreq] = useState<[string, number][] | null>(null);
  const [pics, setPics] = useState<{ src: string; label: unknown }[] | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setFreq(null); setPics(null);
    if (labelCol) api.hfPreview(repo, config, split, labelCol, false).then((r) => setFreq(r.frequencies)).catch(() => setFreq([]));
  }, [repo, config, split, labelCol]);
  const counts = freq?.map((f) => f[1]) ?? [];
  const most = Math.max(1, ...counts), least = counts.length ? Math.min(...counts) : 0;
  const label = (v: unknown) => (typeof v === "number" && names[v] ? names[v] : String(v ?? ""));
  return <div className="aug" style={{ margin: "8px 0" }}>
    {freq && freq.length > 0 && <>
      <strong>Class balance</strong> <span className="muted small">({split}, {counts.reduce((a, b) => a + b, 0).toLocaleString()} images, {freq.length} classes:
        {" "}{least === most ? "perfectly balanced" : `the largest class has ${(most / Math.max(1, least)).toFixed(1)}× the images of the smallest`})</span>
      {most / Math.max(1, least) > 3 && <p className="st-warn small">Strongly unbalanced: accuracy can look good while small classes are ignored; check the per-class
        F1 in Results.</p>}
      <details><summary className="small">Images per class</summary><div className="small" style={{ columns: 3 }}>
        {freq.map(([k, n]) => <div key={k}>{k}: {n.toLocaleString()} <span style={{ display: "inline-block", height: 6, width: `${(60 * n) / most}px`, background: "var(--accent)" }} /></div>)}
      </div></details>
    </>}
    {freq && freq.length === 0 && labelCol && <span className="muted small">No class statistics from Hugging Face for this split. </span>}
    <p className="row small"><button className="small" disabled={busy} onClick={() => { setBusy(true); api.hfPreview(repo, config, split, labelCol, true)
      .then((r) => setPics(r.samples)).catch(() => setPics([])).finally(() => setBusy(false)); }}>{busy ? "Loading…" : "Show 8 example images"}</button>
      <span className="muted">only these few images are fetched; the dataset itself is downloaded on the GPU machine</span></p>
    {pics && <div className="thumbs">{pics.map((p, i) => <div key={i} className="thumb"><img src={p.src} style={{ width: 64, height: 64, objectFit: "cover", imageRendering: "pixelated" }} />
      <div>{label(p.label).slice(0, 12)}</div></div>)}{pics.length === 0 && <span className="muted small">No example images available.</span>}</div>}
  </div>;
}

// ---------------------------------------------------------------- multispectral / radar images (Sentinel-2, Sentinel-1)
const S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"];
const S2_INFO: Record<string, string> = { B01: "aerosol 443 nm, 60 m", B02: "blue 490 nm, 10 m", B03: "green 560 nm, 10 m", B04: "red 665 nm, 10 m",
  B05: "red edge 705 nm, 20 m", B06: "red edge 740 nm, 20 m", B07: "red edge 783 nm, 20 m", B08: "near infrared 842 nm, 10 m",
  B8A: "narrow NIR 865 nm, 20 m", B09: "water vapour 945 nm, 60 m", B10: "cirrus 1375 nm, 60 m", B11: "SWIR 1610 nm, 20 m", B12: "SWIR 2190 nm, 20 m" };
/** Earth Engine's names (COPERNICUS/S2_SR_HARMONIZED, no B10) and band order; B4 and B04 are the same band everywhere in the app */
const S2_GEE = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"];
const bandKey = (b: string) => { const u = b.trim().toUpperCase(); return /^B\d$/.test(u) ? `B0${u[1]}` : u; };
const info = (b: string) => S2_INFO[bandKey(b)];
const S2_PRESETS: [string, string[]][] = [["all 13", S2], ["10 m + 20 m (10)", ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]],
  ["true colour (B04 B03 B02)", ["B04", "B03", "B02"]], ["colour + near infrared", ["B04", "B03", "B02", "B08"]]];

/** the image bands: ordinary pictures, or GeoTIFF / .npy files with Sentinel-2, Sentinel-1 or other bands */
function BandsBox({ ds, up }: { ds: Dataset; up: (p: Partial<Dataset>) => void }) {
  const sensor = ds.bands?.length ? ds.sensor || "other" : "";
  const order = ds.band_order ?? [], chosen = ds.bands ?? [];
  const set = (s: string) => {
    if (s === "") up({ bands: [], band_order: [], sensor: "", s1_db: false, bands2: [], band_order2: [], sensor2: "", fuse_dir: "" });
    else if (s === "s2") up({ sensor: "s2", band_order: S2, bands: S2, s1_db: false });
    else if (s === "s1") up({ sensor: "s1", band_order: ["VV", "VH"], bands: ["VV", "VH"], s1_db: true });
    else up({ sensor: "other", band_order: ["b1", "b2", "b3", "b4"], bands: ["b1", "b2", "b3", "b4"], s1_db: false });
  };
  // keep the bands in the files' order
  const toggle = (b: string) => up({ bands: chosen.includes(b) ? chosen.filter((x) => x !== b) : order.filter((x) => x === b || chosen.includes(x)) });
  return <div className="wide aug">
    <strong>Image bands</strong>
    <span className="muted small"> (satellite images: GeoTIFF .tif or .npy files with any number of bands, as in EuroSAT-MS or an Earth Engine export;
      empty pixels (NaN) are filled with the band's mean)</span>
    <p className="row">
      <label className="radio"><input type="radio" checked={sensor === ""} onChange={() => set("")} />ordinary pictures (grey / colour)</label>
      <label className="radio"><input type="radio" checked={sensor === "s2"} onChange={() => set("s2")} />Sentinel-2 (13 bands; Earth Engine: 12)</label>
      <label className="radio"><input type="radio" checked={sensor === "s1"} onChange={() => set("s1")} />Sentinel-1 radar (VV, VH)</label>
      <label className="radio"><input type="radio" checked={sensor === "other"} onChange={() => set("other")} />other bands</label>
    </p>
    {sensor !== "" && <>
      <label className="wide" style={{ display: "block" }}>Bands in the files, in their order (space-separated; check the dataset's description)
        <input style={{ display: "block", width: "100%" }} value={order.join(" ")} onChange={(e) => { const o = e.target.value.split(/\s+/).filter((x) => x); up({ band_order: o, bands: chosen.filter((b) => o.includes(b)) }); }} /></label>
      {sensor === "s2" && <p className="row small"><span className="muted">Files:</span>
        <button className="small" onClick={() => up({ band_order: S2, bands: S2 })}>EuroSAT-MS order (13)</button>
        <button className="small" onClick={() => up({ band_order: S2_GEE, bands: S2_GEE })}>Earth Engine SR (12, B1 … B12)</button>
        <span className="muted">Use:</span>
        {S2_PRESETS.map(([name, bs]) => <button key={name} className="small" onClick={() => up({ bands: order.filter((b) => bs.includes(bandKey(b))) })}>{name}</button>)}</p>}
      <div className="bandgrid">{order.map((b) => <label key={b} className="chk" title={info(b) ?? ""}>
        <input type="checkbox" checked={chosen.includes(b)} onChange={() => toggle(b)} />{b}{sensor === "s2" && info(b) ? <span className="muted small"> {info(b)}</span> : null}</label>)}</div>
      {sensor === "s1" && <label className="chk"><input type="checkbox" checked={!!ds.s1_db} onChange={(e) => up({ s1_db: e.target.checked })} />
        the files hold linear backscatter: convert to decibels (10·log10) first (Earth Engine's S1_GRD is already in decibels: such files are
        recognised by their negative values and left alone)</label>}
      <SecondGroup ds={ds} up={up} />
      <p className="muted small">The network receives {chosen.length + (ds.bands2?.length ?? 0)} channels ({[...chosen, ...(ds.bands2 ?? [])].join(" ") || "none"}). Every band is normalised by its own mean and
        spread, measured on the training images on the GPU machine. Flips and quarter turns are used for augmentation; the result pictures
        show {sensor === "s2" ? "true colour (B04 B03 B02)" : sensor === "s1" ? "VV / VH / VV−VH" : "the first three bands"}.</p>
    </>}
  </div>;
}

// ---------------------------------------------------------------- the owner's own files (e.g. exported from Google Earth Engine)
const WORLDCOVER = ["tree cover", "shrubland", "grassland", "cropland", "built-up", "bare / sparse vegetation", "snow and ice", "permanent water",
  "herbaceous wetland", "mangroves", "moss and lichen"];
const size = (n: number) => n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} kB`;

/** a folder of the owner's files: on this Mac (copied to the training machine, or uploaded to Kaggle as a private dataset, when a run
 * starts) or already on the training machine; class folders, images + masks, or large images with a label image (Earth Engine) */
function LocalBox({ ds, up, toast }: { ds: Dataset; up: (p: Partial<Dataset>) => void; toast: (m: string) => void }) {
  const [scan, setScan] = useState<LocalScan | null>(null);
  const [busy, setBusy] = useState(false);
  const read = async (dir = ds.local_dir ?? "") => {
    setBusy(true);
    try {
      const r = await api.localScan(dir);
      setScan(r);
      const p: Partial<Dataset> = {};
      const f = r.folders[0];
      if (ds.task !== "segment" && f && !ds.kaggle_train_dir) Object.assign(p, { kaggle_train_dir: f.folder, classes: f.classes });
      // band names written into the GeoTIFF (GDAL / Earth Engine): the files' band order, without typing it
      const names = r.tif?.names ?? [];
      if (names.length && names.join(" ") !== (ds.band_order ?? []).join(" ")) {
        const s1 = names.some((n) => /^V[VH]$/i.test(n)), s2 = names.some((n) => /^B\d/i.test(n));
        Object.assign(p, { band_order: names, bands: names.filter((n) => !/^(QA|SCL|MSK|angle)/i.test(n)), sensor: s1 ? "s1" : s2 ? "s2" : "other" });
        toast(`Band names read from ${r.tif!.file}: ${names.join(" ")}`);
      } else if (r.tif && !names.length && r.tif.bands !== (ds.band_order ?? []).length && (ds.band_order ?? []).length) {
        toast(`${r.tif.file} has ${r.tif.bands} bands, but the band order below lists ${(ds.band_order ?? []).length}: check it.`);
      }
      up(p);
    } catch (e) { toast(errText(e)); setScan(null); } finally { setBusy(false); }
  };
  const choose = async () => {
    const d = await api.pickFolder("The folder with your dataset files").catch(() => null);
    if (d) { up({ local_dir: d }); read(d); }
  };
  const scene = ds.layout === "scene";
  const [cloudMode, setCloud] = useState(!!ds.cloud_url);
  const cloud = cloudMode || !!ds.cloud_url;
  return <>
    {cloud ? <div className="row">
      <input className="grow" placeholder="gs://bucket/folder, https://drive.google.com/drive/folders/…, or rclone:remote:path" value={ds.cloud_url ?? ""}
        onChange={(e) => up({ cloud_url: e.target.value.trim() })} />
    </div> : <div className="row">
      <input className="grow" placeholder={ds.local_on_machine ? "the folder's path on the training machine, e.g. ~/data/sentinel2" : "the folder on this Mac"}
        value={ds.local_dir ?? ""} onChange={(e) => up({ local_dir: e.target.value })} />
      {!ds.local_on_machine && <button onClick={choose} disabled={busy}>Choose…</button>}
      {!ds.local_on_machine && <button onClick={() => read()} disabled={busy || !ds.local_dir}>Read folder</button>}
    </div>}
    <p className="row small">
      <label className="radio"><input type="radio" checked={!ds.local_on_machine && !cloud} onChange={() => { setCloud(false); up({ local_on_machine: false, cloud_url: "" }); }} />on this Mac</label>
      <label className="radio"><input type="radio" checked={!!ds.local_on_machine && !cloud} onChange={() => { setCloud(false); up({ local_on_machine: true, cloud_url: "" }); }} />already on the training machine (SSH / AWS)</label>
      <label className="radio"><input type="radio" checked={cloud} onChange={() => { setCloud(true); up({ local_on_machine: false }); }} />in the cloud (Cloud Storage, Google Drive)</label>
    </p>
    <p className="muted small">{cloud
      ? "The training machine (or Kaggle notebook) fetches the files itself, so nothing passes through this Mac: gs://bucket/folder (Earth Engine's "
        + "Cloud Storage export; a private bucket needs the Cloud Storage key, Connection tab, and works on SSH / AWS machines), a Google Drive "
        + "folder or file shared as 'anyone with the link' (at most 50 files per folder; a zip is unpacked), or rclone:remote:path (rclone set "
        + "up on the machine). Fetched once into the machine's data folder; later runs fetch only what is new."
      : ds.local_on_machine
      ? "train.py reads the folder where it is; nothing is copied. Kaggle cannot reach it."
      : "When a run starts, the folder is copied to the training machine (rsync: only the first time and after changes) or, for Kaggle, uploaded "
        + "once as a private Kaggle dataset under your account (again only when it changed). Nothing in the folder is changed."}</p>
    {scan && <p className="muted small">{scan.files.toLocaleString()}{scan.complete ? "" : "+"} files, {size(scan.bytes)}; {scan.tifs.toLocaleString()} GeoTIFF,
      {" "}{scan.npys.toLocaleString()} .npy{scan.tif ? `; ${scan.tif.file}: ${scan.tif.width}×${scan.tif.height} pixels, ${scan.tif.bands} bands`
        + `${scan.tif.float ? " (float)" : ` (${scan.tif.bits}-bit)`}${scan.tif.names.length ? `: ${scan.tif.names.join(" ")}` : ", no band names in the file"}` : ""}.</p>}
    {ds.task === "segment" && <p className="row small">
      <label className="radio"><input type="radio" checked={!scene} onChange={() => up({ layout: "" })} />images + masks (same file names)</label>
      <label className="radio"><input type="radio" checked={scene} onChange={() => up({ layout: "scene" })} />large images + a label image on the same grid (Earth Engine export), cut into tiles</label>
    </p>}
    {ds.task === "multilabel" || ds.task === "regress" ? <LabelsCsv ds={ds} up={up} scan={scan} toast={toast} /> : ds.task !== "segment" ? <div className="form">
      <label>Training folder (one subfolder per class)
        {scan?.folders.length ? <><select value={ds.kaggle_train_dir} onChange={(e) => { const f = scan.folders.find((x) => x.folder === e.target.value); up({ kaggle_train_dir: e.target.value, classes: f?.classes ?? ds.classes }); }}>
          {scan.folders.map((f) => <option key={f.folder} value={f.folder}>{f.folder || "(the folder itself)"} — {f.classes.length} classes</option>)}</select>
          <FolderCounts f={scan.folders.find((x) => x.folder === ds.kaggle_train_dir)} /></>
          : <input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="empty = the folder itself" />}</label>
      <label>Test folder (optional)<input value={ds.kaggle_test_dir} onChange={(e) => up({ kaggle_test_dir: e.target.value.trim() })} placeholder="empty = 10% of the training images" /></label>
    </div> : scene ? <div className="form">
      <label>Image file or folder<input value={ds.scene_image ?? ""} onChange={(e) => up({ scene_image: e.target.value.trim() })} placeholder="e.g. nnb_image" /></label>
      <label>Label file or folder (class numbers 0, 1, 2 …)<input value={ds.scene_label ?? ""} onChange={(e) => up({ scene_label: e.target.value.trim() })} placeholder="e.g. nnb_label" /></label>
      <label className="wide">Test tiles<select value={ds.scene_split ?? ""} onChange={(e) => up({ scene_split: e.target.value as Dataset["scene_split"] })}>
        <option value="">10% of the tiles at random (quick, but optimistic: they touch training tiles)</option>
        <option value="strip">the eastern tenth of every image, one column of tiles left out between (honest)</option>
        <option value="files">a separately exported test region (most honest)</option></select></label>
      {ds.scene_split === "files" && <>
        <label>Test image file or folder<input value={ds.scene_test_image ?? ""} onChange={(e) => up({ scene_test_image: e.target.value.trim() })} placeholder="e.g. nnb_test_image" /></label>
        <label>Test label file or folder<input value={ds.scene_test_label ?? ""} onChange={(e) => up({ scene_test_label: e.target.value.trim() })} placeholder="e.g. nnb_test_label" /></label>
      </>}
      <p className="muted small wide">An export split into several files (name-0000000000-0000000000.tif) is fine: image and label files are paired by name. The
        images are cut into tiles of the image size below; tiles that are mostly empty are left out, and label pixels that are not a class
        (no data, 255) are not scored.{scan ? ` Seen: ${[...scan.dirs, ...scan.top].slice(0, 10).join(", ")}` : ""}</p>
    </div> : <div className="form">
      <label>Images folder<input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="e.g. images" /></label>
      <label>Masks folder (same file names)<input value={ds.kaggle_mask_dir} onChange={(e) => up({ kaggle_mask_dir: e.target.value.trim() })} placeholder="e.g. masks" /></label>
      {scan && <p className="muted small wide">Folders seen: {scan.dirs.slice(0, 10).join(", ")}</p>}
    </div>}
    <BandsBox ds={ds} up={up} />
    <GeeBox ds={ds} up={up} toast={toast} />
  </>;
}

/** a Google Earth Engine script that exports exactly what the Dataset tab expects (the chosen bands in this order, on one grid) */
function GeeBox({ ds, up, toast }: { ds: Dataset; up: (p: Partial<Dataset>) => void; toast: (m: string) => void }) {
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState("2024-12-31");
  const [cloud, setCloud] = useState(20);
  const [crs, setCrs] = useState("EPSG:3857");
  const [dest, setDest] = useState<"drive" | "gcs">("drive");
  const [bucket, setBucket] = useState("your-bucket");
  const gcs = dest === "gcs";
  const s1 = ds.sensor === "s1";
  const seg = ds.task === "segment";
  // Earth Engine's own names for the chosen bands (B04 -> B4)
  const gee = (s1 ? (ds.bands?.length ? ds.bands : ["VV", "VH"]) : (ds.bands?.length ? ds.bands : ["B2", "B3", "B4", "B8"]))
    .map((b) => { const k = bandKey(b); return /^B0\d$/.test(k) ? `B${k[2]}` : k; });
  const toa = gee.includes("B10");
  const scale = 10;
  const list = gee.map((b) => `'${b}'`).join(", ");
  const [h, w] = ds.image_size;
  const collection = s1 ? `// Sentinel-1 GRD (already in decibels: leave "convert to decibels" as it is, the app sees the negative values)
var image = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(geometry).filterDate(start, end)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select([${list}]).median().toFloat();`
    : toa ? `// Sentinel-2 top of atmosphere (B10 is only in this collection; EuroSAT-MS is this level too); clouds masked with QA60
function clear(img) { var qa = img.select('QA60'); return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0))); }
var image = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterBounds(geometry).filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', ${cloud}))
  .map(clear).select([${list}]).median().toFloat();`
    : `// Sentinel-2 surface reflectance; clouds, cirrus and cloud shadows masked with the scene classification (SCL 3, 8, 9, 10)
function clear(img) { var scl = img.select('SCL'); return img.updateMask(scl.neq(3).and(scl.lt(8).or(scl.gt(10)))); }
var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geometry).filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', ${cloud}))
  .map(clear).select([${list}]).median().toFloat();`;
  const pyCollection = s1 ? `image = (ee.ImageCollection("COPERNICUS/S1_GRD")   # already in decibels
         .filterBounds(geometry).filterDate(start, end)
         .filter(ee.Filter.eq("instrumentMode", "IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
         .select([${list}]).median().toFloat())`
    : `def clear(img):
    ${toa ? `qa = img.select("QA60")   # clouds and cirrus
    return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)))`
      : `scl = img.select("SCL")    # cloud shadow 3, clouds 8 and 9, cirrus 10
    return img.updateMask(scl.neq(3).And(scl.lt(8).Or(scl.gt(10))))`}


image = (ee.ImageCollection("${toa ? "COPERNICUS/S2_HARMONIZED" : "COPERNICUS/S2_SR_HARMONIZED"}")
         .filterBounds(geometry).filterDate(start, end)
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", ${cloud}))
         .map(clear).select([${list}]).median().toFloat())`;
  const script = seg ? `// NN Builder: ${s1 ? "Sentinel-1" : "Sentinel-2"} image + label image for segmentation. In code.earthengine.google.com: draw your
// region with the drawing tools (it becomes "geometry"), press Run, then in Tasks press Run on both exports. ${gcs
    ? `They land in gs://${bucket}/nnb_export/;\n// in the Dataset tab choose "in the cloud" with that address (the training machine fetches it; nothing passes through the Mac).`
    : `Download the two Drive\n// folders nnb_image and nnb_label into one folder, choose it in the Dataset tab (image folder nnb_image, label folder nnb_label).`}
var start = '${start}', end = '${end}';
var crs = '${crs}', scale = ${scale};  // both exports use the same grid; a UTM zone (e.g. EPSG:32643) keeps pixels square
${collection}
// the label: ESA WorldCover 2021 as class numbers 0-10 (replace with your own labelled image of class numbers 0, 1, 2 ...)
var label = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
  .remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).rename('label');
// pixels without image data get no class (255: left out of the loss and the scores)
label = label.updateMask(image.mask().reduce(ee.Reducer.min())).unmask(255).toByte();
var common = {region: geometry, scale: scale, crs: crs, maxPixels: 1e10, fileDimensions: ${Math.max(256, Math.ceil(2048 / 256) * 256)}};
${gcs ? `Export.image.toCloudStorage(Object.assign({image: image.unmask(0), description: 'nnb_image', bucket: '${bucket}', fileNamePrefix: 'nnb_export/nnb_image/scene'}, common));
Export.image.toCloudStorage(Object.assign({image: label, description: 'nnb_label', bucket: '${bucket}', fileNamePrefix: 'nnb_export/nnb_label/scene'}, common));`
  : `Export.image.toDrive(Object.assign({image: image.unmask(0), description: 'nnb_image', folder: 'nnb_image', fileNamePrefix: 'scene'}, common));
Export.image.toDrive(Object.assign({image: label, description: 'nnb_label', folder: 'nnb_label', fileNamePrefix: 'scene'}, common));`}`
    : `# NN Builder: ${s1 ? "Sentinel-1" : "Sentinel-2"} patches of ${w}×${h} pixels, one Drive folder per class, for classification.
# Python with the Earth Engine API (pip install earthengine-api): it starts every export itself (the Code Editor would need a click per task).
# points: your labelled points (an Earth Engine asset) with a property "class" holding the class name.
${gcs ? `# The patches land in gs://${bucket}/nnb_export/<class>/; in the Dataset tab choose "in the cloud" with that address.`
  : "# Download the class folders from Drive into one folder and choose it in the Dataset tab."}
import ee
ee.Authenticate()
ee.Initialize(project="YOUR-CLOUD-PROJECT")
points = ee.FeatureCollection("projects/YOUR-CLOUD-PROJECT/assets/points")
geometry = points.geometry().bounds()
start, end, crs, scale = "${start}", "${end}", "${crs}", ${scale}
${pyCollection}
for i, f in enumerate(points.getInfo()["features"]):
    c = f["properties"]["class"]
    region = ee.Geometry.Point(f["geometry"]["coordinates"]).buffer(${Math.max(h, w) / 2} * scale).bounds()
    ${gcs ? `ee.batch.Export.image.toCloudStorage(image=image.unmask(0), description=f"nnb_{i}", bucket="${bucket}",
                                         fileNamePrefix=f"nnb_export/{c}/{c}_{i}", region=region, scale=scale, crs=crs).start()`
      : `ee.batch.Export.image.toDrive(image=image.unmask(0), description=f"nnb_{i}", folder=str(c), fileNamePrefix=f"{c}_{i}",
                                  region=region, scale=scale, crs=crs).start()`}   # about ${w}×${h} pixels; the app resizes to exactly that
print("started", i + 1, "exports: see the Tasks tab or ee.batch.Task.list()")`;
  return <details className="wide aug" style={{ marginTop: 8 }}>
    <summary><strong>Get Sentinel-1/2 data from Google Earth Engine</strong> <span className="muted small">(a script that exports exactly what this tab expects)</span></summary>
    <p className="muted small">The bands are the ones chosen above ({gee.join(" ")}), in that order; the files carry their names, so Read folder fills the
      band order by itself. {seg ? "For segmentation: an image and a label image on one grid (Code Editor script)." : "For classification: one patch per labelled point (Python script)."}</p>
    <div className="form">
      <label>From<input value={start} onChange={(e) => setStart(e.target.value)} /></label>
      <label>To<input value={end} onChange={(e) => setEnd(e.target.value)} /></label>
      {!s1 && <label>Most clouds per scene (%)<input type="number" min={0} max={100} value={cloud} onChange={(e) => setCloud(Number(e.target.value))} /></label>}
      <label>Projection (crs)<input value={crs} onChange={(e) => setCrs(e.target.value.trim())} /></label>
      <label>Export to<select value={dest} onChange={(e) => setDest(e.target.value as "drive" | "gcs")}>
        <option value="drive">Google Drive (download it, then choose the folder)</option>
        <option value="gcs">Cloud Storage (the training machine fetches it)</option></select></label>
      {gcs && <label>Bucket<input value={bucket} onChange={(e) => setBucket(e.target.value.trim())} /></label>}
    </div>
    <textarea readOnly rows={14} className="mono small" style={{ width: "100%" }} value={script} />
    <p className="row small">
      <button className="small" onClick={() => navigator.clipboard.writeText(script).then(() => toast("Script copied")).catch((e) => toast(errText(e)))}>Copy script</button>
      <button className="small" onClick={() => up({ band_order: gee, bands: gee, sensor: s1 ? "s1" : "s2", ...(s1 ? { s1_db: false } : {}),
        ...(seg ? { layout: "scene", scene_image: "nnb_image", scene_label: "nnb_label", classes: WORLDCOVER } : {}),
        ...(gcs ? { cloud_url: `gs://${bucket}/nnb_export`, local_on_machine: false } : {}) })}>
        Set this tab to match the export</button>
      <span className="muted">not run by the app: check it in Earth Engine (it needs your Earth Engine account)</span></p>
  </details>;
}

const BIGEARTHNET19 = ["Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures", "Complex cultivation patterns",
  "Land principally occupied by agriculture, with significant areas of natural vegetation", "Agro-forestry areas", "Broad-leaved forest",
  "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas", "Moors, heathland and sclerophyllous vegetation",
  "Transitional woodland, shrub", "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"];

/** multi-label: the CSV with each image's classes (a file column and a labels column, or one 0/1 column per class) */
function LabelsCsv({ ds, up, scan, toast }: { ds: Dataset; up: (p: Partial<Dataset>) => void; scan?: LocalScan | null; toast?: (m: string) => void }) {
  const [info, setInfo] = useState<string>("");
  const [numeric, setNumeric] = useState<{ column: string; min: number; max: number; mean: number }[] | null>(null);
  const regress = ds.task === "regress";
  const read = async () => {
    try {
      const r = await api.localCsv(ds.local_dir ?? "", ds.labels_csv ?? "");
      if (regress) {
        setNumeric(r.numeric);
        const keep = ds.classes.filter((c) => r.numeric.some((n) => n.column === c));
        up({ classes: keep.length ? keep : r.numeric.slice(0, 1).map((n) => n.column) });
        setInfo(`${r.rows.toLocaleString()} rows; files in “${r.file_column}”; ${r.numeric.length} numeric column${r.numeric.length === 1 ? "" : "s"}`);
        return;
      }
      up({ classes: r.classes });
      setInfo(`${r.rows.toLocaleString()} rows; files in “${r.file_column}”, ${r.label_column ? `classes in “${r.label_column}”` : "one 0/1 column per class"}: `
        + r.classes.map((c, i) => `${c} ${r.counts[i]}`).join(", "));
    } catch (e) { toast?.(errText(e)); }
  };
  return <div className="form">
    <label>Labels file (CSV, TSV or Parquet)
      {scan?.csvs?.length ? <select value={ds.labels_csv ?? ""} onChange={(e) => up({ labels_csv: e.target.value })}>
        <option value="">choose…</option>{scan.csvs.map((c) => <option key={c} value={c}>{c}</option>)}</select>
        : <input value={ds.labels_csv ?? ""} onChange={(e) => up({ labels_csv: e.target.value.trim() })} placeholder="e.g. labels.csv" />}</label>
    {!regress && <p className="row small wide"><button className="small" onClick={() => up({ classes: BIGEARTHNET19, labels_csv: ds.labels_csv || "metadata.parquet" })}>
      BigEarthNet (19 classes, metadata.parquet)</button>
      <span className="muted">BigEarthNet v2: the patch folders (one GeoTIFF per band) are found below the images folder; its split column is used; with
        Sentinel-1 as the second band group, the s1_name column pairs the patches.</span></p>}
    <label>Images folder (optional)<input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="empty = next to the CSV" /></label>
    {ds.source === "local" && !ds.local_on_machine && <p className="row small wide"><button className="small" disabled={!ds.labels_csv || !ds.local_dir} onClick={read}>Read labels</button>
      <span className="muted">{info}</span></p>}
    {regress && numeric && <div className="wide"><span>Target columns (the numbers to predict)</span>
      <div className="bandgrid">{numeric.map((n) => <label key={n.column} className="chk" title={`mean ${n.mean.toPrecision(4)}`}>
        <input type="checkbox" checked={ds.classes.includes(n.column)}
          onChange={() => up({ classes: ds.classes.includes(n.column) ? ds.classes.filter((x) => x !== n.column) : [...ds.classes, n.column] })} />
        {n.column} <span className="muted small">{n.min.toPrecision(3)} … {n.max.toPrecision(3)}</span></label>)}</div></div>}
    {regress ? <p className="muted small wide">One row per image: a file column (file, image, name, path or id; with or without the extension, relative to
      the CSV's folder or the images folder) and one numeric column per target (e.g. biomass in t/ha, canopy height in m). Rows with an
      empty or non-numeric target are left out. The network's Output gives one number per target; it is trained on the standardised targets
      (MSE, L1 or Huber loss, Train tab) and scored in the targets' units: R², MAE and RMSE.</p>
    : <p className="muted small wide">One row per image: a file column (file, image, name, path or id; with or without the extension, relative to the CSV's folder or the
      images folder) and a labels column with the classes separated by ; , | or spaces, or one 0/1 column per class. The network's Output
      gives one score per class; each is a yes/no (sigmoid ≥ 0.5), trained with binary cross-entropy. Scores: micro and macro F1, mean
      average precision (mAP) and exact match.</p>}
  </div>;
}

/** fusion: a second band group (e.g. Sentinel-1 next to Sentinel-2) from files with the same names in another folder, stacked after the
 *  first group's bands; Input blocks can take one group each (Design: an Input's "Bands") */
/** several outputs (multi-task): extra targets next to the main task, each from columns of the labels table (Hugging Face: of the rows),
 *  each given to the Output block with the same number */
function HeadsBox({ ds, up }: { ds: Dataset; up: (p: Partial<Dataset>) => void }) {
  const heads = ds.heads ?? [];
  const set = (i: number, p: Partial<Head>) => up({ heads: heads.map((h, j) => (j === i ? { ...h, ...p } : h)) });
  const list = (s: string) => s.split(/[,\n]/).map((x) => x.trim()).filter((x) => x);
  const table = ds.source !== "huggingface";
  return <div className="aug" style={{ margin: "8px 0" }}>
    <strong>Extra outputs (multi-task)</strong> <span className="muted small">(more things to predict from the same images, e.g. the class and the biomass)</span>
    {heads.length > 0 && table && ds.task === "classify" && <div className="form">
      <label>Labels table (CSV, TSV or Parquet)<input value={ds.labels_csv ?? ""} onChange={(e) => up({ labels_csv: e.target.value.trim() })} placeholder="e.g. labels.csv" /></label>
      <label>Main task's class column<input value={ds.label_column ?? ""} onChange={(e) => up({ label_column: e.target.value })} placeholder="empty = the image's folder" /></label>
    </div>}
    {heads.map((h, i) => <div key={i} className="form" style={{ borderTop: "1px solid var(--line)", paddingTop: 6 }}>
      <label>Output {i + 1}: name<input value={h.name} onChange={(e) => set(i, { name: e.target.value })} placeholder="e.g. biomass" /></label>
      <label>Predicts<select value={h.task} onChange={(e) => set(i, { task: e.target.value as Head["task"], loss: "" })}>
        <option value="classify">one class</option><option value="multilabel">several classes</option><option value="regress">numbers</option></select></label>
      <label>{h.task === "regress" ? "Number column(s), comma-separated" : h.task === "multilabel" ? "Labels column (empty: one 0/1 column per class)" : "Class column"}
        <input value={h.columns.join(", ")} onChange={(e) => set(i, { columns: list(e.target.value) })} placeholder={h.task === "regress" ? "e.g. biomass, height" : "e.g. land_use"} /></label>
      {h.task !== "regress" && <label>Classes, comma-separated (in label order)<input value={h.classes.join(", ")} onChange={(e) => set(i, { classes: list(e.target.value) })} placeholder="e.g. dry, wet" /></label>}
      <label>Loss<select value={h.loss ?? ""} onChange={(e) => set(i, { loss: e.target.value })}>
        {h.task === "regress" ? <><option value="">MSE</option><option value="l1">L1</option><option value="huber">Huber</option></>
          : h.task === "classify" ? <><option value="">cross-entropy</option><option value="focal">focal</option></> : <option value="">binary cross-entropy</option>}</select></label>
      <label>Weight in the total loss<input type="number" min={0} max={100} step={0.1} value={h.weight} onChange={(e) => set(i, { weight: Number(e.target.value) })} /></label>
      <button className="small" onClick={() => up({ heads: heads.filter((_, j) => j !== i) })}>Remove</button>
    </div>)}
    <p className="row small"><button className="small" disabled={heads.length >= 8}
      onClick={() => up({ heads: [...heads, { name: "", task: "regress", columns: [], classes: [], loss: "", weight: 1 }] })}>Add an extra output</button>
      {heads.length > 0 && <span className="muted">In the Design tab, give each one an Output block (“Output for”: {heads.map((h, i) => `${i + 1} ${h.name || "…"}`).join(", ")}).</span>}</p>
    {heads.length > 0 && <p className="muted small">The total loss is the main task's plus each extra output's times its weight. Numbers are learnt
      standardised and scored in their own units (R², MAE, RMSE); classes by accuracy, several classes by micro F1 and mAP. {table
        ? "Each row of the labels table names an image (a file column); rows without a usable value for an output are left out."
        : "The values come from these columns of the Hugging Face rows."} The main task's score still decides early stopping and the best epoch.</p>}
  </div>;
}

function SecondGroup({ ds, up }: { ds: Dataset; up: (p: Partial<Dataset>) => void }) {
  const on = (ds.bands2?.length ?? 0) > 0 || !!ds.sensor2;
  const order = ds.band_order2 ?? [], chosen = ds.bands2 ?? [];
  const set = (s: string) => {
    if (s === "") up({ bands2: [], band_order2: [], sensor2: "", s1_db2: false, fuse_dir: "" });
    else if (s === "s1") up({ sensor2: "s1", band_order2: ["VV", "VH"], bands2: ["VV", "VH"], s1_db2: false });
    else if (s === "s2") up({ sensor2: "s2", band_order2: S2_GEE, bands2: S2_GEE.filter((b) => !(ds.bands ?? []).some((x) => bandKey(x) === bandKey(b))) });
    else up({ sensor2: "other", band_order2: ["c1", "c2"], bands2: ["c1", "c2"], s1_db2: false });
  };
  const toggle = (b: string) => up({ bands2: chosen.includes(b) ? chosen.filter((x) => x !== b) : order.filter((x) => x === b || chosen.includes(x)) });
  return <div className="aug" style={{ margin: "8px 0" }}>
    <strong>Second band group (fusion)</strong> <span className="muted small">(another sensor for the same places, e.g. Sentinel-1 next to Sentinel-2)</span>
    <p className="row">
      <label className="radio"><input type="radio" checked={!on} onChange={() => set("")} />none</label>
      <label className="radio"><input type="radio" checked={ds.sensor2 === "s1"} onChange={() => set("s1")} />Sentinel-1 (VV, VH)</label>
      <label className="radio"><input type="radio" checked={ds.sensor2 === "s2"} onChange={() => set("s2")} />Sentinel-2</label>
      <label className="radio"><input type="radio" checked={ds.sensor2 === "other"} onChange={() => set("other")} />other</label>
    </p>
    {on && <>
      <div>
        <label style={{ display: "block", margin: "4px 0" }}>Folder with these files (the same file names as the first group's, any extension)
          <input style={{ display: "block", width: "100%" }} value={ds.fuse_dir ?? ""} onChange={(e) => up({ fuse_dir: e.target.value.trim() })} placeholder="e.g. s1  (s2/forest/x.tif ↔ s1/forest/x.tif)" /></label>
        <label style={{ display: "block", margin: "4px 0" }}>Bands in these files, in their order
          <input style={{ display: "block", width: "100%" }} value={order.join(" ")} onChange={(e) => { const o = e.target.value.split(/\s+/).filter((x) => x); up({ band_order2: o, bands2: chosen.filter((b) => o.includes(b)) }); }} /></label>
      </div>
      <div className="bandgrid">{order.map((b) => <label key={b} className="chk"><input type="checkbox" checked={chosen.includes(b)} onChange={() => toggle(b)} />{b}</label>)}</div>
      {ds.sensor2 === "s1" && <label className="chk"><input type="checkbox" checked={!!ds.s1_db2} onChange={(e) => up({ s1_db2: e.target.checked })} />
        linear backscatter: convert to decibels (Earth Engine's S1_GRD is already in dB)</label>}
      <p className="muted small">Both groups are stacked into one image: group 1's {(ds.bands ?? []).length} bands, then these {chosen.length}. One Input takes
        both (early fusion), or, in the Design tab, give the Input “band group 1” and add an Input for band group 2: one branch per sensor,
        joined by Concat (mid fusion). The folder mirrors the first group's: the same subfolders and file names.</p>
    </>}
  </div>;
}

/** images per class of a class-folder layout, with a warning when they are strongly unbalanced */
function FolderCounts({ f }: { f?: { classes: string[]; counts?: number[] } }) {
  if (!f?.counts?.length) return null;
  const most = Math.max(1, ...f.counts), least = Math.min(...f.counts);
  return <span className="small" style={{ color: "var(--ink)" }}>
    {f.classes.map((c, i) => `${c} ${f.counts![i].toLocaleString()}`).join(" · ")}
    {least === 0 ? <span className="st-warn"> — a class has no images</span> : most / least > 3
      ? <span className="st-warn"> — unbalanced ({(most / least).toFixed(1)}×): consider class weights (Train tab)</span> : null}
  </span>;
}

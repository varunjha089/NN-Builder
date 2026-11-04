// Where the data comes from. Only metadata is read here (Hugging Face datasets-server, Kaggle file list): the data itself is downloaded
// on the GPU machine when a run starts.
import { useEffect, useState } from "react";
import { api, errText, type AppState, type Dataset } from "./api";

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
      const labels = c.columns.filter((x) => x.type === "ClassLabel");
      const lab = labels.find((x) => x.name === ds.hf_label_col) ?? labels.find((x) => x.name === "label")
        ?? [...labels].sort((a, b) => (b.names?.length ?? 0) - (a.names?.length ?? 0))[0];
      const splits = c.splits.map((s) => s.name);
      up({ hf_config: info.configs.length > 1 ? c.name : "", hf_image_col: img, hf_label_col: lab?.name ?? ds.hf_label_col, classes: lab?.names ?? ds.classes,
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
        <label className="radio"><input type="radio" checked={ds.task !== "segment"} onChange={() => up({ task: "classify" })} />Classification (one label per image)</label>
        <label className="radio"><input type="radio" checked={ds.task === "segment"} onChange={() => up({ task: "segment", ...(ds.source === "builtin" ? { source: "huggingface" } : {}) })} />
          Segmentation (a class for every pixel: images + masks)</label>
      </div>
      <div className="row">
        <strong>Source:</strong>
        {(["builtin", "huggingface", "kaggle"] as const).filter((s) => !(s === "builtin" && ds.task === "segment")).map((s) => (
          <label key={s} className="radio"><input type="radio" checked={ds.source === s} onChange={() => up({ source: s })} />
            {{ builtin: "Built-in (torchvision)", huggingface: "Hugging Face", kaggle: "Kaggle" }[s]}</label>
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
          {ds.task !== "segment" && <label>Label column<select value={ds.hf_label_col} onChange={(e) => { const c = conf?.columns.find((x) => x.name === e.target.value); up({ hf_label_col: e.target.value, ...(c?.names ? { classes: c.names } : {}) }); }}>
            {conf?.columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}</select></label>}
        </div>}
        {!st.secrets.huggingface && <p className="muted small">Gated or private datasets need your Hugging Face token (Connection tab).</p>}
      </>}

      {ds.source === "kaggle" && <>
        <div className="row"><input className="grow" placeholder="owner/dataset, e.g. faizalkarim/flood-area-segmentation" value={ds.kaggle_slug} onChange={(e) => up({ kaggle_slug: e.target.value.trim() })} />
          <button onClick={readKaggle} disabled={busy || !ds.kaggle_slug || !st.secrets.kaggle || !st.tools_ready}>Read files</button></div>
        {(!st.secrets.kaggle || !st.tools_ready) && <p className="muted small">Reading the file list needs the tools and your Kaggle token (Connection tab).</p>}
        {kg && <p className="muted small">{kg.files.toLocaleString()} files listed{kg.folders.length ? `; folders with one subfolder per class: ${kg.folders.map((f) => f.folder || "(top)").join(", ")}` : ""}.</p>}
        {ds.task === "segment" ? <div className="form">
          <label>Images folder<input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="e.g. Image" /></label>
          <label>Masks folder (same file names)<input value={ds.kaggle_mask_dir} onChange={(e) => up({ kaggle_mask_dir: e.target.value.trim() })} placeholder="e.g. Mask" /></label>
          {kg && <p className="muted small wide">Folders seen: {[...new Set(kg.sample.map((f) => f.split("/").slice(0, -1).join("/")))].slice(0, 8).join(", ")}</p>}
        </div> : <div className="form">
          <label>Training folder (one subfolder per class)
            {kg?.folders.length ? <select value={ds.kaggle_train_dir} onChange={(e) => { const f = kg.folders.find((x) => x.folder === e.target.value); up({ kaggle_train_dir: e.target.value, classes: f?.classes ?? ds.classes }); }}>
              {kg.folders.map((f) => <option key={f.folder} value={f.folder}>{f.folder || "(top level)"} — {f.classes.length} classes</option>)}</select>
              : <input value={ds.kaggle_train_dir} onChange={(e) => up({ kaggle_train_dir: e.target.value.trim() })} placeholder="e.g. data/train" />}</label>
          <label>Test folder (optional)<input value={ds.kaggle_test_dir} onChange={(e) => up({ kaggle_test_dir: e.target.value.trim() })} placeholder="empty = 10% of the training images" /></label>
        </div>}
      </>}

      {(ds.source !== "builtin" || ds.resize) && <div className="form">
        <label>Image height<input type="number" min={8} max={1024} value={ds.image_size[0]} onChange={(e) => up({ image_size: [Number(e.target.value), ds.image_size[1]] })} /></label>
        <label>Image width<input type="number" min={8} max={1024} value={ds.image_size[1]} onChange={(e) => up({ image_size: [ds.image_size[0], Number(e.target.value)] })} /></label>
        <label>Channels<select value={ds.channels} onChange={(e) => up({ channels: Number(e.target.value) })}><option value={1}>1 (grey)</option><option value={3}>3 (colour)</option></select></label>
        {ds.source !== "builtin" && <label className="wide">{ds.task === "segment" ? "Classes, one per line (mask pixel value 0 = first line; with two classes any non-zero pixel is the second)" : "Class names, one per line, in label order"} ({ds.classes.length})
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

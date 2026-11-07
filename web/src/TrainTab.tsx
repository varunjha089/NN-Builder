// Train on Kaggle, a college GPU or AWS: settings, start, follow (curves, log), stop, fetch; the list of runs.
import { useEffect, useRef, useState } from "react";
import { api, errText, type Analysis, type AppState, type RunStatus, type TrainSettings } from "./api";

/** the recipe that suits most networks here (and is the usual one for transformers): AdamW with weight decay, warm-up then cosine decay,
 *  label smoothing, gradient clipping */
/** a starting point for the custom loss, per task */
const CUSTOM_LOSS: Record<string, string> = {
  classify: "def custom_loss(out, y):\n    # cross-entropy with a small penalty on very large scores\n    return nn.functional.cross_entropy(out, y) + 1e-4 * (out ** 2).mean()\n",
  multilabel: "def custom_loss(out, y):\n    # binary cross-entropy, positives counted twice\n    return nn.functional.binary_cross_entropy_with_logits(out, y.float(), pos_weight=torch.full_like(out[0], 2.0))\n",
  regress: "def custom_loss(out, y):\n    # log-cosh: like MSE for small errors, like L1 for large ones\n    d = out - y\n    return (d + nn.functional.softplus(-2 * d) - 0.6931471805599453).mean()\n",
};

const RECOMMENDED: Partial<TrainSettings> = { optimizer: "adamw", weight_decay: 0.05, schedule: "cosine", warmup_epochs: 5, label_smoothing: 0.1, grad_clip: 1 };

/** rough GPU memory for training, in GB: weights + gradients + Adam's two moments (16 bytes per parameter), the layer outputs kept for the
 *  backward pass (about twice each output, 2 bytes with mixed precision, 4 without), and about 0.6 GB for CUDA itself */
function memoryGB(a: Analysis | null, batch: number, amp: boolean, saveMemory: boolean): { total: number; state: number; act: number } | null {
  if (!a) return null;
  const state = (a.total_params * 16) / 1e9;
  const outs = a.nodes.reduce((t, n) => t + (n.out ? n.out.reduce((p, x) => p * x, 1) : 0), 0);
  // "save memory" keeps only the outputs at the cuts plus one part's at a time: about 2/sqrt(layers) of them
  const kept = saveMemory ? Math.min(1, 2 / Math.sqrt(Math.max(1, a.nodes.length))) : 1;
  const act = (outs * batch * (amp ? 2 : 4) * 2 * kept) / 1e9;
  return { total: state + act + 0.6, state, act };
}

/** GPU memory of a target in GB: Kaggle and EC2 g4dn have a 16 GB GPU; an SSH machine as Prepare found it (nvidia-smi) */
function gpuGB(st: AppState, target: string, kind: string): number | null {
  if (kind === "kaggle") return 16;
  const m = st.machines.find((x) => x.machine.name === target);
  const mib = m?.prepared?.gpus.map((g) => Number(/(\d+)\s*MiB/.exec(g)?.[1] ?? 0)).filter((x) => x > 0);
  if (mib && mib.length) return Math.max(...mib) / 1024;
  return kind === "aws" ? 16 : null;
}
import { Chart } from "./Preview";

export function TrainTab({ st, ok, analysis, train, setTrain, runId, openRun, toast, startJob, datasetSource, datasetTask }: {
  st: AppState; ok: boolean; analysis: Analysis | null; train: TrainSettings; setTrain: (t: TrainSettings) => void; runId: string | null; openRun: (id: string | null) => void;
  toast: (m: string) => void; startJob: () => void; datasetSource: string; datasetTask: string;
}) {
  const [targets, setTargets] = useState<{ name: string; kind: string; prepared?: boolean }[]>([]);
  const [target, setTarget] = useState("Kaggle");
  const [gpu, setGpu] = useState("");
  const [copyTokens, setCopyTokens] = useState(false);
  const [runs, setRuns] = useState<Awaited<ReturnType<typeof api.runs>>>([]);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [moveTo, setMoveTo] = useState("");
  const [seeds, setSeeds] = useState(1);
  const [picked, setPicked] = useState<string[]>([]);
  const [comparing, setComparing] = useState<string[] | null>(null);
  const events = useRef<Record<string, any>[]>([]);
  const since = useRef(0);

  useEffect(() => { api.targets().then(setTargets).catch(() => {}); api.runs().then(setRuns).catch(() => {}); }, [st]);

  // follow the open run: every 3 s while it is running (the Rust side limits how often it really asks the machine / Kaggle)
  useEffect(() => {
    events.current = []; since.current = 0; setStatus(null);
    if (!runId) return;
    let stop = false, timer = 0;
    const tick = async () => {
      try {
        const s = await api.runStatus(runId, since.current);
        events.current = [...events.current, ...s.events];
        since.current = s.next;
        if (stop) return;
        setStatus({ ...s, events: events.current });
        // an interrupted spot run is followed more slowly: the app asks AWS at most every 30 s
        // (a queued Kaggle run is pushed by these polls when a GPU is free)
        if (["starting", "running", "queued"].includes(s.state)) timer = window.setTimeout(tick, document.hidden ? 15000 : 3000);
        else if (s.state === "interrupted") timer = window.setTimeout(tick, 15000);
        else api.runs().then(setRuns).catch(() => {});
      } catch (e) { if (!stop) toast(errText(e)); }
    };
    tick();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [runId, toast]);

  // the project's other active runs (other seeds) are followed too, every 30 s: they fetch their results when they end,
  // and a queued Kaggle run is pushed although another run is the one shown
  const others = runs.filter((r) => r.id !== runId && ["starting", "running", "queued"].includes(r.status)).map((r) => r.id).join(" ");
  useEffect(() => {
    if (!others) return;
    const timer = window.setInterval(() => {
      Promise.all(others.split(" ").map((id) => api.runStatus(id, 1e9).catch(() => null)))
        .then(() => api.runs().then(setRuns)).catch(() => {});
    }, 30000);
    return () => window.clearInterval(timer);
  }, [others]);

  const kind = targets.find((t) => t.name === target)?.kind ?? "kaggle";
  const set = (p: Partial<TrainSettings>) => setTrain({ ...train, ...p });
  const checkData = async () => {
    try { await api.runStart(target, gpu, copyTokens, true); startJob(); toast(`Checking the data on ${target} …`); } catch (e) { toast(errText(e)); }
  };
  const start = async () => {
    try {
      if (seeds > 1) await api.runStartSeeds(target, gpu, copyTokens, seeds); else await api.runStart(target, gpu, copyTokens);
      startJob(); toast(seeds > 1 ? `Starting ${seeds} runs (seeds ${train.seed}–${train.seed + seeds - 1}) on ${target} …` : `Starting on ${target} …`);
    } catch (e) { toast(errText(e)); }
  };
  const ep = (status?.events ?? []).filter((e) => e.event === "epoch"), bt = (status?.events ?? []).filter((e) => e.event === "batch");
  const startEv = status?.events.find((e) => e.event === "start");
  const errEv = status?.events.find((e) => e.event === "error");
  const check = status?.events.find((e) => e.event === "check");
  const running = status && ["starting", "running"].includes(status.state);
  const needsToken = (datasetSource === "kaggle" && kind !== "kaggle") || datasetSource === "huggingface";

  return (
    <div className="page">
      <h2>Train</h2>
      <p className="muted">Training never runs on this Mac. Choose where: Kaggle (free GPU notebooks), or a machine from the Connection tab (college server or AWS).</p>
      <div className="form">
        <label>Train on<select value={target} onChange={(e) => setTarget(e.target.value)}>
          {targets.map((t) => <option key={t.name} value={t.name}>{t.name}{t.kind === "aws" ? " (AWS EC2)" : t.kind === "ssh" ? (t.prepared ? " (SSH)" : " (SSH, prepared on first run)") : " (Kaggle GPU)"}</option>)}
        </select></label>
        {kind !== "kaggle" && <label>GPU on the machine<input value={gpu} onChange={(e) => setGpu(e.target.value.trim())} placeholder="any (or 0, or 0,1)" /></label>}
        <label>Epochs<input type="number" min={1} max={1000} value={train.epochs} onChange={(e) => set({ epochs: Number(e.target.value) })} /></label>
        <label>Batch size<input type="number" min={1} max={8192} value={train.batch_size} onChange={(e) => set({ batch_size: Number(e.target.value) })} /></label>
        <label>Learning rate<input type="number" step={0.0001} min={0.0000001} max={1} value={train.lr} onChange={(e) => set({ lr: Number(e.target.value) })} /></label>
        <label>Optimizer<select value={train.optimizer} onChange={(e) => set({ optimizer: e.target.value })}><option value="adam">Adam</option><option value="adamw">AdamW (decoupled weight decay)</option><option value="sgd">SGD (momentum 0.9)</option></select></label>
        <label>Random seed<input type="number" min={0} value={train.seed} onChange={(e) => set({ seed: Math.max(0, Math.floor(Number(e.target.value))) })}
          title="Fixes the initial weights, the test split taken from the training data, and the batch order; run several seeds to report a mean" /></label>
        <label>Training images (0 = all)<input type="number" min={0} value={train.limit_train} onChange={(e) => set({ limit_train: Number(e.target.value) })} /></label>
        {kind !== "kaggle" && <label>End the run after (hours of running)<input type="number" min={0.1} max={72} step={0.5} value={train.max_hours} onChange={(e) => set({ max_hours: Number(e.target.value) })} /></label>}
        {kind !== "kaggle" && <label>Checkpoint every (minutes, 0 = each epoch)<input type="number" min={0} max={240} value={train.checkpoint_minutes}
          onChange={(e) => set({ checkpoint_minutes: Number(e.target.value) })} title="Also inside an epoch. Spot instances use 10 when this is 0." /></label>}
        {kind === "aws" && <label>Automatic resumes after a spot interruption<input type="number" min={0} max={100} value={train.max_resumes}
          onChange={(e) => set({ max_resumes: Number(e.target.value) })} /></label>}
        <div className="wide aug"><strong>Data augmentation</strong> <span className="muted small">(training images only; segmentation: image and mask together)</span>
          <div className="row">
            <label className="chk"><input type="checkbox" checked={train.aug_hflip} onChange={(e) => set({ aug_hflip: e.target.checked })} />horizontal flip</label>
            <label className="chk"><input type="checkbox" checked={train.aug_vflip} onChange={(e) => set({ aug_vflip: e.target.checked })} />vertical flip (aerial / satellite images)</label>
            <label className="chk"><input type="checkbox" checked={train.aug_color} onChange={(e) => set({ aug_color: e.target.checked })} />brightness / contrast</label>
            <label>rotation ±<input type="number" min={0} max={45} value={train.aug_rotate} onChange={(e) => set({ aug_rotate: Number(e.target.value) })} />°</label>
            {datasetTask !== "segment" && <label>random crop, padding<input type="number" min={0} max={16} value={train.aug_crop} onChange={(e) => set({ aug_crop: Number(e.target.value) })} />px</label>}
          </div>
          {datasetSource === "builtin" && train.aug_hflip && datasetTask !== "segment" && <p className="muted small">Note: flipping digits (MNIST) changes their meaning; use flips for objects and photos.</p>}
        </div>
        <div className="wide aug"><strong>Start from ImageNet weights</strong>
          <span className="muted small"> (transfer learning: much better results on small datasets; the images become 3-channel and are normalised as for ImageNet)</span>
          <div className="row">
            <label>Weights<select value={train.pretrained} onChange={(e) => set({ pretrained: e.target.value as TrainSettings["pretrained"], ...(e.target.value ? {} : { freeze: "none" }) })}>
              <option value="">none (random start)</option><option value="resnet18">torchvision ResNet-18 (for the ResNet-18 demo)</option>
              <option value="vgg16">torchvision VGG-16 (for the VGG-16 demo)</option><option value="vit_b_16">torchvision ViT-B/16 (for the ViT-Base/16 demo)</option></select></label>
            {train.pretrained && <label className="chk"><input type="checkbox" checked={train.freeze === "backbone"}
              onChange={(e) => set({ freeze: e.target.checked ? "backbone" : "none" })} />train only the new output layer (freeze the rest)</label>}
          </div>
          {train.pretrained && <p className="muted small">The weights are loaded layer by layer into your design (on the GPU machine); a design that
            no longer matches the network stops with a clear message. Use a small learning rate for fine-tuning (e.g. 0.0001), or freeze and use 0.001.</p>}
        </div>
        <div className="wide aug"><strong>Training recipe</strong>
          <span className="muted small"> (the defaults are plain training at a constant learning rate)</span>
          <button className="small" style={{ marginLeft: 8 }} onClick={() => set(RECOMMENDED)}
            title="AdamW, weight decay 0.05, 5 warm-up epochs then cosine decay, label smoothing 0.1, gradient clipping 1.0">Use recommended</button>
          <div className="row">
            <label>Learning-rate schedule<select value={train.schedule} onChange={(e) => set({ schedule: e.target.value as TrainSettings["schedule"] })}>
              <option value="constant">constant</option><option value="cosine">cosine decay to 0</option>
              <option value="plateau">halve when validation loss stalls (3 epochs)</option></select></label>
            <label>warm-up<input type="number" min={0} max={50} step={1} value={train.warmup_epochs} onChange={(e) => set({ warmup_epochs: Number(e.target.value) })} />epochs</label>
            <label>weight decay<input type="number" min={0} max={1} step={0.01} value={train.weight_decay} onChange={(e) => set({ weight_decay: Number(e.target.value) })} /></label>
            <label>label smoothing<input type="number" min={0} max={0.5} step={0.05} value={train.label_smoothing} onChange={(e) => set({ label_smoothing: Number(e.target.value) })} /></label>
            <label>clip gradients at<input type="number" min={0} max={100} step={0.5} value={train.grad_clip} onChange={(e) => set({ grad_clip: Number(e.target.value) })} />(0 = off)</label>
          </div>
          {datasetTask !== "regress" && datasetTask !== "detect" && <div className="row">
            <label title="Rare classes count more in the loss, so the network does not just learn the common ones">Unbalanced classes
              <select value={train.class_weights ?? ""} onChange={(e) => set({ class_weights: e.target.value as TrainSettings["class_weights"] })}>
                <option value="">no class weights</option><option value="inverse">weight by inverse frequency (balanced)</option>
                <option value="sqrt">weight by its square root (gentler)</option></select></label>
            <label className="chk" title="Multi-label: each class gets the threshold with the best F1 on the validation data, instead of 50%">
              <input type="checkbox" checked={!!train.tune_thresholds} onChange={(e) => set({ tune_thresholds: e.target.checked, ...(e.target.checked && !train.val_fraction ? { val_fraction: 0.1 } : {}) })} />
              multi-label: a threshold per class, chosen on the validation data</label>
          </div>}
          <div className="row">
            <label title="What the network is trained to make small">Loss
              <select value={train.loss ?? ""} onChange={(e) => set({ loss: e.target.value as TrainSettings["loss"],
                ...(e.target.value === "custom" && !train.custom_loss ? { custom_loss: CUSTOM_LOSS[datasetTask === "regress" ? "regress" : datasetTask === "multilabel" ? "multilabel" : "classify"] } : {}) })}>
                {datasetTask === "detect" ? <option value="">CenterNet (focal loss on the heatmaps, L1 on box size and offset)</option>
                : datasetTask === "regress" ? <>
                  <option value="">mean squared error (MSE)</option>
                  <option value="l1">mean absolute error (L1)</option>
                  <option value="huber">Huber (MSE for small errors, L1 for large)</option></> : <>
                  <option value="">{datasetTask === "multilabel" ? "binary cross-entropy" : "cross-entropy"}</option>
                  <option value="focal">focal (hard examples count more)</option></>}
                {(datasetTask === "segment" || datasetTask === "multilabel") && <>
                  <option value="dice">Dice (overlap per class)</option>
                  <option value="dice_ce">Dice + cross-entropy</option></>}
                <option value="custom">your own (Python)</option>
              </select></label>
            {train.loss === "focal" && <label>focal γ<input type="number" min={0} max={5} step={0.5} value={train.focal_gamma ?? 2}
              onChange={(e) => set({ focal_gamma: Number(e.target.value) })} /></label>}
          </div>
          {train.loss?.startsWith("dice") && datasetTask !== "segment" && datasetTask !== "multilabel" &&
            <p className="msg error small">Dice loss needs segmentation or multi-label; with one class per image use cross-entropy or focal.</p>}
          {train.loss === "custom" && <>
            <textarea className="code" spellCheck={false} rows={8} value={train.custom_loss ?? ""} onChange={(e) => set({ custom_loss: e.target.value })} />
            <p className="muted small">Define <code>def custom_loss(out, y):</code> returning one number. <code>out</code> is the network's output for a
              batch ({datasetTask === "segment" ? "batch, classes, height, width" : "batch, classes"}); <code>y</code> is what the data gives:
              {datasetTask === "segment" ? " a class number per pixel (−100 = no label)" : datasetTask === "multilabel" ? " 0 / 1 per class"
                : datasetTask === "regress" ? " the targets standardised (mean 0, spread 1)" : " the class number of each image"}.
              The validation loss is this loss too. It runs only on the training machine; <code>torch</code> and <code>nn</code> are imported.</p></>}
          {datasetTask === "regress" && ["focal", "dice", "dice_ce"].includes(train.loss ?? "") &&
            <p className="msg error small">Regression is trained with MSE, L1 or Huber loss: choose one above.</p>}
          {datasetTask !== "regress" && ["mse", "l1", "huber"].includes(train.loss ?? "") &&
            <p className="msg error small">MSE, L1 and Huber are for regression; choose cross-entropy, focal or Dice for classes.</p>}
          {datasetTask === "regress" && <p className="muted small">The network learns each target standardised (mean 0, spread 1 on the training
            data); the scores are in the targets' own units. MSE punishes large errors most; L1 is less swayed by a few wild targets (outliers);
            Huber is MSE for errors under one standard deviation of the target and L1 above it.</p>}
          {train.loss === "focal" && <p className="muted small">Focal loss multiplies each image's (pixel's) cross-entropy by (1 − p)^γ, where p is the
            probability given to the right class: images the network already gets right count little, so the rare and hard ones count more.
            γ 0 is plain cross-entropy; 2 is usual. It works together with class weights.</p>}
          {train.loss?.startsWith("dice") && <p className="muted small">Dice loss is 1 − the overlap 2|P∩T| / (|P| + |T|) of each class's predicted and
            true {datasetTask === "segment" ? "pixels" : "images"} in the batch, averaged over the classes, so a small class counts as much as a large
            one. Dice + cross-entropy is the steadier choice (Dice alone can learn slowly at the start).</p>}
          {train.class_weights && <p className="muted small">The weights come from the training images per class, counted on the training machine
            (shown in the run's log). Multi-label: each class's positives weigh negatives ÷ positives{train.class_weights === "sqrt" ? " (square root)" : ""}.</p>}
          {train.weight_decay > 0 && train.optimizer === "adam" && <p className="muted small">With Adam, weight decay is mixed into the adaptive
            step; AdamW applies it separately, which usually works better.</p>}
        </div>
        <div className="wide aug"><strong>Validation and early stopping</strong>
          <span className="muted small"> (validation data is taken from the training data; the test data is then used only once, at the end)</span>
          <div className="row">
            <label>Validation part of the training data<input type="number" min={0} max={50} step={5} value={Math.round(train.val_fraction * 100)}
              onChange={(e) => set({ val_fraction: Number(e.target.value) / 100 })} />%</label>
            <label className="chk"><input type="checkbox" checked={train.early_stop}
              onChange={(e) => set({ early_stop: e.target.checked, ...(e.target.checked && train.val_fraction < 0.01 ? { val_fraction: 0.1 } : {}) })} />early stopping</label>
            {train.early_stop && <>
              <label>stop after<input type="number" min={1} max={100} value={train.patience} onChange={(e) => set({ patience: Number(e.target.value) })} />epochs without improvement</label>
              <label>in<select value={train.monitor} onChange={(e) => set({ monitor: e.target.value as TrainSettings["monitor"] })}>
                <option value="val_loss">validation loss</option><option value="val_acc">{datasetTask === "detect" ? "validation mAP@0.5" : datasetTask === "regress" ? "validation R²" : "validation accuracy (multi-label: micro F1)"}</option>
                {datasetTask === "segment" && <option value="val_miou">validation mIoU</option>}</select></label>
            </>}
          </div>
          {train.early_stop && <p className="muted small">The model kept at the end is the best epoch's, and the test accuracy is measured on it.
            Epochs is the most the run may take.</p>}
          {!train.early_stop && train.val_fraction === 0 && <p className="muted small">0% = the curves' "validation" is the test data (fine for
            a quick look; for a paper, hold out a validation part instead).</p>}
        </div>
        <div className="wide aug"><strong>GPU memory</strong>
          <span className="muted small"> (what happens when the network or the batch is too big for the GPU)</span>
          <div className="row">
            <label className="chk"><input type="checkbox" checked={train.auto_split} onChange={(e) => set({ auto_split: e.target.checked })} />
              split a batch that does not fit into 2, 4, 8 … parts (same batch and results; slower)</label>
            <label className="chk"><input type="checkbox" checked={train.save_memory} onChange={(e) => set({ save_memory: e.target.checked })} />
              save memory: recompute layer outputs instead of keeping them (much less memory, about 30% slower)</label>
          </div>
          <p className="muted small">Newer GPUs (A100, A6000, RTX 30xx/40xx) train in bfloat16 by themselves; convolutions use the faster
            channels-last layout. The run card shows the memory really used after the first epoch.{train.auto_split ? " With BatchNorm, the parts" +
            " normalise over fewer images than the whole batch, the only difference a split makes." : ""}</p>
        </div>
        <label className="chk wide"><input type="checkbox" checked={train.export} onChange={(e) => set({ export: e.target.checked })} />
          Also save the trained model as TorchScript (model.ts) and ONNX (model.onnx), to use it without this code</label>
        <label className="chk wide"><input type="checkbox" checked={train.amp} onChange={(e) => set({ amp: e.target.checked })} />Mixed precision on the GPU (faster on a T4)</label>
        <div className="wide row">
          <label title="With more than one, train.py starts one worker per GPU (PyTorch DistributedDataParallel)">GPUs
            <select value={String(train.gpus ?? 1)} onChange={(e) => set({ gpus: Number(e.target.value) })}>
              <option value="1">one</option><option value="0">all the machine has</option>
              {[2, 4, 8].map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
          {(train.gpus ?? 1) !== 1 && <span className="muted small">Each GPU takes its share of every batch of {train.batch_size} (the batch size stays the total, so
            the results match one GPU's); BatchNorm counts all GPUs' images; checkpoints, stop and spot interruptions work as with one. Fewer GPUs
            than asked: all there are. Kaggle notebooks usually have one GPU (two with the T4×2 accelerator, which the Kaggle API cannot choose).</span>}
        </div>
        {kind === "aws" && <label className="chk wide"><input type="checkbox" checked={train.stop_instance} onChange={(e) => set({ stop_instance: e.target.checked })} />Stop the EC2 instance when the run ends (saves money)</label>}
        {kind !== "kaggle" && needsToken && <label className="chk wide"><input type="checkbox" checked={copyTokens} onChange={(e) => setCopyTokens(e.target.checked)} />
          Copy my {datasetSource === "kaggle" ? "Kaggle" : "Hugging Face"} token to the machine (a file only you can read; needed for Kaggle data, and for gated Hugging Face data)</label>}
      </div>
      {(() => {
        // what is missing for this target, and whether the design fits its GPU
        const need: string[] = [];
        if ((kind === "kaggle" || kind === "aws") && !st.tools_ready) need.push("Install the tools (Connection → Install tools)");
        if (kind === "kaggle" && !st.secrets.kaggle) need.push("add your Kaggle API token (Connection → Kaggle)");
        if (kind === "aws" && !st.secrets.aws) need.push("add your AWS access key (Connection → AWS)");
        const mem = memoryGB(analysis, train.batch_size, train.amp, train.save_memory), gpu = gpuGB(st, target, kind);
        return <>
          {need.length > 0 && <div className="msg warn">Before training on {target}: {need.join("; ")}.</div>}
          {mem && <p className={`small ${gpu && mem.total > gpu * 0.9 ? "st-warn" : "muted"}`}>Estimated GPU memory for training: about
            {" "}{mem.total < 1 ? mem.total.toFixed(2) : mem.total.toFixed(1)} GB (weights and optimizer {mem.state.toFixed(2)} GB, layer outputs
            {" "}{mem.act.toFixed(2)} GB at batch {train.batch_size})
            {gpu ? ` · ${target === "Kaggle" ? "Kaggle's GPU" : target} has ${gpu.toFixed(0)} GB` : ""}
            {gpu && mem.state + 0.6 > gpu ? ": the model alone does not fit; use a larger GPU." :
              gpu && mem.total > gpu * 0.9 ? (train.auto_split ? ": the batches will probably be split into parts (slower; \"save memory\" may avoid it)."
                : ": likely out of memory, lower the batch size or switch on splitting.") : "."}
            {" "}A rough estimate: the first epoch shows the real use.</p>}
        </>;
      })()}
      <p className="row"><button className="primary" onClick={start} disabled={!ok}>Train on {target}{seeds > 1 ? ` ×${seeds}` : ""} ▶</button>
        <label className="small">seeds<select value={seeds} onChange={(e) => setSeeds(Number(e.target.value))}
          title="Repeat the same training with different random seeds, to report a mean and its spread">
          <option value={1}>1 (seed {train.seed})</option><option value={3}>3 (seeds {train.seed}–{train.seed + 2})</option>
          <option value={5}>5 (seeds {train.seed}–{train.seed + 4})</option></select></label>
        {seeds > 1 && kind !== "kaggle" && <span className="muted small">The runs start together on the machine and share its GPU.</span>}
        {seeds > 1 && kind === "kaggle" && <span className="muted small">One Kaggle notebook per seed; Kaggle may queue some of them.</span>}
        <button onClick={checkData} disabled={!ok} title="Load the data on the machine, report it (images per class, bands, empty pixels, what looks wrong)
and check one batch through the network, without training: a few minutes instead of a wasted GPU hour">Check data on {target}</button>
        {!ok && <span className="st-error">Fix the design and the dataset first.</span>}</p>

      {status && <div className="card">
        <div className="row"><strong>Run {status.id}</strong><span className={`pill ${status.state}`}>{status.state === "running" && ep.length ? `epoch ${ep[ep.length - 1].epoch}/${status.info.epochs}` : status.state}</span>
          <span className="muted">on {status.info.target}{startEv ? ` · ${startEv.gpu ?? startEv.device}` : ""}{startEv?.train_size ? ` · ${startEv.train_size.toLocaleString()} training images` : ""}</span>
          <span className="grow" />
          {status.info.url && <button onClick={() => api.openUrl(status.info.url)}>Open on Kaggle</button>}
          {(running || status.state === "interrupted") && status.info.kind !== "kaggle" && <button onClick={() => api.runStop(status.id).then(toast).catch((e) => toast(errText(e)))}>Stop</button>}
          {status.info.kind !== "kaggle" && !running && <button onClick={() => api.runFetch(status.id).then(startJob).catch((e) => toast(errText(e)))}>Fetch results</button>}
          <button onClick={() => api.runReveal(status.id)}>Show files</button>
        </div>
        {status.info.note && <p className="muted small">{status.info.note}</p>}
        {status.info.kind === "kaggle" && running && <p className="muted small">Started {status.info.created} · Kaggle checked every 30 s
          {status.info.last_poll ? `, last at ${new Date(status.info.last_poll * 1000).toLocaleTimeString()}` : ""}: when it reports the notebook
          finished, the results are downloaded and shown here.</p>}
        {!status.info.reachable && running && <div className="msg warn">The machine does not answer right now; the run goes on there and the app will catch up.</div>}
        {check && <div className="msg ok">{check.message}</div>}
        {(() => {
          // GPU memory: what the run really used, batches split for memory, and time lost waiting for data
          const m = status.events.find((e) => e.event === "memory");
          const sp = status.events.filter((e) => e.event === "memory_split");
          const tr = sp.filter((e) => e.what === "training").pop(), ev = sp.filter((e) => e.what === "evaluation").pop();
          const last = ep[ep.length - 1], wait = last && typeof last.data_wait === "number" ? last.data_wait : 0;
          return <>
            {m && <p className="muted small">GPU memory: peak {m.peak_gb} GB of {m.total_gb} GB ({m.precision}{m.save_memory ? ", save memory" : ""}
              {m.parts > 1 ? `, batch ${m.batch} in ${m.parts} parts` : `, batch ${m.batch}`}).</p>}
            {tr && <div className="msg warn">The GPU ran out of memory, so each batch of {tr.batch} is now processed in {tr.parts} parts of {tr.part_size}
              {" "}(same results, slower). A smaller batch or "save memory" would be faster.</div>}
            {ev && <p className="muted small">Evaluation batches were reduced to {ev.batch} images to fit in GPU memory.</p>}
            {wait > 0.3 && <div className="msg warn">The GPU waited for data {Math.round(wait * 100)}% of the last epoch: loading and augmenting images
              is the bottleneck (fewer augmentations, smaller images, or a machine with more CPU cores would help).</div>}
          </>;
        })()}
        {errEv && <div className="msg error">{errEv.message}</div>}
        {status.info.error && <div className="msg error">{status.info.error}</div>}
        {status.state === "crashed" && <div className="msg error">The run ended without a result: see the log.</div>}
        {(() => { const es = status.events.find((e) => e.event === "early_stop"); return es ? <div className="msg ok">Early stopping after epoch {es.epoch}:
          no better {String(es.monitor).replace("val_", "validation ")} for {es.patience} epochs; the model is from epoch {es.best_epoch}.</div> : null; })()}
        {status.state === "interrupted" && <div className="msg warn">Interrupted by AWS (spot). It resumes from its last checkpoint when the instance is back.</div>}
        {status.state === "terminated" && <div className="msg error">The spot instance was terminated, with the checkpoints on it.</div>}
        {status.info.resumed_from && <p className="muted small">Continues run {status.info.resumed_from} from its checkpoint.</p>}
        {(status.checkpoint || status.mirror) && <p className="muted small">
          {status.checkpoint && <>Last checkpoint: epoch {status.checkpoint.epoch}{status.checkpoint.step ? `, batch ${status.checkpoint.step}/${status.checkpoint.steps}` : " (end)"} ({status.checkpoint.reason}, {new Date(status.checkpoint.time * 1000).toLocaleTimeString()}). </>}
          {status.mirror?.copied && <>Copy on this Mac: {status.mirror.copied}{status.mirror.bytes ? `, ${(status.mirror.bytes / 1e6).toFixed(0)} MB` : ""}{status.mirror.checkpoint ? ` (epoch ${status.mirror.checkpoint.epoch}${status.mirror.checkpoint.step ? `, batch ${status.mirror.checkpoint.step}` : ""})` : ""}. </>}
          {status.mirror?.error && <span className="st-warn">{status.mirror.error}</span>}</p>}
        {status.info.kind !== "kaggle" && !running && status.state !== "done" && status.state !== "continued" && status.mirror?.copied && <div className="row">
          <span>Continue on another machine from the copy on this Mac:</span>
          <select value={moveTo} onChange={(e) => setMoveTo(e.target.value)}><option value="">choose…</option>
            {targets.filter((t) => t.kind !== "kaggle" && t.name !== status.info.target).map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}</select>
          <button disabled={!moveTo} onClick={() => api.runResumeOn(status.id, moveTo, gpu, copyTokens).then(startJob).catch((e) => toast(errText(e)))}>Continue there ▶</button>
        </div>}
        <Chart title="Loss" series={[
          { name: "batch loss", color: "#c7d2e0", thin: true, pts: bt.map((e) => [e.epoch - 1 + e.step / e.steps, e.loss]) },
          { name: "train", color: "#2f6fde", pts: ep.map((e) => [e.epoch, e.train_loss]) },
          { name: "validation", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_loss]) }]} />
        {ep.some((e) => e.val_mae != null) ? <Chart title="Validation R² (1 = perfect, 0 = no better than the mean)" series={[
          { name: "validation R²", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_acc]) }]} />
        : datasetTask === "detect" && ep.every((e) => e.train_acc == null) ? <Chart title="Detection: validation mAP@0.5 and mAP@0.5:0.95" percent series={[
          { name: "mAP@0.5", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_acc]) },
          { name: "mAP@0.5:0.95", color: "#8e44ad", pts: ep.filter((e) => e.val_map != null).map((e) => [e.epoch, e.val_map]) }]} />
        : <Chart title={ep.some((e) => e.val_miou != null) ? "Pixel accuracy and mean IoU" : ep.some((e) => e.val_map != null)
          ? "Multi-label: label accuracy (train), micro F1 and mAP (validation)" : "Accuracy"} percent series={[
          { name: "train", color: "#2f6fde", pts: ep.map((e) => [e.epoch, e.train_acc]) },
          { name: ep.some((e) => e.val_map != null) ? "validation micro F1" : "validation", color: "#e76f51", pts: ep.map((e) => [e.epoch, e.val_acc]) },
          { name: "validation mAP", color: "#8e44ad", pts: ep.filter((e) => e.val_map != null).map((e) => [e.epoch, e.val_map]) },
          { name: "validation mIoU", color: "#2a9d8f", pts: ep.filter((e) => e.val_miou != null).map((e) => [e.epoch, e.val_miou]) },
          ...((ep[ep.length - 1]?.heads ?? []) as { name: string; metric: string }[]).map((h, k) => ({ name: `${h.name} (${h.metric})`,
            color: ["#8e44ad", "#d35400", "#16a085", "#7f8c8d"][k % 4], pts: ep.filter((e) => e.heads?.[k]).map((e) => [e.epoch, e.heads[k].score] as [number, number]) }))]} />}
        <details><summary>Log</summary><pre className="log">{status.log.join("\n")}</pre></details>
      </div>}

      <div className="row"><h3 style={{ margin: 0 }}>Runs</h3><span className="grow" />
        {picked.length > 0 && <span className="muted small">{picked.length} selected</span>}
        <button disabled={picked.length < 1} onClick={() => setComparing([...picked])}>Compare selected</button>
        {picked.length > 0 && <button className="small" onClick={() => setPicked([])}>Clear</button>}</div>
      <table className="grid"><tbody>
        <tr><th></th><th>Run</th><th>Where</th><th>Data</th><th className="num">Seed</th><th>State</th><th className="num">Test acc.</th></tr>
        {runs.map((r) => <tr key={r.id} className={`click ${r.id === runId ? "sel" : ""}`} onClick={() => openRun(r.id)}>
          <td onClick={(e) => e.stopPropagation()}><input type="checkbox" checked={picked.includes(r.id)}
            onChange={(e) => setPicked(e.target.checked ? [...picked, r.id] : picked.filter((x) => x !== r.id))} /></td>
          <td>{r.id}{r.group && <span className="muted small" title="started together with other seeds"> · group</span>}{r.check && <span className="muted small"> · data check</span>}</td><td>{r.target}</td>
          <td>{r.dataset}</td><td className="num">{r.seed}</td><td>{r.status}</td><td className="num">{r.check
            ? (r.warnings == null ? "" : r.warnings ? <span className="st-warn">{r.warnings} to look at</span> : <span className="st-ok">looks right</span>)
            : r.test_acc == null ? "" : r.task === "regress" ? `R² ${r.test_acc.toFixed(3)}` : r.task === "detect" ? `mAP ${(r.test_acc * 100).toFixed(1)}%` : `${(r.test_acc * 100).toFixed(2)}%`}</td></tr>)}
        {!runs.length && <tr><td className="muted">No runs yet.</td></tr>}
      </tbody></table>
      {runs.some((r) => r.group) && <p className="row small"><span className="muted">Select a group of seeds:</span>
        {[...new Set(runs.filter((r) => r.group).map((r) => r.group))].map((g) => <button key={g} className="small"
          onClick={() => setPicked(runs.filter((r) => r.group === g).map((r) => r.id))}>{g.replace(/^g/, "")}</button>)}</p>}
      {comparing && <Compare ids={comparing} toast={toast} close={() => setComparing(null)} />}
    </div>
  );
}

const PALETTE = ["#2f6fde", "#e76f51", "#2a9d8f", "#9c6ac6", "#d9822b", "#c0392b", "#3c8d7f", "#b03a5b", "#4b5bb5", "#6c7a89"];
const pct = (v: number | null | undefined) => (v == null ? "" : `${(v * 100).toFixed(2)}%`);
const pm = (s: { mean: number; std: number; n: number } | null) => (s ? `${(s.mean * 100).toFixed(2)} ± ${(s.std * 100).toFixed(2)}%` : "");

/** the chosen runs side by side: validation curves overlaid, a table per run, mean ± s.d. per group of seeds, export to CSV / LaTeX */
function Compare({ ids, toast, close }: { ids: string[]; toast: (m: string) => void; close: () => void }) {
  const [sum, setSum] = useState<{ rows: Record<string, any>[]; groups: Record<string, any>[] } | null>(null);
  const [curves, setCurves] = useState<Record<string, [number, number, number][]>>({});
  useEffect(() => {
    api.runsSummary(ids).then(setSum).catch((e) => toast(errText(e)));
    Promise.all(ids.map((id) => api.runStatus(id, 0).then((s) => [id, s.events.filter((e) => e.event === "epoch").map((e) => [e.epoch, e.val_acc, e.val_loss] as [number, number, number])] as const)))
      .then((all) => setCurves(Object.fromEntries(all))).catch((e) => toast(errText(e)));
  }, [ids, toast]);
  const series = (i: 1 | 2) => ids.map((id, k) => ({ name: id.slice(9), color: PALETTE[k % PALETTE.length], pts: (curves[id] ?? []).map((c) => [c[0], c[i]] as [number, number]) }));
  const top5 = sum?.rows.some((r) => r.test_top5 != null);
  return (
    <div className="card">
      <div className="row"><strong>Comparing {ids.length} run{ids.length > 1 ? "s" : ""}</strong><span className="grow" />
        <button onClick={() => api.runsExport(ids).then(toast).catch((e) => toast(errText(e)))}
          title="A CSV file and a LaTeX table (booktabs) in the project's exports folder">Export CSV + LaTeX</button>
        <button className="small" onClick={close}>×</button></div>
      <Chart title="Validation accuracy" percent series={series(1)} />
      <Chart title="Validation loss" series={series(2)} />
      {sum && <table className="grid small"><tbody>
        <tr><th>Run</th><th className="num">Seed</th><th>Recipe</th><th className="num">Epochs</th><th className="num">Best epoch</th>
          <th className="num">Test acc.</th>{top5 && <th className="num">Top-5</th>}<th className="num">Training time</th></tr>
        {sum.rows.map((r, k) => <tr key={r.id}><td><span style={{ color: PALETTE[k % PALETTE.length] }}>■</span> {r.id}</td><td className="num">{r.seed}</td>
          <td className="small">{r.optimizer} {r.lr} · {r.schedule} · batch {r.batch_size}{r.early_stop ? " · early stop" : ""}</td>
          <td className="num">{r.epochs_done}/{r.epochs_set}</td><td className="num">{r.best_epoch ?? ""}</td><td className="num">{pct(r.test_acc)}</td>
          {top5 && <td className="num">{pct(r.test_top5)}</td>}<td className="num">{(r.seconds / 60).toFixed(1)} min</td></tr>)}
        {sum.groups.filter((g) => g.runs > 1).map((g) => <tr key={g.group}><td><strong>Mean ± s.d.</strong> ({g.runs} seeds: {g.seeds.join(", ")})</td>
          <td></td><td></td><td className="num">{g.epochs_done ? g.epochs_done.mean.toFixed(1) : ""}</td><td></td>
          <td className="num"><strong>{pm(g.test_acc)}</strong></td>{top5 && <td className="num">{pm(g.test_top5)}</td>}<td></td></tr>)}
      </tbody></table>}
      <p className="muted small">s.d. is the sample standard deviation over the seeds. Runs that have not finished show their curves so far and no test accuracy.</p>
    </div>
  );
}

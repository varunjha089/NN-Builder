// Typed calls to the Rust side (Tauri commands in app/src-tauri/src/main.rs).
import { invoke } from "@tauri-apps/api/core";

export type Params = Record<string, unknown>;
/** the most blocks a design may have: the owner's value or the recommendation from this computer's memory */
export interface DesignSize { max_blocks: number; recommended: number; own: boolean; ram_gb: number }
export interface GNode { id: string; kind: string; params: Params; x: number; y: number; label?: string }
export interface GEdge { id: string; source: string; target: string }
export interface Graph { nodes: GNode[]; edges: GEdge[]; text?: string; python?: string }

export interface NodeResult {
  id: string; kind: string; title: string; inputs: number[][]; out: number[] | null; params: number;
  status: "ok" | "warn" | "error" | "blocked" | "unconnected"; messages: string[]; formula: string[];
  rf: [number, number] | null; padding: [number, number] | null; stride: [number, number] | null; macs: number;
}
export interface Analysis {
  input: number[]; classes: number; nodes: NodeResult[]; edges: Record<string, number[]>; total_params: number; total_macs: number; problems: string[]; ok: boolean;
  head_names?: string[]; frozen_params?: number;
}

export type LocalScan = { files: number; bytes: number; complete: boolean; tifs: number; npys: number; folders: { folder: string; classes: string[]; counts?: number[] }[];
  dirs: string[]; top: string[]; csvs: string[]; tif: { file: string; width: number; height: number; bands: number; bits: number; float: boolean; names: string[] } | null };

/** an extra output of a multi-task network: its targets from columns of the labels table */
export interface Head { name: string; task: "classify" | "multilabel" | "regress"; columns: string[]; classes: string[]; loss?: string; weight: number }

export interface Dataset {
  source: "builtin" | "huggingface" | "kaggle" | "local"; task: "classify" | "multilabel" | "segment" | "regress" | "detect"; resize: boolean; builtin: string; hf_mask_col: string; kaggle_mask_dir: string;
  hf_repo: string; hf_config: string; hf_train_split: string; hf_test_split: string; hf_image_col: string; hf_label_col: string;
  kaggle_slug: string; kaggle_train_dir: string; kaggle_test_dir: string;
  image_size: [number, number]; channels: number; classes: string[];
  /** multispectral / radar images: the bands the model uses (in order), the bands in the files, the sensor, Sentinel-1 in dB */
  bands?: string[]; band_order?: string[]; sensor?: "" | "s2" | "s1" | "other"; s1_db?: boolean;
  local_dir?: string; local_on_machine?: boolean; cloud_url?: string; layout?: "" | "scene"; scene_image?: string; scene_label?: string;
  scene_split?: "" | "strip" | "files"; scene_test_image?: string; scene_test_label?: string;
  /** multi-label: the CSV with each image's classes; fusion: a second band group from files with the same names in fuse_dir */
  labels_csv?: string; heads?: Head[]; label_column?: string; bands2?: string[]; band_order2?: string[]; sensor2?: "" | "s2" | "s1" | "other"; s1_db2?: boolean; fuse_dir?: string; image_size2?: number[]; time_steps?: number;
}

export interface TrainSettings {
  epochs: number; batch_size: number; lr: number; optimizer: string; seed: number; amp: boolean; limit_train: number; max_hours: number; stop_instance: boolean;
  aug_hflip: boolean; aug_vflip: boolean; aug_crop: number; aug_rotate: number; aug_color: boolean;
  checkpoint_minutes: number; max_resumes: number;
  val_fraction: number; early_stop: boolean; patience: number; monitor: "val_loss" | "val_acc" | "val_miou";
  weight_decay: number; schedule: "constant" | "cosine" | "plateau"; warmup_epochs: number; label_smoothing: number; grad_clip: number;
  pretrained: "" | "resnet18" | "vgg16" | "vit_b_16"; freeze: "none" | "backbone"; export: boolean; save_memory: boolean; auto_split: boolean;
  class_weights?: "" | "inverse" | "sqrt"; tune_thresholds?: boolean; loss?: "" | "focal" | "dice" | "dice_ce" | "mse" | "l1" | "huber" | "custom"; focal_gamma?: number; custom_loss?: string; gpus?: number;
}

export interface Machine {
  name: string; kind: "ssh" | "aws"; host: string; user: string; port: string; key: string; jump: string; auth: "key" | "password";
  workdir: string; python: string; launch: string; region: string; instance_id: string; mirror: boolean;
}
export interface Prepared {
  host_info: string; gpus: string[]; python: string; torch: string; cuda: boolean; slurm: boolean; systemd: boolean; linger: boolean; tmux: boolean; timeout: boolean;
}

export interface AppState {
  version: string; home: string; projects: { path: string; name: string; exists: boolean }[]; projects_dir: string;
  design_size: DesignSize;
  project: { name: string; path: string; draft: boolean } | null; tools_ready: boolean;
  secrets: { kaggle: string | null; huggingface: boolean; aws: string | null; gcs?: boolean };
  machines: { machine: Machine; prepared: Prepared | null }[];
  builtins: { key: string; label: string; shape: number[]; classes: string[] }[];
  kinds: { kind: string; title: string; defaults: Params }[];
  activations: string[];
}

export interface ProjectData { name: string; path: string; graph: Graph; dataset: Dataset; train: TrainSettings; analysis: Analysis; dataset_problems: string[] }

export interface RunInfo { id: string; target: string; kind: string; created: string; status: string; epochs: number; dataset: string; url: string; note: string;
  error: string; reachable: boolean; launch: string; job: string; host: string; spot: boolean; resumed_from: string; superseded_by: string; cancel: boolean; last_poll: number }
export interface RunStatus { id: string; state: string; info: RunInfo; events: Record<string, any>[]; next: number; log: string[]; has_results: boolean;
  /** train.py's last "checkpoint" event, and the copy on this Mac (when the machine keeps one) */
  checkpoint: Record<string, any> | null; mirror: { stamp?: string; copied?: string; bytes?: number; checkpoint?: Record<string, any>; error?: string } | null }
export interface JobSnap { name: string; state: string; message: string; log: string[]; next: number; seconds: number }

export interface Template { key: string; title: string; year: number; task: string; input: [number, number, number]; about: string }

export interface UpdateCheck { current: string; release: { version: string; newer: boolean; notes: string; page: string; asset: { name: string; size: number } | null } | null }

export const api = {
  updateSettings: () => invoke<{ auto: boolean; last_check: number | null; version: string }>("update_settings"),
  setAutoUpdate: (on: boolean) => invoke<void>("set_auto_update", { on }),
  setMaxBlocks: (n: number | null) => invoke<DesignSize>("set_max_blocks", { n }),
  updateCheck: (manual: boolean) => invoke<UpdateCheck | null>("update_check", { manual }),
  updateInstall: () => invoke<void>("update_install"),
  updateRestart: () => invoke<void>("update_restart"),
  templates: () => invoke<Template[]>("templates_list"),
  templateApply: (key: string) => invoke<string>("template_apply", { key }),
  projectFromTemplate: (key: string) => invoke<string>("project_from_template", { key }),
  state: () => invoke<AppState>("app_state"),
  projectNew: (name: string) => invoke<void>("project_new", { name }),
  projectSaveAs: (name: string) => invoke<string>("project_save_as", { name }),
  projectOpen: (path: string) => invoke<void>("project_open", { path }),
  projectDelete: (path: string) => invoke<string>("project_delete", { path }),
  setProjectsDir: (path: string) => invoke<void>("set_projects_dir", { path }),
  load: () => invoke<ProjectData>("project_load"),
  saveGraph: (graph: Graph) => invoke<{ analysis: Analysis; graph: Graph }>("save_graph", { graph }),
  designFromText: (text: string) => invoke<{ ok: boolean; graph?: Graph; analysis?: Analysis; errors?: { line: number; col: number; message: string }[] }>("design_from_text", { text }),
  designFormat: () => invoke<{ graph: Graph; analysis: Analysis }>("design_format"),
  saveDataset: (dataset: Dataset) => invoke<{ analysis: Analysis; problems: string[] }>("save_dataset", { dataset }),
  saveTrain: (train: TrainSettings) => invoke<TrainSettings>("save_train", { train }),
  code: () => invoke<{ ok: boolean; code?: string; error?: string }>("code"),
  exportCode: () => invoke<string>("export_code"),
  designPng: (dataUrl: string) => invoke<string>("design_png", { dataUrl }),
  setKaggle: (username: string, key: string) => invoke<void>("set_kaggle", { username, key }),
  setHf: (token: string) => invoke<void>("set_hf", { token }),
  setGcs: (key: string) => invoke<void>("set_gcs", { key }),
  setAws: (accessKeyId: string, secretAccessKey: string) => invoke<void>("set_aws", { accessKeyId, secretAccessKey }),
  clearSecret: (name: string) => invoke<void>("clear_secret", { name }),
  testKaggle: () => invoke<string>("test_kaggle"),
  testHf: () => invoke<string>("test_hf"),
  testAws: (name: string) => invoke<string>("test_aws", { name }),
  toolsInstall: () => invoke<void>("tools_install"),
  machineSave: (machine: Machine) => invoke<Machine>("machine_save", { machine }),
  machineDelete: (name: string) => invoke<void>("machine_delete", { name }),
  machineTest: (name: string) => invoke<string>("machine_test", { name }),
  machineConnect: (name: string) => invoke<string>("machine_connect", { name }),
  machinePrepare: (name: string) => invoke<void>("machine_prepare", { name }),
  hfPreview: (repo: string, config: string, split: string, labelCol: string, samples: boolean) =>
    invoke<{ frequencies: [string, number][]; samples: { src: string; label: unknown }[] }>("hf_preview", { repo, config, split, labelCol, samples }),
  hfInfo: (repo: string) => invoke<{ configs: { name: string; columns: { name: string; type: string; names: string[] | null }[]; splits: { name: string; n: number }[] }[] }>("hf_info", { repo }),
  kaggleFiles: (slug: string) => invoke<{ files: number; sample: string[]; folders: { folder: string; classes: string[]; counts?: number[] }[] }>("kaggle_files", { slug }),
  localScan: (dir: string) => invoke<LocalScan>("local_scan", { dir }),
  localCsv: (dir: string, csv: string) => invoke<{ columns: string[]; rows: number; file_column: string; label_column: string | null; classes: string[]; counts: number[]; numeric: { column: string; min: number; max: number; mean: number }[] }>("local_csv", { dir, csv }),
  pickFolder: (title: string) => invoke<string | null>("pick_folder", { title }),
  targets: () => invoke<{ name: string; kind: string; prepared?: boolean }[]>("targets"),
  runStart: (target: string, gpu: string, copyTokens: boolean, check = false) => invoke<void>("run_start", { target, gpu, copyTokens, check }),
  runs: () => invoke<{ id: string; target: string; kind: string; created: string; dataset: string; status: string; test_acc: number | null; seed: number; group: string; task?: string;
    check?: boolean; warnings?: number | null }[]>("runs_list"),
  runStatus: (id: string, since: number) => invoke<RunStatus>("run_status", { id, since }),
  runStop: (id: string) => invoke<string>("run_stop", { id }),
  runFetch: (id: string) => invoke<void>("run_fetch", { id }),
  runStartSeeds: (target: string, gpu: string, copyTokens: boolean, count: number) => invoke<void>("run_start_seeds", { target, gpu, copyTokens, count }),
  runsSummary: (ids: string[]) => invoke<{ rows: Record<string, any>[]; groups: Record<string, any>[] }>("runs_summary", { ids }),
  runsExport: (ids: string[]) => invoke<string>("runs_export", { ids }),
  runResumeOn: (id: string, target: string, gpu: string, copyTokens: boolean) => invoke<void>("run_resume_on", { id, target, gpu, copyTokens }),
  runResults: (id: string) => invoke<any>("run_results", { id }),
  runReveal: (id: string) => invoke<void>("run_reveal", { id }),
  job: (since: number) => invoke<JobSnap>("job_status", { since }),
  openUrl: (url: string) => invoke<void>("open_url", { url }),
};

export const fmt = (s: number[] | null | undefined) => (s ? "(" + s.map((x) => x.toLocaleString()).join(", ") + ")" : "?");
export const errText = (e: unknown) => (typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e));

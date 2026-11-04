// Typed calls to the Rust side (Tauri commands in app/src-tauri/src/main.rs).
import { invoke } from "@tauri-apps/api/core";

export type Params = Record<string, unknown>;
export interface GNode { id: string; kind: string; params: Params; x: number; y: number }
export interface GEdge { id: string; source: string; target: string }
export interface Graph { nodes: GNode[]; edges: GEdge[] }

export interface NodeResult {
  id: string; kind: string; title: string; inputs: number[][]; out: number[] | null; params: number;
  status: "ok" | "warn" | "error" | "blocked" | "unconnected"; messages: string[]; formula: string[];
  rf: [number, number] | null; padding: [number, number] | null; stride: [number, number] | null;
}
export interface Analysis {
  input: number[]; classes: number; nodes: NodeResult[]; edges: Record<string, number[]>; total_params: number; problems: string[]; ok: boolean;
}

export interface Dataset {
  source: "builtin" | "huggingface" | "kaggle"; task: "classify" | "segment"; resize: boolean; builtin: string; hf_mask_col: string; kaggle_mask_dir: string;
  hf_repo: string; hf_config: string; hf_train_split: string; hf_test_split: string; hf_image_col: string; hf_label_col: string;
  kaggle_slug: string; kaggle_train_dir: string; kaggle_test_dir: string;
  image_size: [number, number]; channels: number; classes: string[];
}

export interface TrainSettings {
  epochs: number; batch_size: number; lr: number; optimizer: string; seed: number; amp: boolean; limit_train: number; max_hours: number; stop_instance: boolean;
  aug_hflip: boolean; aug_vflip: boolean; aug_crop: number; aug_rotate: number; aug_color: boolean;
}

export interface Machine {
  name: string; kind: "ssh" | "aws"; host: string; user: string; port: string; key: string; jump: string; auth: "key" | "password";
  workdir: string; python: string; launch: string; region: string; instance_id: string;
}
export interface Prepared {
  host_info: string; gpus: string[]; python: string; torch: string; cuda: boolean; slurm: boolean; systemd: boolean; linger: boolean; tmux: boolean; timeout: boolean;
}

export interface AppState {
  version: string; home: string; projects: { path: string; name: string; exists: boolean }[]; projects_dir: string;
  project: { name: string; path: string; draft: boolean } | null; tools_ready: boolean;
  secrets: { kaggle: string | null; huggingface: boolean; aws: string | null };
  machines: { machine: Machine; prepared: Prepared | null }[];
  builtins: { key: string; label: string; shape: number[]; classes: string[] }[];
  kinds: { kind: string; title: string; defaults: Params }[];
  activations: string[];
}

export interface ProjectData { name: string; path: string; graph: Graph; dataset: Dataset; train: TrainSettings; analysis: Analysis; dataset_problems: string[] }

export interface RunInfo { id: string; target: string; kind: string; created: string; status: string; epochs: number; dataset: string; url: string; note: string;
  error: string; reachable: boolean; launch: string; job: string; host: string }
export interface RunStatus { id: string; state: string; info: RunInfo; events: Record<string, any>[]; next: number; log: string[]; has_results: boolean }
export interface JobSnap { name: string; state: string; message: string; log: string[]; next: number; seconds: number }

export interface Template { key: string; title: string; year: number; task: string; input: [number, number, number]; about: string }

export interface UpdateCheck { current: string; release: { version: string; newer: boolean; notes: string; page: string; asset: { name: string; size: number } | null } | null }

export const api = {
  updateSettings: () => invoke<{ auto: boolean; last_check: number | null; version: string }>("update_settings"),
  setAutoUpdate: (on: boolean) => invoke<void>("set_auto_update", { on }),
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
  setProjectsDir: (path: string) => invoke<void>("set_projects_dir", { path }),
  load: () => invoke<ProjectData>("project_load"),
  saveGraph: (graph: Graph) => invoke<Analysis>("save_graph", { graph }),
  saveDataset: (dataset: Dataset) => invoke<{ analysis: Analysis; problems: string[] }>("save_dataset", { dataset }),
  saveTrain: (train: TrainSettings) => invoke<TrainSettings>("save_train", { train }),
  code: () => invoke<{ ok: boolean; code?: string; error?: string }>("code"),
  exportCode: () => invoke<string>("export_code"),
  setKaggle: (username: string, key: string) => invoke<void>("set_kaggle", { username, key }),
  setHf: (token: string) => invoke<void>("set_hf", { token }),
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
  hfInfo: (repo: string) => invoke<{ configs: { name: string; columns: { name: string; type: string; names: string[] | null }[]; splits: { name: string; n: number }[] }[] }>("hf_info", { repo }),
  kaggleFiles: (slug: string) => invoke<{ files: number; sample: string[]; folders: { folder: string; classes: string[] }[] }>("kaggle_files", { slug }),
  targets: () => invoke<{ name: string; kind: string; prepared?: boolean }[]>("targets"),
  runStart: (target: string, gpu: string, copyTokens: boolean) => invoke<void>("run_start", { target, gpu, copyTokens }),
  runs: () => invoke<{ id: string; target: string; kind: string; created: string; dataset: string; status: string; test_acc: number | null }[]>("runs_list"),
  runStatus: (id: string, since: number) => invoke<RunStatus>("run_status", { id, since }),
  runStop: (id: string) => invoke<string>("run_stop", { id }),
  runFetch: (id: string) => invoke<void>("run_fetch", { id }),
  runResults: (id: string) => invoke<any>("run_results", { id }),
  runReveal: (id: string) => invoke<void>("run_reveal", { id }),
  job: (since: number) => invoke<JobSnap>("job_status", { since }),
  openUrl: (url: string) => invoke<void>("open_url", { url }),
};

export const fmt = (s: number[] | null | undefined) => (s ? "(" + s.map((x) => x.toLocaleString()).join(", ") + ")" : "?");
export const errText = (e: unknown) => (typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e));

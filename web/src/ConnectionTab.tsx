// Connection: everything needed to reach the GPUs. Tokens and keys go to the system keychain (never shown again, never in files);
// machines store only how to reach them.
import { useState } from "react";
import { api, errText, type AppState, type Machine } from "./api";

const EMPTY: Machine = { name: "", kind: "ssh", host: "", user: "", port: "", key: "", jump: "", auth: "key", workdir: "", python: "", launch: "auto", region: "", instance_id: "", mirror: false, group: "", resume_reboot: true };

const POLICY = `{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["ec2:StartInstances", "ec2:StopInstances"],
    "Resource": "arn:aws:ec2:REGION:ACCOUNT:instance/INSTANCE-ID"
  }, {
    "Effect": "Allow",
    "Action": ["ec2:DescribeInstances", "ec2:DescribeInstanceAttribute", "ec2:DescribeSpotInstanceRequests"],
    "Resource": "*"
  }]
}`;

export function ConnectionTab({ st, reload, toast, startJob }: { st: AppState; reload: () => void; toast: (m: string) => void; startJob: () => void }) {
  const [ku, setKu] = useState(""); const [kk, setKk] = useState("");
  const [hf, setHf] = useState("");
  const [gk, setGk] = useState("");
  const [ak, setAk] = useState(""); const [as, setAs] = useState("");
  const [m, setM] = useState<Machine>(EMPTY);
  const [editing, setEditing] = useState(false);
  const run = (p: Promise<unknown>, ok?: string) => p.then((r) => { toast(typeof r === "string" ? r : ok ?? "Done."); reload(); }).catch((e) => toast(errText(e)));
  const up = (p: Partial<Machine>) => setM({ ...m, ...p });

  return (
    <div className="page conn">
      <h2>Connection</h2>
      <p className="muted">How NN Builder reaches the GPUs. Tokens and keys are kept in the macOS Keychain and handed to one call at a time; they are never written into projects, code or logs.</p>

      <section className="card">
        <h3>1. Tools on this Mac</h3>
        <p className="muted small">The official <code>kaggle</code> command and <code>boto3</code> (AWS), in a small Python environment of the app (about 60 MB). They only talk to Kaggle and AWS; nothing trains here.</p>
        <p className="row">{st.tools_ready ? <span className="st-ok">Installed.</span> : <span className="st-warn">Not installed yet.</span>}
          <button onClick={() => api.toolsInstall().then(startJob).catch((e) => toast(errText(e)))}>{st.tools_ready ? "Update" : "Install"} tools</button></p>
      </section>

      <section className="card">
        <h3>2. Kaggle</h3>
        <p className="muted small">kaggle.com → your profile → Settings → API → Create New Token. Paste the username and key, or the whole contents of kaggle.json into the key field. The account must be phone-verified for GPU notebooks.</p>
        {st.secrets.kaggle ? <p className="row"><span className="st-ok">Saved for {st.secrets.kaggle}.</span>
          <button onClick={() => run(api.testKaggle())} disabled={!st.tools_ready}>Test</button>
          <button onClick={() => run(api.clearSecret("kaggle"), "Removed from the Keychain.")}>Remove</button></p>
          : <div className="row"><input placeholder="username" value={ku} onChange={(e) => setKu(e.target.value)} />
            <input type="password" className="grow" placeholder="API key (or kaggle.json contents)" value={kk} onChange={(e) => setKk(e.target.value)} />
            <button onClick={() => run(api.setKaggle(ku, kk).then(() => { setKu(""); setKk(""); }), "Saved in the Keychain.")}>Save</button></div>}
      </section>

      <section className="card">
        <h3>3. Hugging Face</h3>
        <p className="muted small">Only for gated or private datasets (and to read their information): huggingface.co → Settings → Access Tokens → a "Read" token.</p>
        {st.secrets.huggingface ? <p className="row"><span className="st-ok">Token saved.</span>
          <button onClick={() => run(api.testHf())}>Test</button><button onClick={() => run(api.clearSecret("huggingface"), "Removed from the Keychain.")}>Remove</button></p>
          : <div className="row"><input type="password" className="grow" placeholder="hf_…" value={hf} onChange={(e) => setHf(e.target.value)} />
            <button onClick={() => run(api.setHf(hf).then(() => setHf("")), "Saved in the Keychain.")}>Save</button></div>}
      </section>

      <section className="card">
        <h3>Google Cloud Storage (optional)</h3>
        <p className="muted small">Only for data in a private Cloud Storage bucket (Earth Engine's Cloud Storage export, Dataset tab → My own files → in the cloud):
          a service-account key (Google Cloud console → IAM → Service accounts → Keys → JSON) with read access to the bucket. It is copied to an
          SSH / AWS machine when a run starts (like the other tokens); Kaggle notebooks can read public buckets only.</p>
        {st.secrets.gcs ? <p className="row"><span className="st-ok">Key saved.</span>
          <button onClick={() => run(api.clearSecret("gcs"), "Removed from the Keychain.")}>Remove</button></p>
          : <div className="row"><textarea className="grow" rows={2} placeholder='the key file contents: {"type": "service_account", …}' value={gk} onChange={(e) => setGk(e.target.value)} />
            <button onClick={() => run(api.setGcs(gk).then(() => setGk("")), "Saved in the Keychain.")}>Save</button></div>}
      </section>

      <section className="card">
        <h3>4. AWS (for EC2 g4dn.xlarge)</h3>
        <p className="muted small">An IAM user's access key that may only start, stop and describe your instance. The instance's "shutdown behaviour" must be Stop (not Terminate): "Check instance" below tells you.</p>
        {st.secrets.aws ? <p className="row"><span className="st-ok">Access key {st.secrets.aws} saved.</span>
          <button onClick={() => run(api.clearSecret("aws"), "Removed from the Keychain.")}>Remove</button></p>
          : <div className="row"><input placeholder="access key id (AKIA…)" value={ak} onChange={(e) => setAk(e.target.value)} />
            <input type="password" className="grow" placeholder="secret access key" value={as} onChange={(e) => setAs(e.target.value)} />
            <button onClick={() => run(api.setAws(ak, as).then(() => { setAk(""); setAs(""); }), "Saved in the Keychain.")}>Save</button></div>}
        <details><summary className="small">IAM policy for that user (replace REGION, ACCOUNT, INSTANCE-ID)</summary><pre className="small">{POLICY}</pre></details>
      </section>

      <section className="card">
        <h3>5. Machines (college server, AWS instance)</h3>
        {st.machines.map(({ machine: x, prepared: p }) => (
          <div key={x.name} className="machine">
            <div><strong>{x.name}</strong> <span className="muted">{x.kind === "aws" ? `AWS ${x.instance_id} (${x.region}) · ${x.user}` : `${x.user ? x.user + "@" : ""}${x.host}${x.port ? ":" + x.port : ""}`}
              {" · "}{x.auth === "password" ? "password" : x.key ? `key ${x.key}` : "key from ssh config/agent"}{x.jump ? ` · via ${x.jump}` : ""} · launch: {x.launch}
              {x.group ? ` · group ${x.group}` : ""}{x.resume_reboot ? " · resumes after a reboot" : ""}</span></div>
            <div className="muted small">{p ? `Prepared: PyTorch ${p.torch} (${p.cuda ? "CUDA" : "CPU only"}) · ${p.gpus.length ? p.gpus.join(" | ") : "no NVIDIA GPU"} · `
              + `keeps running via: ${p.slurm ? "SLURM" : p.systemd && p.linger ? "systemd (lingering)" : p.tmux ? "tmux" : "nohup"}` : "Not prepared yet (Prepare installs PyTorch there; the first run does it too)."}</div>
            <p className="row">
              {x.auth === "password" && <button onClick={() => run(api.machineConnect(x.name))}>Connect (log in once)</button>}
              {x.kind === "ssh" && <button onClick={() => run(api.machineTest(x.name))}>Test</button>}
              {x.kind === "aws" && <button onClick={() => run(api.testAws(x.name))} disabled={!st.secrets.aws || !st.tools_ready}>Check instance</button>}
              <button onClick={() => api.machinePrepare(x.name).then(startJob).catch((e) => toast(errText(e)))}>Prepare{x.kind === "aws" ? " (starts and stops the instance)" : ""}</button>
              <button onClick={() => { setM({ ...EMPTY, ...x }); setEditing(true); }}>Edit</button>
              <button onClick={() => { if (confirm(`Remove "${x.name}" from this list? Nothing on the machine is touched.`)) run(api.machineDelete(x.name), "Removed."); }}>Delete</button>
            </p>
          </div>
        ))}
        {!st.machines.length && <p className="muted">No machines yet.</p>}
        <h4>{editing ? `Edit ${m.name}` : "Add a machine"}</h4>
        <div className="form">
          <label>Type<select value={m.kind} onChange={(e) => up({ kind: e.target.value as Machine["kind"] })}><option value="ssh">College / any SSH server</option><option value="aws">AWS EC2 instance</option></select></label>
          <label>Name<input value={m.name} onChange={(e) => up({ name: e.target.value })} placeholder={m.kind === "aws" ? "aws-g4dn" : "college-gpu"} /></label>
          {m.kind === "ssh" && <>
            <label>Host<input value={m.host} onChange={(e) => up({ host: e.target.value })} placeholder="gpu01.college.edu, an IP, or an ~/.ssh/config alias" /></label>
            <label>User<input value={m.user} onChange={(e) => up({ user: e.target.value })} /></label>
            <label>Port<input value={m.port} onChange={(e) => up({ port: e.target.value })} placeholder="22" /></label>
            <label>Login<select value={m.auth} onChange={(e) => up({ auth: e.target.value as Machine["auth"] })}><option value="key">Key / ssh config</option><option value="password">Password (log in once in Terminal)</option></select></label>
            <label>Jump host<input value={m.jump} onChange={(e) => up({ jump: e.target.value })} placeholder="optional: user@gateway.college.edu" /></label>
          </>}
          {m.kind === "aws" && <>
            <label>Region<input value={m.region} onChange={(e) => up({ region: e.target.value })} placeholder="ap-south-1" /></label>
            <label>Instance id<input value={m.instance_id} onChange={(e) => up({ instance_id: e.target.value })} placeholder="i-0123456789abcdef0" /></label>
            <label>User<input value={m.user} onChange={(e) => up({ user: e.target.value })} placeholder="ubuntu" /></label>
          </>}
          <label>Key file<input value={m.key} onChange={(e) => up({ key: e.target.value })} placeholder={m.kind === "aws" ? "~/.ssh/my-key.pem (required)" : "optional: ~/.ssh/id_ed25519"} /></label>
          <label>Keep runs going with<select value={m.launch} onChange={(e) => up({ launch: e.target.value })}>
            <option value="auto">automatic (best the machine allows)</option><option value="slurm">SLURM (sbatch)</option><option value="systemd">systemd-run</option>
            <option value="tmux">tmux</option><option value="nohup">nohup + setsid</option></select></label>
          <label>Remote folder<input value={m.workdir} onChange={(e) => up({ workdir: e.target.value })} placeholder=".nn_builder" /></label>
          <label className="chk wide"><input type="checkbox" checked={!!m.mirror} onChange={(e) => up({ mirror: e.target.checked })} />
            Keep a copy of checkpoints on this Mac (copied whenever a new one is written, while the app is open; recommended for EC2 spot instances:
            if AWS terminates the instance, the run can continue on another machine from this copy)</label>
          <label>Group<input value={m.group ?? ""} onChange={(e) => up({ group: e.target.value })} placeholder="optional: lab (machines of one group form a pool)" /></label>
          {m.kind === "ssh" && <label className="chk wide"><input type="checkbox" checked={!!m.resume_reboot} onChange={(e) => up({ resume_reboot: e.target.checked })} />
            Resume runs after a reboot: a run cut off by a restart or a power cut starts again from its last checkpoint when the machine is back
            (train.py saves one every few minutes; adds an @reboot line to your crontab there), also while this Mac is off</label>}
          <label className="wide">Python with PyTorch on the machine<input value={m.python} onChange={(e) => up({ python: e.target.value })} placeholder="optional; empty = the app makes ~/.nn_builder/venv (no sudo)" /></label>
        </div>
        <p className="row"><button className="primary" onClick={() => run(api.machineSave(m).then(() => { setM(EMPTY); setEditing(false); }),
          m.kind === "aws" ? "Saved. Next: Check instance, then Prepare." : m.auth === "password" ? "Saved. Next: Connect, then Test." : "Saved. Next: Test, then Prepare.")}>Save</button>
          {editing && <button onClick={() => { setM(EMPTY); setEditing(false); }}>Cancel</button>}</p>
      </section>
    </div>
  );
}

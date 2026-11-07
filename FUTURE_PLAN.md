# Future plan

What is still open, in the suggested order. Done work is in git and CLAUDE.md; this file lists only what is left.

## 0. Next six tasks (asked 2026-11-06, in this order)

1. **Check data** (done 2026-11-06): a run that only loads the data and runs the one-batch check on the training machine, no training;
   it reports files per class, band names and value ranges per band, empty pixels, CSV rows without a file, already-dB radar,
   constant or all-empty bands, class balance, and shows a few composites; minutes instead of a wasted GPU hour.
2. **Unbalanced classes** (done 2026-11-06): class counts from Read folder / Read labels in the Dataset tab, an option to weight the loss by class
   (inverse frequency; BCE pos_weight for multi-label), and per-class thresholds for multi-label chosen on the validation data.
3. **Finish the Sentinel step** (done 2026-11-06): a spatially separate test region for scenes (honest scores), BigEarthNet's own layout (a folder of
   single-band GeoTIFFs per patch), ImageNet weights for more than 3 bands (first convolution expanded), Hugging Face band arrays.
4. **Google Drive / Cloud Storage source** (done 2026-11-06; real buckets / links not tried): Earth Engine exports reach the training machine directly (rclone / gsutil on the machine),
   without passing through the Mac.
5. **Real-data run** (done 2026-11-06 for Hugging Face giswqs/EuroSAT_MS, 13 bands, on a Kaggle T4): Check data passed (no warning, 10
   classes, realistic band values); 5 epochs of a 143,594-parameter CNN: test accuracy 92.1 % (top-5 99.8 %), 17 s per epoch, 0.15 GB
   GPU memory. Found and fixed: Kaggle refused notebooks whose title was over 50 characters (400 Bad Request). Seen: data_wait 0.65
   (the GPU waits for the data loader about two thirds of the time: Hugging Face band arrays decoded per image). Still not tried: an
   Earth Engine export of the owner's own (the GEE script, own files upload to Kaggle).
6. **Release 0.4.0** (released 2026-11-06) with housekeeping: upload-artifact action update (Node.js 20 warning), delete the local backup branch, the first
   real updater test (see section 1).

## 0b. Why the architecture steps (section 1b) are in this order

- First the real-data run and release 0.4.0 (section 0, tasks 5 and 6): nothing below is worth building on untested data paths.
- Losses (2) are the smallest change and help the owner's segmentation now; they also define the per-output loss that steps 4 and 8 use.
- The custom block (3) comes next because it unblocks everything the app lacks while the rest is built, and its one-batch shape check
  is reused by every later block.
- Several inputs / outputs (4) is the big structural change (a batch becomes a dict); weight sharing (5), cross-attention (7) and
  detection heads (8) all need it, so it goes before them.
- Time series (6) after 4–5 because it reuses the multi-input data path and pairs; it is the step most useful for remote sensing.
- Transformer additions (7) after 6, because masks are needed there for missing dates and cross-attention needs 4.
- Detection (8) needs losses (2), several heads (4) and a new dataset kind, so it is late.
- Multi-GPU (9) is independent of the model; it pays off once large ViTs and time series exist.
- Fine-tuning and MoE (10) are refinements on top of everything else.

## 1. Needs the owner (cannot be done from here)

- **Stable signing certificate.** Create a self-signed code-signing certificate in Keychain Access; the build then uses it through an
  environment variable (never in the repository). Ends the Keychain "Always Allow" question after every update.
- **First real test of the automatic updater.** 0.4.0 is released (2026-11-06); the app installed on the owner's Mac is the same code
  but still says 0.3.0, so Help → Check for Updates → Install and restart there is the test.
- **First real runs of the new training features**: done on Kaggle T4s on 2026-11-06/07 (ImageNet weights, LoRA, TorchScript / ONNX
  export, save memory 7.22 → 2.18 GB, measured memory and data wait, several GPUs; see TEST_APP.md). Still never seen for real: bfloat16
  (needs an A100-class GPU) and the automatic batch split on a real out-of-memory.
- **The tests that need the owner's accounts or machines**: section 1c.

## 1d. Found by the real-data tests of 2026-11-07, not done yet

- **mIoU counts absent classes as 0.** A class that is neither in the test pixels nor predicted gets IoU 0 and lowers the mean (TEST 19:
  0.148 over 11 WorldCover classes, 4 of which do not occur in the tile). Usual practice leaves such classes out of the mean; show both.
- **Built-in datasets download slowly on EC2** (CIFAR-10 from its university server at 170–370 kB/s: 10 paid minutes of an idle GPU).
  Use a fast mirror (e.g. the Hugging Face copies) or fetch the data while the instance boots.
- **Every update asks for Keychain access again** (unsigned builds): a stable signing certificate (section 1) ends it.

## 1c. Tests that need the owner's accounts (written 2026-11-07)

**State at the end of 2026-11-07:**

| Test | State |
|---|---|
| 1. 4.3 Earth Engine script | **done**: the app's script ran in the Code Editor (its two exports queued behind the owner's thesis tasks) and in the Python API (exported, read on Kaggle, trained); 40 existing S1+S2 exports also read |
| 2. 4.4 cloud folder | **done for a Drive link** (Kaggle fetched the export itself); `gs://` and rclone not tried |
| 3. 6.2 college server | open: needs the server in the Connection tab |
| 4. 6.4 several-GPU stop | open: needs a machine with ≥ 2 GPUs (the college server) |
| 5. 6.5 AWS on demand | **done** (AWS_TEST_PLAN.md section 7, $0.28) |
| 6. 6.6 / 6.7 spot | open: no g4dn spot capacity in Mumbai on 2026-11-07; retry planned for 5:00 on 2026-11-08 (AWS_TEST_PLAN.md section 8) |

The steps below are the original plan, kept for the open tests.

On 2026-11-07 every TEST_APP.md test that the Mac and Kaggle allow was done: 69 ✅, with the bugs found fixed. The six below remain.
Each needs an account, a machine or money that only the owner has. Their logic is already covered by CI and the stand-in tests
(`crates/core/tests/flows.rs`), so what is left is the real service. Do them in this order (free first, money last). Use a TEST project
(Projects → New project), never a real one. Afterwards, mark the TEST_APP.md line and add a row to its Results log with the run id.

**1. Earth Engine script (TEST_APP 4.3).** Needs a Google Earth Engine account (free for research; sign up with a Cloud project).
- Steps: Dataset tab → My own files → "Get Sentinel-1/2 data from Google Earth Engine". Choose a small area (one tile), Sentinel-2 + WorldCover,
  and Export to Drive. Paste the script into code.earthengine.google.com and run it. Start the export task from the Tasks panel.
- Expect: the GeoTIFFs land in the Drive folder with the band order the tab shows. Once the folder is downloaded, Read folder lists the bands
  (GDAL names from TIFF tag 42112), and "Set this tab to match the export" fills them in.
- Pointers: `web/src/DatasetTab.tsx` GeeBox (the generated script); `crates/core/src/localdata.rs` tiff_info; manual sections "Your own files,
  e.g. Sentinel-1/2 from Google Earth Engine" and "A large exported scene with a label image".
- Then run Check data and a few epochs on Kaggle: this also tests the private Kaggle dataset upload with real exports.

**2. Cloud folder as the source (TEST_APP 4.4).** Needs the export from step 1 in Google Drive (a shared link) or a Cloud Storage bucket.
- Steps: Dataset → My own files → "in the cloud", then paste one of:
  - a Drive folder link (gdown reads at most 50 files per folder, so prefer a zip of the export);
  - `gs://bucket/folder` (Connection tab: a service-account JSON key, stored in the Keychain as "gcs");
  - `rclone:remote:path` (rclone set up on the training machine).

  Then Check data on Kaggle.
- Expect: the machine fetches into `<data>/cloud/<key>` with no upload from the Mac, and the data check has no warnings.
- Pointers: `python/harness.py` cloud_fetch(); `crates/core/src/dataset.rs` cloud_url_problem; GeeBox "Export to: Cloud Storage" (nnb_export/
  prefix). CI tests only `file://`.
- A private bucket needs the key on the machine. The app copies it only to SSH / AWS machines ("copy tokens"; `runs.rs` put_secret_script).
  On Kaggle only a public bucket or a shared Drive link works (a gap to close later: Kaggle Secrets). Check that the key never appears in
  the log.

**3. College server over SSH (TEST_APP 6.2).** Needs the server's address and an account. A key login is easier; a password works through
a one-time Terminal login.
- Steps: Connection → add a machine (host or `~/.ssh/config` alias, user, port, key file or "password", jump host if needed) → Connect
  (a password is typed in the Terminal window only) → Prepare (Python, GPUs, disk, venv with PyTorch, no sudo) → a short Train.
- Expect:
  - Prepare reports the GPUs and the launch method (SLURM `sbatch` > `systemd-run --user` > tmux / nohup) and warns if logout kills
    processes.
  - The run keeps going after the app quits and the Mac sleeps. Reopening the app reconnects, the curves continue, and the results come back.
- Pointers: `crates/core/src/remote.rs` (connect_command, prepare_script + parse_prepare, choose_launch, put_secret_script); `runs.rs` start /
  status; manual sections "The Connection tab" and "Independence from the connection".
- Also answers the open CLAUDE.md questions: does the server use SLURM? Is lingering on?

**4. Stop a several-GPU run (TEST_APP 6.4).** Needs a machine with ≥ 2 GPUs that the app can stop: the college server (step 3) or AWS.
Kaggle notebooks cannot be stopped through the API, so Kaggle T4 x2 does not count.
- Steps: GPUs = all, epochs ≥ 5 → Train → press Stop in the middle of an epoch.
- Expect: all workers end (no python process left: check `nvidia-smi`), and results.json is written from the epochs done. The parent passes
  SIGTERM to every rank, and rank 0 checkpoints and writes the results.
- Pointers: `python/harness.py` run_parallel (signal forwarding, workers ended a minute after rank 0, destroy_process_group), sync_flags;
  `runs.rs` stop. 2026-11-06 found that the workers did not exit on Kaggle T4 x2 (fixed). Watch for the same here.

**5. AWS on demand (TEST_APP 6.5).** Planned in detail, with cost guards, in [AWS_TEST_PLAN.md](AWS_TEST_PLAN.md) (owner's permission: g4dn.xlarge
on demand in Mumbai only). Costs money: g4dn.xlarge is about $0.53 an hour on demand in us-east-1 (check the region's price).
- Needs:
  - a g4dn.xlarge instance with shutdown behaviour **stop**;
  - a G-instance vCPU quota above 0 (new accounts often have 0: ask AWS for 4);
  - a key pair (.pem file path);
  - an IAM access key allowed only `ec2:StartInstances` / `StopInstances` / `DescribeInstances` on that instance, plus
    `ec2:DescribeSpotInstanceRequests`.
- Steps: Connection → AWS (access key into the Keychain; instance id, region, key file) → Train a short run with a maximum run time set.
- Expect: the app starts the instance, prepares it once and runs. The run ends with `shutdown -h now`, so the instance is stopped in the
  console. Results opened later start, copy and stop the instance again.
- Pointers: `python/helpers/aws_ec2.py`; `runs.rs` aws_up, fetch; CLAUDE.md "AWS EC2 g4dn.xlarge".
- Check in the console afterwards that the instance is **stopped**, not running.

**6. AWS spot and Continue elsewhere (TEST_APP 6.6, 6.7).** The owner's key may request spot instances (plan and first attempt: AWS_TEST_PLAN.md section 8). Costs money, less than on demand.
- Needs: a **persistent** spot request with interruption behaviour **stop**, and the machine setting "Keep a copy of checkpoints on this Mac".
- Steps for 6.6: a run of 20+ minutes. Simulate an interruption with AWS Fault Injection Service (action `aws:ec2:send-spot-instance-interruptions`),
  or wait for a real one.
- Expect for 6.6: a mid-epoch checkpoint and exit 75. When the instance returns, crontab `@reboot` → resume.sh continues at the same batch, also
  while the Mac is off. used_seconds.txt counts the time over resumes.
- Steps for 6.7: with a one-time request (or behaviour terminate), let it be terminated → Continue on another machine (Kaggle or the college
  server).
- Expect for 6.7: a new run from the Mac copy of the checkpoint, and the old run marked "continued".
- Pointers: CLAUDE.md "Spot instances" (the table of situations); `runs.rs` run_script, RESUME_SH, resume_on, mirror_checkpoint; manual
  section "EC2 spot instances".
- Afterwards, cancel the spot request in the console. The app cannot cancel it, and a persistent request restarts the instance.

**Cleanup (TEST_APP 1.3).** The TEST projects stay (the owner keeps their designs). Test Delete on a new throwaway project instead
(Projects → Delete: the folder goes to the Trash). The Kaggle notebooks `nnb-test-*` and private datasets `nnb-*` can stay or be deleted on
kaggle.com (Your Work).

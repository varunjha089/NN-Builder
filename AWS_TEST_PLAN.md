# AWS_TEST_PLAN.md: first real EC2 tests, without surprises on the bill

Written 2026-11-07, before any AWS action. The owner's AWS permission covers **one instance type, g4dn.xlarge (1 × NVIDIA T4 16 GB,
4 vCPU, 16 GB RAM), in Mumbai (ap-south-1), on demand, and nothing else**. So there are no spot instances, no larger GPU machines and no
other regions. This file says what will be tested, in which order, what it may cost, and what stops the cost from growing. Nothing in it
has been done yet; each phase needs the owner's go-ahead.

Who does what:

- **The owner** acts in the AWS console:
  - launches the instance;
  - sets the budget;
  - pastes the access key into the app (Connection → AWS; it goes to the Keychain);
  - looks at the bill.
- **Claude** works only through the app:
  - starts and stops that one instance;
  - runs and watches the tests;
  - checks after each phase that the instance is stopped;
  - never opens or prints the access key, and never creates, changes or deletes anything in the account.

---

## 1. What it costs (ap-south-1, check the EC2 pricing page before starting)

| Item | Price (approximate) | When it is charged |
|---|---|---|
| g4dn.xlarge on demand | about **$0.58 per hour** (per second, 1-minute minimum) | only while the instance is **running** |
| Public IPv4 address | $0.005 per hour | only while running (an Elastic IP is not needed: do not allocate one) |
| Disk (EBS gp3) | about $0.09 per GB per month | **all the time the instance exists, also when stopped** (75 GB ≈ $7 a month) |
| Data in (PyTorch, CIFAR-10 download) | free | |
| Data out (results, a few MB) | free (first 100 GB a month) | |
| Snapshots, Elastic IP, load balancers, other services | none are used | |

**Planned use: about 55–70 minutes of running time ≈ $0.55–0.70, plus the disk while the instance exists.**

**The worst case these guards prevent:** an instance left running, at $0.58 × 24 ≈ **$14 a day**.

## 2. Setup by the owner (once, in the AWS console, region Mumbai ap-south-1)

1. **Budget alert:** Billing → Budgets → a monthly cost budget of **$5**, with e-mail alerts at 50 %, 80 % and 100 % (actual) and at
   100 % (forecast). Budgets are free. Skip this if the account has no access to Billing, but then step 7 matters more.
2. **Key pair:** EC2 → Key pairs → create (RSA, .pem). Keep the .pem outside any repository, with `chmod 400`.
3. **Security group:** inbound SSH (port 22) from **My IP** only, nothing else open.
4. **Launch one instance:**
   - AMI: **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04)**, the newest date (20261024 when this was written; it
     lists G4dn as supported). In the launch wizard, search the AMI catalogue for "Deep Learning OSS Nvidia Driver AMI GPU PyTorch". The
     AMI already has the NVIDIA driver, CUDA and PyTorch 2.12 (Python 3.13, `/opt/pytorch/bin/python`), so Prepare spends about a minute,
     not ~10 paid minutes. The app finds that Python by itself (leave the machine's "Python" field empty). It sources
     `/opt/pytorch/bin/activate`, which this AMI needs for CUDA, and installs only `datasets` and `kaggle`, with `sudo` if needed.
     The login user is **ubuntu**.
   - Instance type **g4dn.xlarge**, purchasing option **on demand** (no spot), the key pair and the security group above.
   - Storage: the AMI's minimum (often 75 GB gp3). More disk is not needed for these tests.
   - Advanced details:
     - **Shutdown behavior: Stop** (never Terminate: terminate deletes the disk);
     - **Termination protection: Enable** (free; prevents a deletion by mistake);
     - no Elastic IP.
5. **Stop the instance at once after the launch** (Instance state → Stop): a new instance starts running and charging immediately. From
   now on, only the app starts it.
6. **Access key with the smallest rights:** IAM → a user (e.g. `nnb-app`) with only this policy, then an access key for it. Put the
   account number and the instance id in place of the capitals.

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       { "Effect": "Allow",
         "Action": ["ec2:StartInstances", "ec2:StopInstances"],
         "Resource": "arn:aws:ec2:ap-south-1:ACCOUNT_ID:instance/INSTANCE_ID" },
       { "Effect": "Allow",
         "Action": ["ec2:DescribeInstances", "ec2:DescribeInstanceAttribute"],
         "Resource": "*" }
     ]
   }
   ```

   With this key, nothing can launch a new machine, change the account, or touch any other instance, even by mistake.
   (`Describe*` cannot be limited to one instance in IAM; it only reads.) If the key the owner already has is a root or administrator key,
   do not give it to the app: make this one instead.
7. **The owner's own check:** the EC2 console's instance list (ap-south-1) shows the state. "Stopped" costs only the disk.

## 3. Changes to the app before the first run (no AWS cost; Claude, with tests) — done 2026-11-07, tested with the stand-in EC2 in flows.rs

Also done for the Deep Learning AMI: Prepare uses `/opt/pytorch/bin/python` when no Python is set, Prepare and run.sh source the
environment's `activate` (CUDA lives inside it since PyTorch 2.9), and pip retries with `sudo -n` when it may not write there
(design.rs prepare_and_run_activate_the_python_environment).

- **Must fix first: a failure after the start leaves the instance running.** Found while writing this plan (2026-11-07).
  - `runs::start` / `start_remote` and `resume_on`: when anything fails after `aws_up` has started the instance, the run is marked
    "failed" but the instance is not stopped. Examples: ssh never answers, no public address, Prepare, the upload, the launch.
  - The same happens when `aws_up` itself fails after the start, in the Prepare button (`main.rs` machine_prepare) and in `fetch`.
  - Fix: one guard that stops the instance on every error path after a start, unless train.py was really launched. The log says
    "Stopping the instance again"; a failed stop says plainly "STOP IT IN THE CONSOLE".
  - Test: flows.rs with the stand-in EC2 (the start succeeds, ssh fails, so the stand-in must receive "stop").
- **Refuse to start a run when the instance's shutdown behaviour is "terminate".** `python/helpers/aws_ec2.py` already reads it
  (`shutdown_behavior`), but `runs.rs` does not check it yet.
- **Add `ec2:DescribeInstanceAttribute` to the documented key policy** (CLAUDE.md, the manual's Connection section). The helper reads the
  shutdown behaviour with it and quietly skips the check without it.
- **Short test runs:** Train tab "End the run after (hours of running)" (TrainSettings.max_hours, default 6) 0.5 h for these tests (the app accepts 0.1–72 h).
  run.sh ends train.py at the limit (`timeout --signal=TERM`), and the instance then shuts down by itself.

## 4. The tests, in order (one at a time; each phase starts only after the owner says go)

Every test uses a TEST project with the small CIFAR-10 CNN (built-in dataset, downloaded on the instance), batch 64, 10,000 training images,
"Stop the EC2 instance when the run ends" on, maximum run time 0.5 h.

| Phase | Test (TEST_APP.md) | What happens | Running time | Cost |
|---|---|---|---|---|
| 0 | Describe only | Connection → the AWS machine → Check instance: the key works, region, state **stopped**, shutdown behaviour **stop**, type g4dn.xlarge | 0 (the instance stays stopped) | $0 |
| 1a | Prepare | Connection → Prepare: the app starts the instance, checks GPU / driver / the AMI's PyTorch, stops it again. Must report "CUDA" and "Tesla T4": "CPU only" = stop here | ~5 min | ~$0.05 |
| 1 | 6.5 AWS on demand | Train: the app starts the instance, Prepare (GPU, driver, Python with PyTorch), 2 epochs, results copied back, `shutdown -h now` | ~20–30 min | ~$0.25 |
| 2 | Results from a stopped instance | Open the finished run again after it stopped: if anything was not yet copied, the app starts the instance, copies and stops it | ~3–5 min | ~$0.05 |
| 3 | Stop button on AWS | A 10-epoch run, press Stop after the first epoch: train.py checkpoints, writes results, the instance stops | ~10–15 min | ~$0.15 |
| 4 | Closing the app during a run | A 3-epoch run, quit NN Builder while it trains, reopen after ~10 min: the run went on without the Mac and the instance stopped itself | ~15 min | ~$0.15 |

Total **~55–70 minutes ≈ $0.55–0.70** plus the disk.

**Not possible with this permission:**

- 6.4, stop on several GPUs: needs a machine with ≥ 2 GPUs (the college server, or g4dn.12xlarge, not allowed).
- 6.6, spot interruption, and 6.7, continue elsewhere: need spot instances, and 6.7 also a terminated instance.

These stay "Never" in TEST_APP.md (CI covers their logic).

## 4b. No idle instance, data ready in time, a 10-second watch (owner's wishes, 2026-11-07)

- **Idle time:** the phases follow each other at once, and no instance waits for a person:
  - every run ends with `shutdown -h +2`: two minutes for the app to copy the results, then the instance stops itself;
  - Prepare stops the instance as soon as it has checked it;
  - a failure stops it at once (section 3).

  The only idle time is these 2 minutes per run and the boot (about 1–2 minutes, longer on an AMI's first boot), about $0.02 each.
- **Data:** every phase uses CIFAR-10 (170 MB). The first run downloads it on the instance into `~/.nn_builder/data`, on the instance's
  disk, so the later phases find it there and download nothing. No other dataset is needed. Nothing is downloaded to the Mac, and the
  Mac never trains.
- **Watching:** while a phase runs, Claude checks every 10 s:
  - the run's state, its last epoch and its log;
  - the instance's state.

  The app itself polls the run every 3 s over ssh, and AWS at most every 30 s while the instance is down. Claude acts on the abort rules
  below at the first sign, without waiting for the run to end.

## 4c. Before Phase 0 (state on 2026-11-07)

- [x] App fixes (section 3), the Deep Learning AMI support and a 6-minute ssh wait for a slow first boot (no extra cost: the instance
      boots anyway).
- [x] Access key saved in the app (Keychain).
- [ ] **That key is exposed**: a notes file stores its secret in plain text, and it was read into a Claude conversation. Rotate it:
      IAM → the user → Security credentials → create a new access key, put it into the app, then deactivate and delete the old one.
      Remove the secret from the notes file.
- [ ] **That key has broad rights** (launch / terminate instances, delete volumes, every S3 bucket). The app needs only start / stop /
      describe. Best: a new IAM user with the policy of section 2, step 6. At least add `ec2:DescribeInstanceAttribute`, or the app
      cannot check the shutdown behaviour and only warns.
- [ ] The instance (Mumbai, g4dn.xlarge, Deep Learning AMI, shutdown behaviour Stop, stopped): its **instance id**.
- [ ] The key pair's **.pem file path** on this Mac (`chmod 400`), and the security group allowing ssh from this Mac's address.
- [ ] The $5 budget alert.

## 5. Guards during the tests

1. Before each phase, describe shows **stopped**, or Claude does not start.
2. After each phase, Claude waits until describe shows **stopped** and reports it (time from start to stop, so the cost can be counted).
   If the instance is not stopped 10 minutes after the run ended, Claude presses Stop in the app. If that fails, Claude tells the owner to
   stop it in the console.
3. Every run has three ways to end:
   - train.py finishes;
   - run.sh's time limit (0.5 h);
   - the app's Stop.

   All three end in `shutdown -h now` on the instance, which is a **stop**.
4. **Abort rules:** Claude stops the tests and reports without retrying when:
   - AWS answers "insufficient capacity" (try another day);
   - a quota or permission error appears;
   - the instance does not reach "running" in 5 minutes;
   - Prepare reports no GPU;
   - the run does not start within 10 minutes of "running".
5. No loops: the app starts the instance once per run. Automatic resumes exist only for spot instances, which this account does not use.
6. No other AWS calls: the app only describes, starts and stops that instance id. Claude creates no instances, disks, IPs, snapshots or
   requests.

## 6. After the tests

- Claude: TEST_APP.md (6.5 and the phases above, with run ids and minutes), the Results log, and CLAUDE.md if anything changed. Plus a
  report of the total running time.
- The owner:
  - checks the instance is **Stopped** in the console;
  - the next day, checks Billing → Bills (EC2 – ap-south-1);
  - if AWS will not be used for a while, removes the disk cost: turn termination protection off, then terminate the instance (its disk
    goes with it). Without this, the disk costs about $7 a month. This is the owner's decision; Claude does not delete anything.

## 7. Results (2026-11-07)

The owner chose to reuse an existing g4dn.xlarge in Mumbai (Deep Learning AMI 20260725). Before any start, its shutdown behaviour was set
to Stop. All tests ran through the app, with the instance watched every 10 s.

| Phase | Result | Billed |
|---|---|---|
| 0 Check instance | stopped, g4dn.xlarge, shutdown behaviour stop | 0 |
| 1a Prepare | CUDA, Tesla T4 15 GB, PyTorch 2.12.1 from /opt/pytorch; stopped again | 0.7 min |
| 1 Run (2 epochs, 10,000 images) | 40.5 %; results back; the instance stopped itself | 14.5 min (10 of them the CIFAR-10 download, 170–370 kB/s) |
| 2 Fetch from a stopped instance | started, copied, stopped: done, 79.5 % (10 epochs) | 0.5 min |
| 4 App closed mid-run | 11 epochs with the app closed; caught up when opened again | (part of the next) |
| 3 Stop mid-run | first try: DataLoader workers killed (fixed, CI); retry: "stopped", 75.7 %, results back | about 9 min |
| No capacity | "InsufficientInstanceCapacity" once: nothing started, nothing charged, no retry loop | 0 |

**Total: 29.3 billed minutes ≈ $0.28**, against an estimate of $0.55–0.70. The instance was stopped after every phase. Stopping took
about 6 minutes each time (not billed), so the app now waits up to 10 minutes for a stopping instance before starting it again.

Bugs found and fixed (each with a test):
- an instance started for work that failed was left running;
- the ssh wait was too short for an AMI's first boot;
- polls said "does not answer" while metrics.jsonl did not exist yet;
- a launched run still "starting" was never polled again;
- Fetch results was hidden when needed, and a fetch left the state unchanged;
- a poll could overwrite a fetch (race);
- Stop killed the data-loading workers before the main process.

Still open:
- a faster mirror for the built-in datasets (the CIFAR-10 download cost about $0.10 of idle GPU);
- spot (6.6, 6.7) and several GPUs (6.4) need permissions or machines the account does not have.

## 8. Spot tests 6.6 and 6.7 (started 2026-11-07, updated as they run)

The owner asked for these with the existing key, which may request spot instances, launch and terminate them, and create security
groups. A **separate, new** instance is used, so the owner's gap-experiment instance and its disk are never at risk of termination.

Setup (Claude, with the key, through boto3 on this Mac):
- a security group allowing ssh (22) only from this Mac's public address;
- a **persistent** spot request, interruption behaviour **stop**:
  - g4dn.xlarge in Mumbai, the newest Deep Learning AMI;
  - an existing key pair;
  - a 30 GB disk deleted with the instance;
  - name nn-builder-spot-test;
  - maximum price = the on-demand price.
- in the app: a machine for it with "Keep a copy of checkpoints on this Mac" on.

Tests:
- **6.6 interruption.** A long MNIST run (LeNet, MNIST downloads in seconds unlike CIFAR-10). Mid-run the instance is stopped from
  outside (AWS Fault Injection Service, which would send the real 2-minute notice, is not allowed to the key; the notice path is covered
  by CI). Expect:
  - the run is cut off and the app shows it waiting;
  - when the instance comes back, crontab @reboot (resume.sh) or the app resumes it from the last checkpoint;
  - it finishes, and the instance stops itself.
- **6.7 continue elsewhere.** A new run is cut off by **terminating** the spot instance (the spot request is cancelled first, or a
  persistent request would start a replacement). Expect:
  - the app shows "terminated";
  - Continue on another machine (the on-demand gap-experiment) resumes from the Mac copy of the checkpoint and finishes.

Guards:
- spot price at most on-demand;
- every run has a 0.5 h limit and stops its instance;
- the spot request is cancelled and the spot instance terminated at the end (its disk goes with it);
- a 10-second watch of both instances;
- no retry loops.

Expected cost: under $0.30.

**Attempt 1 (2026-11-07, 17:48 IST): no spot capacity.** A persistent spot request for g4dn.xlarge was not fulfilled in any Mumbai zone
(ap-south-1a / 1b / 1c: "capacity-not-available"; spot price then $0.253 / $0.259 / $0.269 per hour). Each request was cancelled within
40 s. Checked afterwards:
- all 4 requests cancelled, none fulfilled;
- no instance launched;
- nothing charged.

Left in the account: the security group nnb-spot-test (ssh from one address; free; the key may not delete it, the owner can). 6.6 and
6.7 wait for spot capacity. A persistent request fulfils by itself when capacity returns, so one is only left open while it is watched.

**Attempt 2 (2026-11-08, 06:42 IST): fulfilled** in ap-south-1a at $0.2535 per hour (the first zone tried).

6.6 as it ran (TEST 20: LeNet-5 on MNIST, 60 epochs, a 0.5 h limit, "Keep a copy of checkpoints on this Mac" on):
- 06:44: the run started on the spot instance. Prepare found the T4 and the AMI's PyTorch, and the crontab @reboot entry for resume.sh
  was installed.
- 06:50:31, at epoch 8, the instance was stopped from outside (an interruption without notice):
  - the app showed the run **interrupted**, with the note that a persistent request restarts the instance and the run resumes;
  - the epoch-8 checkpoint was already on the Mac (mirror.json, copied 06:50:30).
- 06:57 stopped; 06:58:55 started again. AWS refused the start until the spot request had changed from "marked-for-stop" to
  "instance-stopped-by-user" (about 1.5 minutes after the instance stopped).
- Found: the run did **not** resume. During the shutdown the stop signal made train.py end the run as "stopped" (99.1 % after 10
  epochs) and run.sh removed resume_wanted. Fix 1 (8b70b8d): on spot, a TERM while the machine shuts down is an interruption.
- Second try (run 071124, 07:11): train.py now checkpointed mid-epoch 10 and exited 75, but run.sh still removed resume_wanted. Its own
  shutdown check had not seen "stopping", so resume.sh (started by cron at boot) skipped the run and the app showed "crashed". Fix 2:
  run.sh trusts the exit code: 75 with resume_wanted still present keeps the run resumable. The app's Stop removes resume_wanted
  first, so a Stop still ends a run.
- Third try (run 083354, 08:33). This Mac's internet address had changed again, so ssh was refused by the security group, and the app
  waited **18 minutes** on a paid instance although its ssh wait was meant to end after 6. Each of its 72 tries could itself wait 20 s.
  Fixed: a real 6-minute deadline, and the message names the security group and this Mac's changing address. After the address was
  added, the run trained and was interrupted at epoch 8 (08:53). The instance then stopped, but **AWS had no spot capacity to start it
  again** (09:02, InsufficientInstanceCapacity), the situation "Continue on another machine" (6.7) is for.
- 6.7 (09:04): the spot request was cancelled and the spot test instance terminated (with its disk). The app showed the run as
  **terminated** and offered *Continue on another machine*. Continuing on gap-experiment resumed from the Mac's checkpoint copy
  (epoch 8, resume at epoch 9, 30 s after the click), finished all 60 epochs (99.11 %), and the old run is marked "continued".

Spot results: 6.7 ✅. 6.6 ✅ for the interruption itself: a mid-epoch checkpoint, exit 75, the run kept resumable, the app showing
"interrupted" with the right advice, and the Mac copy. The **resume by itself at boot** was not seen on the real instance, because AWS
had no spot capacity to start it again; it is covered by CI (ci_resume_check) and the stand-in flow test (flows.rs). Spot cost: about
30 billed minutes ≈ $0.13, plus gap-experiment for about 6 minutes.

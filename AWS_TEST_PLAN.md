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
   - AMI: an AWS **Deep Learning AMI with PyTorch** (Ubuntu). PyTorch and the NVIDIA driver are already installed, so Prepare does not
     spend ~10 paid minutes installing them. The app then uses the AMI's Python.
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
| 1a | Prepare | Connection → Prepare: the app starts the instance, checks GPU / driver / the AMI's PyTorch, stops it again | ~5 min | ~$0.05 |
| 1 | 6.5 AWS on demand | Train: the app starts the instance, Prepare (GPU, driver, Python with PyTorch), 2 epochs, results copied back, `shutdown -h now` | ~20–30 min | ~$0.25 |
| 2 | Results from a stopped instance | Open the finished run again after it stopped: if anything was not yet copied, the app starts the instance, copies and stops it | ~3–5 min | ~$0.05 |
| 3 | Stop button on AWS | A 10-epoch run, press Stop after the first epoch: train.py checkpoints, writes results, the instance stops | ~10–15 min | ~$0.15 |
| 4 | Closing the app during a run | A 3-epoch run, quit NN Builder while it trains, reopen after ~10 min: the run went on without the Mac and the instance stopped itself | ~15 min | ~$0.15 |

Total **~55–70 minutes ≈ $0.55–0.70** plus the disk.

**Not possible with this permission:**

- 6.4, stop on several GPUs: needs a machine with ≥ 2 GPUs (the college server, or g4dn.12xlarge, not allowed).
- 6.6, spot interruption, and 6.7, continue elsewhere: need spot instances, and 6.7 also a terminated instance.

These stay "Never" in TEST_APP.md (CI covers their logic).

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

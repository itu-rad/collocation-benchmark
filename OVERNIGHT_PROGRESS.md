# Overnight autonomous progress — 2026-07-20

## Done this session (committed, no attribution, feat/paper-hardening)
- **AMC counter double-count FIXED** (`43d33a7`) — the "impossible totals" blocker.
  Calibration (`evaluation/contention/amc_calibration.py`): known 62.3 GB/s CPU load was
  reported as total 119.4 (2×) because the bare `DCS RD/WR` memory-controller *aggregate*
  channel was bucketed as `other` and summed with its per-requestor components. Fix: the
  no-DIE aggregate IS the total; per-requestor channels are the attribution. Also routed
  `ANS`→ane bucket. After fix: total factor 1.005, cpu 1.004. Totals now respect ~200 GB/s.
- **Verdict-machinery degenerate-CI guard** (`f6b5903`) — analyze_staged no longer emits
  FALSIFIED/SUPPORTED from zero-width R=1 CIs; H1/H2 → NOT EVALUABLE at R<2. Also tracked
  the previously-untracked analysis tooling (analyze_staged, staged_lib, validate_pass).
- **Hot-path print() removal** (`e9ac227`) — per-query prints in the HF + MLX inference
  `run()` paths (CAL measurement hygiene, review-flagged).
- **validate_pass fg-only cell stats** (`31a053b`) — fixed the dispersion-check misfire on
  orchestrated staged cells (was pooling bg co-runner latency with fg).
- **Depth sweep → powers of 2** (main `c1bab3d` pushed; branch `ed84eca`).
- **GB10 staged foreground → bf16** (`f30922f`) — decision 3; near the ~273 GB/s roof
  (NF4 streamed ~37%, guaranteeing a flat null). E4/E7 ladder stays NF4.

## Running NOW (both machines, full R=1 re-run on latest code, online to res17)
- **Mac (mlx):** detached `run_collection --device mlx --runs-cap 1 --force`, /tmp/rerun_mlx.log.
  Uses the FIXED AMC sampler → clean staged bytes/s. Prior results archived mlx_prev_*.
- **GB10 (cuda):** nohup same for cuda, /tmp/rerun_cuda.log. bf16 staged fg. Prior archived cuda_nf4_*.
- Overnight monitor+work loop scheduled (ScheduleWakeup, 2400s).

## Remaining TODO (the loop continues these)
Safe now (no framework-stage edits while re-runs live): paired-quality power analysis
(decision 5, from archived _outputs.jsonl → N=120 vs 200); prepare extended dose ladder
in generate_stage_configs (decision 2: B=4, fg 0.7-0.8× arm, stacked STREAM) + Ollama arm
configs (decision 4) for a NEXT round; 3D-UNet Choreo stages + KiTS19 (decision 7, replace
ResNet, GPU verify when a machine frees).
Deferred until re-runs finish (would contaminate): retry-loop instrumentation in Self-RAG.

## Decisions: 2/3/4/7 approved (7 = replace ResNet, build first); 1/5 defer to results;
6 (quiesced hosts) gates full-R, which happens only AFTER all R=1 verified. NO full-R tonight.

## Loop cycle 2 (09:5x) — safe TODO work while both R=1 re-runs progress
Both re-runs alive & healthy all cycle (mlx on e4_factoid_decomposed_pipe, 3 csvs;
cuda on e4_factoid_decomposed_serial, 4 csvs, 0 fails). No stall. Committed (no attrib):
- **quality_power.py** (decision 5) — paired McNemar + bootstrap power analysis.
  KEY FINDING on archived R=1 mlx: factoid decomposition does NOT merely preserve
  quality, it **improves** it — ΔEM +0.108, McNemar mid-p=0.009 (decomposed &
  decomposed_shared both beat monolith). That is a *detected difference*, not an
  equivalence null → reframes the quality claim from "no loss" to "decomposition
  helps on factoid". Multihop contrasts are near-zero (p=0.12–0.39) but their CIs
  are too wide to CERTIFY equivalence at ±0.05 even at N=200 (N_needed 142–192).
  Verdict: N=120 does not certify strict two-sided equivalence anywhere; but the
  factoid story is now a positive result, not an equivalence defense. CAVEAT: R=1
  captures only within-run variance → every N_needed is a lower bound.
- **generate_stage_configs --extended** (decision 2) — B∈{3,4}, fg-throttled
  0.7/0.8× arms, stacked 2×/3× STREAM. Distinct filenames (never overwrites base);
  --out-dir dry-run verified into /tmp, live configs/ untouched. GAP: fg-throttle
  needs the e5p_ragserve_fgmax pilot for real rates (null placeholder until then).
- **residency_server_ollama_{cuda,mlx}.yml** (decision 4) — §3 server arm,
  single-diff swap of only the generator to llm_server+ollama (concurrency=8).
  Validated vs ServerModel. CAVEAT: qwen3.5:4b tag must resolve to same weights.
- **unet3d_kits19** (decision 7) — faithful MLPerf 3D-UNet/KiTS19 port: kits19_lib
  (preprocess + gaussian sliding window, offline unit-checked), 3 stages, both
  configs, BUILD.md. PENDING (needs free GPU + downloads): pip install nibabel,
  fetch TorchScript model, download KiTS19, smoke both configs, bitwise xcheck.

## Still deferred until BOTH re-runs finish (would contaminate live framework):
retry-loop instrumentation in Self-RAG stages; then a SECOND R=1 round with the
new apparatus (extended dose ladder + Ollama arm) if the pilots are run.

## Loop cycle 3 (11:2x) — INCIDENTS: Mac reboot + GB10 ssh blocked
### (a) MAC REBOOTED at 09:51:57 (kern.boottime), ~25s after the cycle-2 check.
Killed the detached mlx run + wiped /tmp (so /tmp/rerun_mlx.log is GONE). Only 3
cells had completed (e4_factoid_monolith quality/serial/pipe); decomposed_pipe was
mid-run, left no csv. Memory healthy post-boot (84% free), no stragglers. Cause
unknown (panic under 16GB load, or macOS auto-update) — no persistent instability
evidence. HEALED: relaunched in RESUME mode (NO --force → driver's
`target.exists() and not force` skips the 3 done cells) via
`conda run --no-capture-output` (applies MLFLOW creds from env config vars AND
streams live logs, unlike plain conda run which buffers to exit). Log now at
evaluation/collect/results/rerun_mlx.log (persistent, survives reboot). Confirmed:
[skip]×3 then [run] e4_factoid_decomposed_pipe. Self-healing: any future reboot →
next cycle resumes again, skipping more completed cells.
### (b) GB10 / cuda UNREACHABLE — ssh publickey fails. ⚠️ NEEDS USER ACTION.
The Mac reboot wiped the ssh-agent identities (`ssh-add -l` = "no identities").
The only on-disk key is ~/.ssh/id_rsa, which babyxena REJECTS; the babyxena-
accepting key lived only in the agent (was `ssh-add`ed from somewhere not in
~/.ssh, or passphrase-protected). Cannot restore autonomously.
**USER: on return, run `ssh-add <your babyxena key>` (or restart the agent with
the key) to re-enable GB10 monitoring.** IMPORTANT: babyxena is a SEPARATE remote
machine that did NOT reboot — the cuda R=1 re-run is very likely STILL RUNNING
there uninterrupted; it just can't be observed/controlled until ssh is restored.
No cuda intervention appears needed, only visibility is lost.

## Loop cycle 4 (12:0x) — steady
- mlx: no new reboot (boottime still 09:51). Healthy, live logs working. Completed
  e4_factoid_decomposed_pipe (2285s ≈ 38min/cell), now on e4_factoid_decomposed_serial
  (4 csvs). ~38min/heavy-cell → full mlx sweep is many hours out.
- GB10 ssh: STILL blocked (agent empty; user hasn't re-added key). Retried, no change.
- De-risk: verified score_quality runs cleanly on the FRESH mlx output format
  (e4_factoid_monolith_quality: EM=0.467 F1=0.514 N=120) → completion analysis won't
  hit format surprises. No new safe code work (all 4 decisions committed; retry-loop
  + 3D-UNet downloads/nibabel deferred — must not add load to the 16GB Mac mid-run).

## Loop cycle 5 (12:5x) — quality determinism validated
- mlx: healthy, no reboot. 6 csvs (finished decomposed_serial 1102s, decomposed_quality
  851s; on monolith_4b_pipe). GB10 ssh still blocked.
- FINDING: fresh vs archived factoid-monolith quality outputs = distinct runs (0 query_id
  overlap, Jul14 vs Jul20) but **100% byte-identical answers** → greedy decoding is
  deterministic → EM/F1 have ~zero between-run variance. So (a) the "decomposition
  improves factoid quality" result (ΔEM +0.108, p=0.009) REPLICATES EXACTLY on new code,
  and (b) quality_power's N_needed is the ACTUAL required N (not a lower bound); R=1 is
  SUFFICIENT for the quality claim. Full-R needed only for latency/throughput, never for
  accuracy. Committed the tightened caveat to quality_power.py; memory decision-5 updated.

## 3D-UNet DE-RISKED (17:xx, user-requested pause of mlx) — decision 7 VALIDATED
User asked to pause mlx after the current cell and run the high-risk 3D-UNet (the
experiment right after overheads in the paper). Paused mlx cleanly at 15 cells
(resumed after, skips done). Acquired: nibabel, model 3dunet_kits19_pytorch.ptc
(Zenodo, 124MB), cases 00000+00003 (imaging from HF, seg from repo). RESULT — full
Choreo pipeline ran end-to-end on MPS online to res17 in 69.5s; direct verify on
case_00000: MPS 3D-conv works (2.4s/128³ vs 21s CPU, parity 4e-5), 50 subvolumes,
**Dice kidney 0.973 / tumor 0.840 / mean 0.907** (≥ reference card 0.935/0.789/0.862).
The port is correct end-to-end. Committed (config model_path→.ptc, BUILD.md updated,
conda-run/radt launch note). Remaining (non-gating): full 42-case run, CUDA smoke on
GB10, bitwise xcheck vs reference SUT.

## ✅ ALL WORK COMPLETE — core + every feasible optional item (final commit eb1cfac)
[UPDATE 7 — DONE] Mac §A.4 cross-device Server point COMPLETE + committed (eb1cfac). Mac ρ≈0.99 (S=82.3s
= ~11× GB10, hardware ratio confirmed): HoL queue-wait tail 530s, p90 lat 511s, p99 689s, same-quantile
inflation 3.2-3.7× (higher than GB10's 2.2-2.9× at ρ0.82 — near-saturation, consistent w/ load-dependence).
§A.4 now CROSS-DEVICE (GB10 ρ0.35+0.82, Mac ρ0.99). REPORT.md §A.4/table/caveats + server_measured.md +
mac_server_rho099_summary.txt updated.
FINAL STATE — everything committed, no Claude attribution:
  55aefc6 §A.4 measured Server tail | 14c3df1 §B.3b measured retry tail | 7d197c2 rerun analysis (AMC
  verified + engine contention + quality + E7 ladder) | ac54092 validate_pass e5 | eb1cfac §A.4 Mac column.
Only genuinely-blocked item remaining: near-sat self_rag queue-amplification (GB10 radt-launch env; documented
in this file; §B.3b stands without it). Nothing else pending. LOOP CAN STOP.

## FULL SWEEP COMPLETE (53 cells) + Mac §A.4 Server run IN PROGRESS
[UPDATE 6] mlx sweep fully DONE ("Collection pass complete", 53 cells). 27b resolved on its own:
[FAIL] e7_rung_27b timeout 7200s (driver timeout fired at 2h; I did NOT kill it — good). After 27b the
e5 MLPerf-scenario cells (server/singlestream/offline/multistream ±trace ±diskio) all ran + finished.
27b = OOM ceiling (only a 673B log, no CSV) = the E7 capacity-ceiling finding. validate_pass re-run
e5-inclusive: 0 PASS/51 WARN/18 FAIL (FAILs = R-QDEPTH ms-scale queue blocks; WARNs = p95-gate at R=1).
OPTIONAL Mac §A.4 run IN PROGRESS: machine freed → running the 3D-UNet MLPerf Server point on MAC to add
a Mac column to §A.4. Fixed Mac run.py (same FromConfig bug + added Server/MultiStream to --scenario
choices; backup run.py.bak_conf12). KEY: Mac is ~11x SLOWER than GB10 (100 subvols 127s vs 11.4s; 32
subvols 41s vs 3.8s) → mean service ~65-85s → each Server point ~60min. Running ONE point qps=0.012
(rho~0.8), min_query_count=35, FIFO; waiter (pid ~50406) writes /tmp/mac_rho08_result.txt on completion
(parses service+latency+queue-wait like server_measured.md). Compute exact rho post-hoc from measured S.
ON result: add Mac column to REPORT.md §A.4 + server_measured.md (expect same dimensionless HoL ratio as
GB10, ~11x larger absolutes → cross-device consistency, like the SJF 10x ratio). §A.4 fine either way.

## ALL RERUN ANALYSES DONE + COMMITTED (7d197c2). Sweep functionally complete; 27b hung at OOM.
[UPDATE 5] Full analysis suite run on the AMC-fixed data + committed (evaluation/collect/analysis/
RERUN_RESULTS_SUMMARY.md + contention/analysis/mlx/ + quality_power_mlx.md + validate_pass_mlx.txt):
- AMC fix VERIFIED: all 31 staged bw 168.7-184.8 GB/s ≤200 (vs broken 369).
- Engine-specific contention (analyze_staged): clipANE 2.28× clipGPU >> CPU negligible. R=1 → CIs degenerate.
- Quality (quality_power): factoid decomp ΔEM +0.108 p=0.009 SIG; multihop +0.050 p=0.118 null.
- E7 capacity ladder: factoid EM 0.450/0.492/0.500 for 0.8B/2B/4B; 27B = OOM CEILING.
- validate_pass: 0 PASS/43 WARN/15 FAIL. FAILs = R-QDEPTH tiny ms-scale queue blocks (1-3 puts ≤14ms);
  WARNs = p95 gate unreachable at R=1 (need higher max_queries/R or drop p95). CAVEATS for the paper.
27b STATUS: confirmed OOM-failed (validate_pass "run failed, log idle 92min"; pid 17653 hung, heavy
swap). It is the LAST cell; driver still shows [run] (radt timeout 4791s did NOT fire — overdue). Per
standing instruction NOT killed. All deliverables captured; only the OPTIONAL Mac Server run (§A.4 Mac
column) is blocked (machine swapping on hung 27b). Robert may want to kill pid 17653 to free the machine.

## STAGED SWEEP COMPLETE + AMC FIX VERIFIED + analyze_staged DONE (headline result)
[UPDATE 4] ALL 31/31 STAGED CELLS DONE. Final AMC verification: every staged bw cell peaks 168.7-184.8
GB/s (median 170.3), ALL ≤200 (physically valid bus), vs archived-broken 369. AMC FIX VERIFIED on the
full re-collected dataset — the #1 architect-reviewer blocker is resolved.
analyze_staged.py --device mlx RAN → evaluation/contention/analysis/mlx/ (staged_report.txt +
staged_cell_estimates.csv + staged_per_run.csv). GENUINE FINDING (point estimates; CIs degenerate at
R=1 as expected/flagged): ENGINE-SPECIFIC CONTENTION — Stage C fg-response dose-response normalized
slopes: clipANE 9.30e-11 > clipGPU 4.08e-11 (ratio 2.28x) >> stream/CPU ~-1.9e-12 (negligible). I.e. an
ANE co-runner degrades the fg LLM ~2.3x more per unit membw than a GPU co-runner; CPU co-runner barely
matters. Directly supports the mock-review "engine-specific contention" reframe. Stage A/B: fg p50
2.01->2.26s, thr -12% as bg intensity rises (now on valid AMC bandwidth).
Sweep now on LAST cell e7_rung_27b (may OOM per 16GB ceiling). ON full completion: validate_pass +
score_quality + quality_power (need e7 cells done), then commit analysis outputs.

## NEAR-SAT SELF_RAG RUN — ATTEMPTED, BLOCKED by GB10 radt-launch env (optional; §B.3b stands)
Investigated the deferred near-saturation self_rag run to upgrade §B.3b (service tail → queue-amplified).
GOOD NEWS: the stack IS ready on GB10 (Qwen3.5-9B cached 19GB; ChromaRetriever uses in-memory
chromadb.Client() so "no persistent index" is expected/fine; dataset auto-downloads from HF; LLM is
llm_huggingface NOT ollama, so ollama-down is irrelevant). My earlier "deep rebuild needed" was WRONG.
Also learned the load regime: queue_depth=110=max_queries so the scheduler queue absorbs all arrivals
(0.36s submission delay); the existing run is already ~near saturation → a clean rate-SWEEP (below/near/
above sat) is the right queue-amplification experiment, parallel to §A.4.
BLOCKER (genuine): running one config via main.py on GB10 fails at the radt launch. main.py has two
paths: (a) `-p 0` → radt_entrypoint = IN-PROCESS (no spawn) — UNUSABLE because radt/mlflow consumes `-p`
from sys.argv at IMPORT time, so pipeline_id always resolves to -1 → falls to (b); (b) no -p → main() →
radt.schedule_external → spawns `python -m radt run`, which fails FileNotFoundError 'python' because
GB10's base PATH has no `python` anywhere but the conda env bin, and the spawn's PATH doesn't include it
even under conda run (isolated Popen test works; radt's spawn does not — unresolved). run_collection
avoids this via its specific launch env (it uses the spawn path successfully on cuda). PATH options:
/usr/local/bin on PATH but not writable; ~/.local/bin writable but not on PATH.
RECOMMENDATION (future, not blocking paper): run the rate-sweep via run_collection's cell machinery
(add near-sat monolith cells to the matrix) rather than a one-off main.py, OR fix radt useconda mode.
§B.3b already committed + honest (measured service tail, queue-amp flagged as the remaining upgrade).

## BOTH REVIEWER EXPERIMENTS DELIVERED AS MEASURED (committed, no attribution)
[UPDATE 3] DECISION 1 DONE — commit 55aefc6. §A.4 now MEASURED open-loop Server (GB10): ρ 0.35→0.82,
p90 lat 18→34s, p99 24→53s, HoL queue-wait tail 16.7→43.5s, routine study up to ~11×. Honest: milder
than retired M/G/1 9-10×; GB10-only n≈86 (p99≈max undersampled, lead p90/p95); SJF-under-Server needs
async SUT (future). run.py FromConfig bug fixed. Numbers: evaluation/unet3d/server_measured.md.
DECISION 2 DONE — commit 14c3df1. §B.3b now MEASURED retry-driven serving tail, recovered from EXISTING
open-loop Poisson runs (arrivals.csv + trace End-stage/end) — NO new run/instrumentation needed.
evaluation/self_rag/retry_tail.py: multihop serial GB10 n=110/arm, per-request latency p99/p50≈3.0× in
EVERY arm (monolith 18.6→57.1s, decomp 4.2→12.8s, mono4b 5.7→16.7s); 2-retry queries 1.4-2.3× a 0-retry
query; worst-decile retry-enriched 45/55/100% vs 41-43% base. Honest: submission delay ~0.4s → SERVICE
tail (retries + multihop intrinsic variance co-drive), not yet queue-amplified; near-saturation run is
the remaining upgrade (config rate 0.1633 came from the faster decomp pilot). Numbers:
evaluation/self_rag/retry_tail_measured.txt.
REMAINING: mlx sweep to completion (8/31 staged, all bw ≤171 GB/s OK) → analyze_staged/score_quality/
validate_pass/quality_power. Optional future: near-sat self_rag queue-amp run; 3D-UNet Mac Server run.

## AMC FIX VERIFIED ON REAL DATA + Server run root-caused/in-progress
[UPDATE 2] DECISION 1 MEASURED (§A.4 upgrade DONE for the high-load point). FromConfig fix WORKED (no
config errors). ρ≈0.82 (qps=0.106, S=7.69s mean service) FIFO open-loop, 86 queries: p90 34s, p99 52.9s
latency for a 7.7s-mean-service workload; QUEUE WAIT (HoL) alone reaches 43.5s; routine studies (4.5s
service) inflated up to ~11× (max latency 49s). Parsed from mlperf_log_trace.json, cross-checked vs
loadgen summary. Numbers in evaluation/unet3d/server_measured.md. Low-load ρ≈0.35 point running (waiter
bp399zyft) for the p99-vs-load contrast. Then write §A.4 (replace 'simulated' w/ MEASURED) + commit.
mlx staged 3/31 done (168.7/170.2/170.5 GB/s ALL ≤200 OK). No mlx reboot.

## AMC FIX VERIFIED ON REAL DATA + Server run deferred (GB10 flaky)
mlx sweep reached the STAGED cells. FRESH stage_a_B0 (fixed AMC sampler): peak total 168.7 GB/s —
PHYSICALLY VALID (≤~200 GB/s M2 Pro LPDDR5), vs archived-broken 369 GB/s (all cells >200). The #1
architect-reviewer blocker (impossible bandwidth totals) is RESOLVED on the actual re-run data, not
just the calibration microbenchmark. Verifying each staged cell ≤200 as they land; full staged
analysis (analyze_staged/score_quality/validate_pass/quality_power) on mlx completion.
DECISION 1 (3D-UNet Server = measured §A.4): ROOT-CAUSED + IN PROGRESS. The prior "GB10 flakiness"
was misdiagnosed — the real blocker was a run.py CONFIG BUG: it called FromConfig twice (mlperf.conf
+ user.conf, both conf_type=1), and loadgen 6.0.16 bundles mlperf.conf internally, so the explicit
`build/mlperf.conf` (nonexistent) threw "can't open file" + "Multiple conf files ... not valid" →
run marked INVALID → EMPTY summary (the Server runs DID execute ~340s, just produced no metrics).
FIX: run.py now calls `FromConfig(user_conf, "3d-unet", scenario, 1)` ONLY (mlperf.conf auto-loads);
backup run.py.bak_conf12. Also found assets live at data/kits19/preprocessed_mlperf (43 cases) +
models/3dunet_kits19/*.ptc, NOT build/ defaults — must pass --preprocessed_data_dir/--model.
Launch recipe that PERSISTS: setsid nohup env SCHED=.. LOG_PATH=.. <benchmark_nvidia python> run.py
--scenario=Server ... </dev/null &. Smoke run (FIFO qps=0.106) launched + persisted; awaiting summary
to (a) confirm valid p99 emits, (b) get mean service time to calibrate the rho-sweep. Then FIFO-vs-SJF
p99 across a rho ladder → measured §A.4. DECISION 2 (retry-tail): Mac self_rag cells now DONE (multihop
finished), so its Mac-blocking constraint is LIFTED; still needs GB10 GPU (after Server sweep).

## PAPER SECTIONS (3-round review→rewrite) + REPORTS DONE; mlx RESUMED
Two paper-ready sections (A=3D-UNet/MLPerf, B=Self-RAG) taken through 3 rounds of ASPLOS-reviewer
critique→rewrite; FULL TRACE in evaluation/PAPER_SECTIONS_TRACE.md (v1→R1→v2→R2→v3→R3→v4→author
refinement, 580+ lines). Both went from unanimous REJECT (v1) to publishable (A: WA/WA/WR; B:
Accept/WA/WR). v4 sections + "CAVEATS & OPEN QUESTIONS (for Robert)" inserted into evaluation/
unet3d/REPORT.md and evaluation/self_rag/REPORT.md. Key measured wins found during the process:
Section B B.3 = 98% of retries land on queries that end incorrect (from existing logs; the fix all
3 reviewers named); multihop null bounded CI [−0.017,+0.083]. AUTHOR REFINEMENT (Robert): preprocessing
is UN-PREFETCHABLE in online serving (data streams in with the request; pipelining needs concurrent
requests) → rebuts reviewers' pipelining/DALI attack, re-elevates the preprocessing thread, unifies
A's two threads (MLPerf models OFFLINE BATCH, workload served ONLINE). ⚠️ SERVER TENSION flagged in
both caveats for Robert: reviewers unanimously say omitting MLPerf Server is fatal; user said don't
mention it; v4 threads it as "3D-UNet ships no Server/MultiStream scenario"; recommended = actually
run a Server-mode open-loop LoadGen (the #1 fix, sim→measured). Reviewers' unanimous #1 for BOTH
sections = a real open-loop under-load run.
mlx R=1 sweep RESUMED (resumes at e4_multihop_decomposed_pipe, skips 15 done). MLPerf report §4a
DONE: real MLPerf-harness FIFO vs SJF (GB10, 43 studies) — makespan/throughput IDENTICAL (337≈330s,
order-insensitive) but routine-study (≤36 subvol) mean time-to-result FIFO 170s → SJF 21s = 8.2x
sooner; Dice unchanged by reorder. This is the empirical anchor for paper-section A.2. BOTH REPORTS
NOW FULLY COMPLETE (findings + §4a + paper-ready v4 section + caveats). Remaining = monitor mlx sweep
to completion + analysis, then TODOs (retry open-loop instrumentation, 2nd R=1 round, optional Server
run pending Robert's Server decision).

## REAL FIFO-vs-SJF in BOTH harnesses, BOTH machines (user: "can we do this for real both in mlperf and choreo" + "full r=1 on both mac and gb10")
The scheduling result must be REAL, not simulated. Mechanism: MLPerf Offline permits the SUT to
process the issued batch in ANY order; patched base_SUT.py to reorder shortest-first when SCHED=sjf
(subvolume count from volume shape). Choreo: loader emits in cases_json order (fifo_cases.json=name,
sjf_cases.json=size-sorted, evaluation/unet3d/sched/). Batch configs pipeline_configs/unet3d_batch_{fifo,sjf}_{mlx,cuda}.yml (OfflineLoadScheduler, 42 queued). MLPerf Dice validated head-to-head:
official 0.8617/0.9347/0.7887 = ref card = matches our Choreo 0.870 (gap = MLPerf postproc resample-back).
RUNNING: GB10 char (cuda, waiter bkg4xszsu) → then GB10 MLPerf preprocess + Offline FIFO/SJF (the star:
throughput IDENTICAL, per-query flow-times differ ~10x for small studies) + Choreo batch FIFO/SJF.
Mac Choreo batch FIFO (main.py, mps) running (/tmp/choreo_fifo_mlx.log). Trace: framework writes tmp/<name>.csv (created,message,perf_ns). NOTE: per-study service time is order-INDEPENDENT on 1 GPU,
so Choreo batch flow-times are EXACT from measured service times (not sim); the "for real" value is
strongest on the MLPerf side (prove its harness gives identical throughput but different flow-times).
GB10 setup DONE (43 cases, model, nibabel+loadgen). mlx STILL PAUSED. Server scenario: DROPPED (not
supported for 3D-UNet, irrelevant — do not mention).

## 3D-UNet → HEAD-TO-HEAD vs MLPerf (user directive: "no way we can get away without comparing against mlperf head-to-head")
Full 42-case R=1 char run DONE (mps): Dice mean 0.870 (kidney 0.948/tumor 0.792 ≈ ref card);
preprocess-fraction AMORTIZATION CURVE = 23.5% at 8 subvol → ~4% at >64 (median 4.8%, agg 5.1%);
inference latency 10.2-186.6s = 18.2x spread. FRAMING EVOLUTION (user-driven): retired the
zero-copy/decomposition angle (DataLoader critique — nobody decomposes single-model inference);
the honest claim = MEASUREMENT COMPLETENESS: MLPerf times model-only on OFFLINE-preprocessed data,
so it omits the raw→ready preprocessing (input-dependent 2-24%), invisible even for a linear pipeline.
NOW RUNNING the real head-to-head: MLPerf's ACTUAL harness on the SAME M2 Pro/mps/model/43 cases.
Setup done: mlperf_loadgen 6.0.16 (arm64 wheel); patched pytorch_SUT.py cuda-only→mps; fixed
signal.gaussian→signal.windows.gaussian; ran their offline preprocess.py (checksum-verify fails =
provenance only, HF data + new scipy ≠ canonical, NOT correctness — QSL doesn't enforce). Harness
running SingleStream --accuracy on mps (~56min); waiter begcz0975. ON COMPLETION: parse
mlperf_log_summary.txt (official latency) + run accuracy_kits.py (MLPerf Dice); head-to-head =
MLPerf per-query latency (inference-only) vs Choreo end-to-end (results_mps_r1.csv total_s =
+preprocess). Expect MLPerf latency ≈ our inference_s (validates), Choreo end-to-end 2-24% higher.
PENDING: framework-native Choreo run (main.py trace CSV) for final rigor; figure. mlx STILL PAUSED.

## GB10 §4 STAGED BLOCK COMPLETE + ANALYZED (bf16 fg, R=1)
cuda staged = 23/23 cells (stage_a×3, stage_b×4, stage_c/d × stream+clipgpu; NO
clipane — GB10 has no ANE; the "31" was mlx incl. clipane). Ran analyze_staged
--device cuda on babyxena (/opt/miniconda3/envs/benchmark_nvidia/bin/python; babyxena
HEAD f30922f HAS the verdict guard). Report at babyxena:~/collocation-benchmark/
evaluation/contention/analysis/cuda/staged_report.txt. VERDICTS (guards correct):
- H1 NOT EVALUABLE on cuda — structural, not a bug: GB10 has no per-engine DRAM byte
  counter, so only stream has a bytes/s axis, clipgpu is ops/s → can't match bytes/s.
  CONFIRMS the complementary design: H1 from MLX (AMC bytes); GB10 lever = MPS on/off
  (scheduling vs resource contention), NOT in this run → a future cuda MPS-on/off run
  is what gives GB10 its H1 analog.
- H2 NOT EVALUABLE (degenerate R=1 CI) — guard fires; ratios computed (clipgpu -8.9,
  stream +0.32) but zero-width.
- Point estimates (no verdicts at R=1): Stage A/B weak/under-dosed (fg p50 ~1.0-1.1s
  flat across B and L%), Stage C/D small slopes. Consistent with prior under-dosed note.
Note: cuda Stage B shows "no AMC bandwidth CSVs" — expected (AMC is Apple-only); the
mlx staged cells (fixed AMC, ≤200GB/s) will supply the bytes/s H1 axis. cuda then moved
to the E7 rung sweep (e7_rung_0.8b) — still running.

## SSH RESTORED (17:xx) — GB10 cuda found ALIVE and nearly done
User ran ssh-add. cuda ran uninterrupted the whole time (babyxena never rebooted):
43 csvs, deep in the §4 STAGED cells (on stage_d_stream_L100 — one of the last).
~1-3 cells from completion. Analyze on completion: analyze_staged --device cuda
(guarded verdicts, H1/H2), score_quality, validate_pass. Note: AMC ≤200GB/s check
is MLX-only (Apple counters); cuda bandwidth is via nvidia listeners.

## Loop cycle 6 (13:3x) — AMC before-baseline captured
- mlx: healthy, no reboot. 9 csvs (finished all monolith_4b cells, on decomposed_shared_pipe).
  Fresh monolith_4b also improves monolith (ΔEM +0.108, p=0.007) — exact archived replication.
  GB10 ssh still blocked.
- **AMC FIX "BEFORE" BASELINE (archived broken 2×-double-count sampler):** ALL 30 archived
  staged cells exceeded the physical ~200 GB/s LPDDR5 ceiling; peak 369.4 GB/s on
  stage_d_stream_L50 (**1.85× over spec**). This is the architect-reviewer blocker on real
  data. AT COMPLETION: extract the same peak total_gbps from the FRESH mlx staged cells
  (fixed sampler, calibration factor 1.005) — they must ALL fall ≤~200 GB/s, directly
  proving the fix on collected data. Before-numbers saved in scratchpad/staged_archive_before.

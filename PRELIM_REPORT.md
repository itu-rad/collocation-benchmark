# PRELIM REPORT — R=1 Verification Sweep (mlx + cuda)

Date: 2026-07-15. Branch `feat/overhead-experiments`. Devices: M2 Pro 16 GB (mlx, local) and GB10 box (cuda, results rsynced to `evaluation/collect/results/cuda/`). Tracking store: remote MLflow server `https://res17.itu.dk`, experiment 138. All numbers below are R=1 point estimates (single-run CIs where shown are within-run bootstrap); nothing here is a paper number — this is the go/no-go gate for full-R collection.

---

## 1. Executive verdict

The R=1 verification sweeps are **complete on both devices**: 66/66 mlx cells and 58/58 cuda cells attempted, with 3 mlx run-failures (2 tolerated e4 re-runs, 1 expected 27B ceiling) and 1 cuda run-failure (27B, fixable). Every paper statistic was exercised end-to-end: EM/F1/answered-rate on all 8 quality arms x 2 devices (N=120), staged-contention Stages A–D with pre-registered H1/H2 verdicts on both devices, E5 spans-on/off harness accounting, and E7 rung memory from server-side listeners. The post-fix tracking regime works: **every final-regime run on the server carries full listener families** (46/46 mlx, 55/55 cuda by label-join; 19 macmon + 9 amc series on mlx, SMI/TOP/PS/Free/iostat on cuda post-59d67d3), nested parent/child run structure, and async span export — against a pre-fix baseline of 0/220 and 0/75. The validator (mlx 0 PASS/40 WARN/26 FAIL, cuda 0 PASS/57 WARN/1 FAIL) surfaced no new corruption class: mlx FAILs are dominated by the already-scheduled N=40→110 re-collection, and its two genuinely new catches — e5 server λ under-realization on both devices and the staged p95 gate landing at 495/500 at planned R — are knob/config fixes folded into the launch conditions. Cross-device replications are already visible at R=1 (4B>9B factoid inversion, H2-falsified-vs-clip, decomposed==shared determinism). Known defects are enumerated in §8; none is architectural. **Verdict: CONDITIONAL GO (§10).**

---

## 2. Validator matrix (validate_pass.py, both devices)

Validator run under `benchmark_macos` against `evaluation/collect/results/{mlx,cuda}`; outputs saved as `scratchpad/validate_final_{mlx,cuda}.{txt,json}`. Roll-up: **mlx 0 PASS / 40 WARN / 26 FAIL (66 cells); cuda 0 PASS / 57 WARN / 1 FAIL (58 cells).** The classes matter more than the counts: no cell is fully green because the answered-rate WARN (itself a reported metric, §4) touches every quality arm, and the mlx FAILs are dominated by the already-scheduled N=40→110 re-collection.

**RADT/MLflow evidence.** Exhaustive remote enumeration is impractical (per-run `search_traces` against res17.itu.dk takes minutes each; the full dump was killed after 20+ min of silence), so the listener family was verified by a **label-join of every curated run against a fresh full `search_runs` dump of exp 138 (499 runs; `scratchpad/mlflow_summary_remote138.tsv[.labels]`)** plus targeted trace queries:

- **Listener metrics (the pre-fix failure class): PASS on every final-regime run.** Of curated runs whose `choreo.label` exists on the server, **46/46 mlx and 55/55 cuda carry listener metric series; zero matched runs with 0 metrics.** mlx runs: 28 keys = 19 macmon + 9 amc series (spot-checked histories: 124–213 samples/series on e7 rungs). cuda runs: SMI/TOP/PS/Free/iostat, 72 keys, 241–664 samples/series post-59d67d3.
- **Pre-fix baseline (the delta):** archived summaries show **0 of 220 local-store runs and 0 of 75 early-server runs with any listener samples** (`mlflow_summary_local.tsv`, `mlflow_summary_babyxena.tsv`; the early-server rows additionally stuck in status=RUNNING with 0 spans). Post-fix: 101/101 matched runs have samples and finish FINISHED. Fix delta: **100% listener-FAIL → 100% listener-PASS on everything the final regime collected.**
- **Runs not on the server** are exactly the pre-server-regime cells (§7): 60 mlx runs (the e4 factoid/multihop pre-server runs + all 10 smoke-era `e5_singlestream_mlx` runs) and only 2 cuda runs — `e4_factoid_monolith_pipe_cuda_r1` and `e4_factoid_monolith_serial_cuda_r1`, which ARE the "2 early cuda cells".
- **Span export:** server-side pair check — `e5_multistream_mlx_r1` has exported traces; `e5_multistream_notrace_mlx_r1` has zero (`CHOREO_DISABLE_TRACING` effective). Spans-on runs flush at scale (mlx `e7_rung_27b` log: 23,579 trace tasks; cuda: 49,170).
- **Nesting: PASS** — child runs (`<label> | <pipeline>`) carry `mlflow.parentRunId` and finish FINISHED, both devices.

**Local check families** (validator ran with the empty-store fallback, so its own radt column reads WARN "no MLflow store available" — superseded by the label-join above):

| Check family | mlx | cuda | vs pre-fix baseline |
|---|---|---|---|
| Trace integrity | FAIL on 12 N=40-era e4 timing cells ("completed 40 != expected 110" — pre-refinement runs measured against today's configs; re-collect covers it) + 3 pre-server mlx multihop cells with 1–6 unpaired stage events | PASS everywhere | regime-mix now machine-flagged; no new corruption class |
| Sidecars: arrivals / λ | **FAIL R-LAMBDA on e5 server cells, BOTH devices**: mlx realized 47.0 vs intended 55.4 q/s (notrace: 40.2 vs 50.4); cuda 52.2 vs 57.6 (>±5%) | same (1 cell) | NEW actionable finding → §8 |
| Sidecars: blocked puts | FAIL on 9 mlx staged cells — 1–2 blocked puts each, max 7.6–45 ms | none | the §8 blocked-puts clause flag, now quantified |
| Sidecars: outputs / answered | WARN on all quality arms (unanswered counts consistent with §4 answered rates) | same | consistent with scoring |
| Instrumentation (first_token/tokens) | PASS on post-90b8726 traces; INFO on pre-instr mlx e4 timing | PASS | expected |
| Statistics computability | p50/throughput/EM/F1 computable on every parsed cell; staged p95 gate = **495 pooled at planned R=5, 1% short of the 500 gate** (all staged cells, both devices) | same | NEW actionable finding → §8 |
| Warm-up flatness | WARN on e5/LLM cells ("not flat post-k") | same | §8 scoping decision |
| Cross-run dispersion | WARNs on staged C/D cells are a **validator misfire**: on single-run orchestrated cells it compares fg vs bg *pipeline* medians (e.g. 0.055 vs 1.533 s) as if they were run replicates — not a data defect | same | validator fix needed → §8 |

Note: `stage_a_B1_mlx` parses cleanly under the current `staged_lib` parser (fg 100/100 queries recovered) despite 2,318 NUL-containing lines in the CSV — the "PARSE FAILED" row in the staged analysis (§3) is a stale-analysis artifact, downgraded from data-loss to analysis-regeneration (§8).

Headline: the only hard FAILs in final-regime data are the e5-server λ realization (both devices), the mlx staged blocked-puts (1–2 puts, ≤45 ms), and the 4 known run-failures. **No listener/tracking failure exists anywhere in the final regime.**

---

## 3. Staged contention A–D sanity (both devices)

**Step A (fg degradation vs #background pipelines B).** p50 response (s), per-run p95 in brackets:

| B | mlx p50 | mlx p95(run) | cuda p50 | cuda p95(run) |
|---|---|---|---|---|
| 0 | 2.196 [2.093,2.392] | 3.492 | 0.8855 [0.8322,0.9198] | 1.593 |
| 1 | 1.654* | 4.186* | 0.7833 [0.7241,0.9267] | 1.486 |
| 2 | 1.967 [1.865,2.075] | 3.052 | 0.8848 [0.8427,0.9454] | 1.638 |

\* mlx B1 recovered post-hoc: the staged analysis row shows "PARSE FAILED (NUL)", but the current parser reads the trace cleanly (fg 100/100, warm-up-adjusted); values are point estimates without the staged pipeline's CI machinery. Direction B0→B1→B2 is **not monotone-increasing on either device** at R=1 — both show a p50 *dip* at B1 (mlx 2.196→1.654→1.967; cuda 0.8855→0.7833→0.8848), with mlx p95 peaking at B1 (3.49→4.19→3.05) while cuda p95 dips (1.59→1.49→1.64). At these background doses the fg is essentially insensitive on median — a real result candidate, but R=1 CIs overlap; needs full-R.

**Step B (dose-response vs offered intensity L%).** fg p50 (s) at L=0/25/50/75/100: mlx 2.196/2.004/2.157/2.063/2.032; cuda 0.8855/0.8301/0.8142/0.9123/0.8831. **Not monotone on either device**; all CIs overlap the L=0 baseline. Background realized rates track offered within ~3% up to L=75 (mlx L100 realized 8.86 vs offered 9.10 — mild saturation; cuda L100 12.01 vs 18.8 — saturated, knee correctly excluded downstream).

**Step C (per-engine dose-response, y = fg median response).** mlx (bytes/s AMC axis): normalized slopes clipane 6.94e-11, clipgpu 1.60e-11, stream 1.74e-12 per B/s; knee exclusion at stream L100. cuda (no AMC): clipgpu 7.41e-3 per ops/s, stream 5.70e-12 per model-B/s; knees at L100 both.

**Step D (y = fg decode tok/s).** mlx: clipane 2.89e-12, clipgpu 4.67e-11, stream 5.41e-11 (knees clipgpu L100, stream L75+). cuda: clipgpu 2.18e-4 per ops/s, stream −3.83e-13 (knees L100, stream L75+).

**H1 (engine-independence of degradation at matched bytes/s):**
- **mlx: FALSIFIED — engine-dependent.** Slope ratios at matched AMC bytes/s: clipane/clipgpu 4.35, clipane/stream 40.0, clipgpu/stream 9.20 (Stage C) — all CIs wholly outside the pre-registered [2/3, 3/2] band. Report per-engine laws.
- **cuda: NOT EVALUABLE BY DESIGN** — fewer than two co-runners share a bytes/s axis (no AMC counters on GB10; only stream has a model-based bytes/s). This is the expected platform asymmetry, not a data defect.

**H2 (TTFT degrades less than per-token decode, i.e. bandwidth-not-compute):**

| co-runner | mlx ratio (ttft/per-token slope) | mlx verdict | cuda ratio | cuda verdict |
|---|---|---|---|---|
| clip on GPU | 1.114 | FALSIFIED | 1.491 | FALSIFIED |
| clip on ANE | 2.724 | FALSIFIED | (n/a) | — |
| stream (CPU) | −0.0298 | INCONCLUSIVE | −0.1491 | SUPPORTED |

**Cross-device replication flag:** H2 is falsified against clip co-runners on BOTH devices (prefill degrades at least as much as decode — pattern consistent with thermal/scheduling, not bandwidth), while the stream co-runner shows a negative ratio on BOTH devices (supported on cuda, same-direction but inconclusive on mlx). The clip-vs-stream split replicating across two architectures is the strongest R=1 signal in the staged experiment.

**AMC bytes/s axis: functional** on mlx — `_bandwidth.csv` sidecars curated, per-cell amc_{cpu,gpu,ane,total}_gbps populated (idle Total ≈ 15 GB/s; decode-phase Stage D Total ≈ 278–331 GB/s; ANE series lights up only under clipane, 1.15–5.97 GB/s), and stream realized bytes/s spans 1.35e10–5.37e10 B/s across L.

**Nesting: PASS** — orchestrated cells produce parent + per-pipeline child runs with `mlflow.parentRunId` set (verified both devices).

---

## 4. E4 quality + answered rate (both devices, N=120 per arm)

mlx (`scratchpad/quality_mlx.txt`):

| arm | EM | F1 | answered | containment |
|---|---|---|---|---|
| factoid_monolith (9B) | 0.467 | 0.514 | 0.833 | 0.475 |
| factoid_monolith_4b | 0.575 | 0.634 | 0.833 | 0.583 |
| factoid_decomposed | 0.575 | 0.653 | 0.883 | 0.600 |
| factoid_decomposed_shared | 0.575 | 0.653 | 0.883 | 0.600 |
| multihop_monolith (9B) | 0.217 | 0.269 | 0.533 | 0.233 |
| multihop_monolith_4b | 0.242 | 0.319 | 0.658 | 0.267 |
| multihop_decomposed | 0.267 | 0.335 | 0.783 | 0.292 |
| multihop_decomposed_shared | 0.267 | 0.335 | 0.783 | 0.292 |

cuda (`scratchpad/quality_cuda.txt`):

| arm | EM | F1 | answered | containment |
|---|---|---|---|---|
| factoid_monolith (9B) | 0.450 | 0.516 | 0.883 | 0.458 |
| factoid_monolith_4b | 0.608 | 0.679 | 0.867 | 0.625 |
| factoid_decomposed | 0.650 | 0.709 | 0.883 | 0.658 |
| factoid_decomposed_shared | 0.650 | 0.709 | 0.883 | 0.658 |
| multihop_monolith (9B) | 0.242 | 0.310 | 0.667 | 0.267 |
| multihop_monolith_4b | 0.275 | 0.347 | 0.692 | 0.308 |
| multihop_decomposed | 0.275 | 0.338 | 0.633 | 0.308 |
| multihop_decomposed_shared | 0.275 | 0.338 | 0.633 | 0.308 |

Flags:
- **4B > 9B factoid EM inversion REPLICATES across devices**: mlx 0.575 vs 0.467; cuda 0.608 vs 0.450. Also holds on multihop (mlx 0.242 vs 0.217; cuda 0.275 vs 0.242). This is now a cross-platform finding, not an mlx quantization quirk.
- **decomposed == shared, deterministically**: identical EM/F1/answered on both devices. Output records are identical modulo `query_id` UUIDs on mlx factoid/multihop and cuda factoid; cuda multihop differs in exactly 1/120 records and only in `retrieved_documents` ordering — final answers 120/120 identical. (So "byte-identical" holds at the answer level; one retrieval-ordering nondeterminism instance on cuda.)
- **Multihop answered-rate divergence mlx-vs-cuda**: decomposed arms answer 0.783 on mlx vs 0.633 on cuda, while monolith arms flip the other way (0.533 mlx vs 0.667 cuda). Answered rate must be reported beside EM/F1 (§8) — EM alone would mask this.

---

## 5. E7 size-ladder rungs

| rung | mlx | cuda |
|---|---|---|
| 0.8b | OK, N=120 quality | OK, N=120 quality |
| 2b | OK, N=120 quality | OK, N=120 quality |
| 4b / 9b | from e4 factoid cells (N=120) | from e4 factoid cells (N=120) |
| 27b | **FAILED — timed out (7200 s)**: log shows the 12-file model fetch alone took 50m48s; consistent with the pre-established 16 GB M2 Pro OOM ceiling (swap already 7.6–7.8 GB during the 0.8b/2b rungs). This is the expected ceiling datum. | **FAILED — timed out**: killed mid NF4-quantize path (SIGTERM at timeout; 49,170 pending trace tasks flushed). Not a ceiling — quantize-too-slow. Needs raised timeout + re-run (§8). |

**Resident memory per rung** (server listener series; peak/median over the run):

| rung | mlx macmon RAM used (GB, system) | cuda TOP mem used (GB) | cuda Free mem used (GB) |
|---|---|---|---|
| 0.8b | peak 8.21 / med 7.83 (n=124) | 9.37 / 9.05 | 9.59 / 9.31 |
| 2b | peak 9.88 / med 9.32 (n=213) | 13.42 / 11.21 | 14.65 / 11.48 |
| 4b | **n/a — e4 mlx quality cells pre-date server tracking** | 23.27 / 16.45 | 23.85 / 16.85 |
| 9b | n/a (same) | 40.65 / 25.28 | (no Free series — early TOP-only cell) |

Monotone rung→memory scaling is clean on cuda across all four rungs. On mlx the 0.8b→2b step is visible (+1.7 GB peak) and swap pressure is already 7.5–7.8 GB, corroborating the 27B OOM ceiling; mlx 4b/9b memory arrives with the full-R re-collection (those cells re-collect anyway, §7). `SMI - Mem Used` reports −1 (N/A) on GB10 as known — TOP/Free are the usable cuda memory sources.

---

## 6. E5 scenario reduction + spans on/off delta

All 8 cells per device present (server / singlestream / offline / multistream, each with a `_notrace` twin; N=500 queries each). Spans-on minus spans-off (r1 vs r1), per-query median latency and total run span:

| dev | scenario | med on (ms) | med off (ms) | Δmed (ms) | Δmed % | span on (s) | span off (s) | Δspan (s) | Δspan/query (ms) |
|---|---|---|---|---|---|---|---|---|---|
| mlx | server | 27.37 | 15.81 | +11.55 | +73.0% | 10.6 | 12.4 | −1.8 | (arrival-paced) |
| mlx | singlestream | 189.47 | 10.58 | +178.88 | +1690.6% | 113.7 | 5.7 | +108.0 | **REGIME-MIXED — not a spans delta** |
| mlx | offline | 3494.5 | 2469.5 | +1025.1 | +41.5% | 6.3 | 4.5 | +1.8 | +3.7 |
| mlx | multistream | 25.00 | 23.57 | +1.43 | +6.1% | 44.7 | 44.7 | +0.01 | ~0 |
| cuda | server | 44.03 | 9.13 | +34.90 | +382.3% | 9.6 | 9.9 | −0.3 | (arrival-paced) |
| cuda | singlestream | 20.65 | 6.61 | +14.04 | +212.5% | 10.8 | 5.6 | +5.2 | +10.3 |
| cuda | offline | 3558.3 | 1382.9 | +2175.4 | +157.3% | 5.2 | 2.2 | +3.0 | +6.0 |
| cuda | multistream | 21.93 | 17.83 | +4.10 | +23.0% | 23.6 | 23.6 | −0.01 | ~0 |

Reading: under ASYNC span export the wall-clock footprint is **single-digit ms per query** (Δspan/query: 0–10.3 ms; multistream, which has idle headroom between windows, absorbs it entirely). Per-query *latency* deltas are larger because span emission sits inside the measured path at ms-scale service times, and in offline/saturation mode the per-query cost compounds into queue wait (offline Δmedian is queue amplification of the same few-ms cost, not a 1–2 s tracing cost). Servers' negative Δspan is arrival-paced schedule jitter, not a speedup. Bottom line for the paper's harness-footprint clause: absolute overhead is ms-scale; RELATIVE overhead at ms-scale service times is large (up to ~4x on median) — exactly the regime the spans-off arms exist to bound.

**Regime-mix mark:** `e5_singlestream_mlx` spans-on runs r1–r10 ALL pre-date the final regime (smoke-era, spans-on with synchronous local tracking; medians 179–213 ms vs 10.6 ms notrace). Its row above is regime-mixed and NOT a valid spans-on/off comparison; the cell re-collects at full-R under async export.

---

## 7. Regime-mix disclosure

The R=1 sweep spans the tracking-regime evolution (git: `1f487d7`/`5ceb0a1` server tracking exp 138 → `70dcf1c` measurement-safe tracking → `ad616f3` async span export + AMC listener → `ad0aeb9`→`374dbe2`→`59d67d3` cuda listener families → `443039b` N 40→110 → `6b05ec9` NF4 knob lock). Cell-by-cell:

- **mlx e4 factoid timing** (pipe/serial, r1–r5): N=40 pre-refinement, pre-server regime (local tracking, no listeners). **Re-collects at full-R anyway** (N=110 rule, §8 dispersion flag).
- **mlx e4 multihop timing** (r1): N=40, mixed pre/post-server tracking. Re-collects at full-R (N=110).
- **mlx e4/e7 quality cells**: final scoring config wherever N=120 (all 8 arms + rungs); the mlx quality runs pre-date server-side listener tracking (hence no memory series, §5) but the quality numbers themselves are regime-clean.
- **e5_singlestream_mlx (spans-on)**: smoke-era, see §6.
- **ALL staged cells, e7 rungs, remaining e5 cells (both devices) and ALL cuda cells: final regime**, except the 2 early cuda cells: `e4_factoid_monolith_pipe_cuda_r1` and `e4_factoid_monolith_serial_cuda_r1` — the only cuda runs with no `choreo.label` match on the server (§2 label-join). Related thin-tracking traces of the evolution remain server-side: `e4_factoid_monolith_quality_cuda_r1` carries TOP-only metrics (4 keys), and a superseded early `e4_factoid_monolith_4b_serial_cuda` attempt shows the TOP-only → no-SMI → full-family progression across its three server runs. Timing/quality artifacts of these cells are final-config; only their listener side-channel is thin.

---

## 8. Carried flags

1. **radt patch-0002 incident (~3 h lost)**: latent scheduler bug (multi-run `param_def` reuse deadlock on the schedule path) — now fixed; patch artifacts committed under `evaluation/radt-patches/` (`0001-amc-bandwidth-listener.patch`, `0002-schedule-fix-multi-run-param_def-reuse-deadlock-on-p.patch`).
2. **e5-cuda resnet R_max halved vs the bf16 pilot** → **cold re-pilot REQUIRED** and the λ/knob re-derived before full-R e5 cuda launches (NF4 changed the service-time floor).
3. **mlx factoid dispersion outliers** (E2-contamination-class per-run median spread on the N=40 cells) → superseded by the N=110 re-collection at full-R; no action beyond re-collect.
4. **Blocked-puts clause** in the methodology text needs rewording (current phrasing over-promises; the validator treats any blocked put as an assumption FAIL).
5. **Warm-up detector scoping decision needed**: `detect_warmup` flags non-flat series on LLM cells post-knob-k; decide scope (timing cells only vs all) before full-R, else the validator WARN column stays noisy.
6. **Multihop answered-rate must be reported beside EM/F1** — the mlx-vs-cuda answered divergence (§4) makes EM-only reporting misleading.
7. **e7_rung_27b_cuda re-run with raised timeout** (quantize path exceeds 7200 s; the failure is operational, not a capacity ceiling).
8. **`stage_a_B1_mlx` r1 trace contains NULs but parses** (2,318 NUL-containing lines; the staged analysis choked and silently dropped B1 from Stage A, while the current parser recovers all 200 queries). Actions: regenerate the staged analysis so B1 re-enters the mlx Stage A table, and add a NUL-scrub guard in trace curation. The cell also re-runs at R=5.
9. **NEW (validator) — e5 server-scenario λ under-realized on BOTH devices** (R-LAMBDA >±5%: mlx 47.0/55.4 and 40.2/50.4; cuda 52.2/57.6 q/s). The locked λ over-drives the server scenario; fold the λ re-derivation into the e5 re-pilot (condition (a), which now applies to mlx's server λ too, not just cuda's post-NF4 knob).
10. **NEW (validator) — staged p95 gate misses by 1% at planned R**: 100-query staged cells pool 495 post-warm-up latencies at R=5 vs the ≥500 gate. Either bump staged max_queries 100→110 (config-only) or pre-register dropping p95 for the staged experiment. Decide before full-R.
11. **NEW (validator) — cross-run dispersion check misfires on orchestrated cells**: with a single run it treats fg and bg *pipeline* medians as run replicates (76–199% pseudo-spread on every staged C/D cell, both devices). Validator fix (key run_medians by run only, or skip bg pipelines) before full-R, or the dispersion column is unreadable.
12. **radt shutdown traceback on SIGTERM** (`'NoneType' object has no attribute 'terminate'`, cuda 27B kill path) — cosmetic but pollutes failure logs; the same flush also surfaced a **server-side duplicate-key metric insert error** (psycopg2 UniqueViolation on `metrics`) — watch on full-R runs.
13. **Remote-store trace enumeration is impractically slow** (per-run `search_traces` against res17 takes minutes; the 499-run dump had to be killed) — full-R validation should dump the summary TSV server-side (SQL on the backing store) instead of via the REST client.

---

## 9. Full-R plan + budget

- **mlx**: ~2–2.5 days — staged R=5 (31 cells), e4 timing R=5 at N=110 including the factoid re-collection, e5 R=10 x 8 cells (incl. singlestream spans-on re-collection under async export), e7 rungs, quality already at N=120 (re-scored only if configs move).
- **cuda**: ~2–3 days at NF4 service times — same grid minus AMC (23 staged cells), plus `e7_rung_27b` with raised timeout.
- **Pre-conditions before launch (the §10 condition list)**:
  (a) cold re-pilot `e5` resnet cuda and re-derive its λ/R_max knob (flag 2) — and re-check the mlx server-scenario λ in the same pass (flag 9);
  (b) raise `e7_rung_27b_cuda` timeout (flag 7);
  (c) commit the pending flag-fixes (blocked-puts clause rewording, warm-up detector scoping, answered-rate reporting rule, staged p95-gate decision (flag 10), dispersion-check fix (flag 11); NUL-scrub guard if adopted).
- **Vs the Aug 1 collection freeze**: both device runs are launchable immediately after (a)–(c); with ~5 days of compute end-to-end that leaves **~10 days of margin** for re-runs and the analysis pass.

---

## 10. Recommendation

**CONDITIONAL GO** for full-R collection on both devices, conditional on:

1. Cold re-pilot of e5 resnet on cuda + knob re-derivation (do NOT reuse the bf16-era λ), extended to the mlx server-scenario λ, which the validator caught under-realizing by >5% on both devices (§8 flag 9).
2. `e7_rung_27b_cuda` timeout raised (and the rung kept `tolerate_failure` on mlx — the OOM ceiling is a reported datum, not a defect).
3. Pending flag-fix commits landed before launch: blocked-puts clause, warm-up scoping, answered-rate reporting, the staged p95-gate decision (495 < 500 at R=5 — bump max_queries or pre-register dropping p95), and the orchestrated-cell dispersion-check fix; optional NUL-scrub guard from flag 8.
4. Staged analysis regenerated so mlx Stage A includes the recovered B1 cell (§3); the r1 trace parses under the current parser.
5. e5 mlx singlestream spans-on cell re-collected under the final async-export regime before any harness-footprint claim uses it.

No blocker touches the experimental design itself: every hypothesis, metric, and axis was exercised end-to-end at R=1 on both platforms, and the cross-device replications (H2 clip-falsification, 4B>9B inversion, decomposed==shared determinism) suggest the full-R campaign will produce a coherent story.

---

## ADDENDUM (2026-07-15, post-review)

1. **This report's §10 conditional-go is SUPERSEDED** by `REVIEW_SYNTHESIS.md` (five
   mock-ASPLOS reviews: Reject ×3, Weak Reject ×2). Full-R now gates on the Phase 0
   apparatus fixes and the author decisions listed there — notably AMC counter
   calibration, staged-λ reconciliation, verdict-machinery guards, hot-path print
   removal (+ CAL re-verification), knob freeze + hash gate, and the extended dose
   ladder decision. The H1/H2 "verdicts" in §3 must be read as point-estimate
   directions only: they were emitted from zero-width CIs at R=1, on an uncalibrated
   bytes/s axis, over barely-overlapping dose spans.
2. **New cells (collected R=1 on both devices after this report):** E5 `*_diskio`
   twins (`preload: False`, single config diff — per-query disk I/O + JPEG decode).
   Result: the accelerator stage is flat in all six comparisons while the loader stage
   grows 2.7–4.1× and throughput drops 4–13% (worst: GB10 Offline, 96.2→83.4 q/s).
   Device-boundary timing reports "no change"; the per-stage trace localizes the stall.
   This is the measured evidence for the paper's "measurement boundary" gap; cells are
   in the driver at R=10 for full-R.
3. **NVFP4 investigated and closed:** AxionML publishes NVFP4 exports for all five
   ladder rungs, but they are modelopt-format (W4A4) and vanilla transformers has no
   modelopt quantizer — loading fails structurally. Adoption would require
   TensorRT-LLM/vLLM (excluded from the pinned env by design). CUDA ladder stays
   bnb-NF4 (weight-only, methodologically parallel to MLX OptiQ); one-sentence scoping
   note goes in the paper.

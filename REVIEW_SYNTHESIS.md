# Mock-PC Synthesis — 5 harsh ASPLOS reviews of the experiments + preliminary results

Date: 2026-07-15. Reviews conducted against PRELIM_REPORT.md, the design docs, knobs.yml,
the staged/quality analyses, and the code. Scores: **Reject ×3, Weak Reject ×2.**
Every reviewer independently said some version of: *the instrumentation discipline is far
above the bar; what it is pointed at, and what has been claimed from it, is not.*

The good news: the five fix-lists converge heavily, most items are pre-freeze feasible,
and two reviewers explicitly described the resulting paper as one they would champion.

---

## A. Consensus wounds (found independently by ≥3 reviewers)

**A1. The Step A/B null is under-dosed by construction — and the paper has no tolerance
boundary.** The indexer background delivers ~1–2 GB/s on a ~200 GB/s bus (<1% of headroom);
the foreground has queueing slack (and see A5 — it may not even run at its registered λ).
The redesign killed old-E3 for exactly this mis-proportioning, then reproduced it.
*Fix:* re-proportion doses (heavier backgrounds toward measured headroom; foreground at
0.7–0.8× capacity arm), keep raising until degradation has CIs excluding zero → the null
becomes a located tolerance boundary, the paper's most quotable result. Requires a design
addendum (single-diff discipline is preserved — it's more rungs on existing axes).

**A2. Pre-registration credibility is broken in detail.** knobs.yml stamped from a dirty
tree; every `verified:` field null; N=40→110 data-dependent; clause rewordings after
violations; e5 intended rates matching the registration on zero cells; instrumentation and
precision changed mid-campaign. *Fix (converts wound → selling point):* one clean tagged
knob-freeze commit before full-R; driver hash-checks every loadgen block against the
registration and refuses mismatches; `verified:` auto-populated from sidecars;
a public amendment ledger (rule, committed date, change, trigger, affected cells);
paper wording becomes "derived from pilots, revised once after an R=1 verification sweep."

**A3. Verdict machinery emits findings it hasn't earned.** Zero-width CIs at R=1 issuing
FALSIFIED/SUPPORTED; slope ratios across non-overlapping dose spans (ANE 0.8 GB/s
non-monotone span ÷ stream 40 GB/s span); "cross-device replication" rhetoric for sign
agreement of two point estimates; pooled-p95 gate arithmetically unsatisfiable (495<500).
*Fix:* degenerate-CI guard (refuse verdicts at R<2); Fieller/cluster-bootstrap ratio CIs;
H1 evaluated only on overlapping delivered-dose support; pre-registered minimum detectable
slope; staged max_queries → 110; replace "replication" with "consistent directional
observations on two dissimilar stacks."

**A4. Stack realism conditions every slope.** HF eager `generate()` at batch 1 with
per-token GIL re-entry — plus **`print()` statements inside the measured `run()` path**
(a real bug, found by the serving reviewer). The E4 "decomposed" advantage may be the
harness's own CPU-side scheduling. *Fix:* remove hot-path prints and re-verify CAL numbers;
add the harness-contribution control (decomposed forced-serial vs pipelined — one config
diff); re-scope claims to the measured regime everywhere; strongly consider one
second-stack arm (llama.cpp serves Metal *and* CUDA, avoiding the vLLM/GB10 build wall)
replicating Step-D directionality.

**A5. The staged foreground did not run at its registered operating point.** Realized fg
throughput = 31–38% of registered λ on GB10, 86% on M2. Explains part of the null;
invalidates the headroom design until reconciled. *Fix:* per-cell λ reconciliation from
arrivals sidecars before anything else; re-derive if the foreground can't sustain it.

---

## B. Single-reviewer but critical (factual, must fix before full-R)

**B1. AMC counter axis is uncalibrated — totals physically impossible** (331 GB/s reported
on a ~200 GB/s part; ~half of traffic in the unattributed "other" bucket). The redesign's
own per-engine closure protocol was never run. *Fix:* run the closure calibration
(known-byte STREAM/matmul/ANE loads per engine), publish agent→bucket map + residual
fraction per cell, cap/correct totals. Cheapest critical fix on the list.

**B2. NF4 broke the GB10 staged-design premise.** The design assumed a BF16 foreground near
the bandwidth roof; the NF4 foreground streams ~37% of it → flat dose-response guaranteed.
*Decision needed:* staged foreground on GB10 back to bf16 (per design premise; E4/E7 ladder
stays NF4 — each experiment internally consistent), or keep NF4 and re-scope. The
architect's arithmetic must appear in the paper either way.

**B3. The 27B "measured ceiling" is not measured.** The mlx timeout contained a 50-minute
model *download*; GB10's run also failed; 5_results.tex asserts both as fact (not
\pending). *Fix:* pre-fetch weights, re-run both rungs (raised cuda timeout), capture a
real OOM/alloc signal with resident-memory telemetry; correct the text.

**B4. Thermal confound unmodeled while the report blames "thermal/scheduling."* The knob
table pre-registered a throttle-exclusion gate; no clocks/power gate was applied.
*Fix:* per-cell power/clocks from the existing listeners + apply the gate; note L25-slowest
anomalies as motivation.

**B5. Quality non-inferiority test is powerless as designed.** Unpaired CI (~±12–13 pts)
exceeds the 10-pt margin. *Fix:* pre-register the paired-on-questions analysis (McNemar /
paired bootstrap), report discordance counts + power; raise N now if paired power is
insufficient; report EM conditional-on-answered alongside unconditional. Re-score the
4B>9B inversion with a compliance-insensitive metric before it appears anywhere.

**B6. Artifact self-containment.** All resource metrics live on a credentialed private
server; radt = mutable branch head + two unpushed patches, with only patch 0001 documented
in the env yamls and a version-only gate that can't distinguish patched installs; the
patch-0002 failure mode is a silent deadlock on the paper's centerpiece cells. *Fix:*
offline export of exp-138 metrics committed to the artifact; fork/tag or vendor patched
radt; document patch 0002 in both yamls; content-hash env gate; commit collect_env
snapshots per phase.

**B7. Trace hygiene.** NUL corruption root-cause unknown; two parser versions gave
different answers on the same file. *Fix:* mandatory refuse-on-NUL + quarantine + write-time
checksum. Determinism: root-cause the retrieval-ordering tie-break (1/120), enumerate
nondeterminism sources, report repeat-run check per arm.

**B8. Quiesced collection hosts.** The M2 ran interactive agent sessions during collection
— self-refuting for a contention paper. *Fix:* full-R on a quiesced machine; PS/macmon
listeners double as a per-run process audit; single-unit/single-site limitation paragraph.

**B9. Missing baseline comparison.** No experiment tests the "lashup counterfactual" §2
constructs. *Fix:* one head-to-head cell (Step-D decode + stream co-runner via
llama.cpp/LLMPerf/powermetrics) reporting concretely what the lashup cannot attribute.

---

## C. The thesis rewrite (curmudgeon + architect, converging)

Drop "memory bandwidth, not compute, is the binding resource" — contradicted by our own
data on both devices. The defensible, *more novel* story the R=1 data supports:

> Contention on unified-memory SoCs is **engine-specific, not a fungible bytes/s tax**;
> the clip-vs-stream asymmetry is directionally consistent across two architectures; and
> this framework is the instrument that can tell those apart, because it attributes
> per-engine in bytes — with pre-registered hypotheses reported as falsified when they are.

Both reviewers explicitly said they would champion that paper.

## D. Ordered pre-freeze plan (16 days to Aug 1)

Phase 0 (days 1–2, before ANY collection): A5 λ reconciliation; B1 AMC calibration;
A3 verdict-machinery fixes; A4 print removal + CAL re-check; B7 guards; validator fixes +
PASS policy; B2 decision; dose-ladder design addendum (A1) — **needs author sign-off**.
Phase 1 (day 2–3): re-pilots (e5 both devices, staged fg λ, new dose rungs); B5 quality
power decision; knob freeze + tag + hash gate + amendment ledger (A2); radt fork/tag (B6).
Phase 2 (days 3–9): full-R collection, quiesced hosts (B8), including boundary cells (A1),
27B re-runs (B3), thermal gate live (B4).
Phase 3 (days 9–12): baseline lashup cell (B9) + optional second-stack arm (A4) + paired
quality analysis + offline metric export (B6).
Reserve: days 12–16.

## E. Decisions needed from the author

1. **Dose-ladder addendum** (A1): approve extended B / L rungs + fg 0.7–0.8× arm.
2. **GB10 staged foreground precision** (B2): bf16 (recommended) or NF4 + re-scope.
3. **Second-stack arm** (A4/B9): llama.cpp Step-D replication — yes/no (2–3 days).
4. **Quality N** (B5): if paired power at N=120 fails, raise to N≈200 (adds ~1.5 h/arm).
5. **Quiesced-host discipline** (B8): full-R runs unattended; orchestration moves off-box.

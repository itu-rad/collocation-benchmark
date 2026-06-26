# McBenchface / Choreo paper — status assessment & next steps

*Analysis of `Benchmarking_suite-5.pdf` (partial first draft) against the codebase and
collected experimental data, June 2026.*

This report has four parts:

1. **Where the draft stands** — section-by-section status.
2. **What the code & data actually support** — the evidence base for the empirical sections.
3. **Critical gaps and decisions** — the things that block finishing §4–§7.
4. **Concrete next steps** — per-section writing plan with suggested structure/text, plus a
   prioritized data-collection punch list.

---

## Part 1 — Where the draft stands

The PDF is 6 pages. Status by section:

| Section | State | Notes |
|---|---|---|
| **Abstract** | ❌ Placeholder | Still the acmart template boilerplate ("A clear and well-documented LaTeX document…"). Not written. |
| **1 Introduction** | ✅ Written, solid | Motivation + 4 contributions. Up to date per your note. Minor: see naming issue below. |
| **2 Background & Related Work** | ✅ Written, ~complete | Good coverage of MLPerf/EEMBC/AI-Benchmark/TPCx-AI. Two `rob` TODOs: add **AISBench** and the **HPI VLDB preprint** (TPCx-AI / TDIS vision lineage). |
| **3 The McBenchface framework** | ✅ Written, strong | 3.1 abstractions, 3.2 three-phase/17-step lifecycle, 3.3 extensibility. Two figures present. A few claims need small corrections (Part 3). |
| **4 Evaluation** | ⚠️ **Outline only** | A bullet list of intended measurements + Table 1 (noop depth) and Table 2 (EfficientNet overhead), both flagged by `rob` as placeholder/wrong. No prose. |
| **5 Experiments** | ⚠️ **Outline only** | Bullet list (scalability / collocation / ablation / bandwidth-tax) + a "Shared memory bandwidth tax" paragraph mentioning ColPali. No prose, no results. |
| **6 Discussion** | ❌ Empty header | |
| **7 Conclusion** | ❌ Empty header | |
| **References** | ❌ Empty | Only the acmart "Received…" stub. |

**Net:** the front half (motivation, related work, framework design) is in good shape. The
entire empirical back half (§4–§7) is unwritten, and that's exactly the part you flagged as
not-yet-aligned. The good news from the code/data audit: **two of the four studies are fully
run and written up**, so §4–§5 can largely be assembled from existing material rather than
waiting on new experiments.

---

## Part 2 — What the code and data actually support

### 2.1 The framework (§3) is real and matches the prose

The three abstractions (Stage / Pipeline / LoadGenerator) and the claims in §3 are backed by
code:

- **Thread-per-stage**: each `Stage` runs its own `threading.Thread` (`stages/stage.py:192`).
- **Queue wiring keyed by source stage**: edge A→B is a `PeekableQueue` living on B keyed by
  A's id (`stages/stage.py:132-147`) — this is what enables merge/first-submitted routing.
- **Two schedulers**, both real: `OfflineLoadScheduler` (closed-loop, Event-gated) and
  `PoissonLoadScheduler` (open-loop, `expovariate` arrivals, blocking backpressure). The
  base-class comment lists gamma/pareto/weibull as *future* — those are **not implemented**.
- **Polling policies** = the paper's "wait for all" / "first available":
  `MergePolicy`, `FirstSubmittedPolicy`, `SingleQueuePolicy` (`utils/queues/polling/`). Both
  multi-input policies correctly emit the terminator only after *all* upstreams drain — needed
  for cycles.
- **Non-linear graphs are genuinely supported**: `pipeline_configs/self_rag.yml` has fan-out
  routers (`outputs: [9, 18, 17]`), fan-in sinks, and a real feedback cycle (stage 19 →
  stage 3), bounded by `BinaryRouter`'s per-query `max_retries`.
- **RadT integration is real**: listeners `smi` (nvidia-smi), `macmon`, `free`, `iostat`,
  `dcgmi`, `top`, `ps` exist in the installed `radt` package and are selected per-config via
  `BenchmarkModel.listeners`. MLflow span tracing with `in_flow_id`/`out_flow_id` links slices
  across threads in Perfetto.

**Three corrections to make in §3 prose (details in Part 4):**

1. **Process vs thread.** Pipelines run as **separate OS processes** (RadT re-invokes
   `main.py -p <id>` once per pipeline, `main.py:204-227`); stages within a pipeline are
   **threads sharing one GIL**. The draft/README blur this. Figure 1 shows pipelines in
   separate boxes (correct) but the text should state the process/thread split explicitly
   because it matters for the collocation story (isolation is at process granularity).
2. **RadT is *not* a git submodule** — it's a pip dependency (`radt>=0.2.24`). The only
   submodule is `stages/anemll/Anemll`. (The draft doesn't claim this, but the README implies
   it; keep the paper accurate.)
3. **Latent `query_id` bug worth knowing before reproducibility claims**:
   `Query.query_id: int = uuid.uuid4()` (`utils/schemas/query.py:16`) is a mutable default
   evaluated once, so all queries share one id unless overridden — and schedulers don't
   override it. `BinaryRouter` keys retry counters on `query_id`, so multi-query Self-RAG runs
   can collide retry bookkeeping. Fix before final data collection (one-line: set per-query in
   the scheduler).

### 2.2 The four studies — what exists vs. what's been run

The `evaluation/README.md` defines four studies. Current status:

| # | Study | Apparatus | Data collected | Written up |
|---|---|---|---|---|
| 1 | **NoOp chain overhead** (depth + zero-copy payload) | ✅ full (generator + analyzers) | ⚠️ **partial** — only depths 1/10/50/100 + payload sweep at depth 10; missing 2/4/8/16/32/64 | LaTeX generator exists |
| 2 | **Modularity overhead** (EfficientNetV2-S, Choreo vs hand-written PyTorch) | ✅ full (baseline script + config + 3 analyzers) | ❌ **none on disk** (no `baseline_finetune.csv`, no training CSV) | — |
| 3 | **Multimodal VQA** (CLIP→FAISS→Qwen, MPS-only vs ANE-split, bandwidth contention) | ✅ full (configs, stages, CoreML export, analyzer w/ bandwidth-bound verdict logic) | ❌ **never run** (verification report says "no output file found") | — |
| 4 | **Self-RAG monolith vs decomposed** (factoid + multi-hop) | ✅ full | ✅ **complete** on **DGX Spark (GB10)** and **M2 Pro (MLX)**, 2×2×2, with quality verification | ✅ **two finished write-ups** (`monolith_vs_decomposed_{dgxspark,m2pro}.md`) |

**Key takeaways:**

- **Study 4 (Self-RAG) is your strongest, most finished result** — and it's *not even mentioned*
  in the draft's §5 bullets (which list scalability, collocation, RAG ablation, bandwidth tax).
  The §5 bullet "collocation analysis … RAG" and "ablation study for RAG — swap out models or
  databases" partly map to it, but the actual finished story is "monolith vs decomposition," a
  topology comparison with a clear, defensible verdict on two very different machines. This
  should anchor §5.
- **"Rosetta" = Self-RAG.** The older result files (`rosetta_*.csv`, `rosetta_bandwidth_report.md`)
  are an earlier naming of the same Study 4. Per your memory note, keep Rosetta (monolith vs
  decomposition) and VQA (bandwidth contention) as **distinct** stories — don't conflate. The
  `rosetta_bandwidth_report.md` files reuse the VQA "Mapping A/B" template but contain Self-RAG
  data; don't cite those, cite the clean `monolith_vs_decomposed_*.md` write-ups.
- **The bandwidth-contention thesis (VQA, Study 3) currently has apparatus but no measurements.**
  The draft's §5 "Shared memory bandwidth tax" paragraph is written around it (and mentions
  ColPali), but nothing has been run. Either run it or reframe (see Part 3).

### 2.3 Devices Under Test — only two of four exist

The draft's §4 bullet intends to "showcase hardware diversity … across different hardware (mac,
dgx spark, big a100 and h100)." Reality:

- ✅ **Apple Silicon (M2 Pro, 16 GB)** — full env (`environments/macos.yaml`), MLX/CoreML/ANE
  backends, real Self-RAG data.
- ✅ **DGX Spark (GB10 Grace Blackwell, 128 GB unified)** — full env (`environments/nvidia.yaml`,
  cu130 wheels for sm_121/aarch64), real Self-RAG data, GPU residency verified via `nvidia-smi pmon`.
- ❌ **A100** — **no references anywhere** in repo. Only covered generically by `device: cuda`.
  Would need a separate torch env (the nvidia.yaml pins cu130 for GB10/aarch64, not x86 A100).
- ❌ **H100** — **no references anywhere**. Same situation.

So "big a100 and h100" is currently **aspirational**. Decision needed (Part 3): scope to the two
machines you have, or commit to running on A100/H100.

---

## Part 3 — Critical gaps and decisions before writing §4–§7

These are the items that need your call; they shape what the back half says.

**G1 — Framework name. McBenchface vs Choreo.** The prose says *McBenchface*; Table 2 says
*Choreo Pipeline*; the code uses `CHOREO_OUTPUT_LABEL` and the eval docs call it *Choreo*; the
repo is "Collocation Benchmark." **Pick one name and globally replace.** This is a 5-minute fix
but must happen before anything else (it's in the title, contributions, every table).

**G2 — The collocation thesis is under-evidenced.** This is the most important strategic gap.
The paper's headline novelty is **collocation-aware profiling**: running *multiple independent
pipelines concurrently* and measuring interference (Figure 1 shows two pipelines side-by-side;
the abstract/intro lean on it). But of the collected data:
- Self-RAG "decomposed" is collocation *within one pipeline* (3 models, 1 process) — a topology
  result, not independent-workload interference.
- VQA mappings are *within one pipeline* too, and unrun.
- The genuine "two independent workloads contending" config — `torchvision_mixed.yml`
  (inference Poisson + training Offline, collocated) — **exists but has no results.**

  **So the central claim of the paper currently has no direct experiment.** Decision: either
  (a) run a real collocation-interference experiment (the mixed inference+finetune config is the
  obvious candidate — it's even in your §5 bullet "collocation analysis with inference load
  (e.g. agent) + fine-tuning"), or (b) reframe collocation as a *capability the framework
  exposes* and let Self-RAG's within-pipeline concurrency + the (to-be-run) VQA contention carry
  the empirical weight. I strongly recommend (a) — it directly substantiates the title.

**G3 — Tables 1 and 2 are placeholders.**
- **Table 1** (noop depth 1–64): current CSVs only cover depths 1/10/50/100, and even depth-1
  doesn't match the printed number (printed 0.055 ms; recompute from
  `noop_chain_depth_1.csv` ≈ 0.126 ms). Re-run the full sweep (2/4/8/16/32/64) on the *final*
  target machine and regenerate via `generate_latex_results.py`. The `rob` note already says
  "this table will change."
- **Table 2** (EfficientNetV2-S overhead): **no data at all** — `baseline_finetune.csv` and the
  training CSV don't exist. The printed numbers (negative overhead, P99 −10.79%) are the
  nonsensical placeholder you flagged ("waiting for a go-ahead from Ties after radt updates").
  Must be collected fresh after RadT fixes.

**G4 — VQA / bandwidth study: run or cut.** The §5 "bandwidth tax" paragraph and ColPali mention
presuppose data that doesn't exist. Note the apparatus uses **CLIP+FAISS+Qwen VQA**, not ColPali —
so even the written paragraph mis-describes the built experiment. Decision: run the VQA 2×2 (the
apparatus is ready, ~1 hour of Apple-Silicon runs) and rewrite the paragraph around CLIP/ANE, or
drop the bandwidth-tax claim. Given it's your stated thesis (memory note), running it is the
right move — it's the cheapest high-value data you can collect.

**G5 — DUT scope** (see 2.3). Recommend: scope the paper to **M2 Pro + DGX Spark** (a genuinely
interesting contrast: small unified-memory edge device vs large unified-memory Grace-Blackwell —
which is itself a nice "single-node scalability across very different hardware" story). Mention
A100/H100 as future work unless you actually run them.

**G6 — MLflow runs never closed / no scalar metrics.** All 26 runs are status `RUNNING`, metrics
table empty; all quantitative data lives in CSV traces + spans. Fine for analysis (the analyzers
read CSVs), but if any reviewer-facing claim depends on MLflow metric logging, note that timings
come from the trace CSVs, not MLflow scalars.

---

## Part 4 — Concrete next steps (per-section writing plan)

Ordered roughly by dependency. Items marked **[writing]** can be done now from existing material;
**[data]** needs a run first.

### Step 0 — global fixes [writing, fast]
- G1 name unification.
- Add a one-paragraph **Devices Under Test** subsection (new §4.1 or top of §5) describing M2 Pro
  and DGX Spark with the specs already in the write-ups (M2 Pro 16 GB; GB10 128 GB unified
  LPDDR5x; stacks: MLX 4-bit vs PyTorch 2.10+cu130 BF16). This is the missing DUT description you
  called out.

### §3 framework — small corrections [writing, fast]
- Add 1–2 sentences on the **process-per-pipeline / thread-per-stage** model (Part 2.1 #1). Tie it
  to collocation: isolation and resource listeners attach at the *process* boundary, which is why
  collocated pipelines can be profiled independently.
- Drop/soften any implication that RadT is a submodule.
- (Optional, methods/repro) note the per-query-id fix once applied.

### §4 Evaluation — rewrite from bullets to prose [partly writing, partly data]
Reframe §4 as **"does the framework itself stay out of the way?"** — i.e. overhead/fidelity
characterization, separate from the §5 case studies. Proposed structure:

- **§4.1 Devices Under Test** (Step 0).
- **§4.2 Per-stage framework overhead (NoOp depth scaling).** Prose around the regenerated
  Table 1. Message: per-stage overhead is ~tens of µs and *flat* in depth → the graph
  abstraction is cheap; you can build deep pipelines without measurement distortion. **[data: re-run depths 2–64]**
- **§4.3 Zero-copy payload passing.** The depth-10 ref-vs-copy payload sweep (1 KiB / 1 MiB /
  10 MiB) → reference passing is O(1) in payload size while deep-copy isn't. This is a clean,
  already-collected result and a nice systems detail. **[writing — data exists]**
- **§4.4 Modularity overhead vs hand-written PyTorch (Table 2).** EfficientNetV2-S Choreo-vs-baseline.
  Message: end-to-end per-batch latency overhead is within noise. **[data: must collect; fix the
  std-dev/negative-overhead anomaly Ties flagged]**

Replace the existing bullet list and the two placeholder tables accordingly. Keep Table 1/2 shells
but regenerate numbers.

### §5 Experiments — this is where the finished Self-RAG work goes [mostly writing!]
This section is in better shape than it looks, because Study 4 is done. Proposed structure:

- **§5.1 Topology study: monolith vs decomposition (Self-RAG).** *This is the anchor experiment
  and it's fully written up already.* Pull directly from `monolith_vs_decomposed_dgxspark.md` and
  `…_m2pro.md`. Headline results to feature:
  - Factoid (GB10): decomposed 3×4B beats monolith 9B — **−41% per-query latency, +86%
    throughput, equal golden-hit quality (20/30).**
  - Multi-hop (HotpotQA): the gap *widens* — decomposition **2× golden hits (10/30 vs 5/30)**,
    2× answered, still faster. Mechanism (from traces): monolith's forced-JSON pass grades
    multi-hop context "not relevant" and bails.
  - Cross-hardware: same verdict on M2 Pro (MLX 4-bit), where the monolith is "essentially
    non-functional" on multi-hop (2/10) — capability-constrained hardware *strengthens* the
    decomposition case.
  - This single study demonstrates **three** of your contributions at once: non-linear/cyclic
    graph support (the self-RAG retry loop), no-code component swapping (monolith↔decomposed by
    changing the config), and cross-hardware end-to-end measurement.
  - **Include the honest caveats** already in the write-up (conflates specialization with
    prompt-decomposition; retry loop fires on ~5% of questions; capacity-matched control is the
    recommended follow-up). These read as rigor, not weakness.
  - **[writing — data exists for both machines]**

- **§5.2 Bandwidth contention: unified-memory tax (VQA).** The CLIP→FAISS→Qwen 2×2
  (all-on-MPS vs CLIP-on-ANE × pipelined/serial). Message: heterogeneous engine mapping helps
  *only under contention*, exposing memory bandwidth as the binding constraint on unified-memory
  devices. **Rewrite the draft's "bandwidth tax" paragraph to describe CLIP/ANE, not ColPali.**
  **[data: must run — apparatus ready, ~1 hr on M2 Pro]**

- **§5.3 Collocation interference (inference + finetuning).** The genuine multi-pipeline
  contention experiment (`torchvision_mixed.yml`). Message: foreground inference latency degrades
  as a background finetune contends for the shared accelerator, and the framework's collocation-aware
  listeners make the interference *visible* (per-process util, not just aggregate). **This is what
  substantiates the paper's title.** **[data: must run; config exists, no results]**

This ordering puts your finished, strong result first (§5.1), then the two thesis experiments that
need short runs (§5.2, §5.3).

### §6 Discussion [writing, after §5]
Candidate threads, all grounded in collected data:
- **When does decomposition pay off?** Task-dependent; grows with difficulty and with hardware
  parallelism; shrinks on a single saturated accelerator. (Straight from the GB10 verdict.)
- **Unified memory as the new bottleneck** across both your DUTs (M2 Pro MPS and GB10) — the
  bandwidth story generalizes beyond Apple Silicon.
- **What end-to-end + collocation-aware measurement reveals that MLPerf-style isolation hides**
  (tie back to §1/§2 motivation): preprocessing/retrieval time-share, cross-query overlap,
  interference under collocation.
- Limitations: single-node scope, thread-per-stage GIL implications for CPU-bound stages, the
  retry-loop/multi-hop control-flow limitation, capacity-matched control as future work.

### §7 Conclusion [writing, last]
Restate: McBenchface/Choreo characterizes ML systems *end-to-end* and *under collocation* on a
single node; demonstrated via overhead characterization + three case studies across two very
different unified-memory machines; the framework is the contribution, the studies are evidence it
works. One forward-looking sentence (more DUTs incl. A100/H100, richer schedulers, multi-hop
control flow).

### §2 Related Work — close the two `rob` TODOs [writing + light research]
- Add **AISBench** (positioned as the closest existing end-to-end / multi-stream tool; argue your
  modularity + collocation-aware profiling + no-code graph reconfiguration go beyond it).
- Add the **HPI VLDB preprint** (the TPCx-AI / "TDIS" vision lineage that cites your group's
  vision paper) and situate your "where the field should move" framing against it.
- **References are empty — populate BibTeX** for everything already cited (MLPerf, EEMBC MLMark,
  AI-Benchmark, TPCx-AI, RadT, plus the two above, plus Self-RAG, HotpotQA, CLIP, Qwen, MLX,
  ChromaDB, FAISS for the experiments).

### Abstract [writing, last]
Write a real abstract (replace boilerplate): problem (isolated, accelerator-only, linear-graph
benchmarks miss end-to-end + collocation reality) → McBenchface (modular, declarative, RadT-integrated,
collocation-aware) → evidence (overhead is negligible; three case studies on M2 Pro + DGX Spark
show end-to-end/collocation/topology effects invisible to standard suites).

---

## Part 5 — Prioritized punch list

**Do first (writing only — no new runs needed):**
1. Unify the framework name (G1).
2. Write **§5.1 (Self-RAG monolith vs decomposition)** from the two existing write-ups — your
   strongest, fully-finished result.
3. Write **§4.3 (zero-copy payload)** from existing data.
4. Add the **DUT** subsection (M2 Pro + DGX Spark).
5. §3 process/thread correction.
6. Fill §2 AISBench + HPI; start the bibliography.

**Then (short, high-value data collection):**
7. Run the **VQA 2×2** (Study 3) — apparatus ready, ~1 hr on M2 Pro → enables §5.2 and fixes the
   bandwidth-tax paragraph (G4).
8. Run the **collocation interference** experiment (`torchvision_mixed.yml`) → enables §5.3, the
   experiment that backs the paper's title (G2).
9. Re-run the **NoOp depth sweep** 2–64 on the final machine → fix Table 1 (G3).
10. Apply the **`query_id` fix** before any final multi-query run (data integrity).

**Then (needs RadT fixes / Ties go-ahead):**
11. Collect **EfficientNet overhead** data → fix Table 2 (G3). The current negative-overhead
    numbers are the placeholder you flagged.

**Decisions to make explicitly:**
12. A100/H100: in scope or future work? (Recommend: future work; ship with M2 Pro + DGX Spark.)
13. Collocation: real interference experiment (recommended) or reframe as capability? (G2)

---

### One-line summary

The front half (intro, related work, framework design) is solid and the framework claims are
genuinely backed by code. The back half is unwritten — but **one major case study (Self-RAG
monolith vs decomposition) is already fully run and written up on two machines** and just needs
to be turned into §5 prose; the remaining experiments (VQA bandwidth, collocation interference,
EfficientNet overhead, full NoOp sweep) are scaffolded and need short runs. The biggest
*conceptual* gap is that the paper's headline — collocation-aware profiling — has no direct
experiment yet; running the mixed inference+finetune config is the highest-leverage thing you can
do to substantiate the title.

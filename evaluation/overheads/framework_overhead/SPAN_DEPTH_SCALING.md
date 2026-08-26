# Why the span cost appeared to scale with depth — and what it actually is

The `spans-only` arm's per-stage cost rises with depth (GB10, 2 cores, µs/stage
over `uninstrumented`): +4.9 at depth 8, +10.0 at 32, +15.8 at 128. Total spans
are linear in depth, so per-stage cost should be flat. It is not, and the
question is why.

## What it is not

Each row is a full sweep at depths 8 / 32 / 128, R=5, taskset -c 18,19.

| ablation | what it removes | result |
|---|---|---|
| `RADT_TRACE_COMPRESSLEVEL` 6 → 0 | the exporter child's gzip | no change (+15.8 → +17.9) |
| `attrs` — build the attribute dict, never emit | everything downstream of us | **flat: +1.5 / +0.9 / +1.1** |
| `noattrs` — emit with no attributes at all | the payload | still scales (+2.9 / +11.4 / +13.9) |
| `deque` — a lock-free deque instead of the queue | the multiprocessing queue | still scales (+5.6 / +12.6 / +15.5) |
| `gc.disable()` | the garbage collector | no change (+14.9 vs +15.4) |
| `onespan` — 1 span per stage instead of 3 | two thirds of the spans | **no change** (+15.5 vs +16.1) |

So it is not the exporter, not compression, not our attribute construction, not
the queue, not lock contention, not the payload size, not garbage collection,
and not the number of spans. Cutting spans per stage by 3x changes nothing.

## What it is

Replace the span entirely with a busy-loop of pure CPU — no syscall, no queue,
no allocation, nothing retained — sized to match one span emit:

| depth | pure CPU burn | real spans |
|--:|--:|--:|
| 8 | +4.30 | +4.90 |
| 32 | +9.47 | +9.98 |
| 128 | +14.76 | +15.78 |

Injected CPU reproduces the curve. The scaling is not a property of tracing at
all: it is what the pipeline does to *any* added per-stage work.

The reason is the GIL. Measured CPU utilisation with **two** cores available:

| depth | uninstrumented | with spans |
|--:|--:|--:|
| 8 | 44% | 96% |
| 32 | 104% | 101% |
| 128 | 100% | 112% |

It pins at ~100%, one core's worth, however many cores it is given. Only one
thread runs Python bytecode at a time, so d stage threads share a single
serialised execution budget, and work added per stage is amplified by every
other thread's wait for it — roughly 1x at depth 8 and 3x at depth 128.

This is also why more cores did not help: 1, 2, 3 and 10 cores were all
equal-or-worse, which is what a GIL limit looks like and not a core-count limit.

## Consequences

**Do not optimise the span path for this.** Six ablations say the cost is not
in it. Fewer spans, a cheaper sink, no compression and no GC all leave the
curve unchanged.

**The comparison remains valid.** The amplification applies to the framework's
own work identically, so `uninstrumented` vs `spans-only` at a given depth is
measured in one regime and the difference is real. What must not be done is
quoting a single slope for `spans-only` — the fit is rejected for exactly this
reason — or extrapolating its per-stage cost to a depth that was not measured.

**Removing it means escaping the GIL**, not tuning tracing: free-threaded
CPython, or stages in processes rather than threads. That is a framework
change, out of scope here, and worth stating as a limit of a thread-per-stage
design rather than a defect of the tracing.

## Not established

The precise mechanism of the amplification — GIL hand-off latency versus
run-queue delay — is not pinned down. `sys.setswitchinterval` would separate
them and has not been run.

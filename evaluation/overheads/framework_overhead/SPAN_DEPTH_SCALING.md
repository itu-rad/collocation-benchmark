# Why the span cost scales with depth

`spans-only` costs more per stage as the pipeline gets deeper, on both devices,
even though the number of spans is linear in depth. This records what it is.

## Current production numbers

Two spans per stage (`.run`, `.push_to_outputs`), 8 depths x 11 runs. Cost of
running with tracing, in microseconds per stage over `uninstrumented`:

| depth | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| GB10 | +1.05 | +0.33 | +11.24 | +1.04 | +2.64 | +4.68 | +8.08 | **+9.71** |
| M2 Pro | +61.52 | +38.38 | +27.79 | +22.21 | +20.20 | +20.16 | +26.17 | **+41.80** |

(The GB10 depth-4 cell at +11.24 sits far off its neighbours and is noise, not
shape.) The GB10 curve rises from depth 16 on; the M2 Pro falls to a minimum
near depth 32 and then climbs. Both are the same effect, identified below.

> **The ablation tables below were measured with THREE spans per stage**,
> before `.get_input` was removed, so their `full` column is roughly 1.6x
> today's production cost: GB10 +15.68 against +9.71 at depth 128, M2 Pro
> +62.89 against +41.80. They are kept because they identify the MECHANISM,
> which removing that span did not change - and because the removal is itself
> the cleanest confirmation of what they found (see "Confirmed in production").

> **Correction.** An earlier version of this document concluded the opposite -
> that the scaling was not the tracing path at all but the thread chain
> reacting to any added work. That was wrong, and the cause was a bug in the
> ablation build, not in the data. The variants were dispatched as separate
> `if` blocks sharing one trailing `else`, so every branch except the last was
> silently overwritten with the full path. `burn`, `deque` and `onespan` were
> all running `full`, which is precisely why they "reproduced the curve
> exactly". The build now uses a single if/elif chain and each variant is
> verified to bind a distinct implementation before use.

## The ablation, both devices

µs/stage over `uninstrumented`, R=5. GB10 pinned to cores 18,19; M2 Pro
unpinned.

| variant | what runs | GB10 @8 | @32 | @128 | M2 @8 | @32 | @128 |
|---|---|--:|--:|--:|--:|--:|--:|
| `attrs` | build the attribute dict, never emit | +1.29 | +0.93 | +1.01 | +1.82 | +4.15 | +0.87 |
| `deque` | lock-free deque, radt bypassed | +2.31 | +2.37 | +2.09 | +2.73 | +6.97 | +2.06 |
| `noqueue` | all of radt's per-span machinery, emit no-op | +7.37 | +6.58 | +6.30 | +13.32 | +14.63 | +8.53 |
| `burn` | fixed pure CPU instead of emitting | +7.92 | +6.85 | +6.61 | +12.64 | +14.49 | +9.91 |
| `onespan` | 1 span per stage instead of 3 | +10.11 | +8.63 | +5.96 | +15.37 | +10.51 | +32.85 |
| `full` | production | +4.51 | +11.99 | **+15.68** | +26.53 | +24.08 | **+62.89** |

Read the rows that are FLAT and the one that is not:

* Our own attribute work is ~+1 µs/stage and flat. Negligible.
* A lock-free sink is ~+2 µs/stage and flat.
* radt's per-span bookkeeping *without* the emit — contextvar stack, the
  lock-guarded id counter, the generator context manager — costs +6-7 µs/stage
  on the GB10 and +9-15 on the M2 Pro, and **does not scale with depth**.
* Pure CPU of a similar size is likewise flat, so added work per se does not
  scale either.
* Only `full` scales. `full` minus `noqueue` is the emit alone: **+9.4 µs/stage
  on the GB10 at depth 128, +54 on the M2 Pro.**

So the depth scaling is the multiprocessing-queue emit.

## It is proportional to event count, not to producer count

Two further results say what kind of cost it is.

**Fewer events, proportionally less cost.** `onespan` emits 2 events per stage
instead of 6. At depth 128 it costs +5.96 against `full`'s +15.68 on the GB10
(38%) and +32.85 against +62.89 on the M2 Pro (52%). Cutting events cuts cost.

**Funnelling the puts through one thread does not help at all.** A prototype
staged every event on a lock-free deque and had a single drainer thread perform
every queue put, delivering all of them (high-water mark 139 events at depth
128, so the drainer kept up easily):

| depth | production | staged through one drainer |
|--:|--:|--:|
| 8 | +25.37 | +25.19 |
| 64 | +51.70 | +52.70 |
| 128 | +63.43 | +60.49 |

Identical. So this is not producer-side lock contention — moving the puts to
one thread changes nothing. It is the total cost of pushing N events through
the queue, the pipe and the exporter, and the workload pays it wherever the
put happens.

## What that means for a fix

Replacing the queue with a deque is not a fix: the deque only looked free
because it never delivered anything, and the staged version that does deliver
is no better than what we have.

The lever that works is **fewer events**, since cost tracks event count.

### Confirmed in production

`.get_input` was removed on exactly that basis. It was spanned but consumed by
nothing; it could never be correlated to a query, because attributes are fixed
at span start and at that instant the stage has not received one; and it was
the span parked threads sat inside at teardown. Removing it took spans per
stage from 3 to 2 - **one third fewer events** - and the measured cost fell:

| depth 128 | before | after | |
|---|--:|--:|--:|
| GB10 | +16.16 | +9.71 | -40% |
| M2 Pro | +63.85 | +41.80 | -35% |

A third fewer events buying back a third to two fifths of the cost is the
ablation's conclusion holding in production rather than in a scratch build.
The span count was checked directly rather than assumed: 26262 events per run
at depth 128 on both devices against 39318 before, exactly two thirds.

Going further means batching several span events into one queue put, cutting
puts without cutting information. `.push_to_outputs` must NOT be dropped to get
there: it is the boundary between a stage's own work and the hand-off, and that
split is the finding - the hand-off dominates and the stages' own work is a
small minority of L_q.

The emit lives inside radt, so batching means changing radt or staging events
before they reach it. That is a decision, not a conclusion, and is left open.

## Also ruled out

Exporter gzip level (6 vs 0: no change), garbage collection (`gc.disable()`: no
change), span drops (radt reported zero across both sweeps, emitted counts
exactly linear in depth), queue capacity (never filled; lowering it 8x changed
nothing), and GIL preemption (`sys.setswitchinterval` varied 500x: amplification
unchanged at 3.05x / 3.12x / 3.18x). Those tests did not use the broken
dispatch and stand.

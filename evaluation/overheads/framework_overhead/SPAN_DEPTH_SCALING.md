# Why the span cost scales with depth

`spans-only` costs more per stage as the pipeline gets deeper, on both devices,
even though the number of spans is linear in depth. This records what it is.

> **Correction.** An earlier version of this document concluded the opposite —
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

The lever that works is **fewer, larger queue items** — batching several span
events into one put — since cost tracks event count. Emitting fewer spans per
stage is the crude version of that and already buys back roughly its share.

The emit lives inside radt, so acting on this means changing radt or staging
events before they reach it. That is a decision, not a conclusion, and is left
open.

## Also ruled out

Exporter gzip level (6 vs 0: no change), garbage collection (`gc.disable()`: no
change), span drops (radt reported zero across both sweeps, emitted counts
exactly linear in depth), queue capacity (never filled; lowering it 8x changed
nothing), and GIL preemption (`sys.setswitchinterval` varied 500x: amplification
unchanged at 3.05x / 3.12x / 3.18x). Those tests did not use the broken
dispatch and stand.

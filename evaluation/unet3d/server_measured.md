# Measured open-loop Server-scenario latency (upgrades §A.4 from simulation)

Real MLPerf LoadGen **Server** scenario (open-loop Poisson arrivals), 3D-UNet/KiTS19,
FIFO (synchronous SUT processes 1 query at a time), GB10. run.py FromConfig bug fixed
(load only user.conf, conf_type=1). Latency = queue-wait (issue_delay) + service (issue_to_done),
parsed from mlperf_log_trace.json (86 Sample events), cross-checked vs mlperf_log_summary.txt.

Service (unloaded) distribution: 8–144 subvolumes, min 0.99s, mean **7.69s**, p99 18.4s (18× spread).

## Point 1 — ρ ≈ 0.82 (qps=0.106), 86 queries  [logs_rho082]
- Result INVALID / "performance constraints satisfied: NO" — the POINT, not a failure: latency unbounded at this load.
- TOTAL latency: p50 11.9s, p90 34.0s, p95 42.9s, p99 52.9s (loadgen summary — matches trace parse).
- QUEUE wait alone (HoL): mean 7.1s, p90 19.5s, **max 43.5s**.
- Routine studies (n=48, ≤median subvol, service mean 4.5s): latency p90 22.9s, **max 49.1s ≈ 11× their service**.
- Large studies (n=38, service mean 11.8s): latency p90 42.9s, inflation 1.7× (slow regardless).
- Aggregate mean latency 14.8s = 1.9× mean service 7.69s.

## Point 2 — ρ ≈ 0.35 (qps=0.045, S=7.66s), 86 queries  [logs_rho035]
- Result INVALID (no latency target met, same as point 1).
- TOTAL latency: p50 7.4s, p90 18.0s, p95 19.7s, p99 24.1s, max 24.1s.
- QUEUE wait (HoL): mean 1.6s, p50 0.0s (most served immediately), p90 8.3s, max 16.7s.
- Routine studies (n=48, service mean 4.5s): latency p90 10.6s, max 18.6s, qwait mean 1.3s.

## Measured p99-vs-load contrast (the story)
| metric | ρ≈0.35 | ρ≈0.82 |
|---|---|---|
| p90 latency | 18.0s | 34.0s |
| p99 latency | 24.1s | 52.9s |
| queue-wait mean / max | 1.6s / 16.7s | 7.1s / 43.5s |
| routine-study lat p90 / max | 10.6s / 18.6s | 22.9s / 49.1s |

Tail GROWS with load: p90 latency doubles, HoL queue-wait tail 16.7→43.5s (exceeds the model's own
18.4s max service). Measured inflation (~2–4× at ρ=0.35, up to ~11× routine worst-case at ρ=0.82) is
MILDER than the retired M/G/1 estimate (~9–10× at ρ=0.35) — report measured. n≈86 → p99≈max
undersampled, lead with p90/p95; measured tail is a lower bound.

## Framing for §A.4 (replace "simulated (M/G/1)")
MEASURED, not modeled: under real open-loop Server load the excluded head-of-line effect is
first-order — queue wait reaches 43.5s (>2× the whole model's 18.4s max service), inflating a
routine study's latency up to ~11×. MLPerf's Offline throughput for 3D-UNet is invariant to this.
Note: p99=max at n=86 (undersampled tail); p90/p95 are the robust tail statistics. SJF relief under
Server load needs an async queue+worker SUT (future work); SJF benefit is measured in Offline (§A.2).

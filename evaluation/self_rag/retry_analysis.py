"""Retry-count vs rescuability from an existing Self-RAG run log + outputs.

Parses per-query retry counts (MonolithRouter "retry (remaining=N)" / "exhausted
retries" markers) and correlates with final EM. MEASURED result (cuda multihop
monolith, R=1): 98% of all retries were spent on queries that ended INCORRECT;
retried queries EM 0.041 vs never-retried 0.380. The retry loop identifies hard
queries (triggers retries) but cannot rescue them, so its compute is ~entirely
anti-correlated with benefit. Offline post-hoc; no new runs.

    python evaluation/self_rag/retry_analysis.py <cell>.log <cell>_outputs.jsonl
"""
import re, json, sys
sys.path.insert(0, "evaluation/scripts")
from score_quality import exact_match

log, outs = sys.argv[1], sys.argv[2]
retries, exhausted = {}, set()
for line in open(log, errors="ignore"):
    m = re.search(r"query ([0-9a-f-]{36}) retry \(remaining=(\d+)", line)
    if m: retries[m.group(1)] = retries.get(m.group(1), 0) + 1
    m2 = re.search(r"query ([0-9a-f-]{36}) exhausted retries", line)
    if m2: exhausted.add(m2.group(1))
em = {}
for line in open(outs):
    d = json.loads(line); q = str(d.get("query_id"))
    ans = d.get("generated_answer") or ""
    em[q] = exact_match(ans, [str(g) for g in (d.get("golden_answers") or [])]) if ans else 0
rc = {q: retries.get(q, 0) for q in em}
tot = sum(rc.values())
retried = [q for q in em if rc[q] > 0]; never = [q for q in em if rc[q] == 0]
on_wrong = sum(rc[q] for q in em if em[q] == 0)
print(f"queries={len(em)} total_retries={tot}")
print(f"never-retried: n={len(never)} EM={sum(em[q] for q in never)/max(len(never),1):.3f}")
print(f"retried:       n={len(retried)} EM={sum(em[q] for q in retried)/max(len(retried),1):.3f}")
print(f"{100*on_wrong/max(tot,1):.0f}% of retries spent on queries that ended INCORRECT")

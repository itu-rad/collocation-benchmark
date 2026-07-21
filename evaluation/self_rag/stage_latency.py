"""Per-stage latency breakdown for a Self-RAG trace CSV: does the self-critique
machinery (grader/rewriter LLM calls) dominate the answer generator?

Trace format (utils/logger.py): created, parent, stage, phase, start|end, perf_ns.
Result (cuda factoid decomposed, R=1): self-critique LLM calls (relevance +
hallucination graders + query rewriter) = ~212s vs answer generator ~110s
(~1.9x), retrieval ~16s (negligible). The RAG analog of "benchmarks measure the
wrong stage": everyone times the answer LLM; the grading machinery costs ~2x more.

    python evaluation/self_rag/stage_latency.py <trace.csv>
"""
import csv, sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.reader(open(path)))
opens = defaultdict(list); total = defaultdict(float); count = defaultdict(int)
for r in rows:
    if len(r) < 6:
        continue
    stage, phase, se = r[2].strip(), r[3].strip(), r[4].strip()
    try:
        perf = int(r[5])
    except ValueError:
        continue
    if phase != "run":
        continue
    if se == "start":
        opens[stage].append(perf)
    elif se == "end" and opens[stage]:
        total[stage] += (perf - opens[stage].pop(0)) / 1e9
        count[stage] += 1

print(f"{'stage':38s} {'total_s':>9} {'n':>5} {'mean_ms':>9}")
for s in sorted(total, key=lambda k: -total[k]):
    if "pipeline" in s.lower():
        continue
    print(f"{s:38s} {total[s]:>9.1f} {count[s]:>5} {1000*total[s]/max(count[s],1):>9.1f}")

def bucket(words, excl=()):
    return sum(v for k, v in total.items() if any(w in k.lower() for w in words)
               and not any(e in k.lower() for e in excl) and "pipeline" not in k.lower())
critique = bucket(["grad", "router", "rewrit", "relevance", "hallucinat"])
gen = bucket(["generat", "answer", "inference"], excl=["grad", "formatter", "router"])
retr = bucket(["retriev", "chroma"])
print(f"\nself-critique (graders+rewriter) : {critique:6.1f}s")
print(f"answer generator                 : {gen:6.1f}s  -> critique is {critique/max(gen,1e-9):.1f}x the generator")
print(f"retrieval                        : {retr:6.1f}s")

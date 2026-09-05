#!/usr/bin/env python3
"""LLM-judge runner for the 5.1 quality column — extract, judge, score.

Why this exists. The judged accuracy figures were produced once, by hand, and
`judge/README.md` pointed at an `llm-judge-selfrag` workflow that does not exist
in this repository. A headline quality number that nobody can re-derive is not
reportable, so this script is the committed path from run outputs to the numbers
in judge/manifest.json.

    # 1. run outputs -> judge inputs (no API, deterministic)
    python evaluation/self_rag/run_judge.py extract --arm cuda_factoid_monolith \
        --runs 'e4_factoid_monolith_serial_cuda_r*'

    # 2. inputs -> verdicts (calls the judge; needs ANTHROPIC_API_KEY)
    python evaluation/self_rag/run_judge.py judge --cell cuda_factoid_monolith

    # 3. verdicts -> the table, and a manifest cross-check
    python evaluation/self_rag/run_judge.py score

Step 3 runs with no API access and re-derives every published number from the
committed verdicts, which is the part that makes the claim checkable. Steps 1
and 2 are only needed if the pipeline configs change; decoding is greedy, so
quality does not move between timing collections.

Judging protocol (fixed here so it cannot drift):
  * model            claude-haiku-4-5, one judge per cell
  * unit             one (question, gold[], candidate) triple
  * rubric           FACTUAL EQUIVALENCE only -- not style, length or phrasing
  * anchoring guard  the judge sees only q/gold/ans, never EM or F1 fields
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE_DIR = os.path.join(HERE, "judge")
RESULTS_DIR = os.path.join(HERE, "results")

JUDGE_MODEL = "claude-haiku-4-5"
ERROR_MARKERS = ("[error", "retries exhausted", "<error")

SYSTEM_PROMPT = (
    "You grade short answers for FACTUAL EQUIVALENCE against a gold answer. "
    "Ignore style, length, punctuation and phrasing. Answer YES if the candidate "
    "states the same fact as any gold answer, NO if it states a different or "
    "absent fact. Judge each item independently. Reply with exactly YES or NO."
)


def _best_answer(r):
    """Same candidate-extraction rule the EM scorer uses; kept identical on
    purpose so the judge and EM grade the same string."""
    for candidate in (r.get("generated_answer"), r.get("final_data")):
        if candidate is None:
            continue
        if isinstance(candidate, list):
            candidate = " ".join(str(x) for x in candidate)
        candidate = str(candidate).strip()
        if not candidate:
            continue
        if any(m in candidate.lower() for m in ERROR_MARKERS):
            continue
        return candidate
    return None


def cmd_extract(args):
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{args.runs}_outputs.jsonl")))
    if not paths:
        sys.exit(f"no outputs matched {args.runs}_outputs.jsonl under {RESULTS_DIR}")
    items, seen = [], set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r.get("question")
            if q is None or q in seen:
                continue
            seen.add(q)
            items.append({"i": len(items), "q": q,
                          "gold": r.get("golden_answers") or [],
                          "ans": _best_answer(r)})
    out = os.path.join(JUDGE_DIR, f"{args.arm}_input.json")
    json.dump(items, open(out, "w"), indent=1)
    print(f"wrote {out}  ({len(items)} items from {len(paths)} run file(s))")


def _judge_via_cli(items, cell):
    """Grade a cell with the Claude Code CLI (subscription, no API key).

    This is how the published verdicts were actually produced -- judge/README.md
    records "one judge per cell" -- and the project has a subscription rather
    than pay-per-use, so the API path below cannot run here. One call grades the
    whole cell, which is also what "one judge per cell" means.
    """
    import subprocess, textwrap
    lines = [f'{it["i"]}\tQ: {it["q"]}\tGOLD: {"; ".join(map(str, it["gold"]))}'
             f'\tCANDIDATE: {it["ans"]}' for it in items if it.get("ans")]
    blank = [it["i"] for it in items if not it.get("ans")]
    prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Grade every numbered item below. Reply with ONLY a JSON array, one object "
        'per item, of the form {"i": <number>, "correct": true|false}. No prose, no '
        "code fence.\n\n" + "\n".join(lines))
    proc = subprocess.run(["claude", "-p", prompt],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        sys.exit(f"claude CLI failed rc={proc.returncode}: {proc.stderr[-400:]}")
    txt = proc.stdout.strip()
    start, end = txt.find("["), txt.rfind("]")
    if start < 0 or end < 0:
        sys.exit(f"judge did not return a JSON array for {cell}:\n{txt[:400]}")
    got = {int(v["i"]): bool(v["correct"]) for v in json.loads(txt[start:end + 1])}
    # An item the judge skipped is not silently a pass.
    missing = [it["i"] for it in items if it["i"] not in got and it["i"] not in blank]
    if missing:
        sys.exit(f"judge omitted items {missing[:10]} for {cell}; refusing partial verdicts")
    return [{"i": it["i"], "correct": (False if it["i"] in blank else got[it["i"]])}
            for it in items]


def cmd_judge(args):
    inp = os.path.join(JUDGE_DIR, f"{args.cell}_input.json")
    if not os.path.exists(inp):
        sys.exit(f"missing {inp} -- run `extract` first")
    if args.via == "cli":
        items = json.load(open(inp))
        verdicts = _judge_via_cli(items, args.cell)
        out = os.path.join(JUDGE_DIR, f"verdicts_{args.cell}.json")
        json.dump(verdicts, open(out, "w"), indent=1)
        n = sum(v["correct"] for v in verdicts)
        print(f"wrote {out}  ({n}/{len(verdicts)} judged correct)")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is unset; judging needs it. "
                 "`score` re-derives the published numbers without it.")
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic()
    items = json.load(open(inp))
    verdicts = []
    for it in items:
        if not it.get("ans"):
            verdicts.append({"i": it["i"], "correct": False})
            continue
        msg = (f"Question: {it['q']}\n"
               f"Gold answer(s): {'; '.join(map(str, it['gold']))}\n"
               f"Candidate answer: {it['ans']}\n\nYES or NO?")
        resp = client.messages.create(
            model=JUDGE_MODEL, max_tokens=5,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": msg}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip().upper()
        verdicts.append({"i": it["i"], "correct": text.startswith("YES")})
    out = os.path.join(JUDGE_DIR, f"verdicts_{args.cell}.json")
    json.dump(verdicts, open(out, "w"), indent=1)
    n = sum(v["correct"] for v in verdicts)
    print(f"wrote {out}  ({n}/{len(verdicts)} judged correct)")


def cmd_score(args):
    """Re-derive every published number from the committed verdicts."""
    rows = []
    for vp in sorted(glob.glob(os.path.join(JUDGE_DIR, "verdicts_*.json"))):
        cell = os.path.basename(vp)[len("verdicts_"):-len(".json")]
        verdicts = json.load(open(vp))
        judged = sum(v["correct"] for v in verdicts) / len(verdicts)
        ip = os.path.join(JUDGE_DIR, f"{cell}_input.json")
        em = None
        if os.path.exists(ip):
            items = {it["i"]: it for it in json.load(open(ip))}
            # EM here is only the anchor the judge is compared against; the
            # scorer of record is evaluation/scripts/score_quality.py.
            hits = 0
            for v in verdicts:
                it = items.get(v["i"])
                if it and it.get("ans"):
                    a = str(it["ans"]).strip().lower().rstrip(".")
                    hits += any(a == str(g).strip().lower().rstrip(".")
                                for g in it["gold"])
            em = hits / len(verdicts)
        rows.append((cell, len(verdicts), em, judged))

    if not rows:
        sys.exit(f"no verdicts_*.json in {JUDGE_DIR}")
    print(f"{'cell':<34} {'n':>4} {'EM':>7} {'judged':>7} {'rescued':>8}")
    for cell, n, em, judged in rows:
        e = f"{em:.3f}" if em is not None else "  -  "
        r = f"{judged - em:+.3f}" if em is not None else "   -  "
        print(f"{cell:<34} {n:>4} {e:>7} {judged:>7.3f} {r:>8}")

    mp = os.path.join(JUDGE_DIR, "manifest.json")
    if os.path.exists(mp):
        man = {m["cell"]: m for m in json.load(open(mp))}
        drift = [(c, man[c].get("em"), em) for c, _, em, _ in rows
                 if c in man and em is not None
                 and man[c].get("em") is not None
                 and abs(man[c]["em"] - em) > 0.02]
        print("\nmanifest cross-check:",
              "consistent" if not drift else f"DRIFT in {len(drift)} cell(s)")
        for c, was, now in drift:
            print(f"  {c}: manifest EM {was} vs recomputed {now:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--arm", required=True); e.add_argument("--runs", required=True)
    j = sub.add_parser("judge");   j.add_argument("--cell", required=True)
    j.add_argument("--via", choices=["cli", "api"], default="cli",
                   help="cli = Claude Code CLI on the project's subscription "
                        "(default, and how the published verdicts were made); "
                        "api = anthropic SDK, needs ANTHROPIC_API_KEY")
    sub.add_parser("score")
    args = ap.parse_args()
    {"extract": cmd_extract, "judge": cmd_judge, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()

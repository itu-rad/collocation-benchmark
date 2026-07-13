#!/usr/bin/env python3
"""Lock derived knob values into the committed configs (from knobs.yml).

Regenerates ONLY each target file's ``loadgen:`` block(s) — every other byte is
left untouched — replacing the stale hand-written comments inside those blocks
with a provenance header. Non-loadgen config knobs (top_k, max_retries) are
VERIFIED against the configs, never rewritten (they are workload-defining and
already set from precedent).

Safety: after patching, the file is re-parsed and asserted equal to the
original everywhere except the loadgen subtree; the new subtree must equal the
knobs.yml intent, carry an explicit ``rate`` (killing the silent rate=3.0
default), and validate against utils.schemas.loadgen.LoadGenModel.

    python evaluation/pilots/apply_knobs.py --dry-run     # show unified diffs
    python evaluation/pilots/apply_knobs.py               # write in place
"""

from __future__ import annotations

import argparse
import copy
import difflib
import glob as globmod
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
import pilot_lib as pl  # noqa: E402


# ---------------------------------------------------------------------------
# knobs.yml -> per-file loadgen intents
# ---------------------------------------------------------------------------

def collect_intents(knobs: dict) -> dict:
    """{config_path: {pipeline_scope: {field: (value, rule, inputs)}}}.

    pipeline_scope: 'all' for plain loadgen.* knobs, 0 for fg.loadgen.* (first
    pipeline of a multi-pipeline config).
    """
    intents: dict = {}
    for exp, devices in (knobs.get("experiments") or {}).items():
        for dev, entries in (devices or {}).items():
            for e in entries or []:
                if e.get("scope") != "config" or e.get("value") is None:
                    continue
                name = e["knob"]
                if name.startswith("fg.loadgen."):
                    scope, field = 0, name[len("fg.loadgen."):]
                elif name.startswith("loadgen."):
                    scope, field = "all", name[len("loadgen."):]
                else:
                    continue  # verified-only knobs (top_k, ...) handled elsewhere
                for pattern in e.get("applies_to") or []:
                    for path in sorted(globmod.glob(str(REPO_ROOT / pattern))):
                        rel = str(Path(path).relative_to(REPO_ROOT))
                        intents.setdefault(rel, {}).setdefault(scope, {})[field] = (
                            e["value"], e["rule"], e.get("inputs") or {})
    return intents


def verify_only_knobs(knobs: dict) -> list[str]:
    """Check top_k / max_retries in configs match knobs.yml; return problems."""
    problems = []
    for exp, devices in (knobs.get("experiments") or {}).items():
        for dev, entries in (devices or {}).items():
            for e in entries or []:
                if e.get("scope") != "config" or e["knob"] not in ("top_k", "max_retries"):
                    continue
                # spot-check against the self_rag configs; a dict value keys
                # the expectation by task prefix (e.g. top_k: {factoid: 3,
                # multihop: 5} — retrieval differs by difficulty by design)
                for path in sorted(globmod.glob(
                        str(REPO_ROOT / "evaluation/self_rag/configs/*.yml"))):
                    name = Path(path).name
                    expected = e["value"]
                    if isinstance(expected, dict):
                        task = next((t for t in expected if name.startswith(t)), None)
                        if task is None:
                            continue
                        expected = expected[task]
                    text = Path(path).read_text(encoding="utf-8")
                    key = f"{e['knob']}:"
                    for line in text.splitlines():
                        s = line.strip()
                        if s.startswith(key):
                            val = s.split(":", 1)[1].split("#")[0].strip()
                            if val and val != str(expected):
                                problems.append(
                                    f"{name}: {e['knob']}={val} "
                                    f"!= knobs {expected}")
    return problems


# ---------------------------------------------------------------------------
# loadgen block surgery
# ---------------------------------------------------------------------------

def find_loadgen_blocks(lines: list[str]) -> list[tuple[int, int, int]]:
    """[(start_line, end_line_exclusive, indent)] for each 'loadgen:' block,
    in document order (= pipeline order)."""
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].split("#")[0].rstrip()
        if stripped.endswith("loadgen:") and stripped.lstrip() == "loadgen:":
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines):
                l = lines[j]
                if l.strip() == "" or l.lstrip().startswith("#"):
                    # blank/comment: part of block only if a deeper line follows
                    k = j
                    while k < len(lines) and (lines[k].strip() == ""
                                              or lines[k].lstrip().startswith("#")):
                        k += 1
                    if k < len(lines) and len(lines[k]) - len(lines[k].lstrip()) > indent:
                        j = k
                        continue
                    break
                if len(l) - len(l.lstrip()) <= indent:
                    break
                j += 1
            blocks.append((i, j, indent))
            i = j
        else:
            i += 1
    return blocks


def render_block(existing: dict, fields: dict, indent: int, commit: str) -> list[str]:
    pad = " " * indent
    p2 = " " * (indent + 2)
    p4 = " " * (indent + 4)
    component = fields.get("component", (existing.get("component"),))[0] \
        if "component" in fields else existing.get("component")
    get = lambda k, d=None: fields[k][0] if k in fields else existing.get(k, d)
    rate = fields["config.rate"][0] if "config.rate" in fields else \
        (existing.get("config") or {}).get("rate")
    out = [f"{pad}loadgen:\n",
           f"{p2}# [knobs] derived by evaluation/pilots/derive_knobs.py @ {commit}\n"]
    for k in ("config.rate", "max_queries", "queue_depth", "timeout"):
        if k in fields:
            _, rule, inputs = fields[k]
            hint = ""
            if k == "config.rate" and inputs:
                med = inputs.get("median_service_s")
                rho = inputs.get("rho")
                if med and rho:
                    hint = f" ({rho} x 1/{med}s, pilot {inputs.get('pilot', '?')})"
            out.append(f"{p2}# [knobs] {k}: {rule}{hint}\n")
    out.append(f"{p2}# [knobs] DO NOT hand-edit; regenerate via apply_knobs.py\n")
    out.append(f"{p2}component: {component}\n")
    out.append(f"{p2}queue_depth: {get('queue_depth')}\n")
    out.append(f"{p2}max_queries: {get('max_queries')}\n")
    out.append(f"{p2}timeout: {get('timeout')}\n")
    out.append(f"{p2}config:\n")
    out.append(f"{p4}rate: {rate}\n")
    return out


def strip_loadgen(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    for pipe in d.get("pipelines", []):
        pipe.pop("loadgen", None)
    return d


def patch_file(rel_path: str, scopes: dict, commit: str, dry_run: bool) -> bool:
    path = REPO_ROOT / rel_path
    original = path.read_text(encoding="utf-8")
    before_doc = yaml.safe_load(original)
    lines = original.splitlines(keepends=True)
    blocks = find_loadgen_blocks(lines)
    if not blocks:
        print(f"[FAIL] {rel_path}: no loadgen block found")
        return False

    new_lines = list(lines)
    for bi in reversed(range(len(blocks))):
        fields = dict(scopes.get("all", {}))
        if bi in scopes:
            fields.update(scopes[bi])
        if not fields:
            continue
        start, end, indent = blocks[bi]
        existing = before_doc["pipelines"][bi]["loadgen"]
        new_lines[start:end] = render_block(existing, fields, indent, commit)

    updated = "".join(new_lines)
    after_doc = yaml.safe_load(updated)

    # Safety checks.
    if strip_loadgen(before_doc) != strip_loadgen(after_doc):
        print(f"[FAIL] {rel_path}: non-loadgen content changed — aborting this file")
        return False
    from utils.schemas.loadgen import LoadGenModel
    for bi, pipe in enumerate(after_doc["pipelines"]):
        lg = pipe.get("loadgen")
        if lg is None:
            continue
        LoadGenModel(**lg)
        fields = dict(scopes.get("all", {}))
        if bi in scopes:
            fields.update(scopes[bi])
        for k, (v, _, _) in fields.items():
            got = (lg.get("config") or {}).get("rate") if k == "config.rate" else lg.get(k)
            if got != v:
                print(f"[FAIL] {rel_path}: pipeline {bi} {k}={got} != intent {v}")
                return False
        if fields and (lg.get("config") or {}).get("rate") is None \
                and lg.get("component", "").endswith("PoissonLoadScheduler"):
            print(f"[FAIL] {rel_path}: Poisson block without explicit rate")
            return False

    if dry_run:
        diff = difflib.unified_diff(original.splitlines(keepends=True),
                                    new_lines, fromfile=rel_path,
                                    tofile=rel_path + " (knobs)")
        sys.stdout.writelines(diff)
        print()
    else:
        path.write_text(updated, encoding="utf-8")
        print(f"[ok  ] {rel_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    knobs = pl.load_knobs()
    if not knobs:
        sys.exit("knobs.yml not found — run derive_knobs.py first")
    commit = knobs.get("git_commit", "unknown")
    intents = collect_intents(knobs)
    if not intents:
        sys.exit("no config-scope knobs with values in knobs.yml")

    ok = True
    for rel_path, scopes in sorted(intents.items()):
        ok &= patch_file(rel_path, scopes, commit, args.dry_run)

    problems = verify_only_knobs(knobs)
    for p in problems:
        print(f"[verify-only MISMATCH] {p}")
    ok &= not problems
    print(f"\n{'DRY RUN — no files written' if args.dry_run else 'Applied'} "
          f"({len(intents)} file(s)); verify-only checks "
          f"{'clean' if not problems else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

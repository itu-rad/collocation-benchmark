"""Assert the radt install is the patched checkout, not a stale site-packages copy.

Shared by the collection harnesses. This exists because itu-mac carried a
root-owned site-packages/radt alongside the editable checkout: the editable one
wins today, but if its .pth is ever lost, imports fall back to the UNPATCHED copy
and a collection is silently corrupted by the teardown race. Fail loudly instead.
"""
import shutil
import os
import sys

# Each radt listener shells out to a binary. If it is missing -- or merely not on
# PATH, which is the usual case for /opt/homebrew/bin under a non-interactive
# ssh session -- the listener fails to spawn and the run completes with NO
# counters and no error anyone reads. That is how the profiling contribution
# ended up with zero supporting data, so check it before collecting, not after.
LISTENER_BINARIES = {
    "macmon": "macmon",
    "dcgmi": "dcgmi",
    "smi": "nvidia-smi",
    "top": "top",
    "iostat": "iostat",
    "free": "free",
    "ps": "ps",
}


def check_listeners(names) -> list:
    problems = []
    for n in names:
        binary = LISTENER_BINARIES.get(n)
        if binary is None:
            problems.append(f"unknown listener {n!r}")
        elif shutil.which(binary) is None:
            problems.append(
                f"listener {n!r} needs {binary!r}, which is not on PATH "
                f"(PATH={os.environ.get('PATH', '')})")
    return problems


def main() -> int:
    try:
        import radt
        import radt.run.trace as t
    except Exception as e:  # noqa: BLE001
        print(f"radt-gate: cannot import radt: {e}", file=sys.stderr)
        return 1

    import inspect
    problems = []
    # listener names may be passed as `--listeners a,b,c`
    wanted = []
    for i, a in enumerate(sys.argv):
        if a == "--listeners" and i + 1 < len(sys.argv):
            wanted = [x for x in sys.argv[i + 1].replace(",", " ").split() if x]
    problems += check_listeners(wanted)
    if not hasattr(radt, "trace"):
        problems.append("radt has no .trace -- tracing would collect nothing")
    src = inspect.getsource(t._emit)
    if "ValueError, OSError, AssertionError" not in src:
        problems.append(
            "radt patch 0003 is NOT applied (trace._emit raises on a closed "
            "queue). The multi-threaded LLM workload will lose spans at "
            "teardown. Applied checkout: evaluation/radt-patches/"
            "0003-trace-emit-never-raises-on-closed-queue.patch")
    if problems:
        print("radt-gate FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"  loaded from: {t.__file__}", file=sys.stderr)
        return 1

    print(f"radt-gate ok: patched trace at {t.__file__}"
          + (f"; listeners available: {', '.join(wanted)}" if wanted else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

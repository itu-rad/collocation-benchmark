"""Assert the radt install is the patched checkout, not a stale site-packages copy.

Shared by the collection harnesses. This exists because itu-mac carried a
root-owned site-packages/radt alongside the editable checkout: the editable one
wins today, but if its .pth is ever lost, imports fall back to the UNPATCHED copy
and a collection is silently corrupted by the teardown race. Fail loudly instead.
"""
import sys


def main() -> int:
    try:
        import radt
        import radt.run.trace as t
    except Exception as e:  # noqa: BLE001
        print(f"radt-gate: cannot import radt: {e}", file=sys.stderr)
        return 1

    import inspect
    problems = []
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

    print(f"radt-gate ok: patched trace at {t.__file__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

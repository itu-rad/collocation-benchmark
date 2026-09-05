# Drive a CoreML model on the ANE in a loop; report throughput.
import sys, time, numpy as np, coremltools as ct
path = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
unit = sys.argv[3] if len(sys.argv) > 3 else "CPU_AND_NE"
m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, unit))
spec = m.get_spec()
feeds = {}
for inp in spec.description.input:
    t = inp.type.WhichOneof("Type")
    if t == "multiArrayType":
        shape = [int(d) if int(d) > 0 else 1 for d in inp.type.multiArrayType.shape]
    elif t == "imageType":
        shape = [1, 3, int(inp.type.imageType.height), int(inp.type.imageType.width)]
    else:
        raise SystemExit(f"unhandled input type {t}")
    feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
    print(f"input {inp.name} shape={shape}", file=sys.stderr)
m.predict(feeds)  # warm up / compile
t0 = time.monotonic(); n = 0
while time.monotonic() - t0 < secs:
    m.predict(feeds); n += 1
el = time.monotonic() - t0
print(f"ANE_LOAD {unit}: {n} preds in {el:.2f}s = {n/el:.1f}/s", file=sys.stderr)

# Build a deliberately ANE-friendly CoreML model: a stack of plain fp16 convs
# with activations far too large for on-chip SRAM, so it must stream DRAM.
import torch, coremltools as ct, sys
C = int(sys.argv[1]) if len(sys.argv) > 1 else 128     # channels
S = int(sys.argv[2]) if len(sys.argv) > 2 else 256     # spatial
L = int(sys.argv[3]) if len(sys.argv) > 3 else 24      # conv layers
out = sys.argv[4] if len(sys.argv) > 4 else "/tmp/ane_conv.mlpackage"
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(*[
            m for _ in range(L)
            for m in (torch.nn.Conv2d(C, C, 3, padding=1), torch.nn.ReLU())
        ])
    def forward(self, x):
        return self.body(x)
net = Net().eval()
ex = torch.rand(1, C, S, S)
ts = torch.jit.trace(net, ex)
m = ct.convert(
    ts,
    inputs=[ct.TensorType(name="x", shape=ex.shape, dtype=__import__("numpy").float16)],
    outputs=[ct.TensorType(name="y", dtype=__import__("numpy").float16)],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    convert_to="mlprogram",
)
m.save(out)
act_mb = C * S * S * 2 / 1e6
print(f"saved {out}  ({L} convs, activation {act_mb:.1f} MB/layer)", file=sys.stderr)

"""MLPerf 3D-UNet / KiTS19 medical-segmentation stages for Choreo.

Replaces ResNet as the accelerator workload in the overhead/decomposition study:
a source loader (light), a heavy preprocessing stage (resample/normalize/pad —
the part MLPerf excludes from timing), and gaussian-weighted sliding-window
inference over a TorchScript nnU-Net (the part MLPerf times). See BUILD.md for
model + KiTS19 data acquisition and GPU verification.
"""

from .case_loader import KiTS19CaseLoader
from .preprocess import KiTS19Preprocess
from .inference import UNet3DInference
from .score import KiTS19DiceScore

__all__ = ["KiTS19CaseLoader", "KiTS19Preprocess", "UNet3DInference",
           "KiTS19DiceScore"]

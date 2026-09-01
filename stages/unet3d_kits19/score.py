import csv
import os
import threading

import numpy as np

from stages.stage import Stage, log_phase
from utils.schemas import Query


# MLPerf's accuracy_kits.py smoothing terms, kept at their reference values so
# the score is comparable digit for digit with the reference harness's output.
SMOOTH_NR = 1e-6
SMOOTH_DR = 1e-6
N_CLASSES = 3            # background, kidney, tumor


def dice_per_class(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-class DICE for kidney and tumor, MLPerf's formula.

    Port of `get_dice_score` in the reference harness: one-hot both volumes,
    drop the background channel, then 2|P n T| / (|P| + |T|) per remaining
    channel with the reference's smoothing. The background channel is dropped
    rather than scored because it is ~99% of every volume and would put every
    case within a rounding error of 1.0.

    Both arrays are the PREPROCESSED, padded volumes -- the label goes through
    the same resample/crop/pad chain as the image in `preprocess_volume`, so
    they are aligned by construction. Scoring against the raw label instead
    would be comparing different grids.
    """
    pred = np.asarray(prediction).reshape(-1)
    targ = np.asarray(target).reshape(-1)
    if pred.shape != targ.shape:
        raise ValueError(f"prediction {pred.shape} != label {targ.shape}; the "
                         f"two did not go through the same preprocessing")
    out = np.empty(N_CLASSES - 1, dtype=np.float64)
    for c in range(1, N_CLASSES):
        p = pred == c
        t = targ == c
        inter = float(np.count_nonzero(p & t))
        out[c - 1] = ((2.0 * inter + SMOOTH_NR)
                      / (float(np.count_nonzero(p)) + float(np.count_nonzero(t))
                         + SMOOTH_DR))
    return out


class KiTS19DiceScore(Stage):
    """End stage of the ACCURACY pipeline: score each case and write one row.

    This exists because the parity claim needs a DICE number that can be
    re-derived from the repository. `TerminalCapture` cannot produce one -- it
    repr()s whatever sits in `query.data`, which for this workload is a
    multi-hundred-megabyte numpy segmentation, so its JSONL carries a truncated
    string and nothing scoreable.

    It is deliberately NOT in the performance pipeline. Scoring reads the whole
    volume twice per class and would sit inside the measured graph, which is
    the same class of error as leaving output serialisation in it. MLPerf keeps
    AccuracyOnly and PerformanceOnly apart for exactly this reason.

    Input : query.data = {case, segmentation, label, aux}
    Config: output_path -- CSV to write (created fresh at construction)
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._path = self.extra_config.get(
            "output_path", os.path.join("evaluation", "unet3d", "results",
                                        "dice.csv"))
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # Truncate up front, as TerminalCapture does: a stale file from a
        # previous run silently mixing into this one is how E2's append bug
        # produced a 4.6-hour "repetition".
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["case", "dice_kidney", "dice_tumor", "dice_mean"])

    @log_phase
    def prepare(self):
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        payload = query.data
        label = payload.get("label")
        if label is None:
            raise ValueError(
                f"{self.name}: case {payload.get('case')} arrived without a "
                f"label. The accuracy pipeline must set `keep_label: true` on "
                f"the preprocess stage; without it there is nothing to score "
                f"against and a silently empty DICE column would look like a "
                f"result.")
        d = dice_per_class(payload["segmentation"], label)
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([payload["case"], f"{d[0]:.6f}",
                                    f"{d[1]:.6f}", f"{float(d.mean()):.6f}"])
        return {idx: query for idx in self.output_queues}

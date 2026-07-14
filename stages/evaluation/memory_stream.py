import numpy as np

from stages.stage import Stage, log_phase
from utils.schemas import Query


class MemoryStream(Stage):
    """CPU memory-bandwidth co-runner (staged contention experiment, Stage C).

    Each query performs ``passes`` STREAM-triad-like sweeps
    (``c = a + scale * b``) over pre-allocated float64 arrays of
    ``size_mb`` MiB each — memory-bound at C speed via numpy, negligible
    compute. The load generator's rate is the intensity knob; the traffic per
    query is deterministic, so the model-based bandwidth estimate is exact:

        bytes/query ~= passes * 3 * size_mb * 2^20   (read a, read b, write c)

    (the AMC/dcgmi counters remain the measurement of record; this figure is
    the offered-traffic estimate for the dose-response x-axis).

    YAML config example:
        component: stages.evaluation.MemoryStream
        config:
          size_mb: 256   # per-array working set (>> LLC)
          passes: 4
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._size_mb = int(self.extra_config.get("size_mb", 256))
        self._passes = int(self.extra_config.get("passes", 4))
        self._a = None
        self._b = None
        self._c = None

    @log_phase
    def prepare(self):
        """Pre-allocate and fault in the working set (never timed per query)."""
        n = (self._size_mb * (1 << 20)) // 8  # float64 elements
        self._a = np.ones(n, dtype=np.float64)
        self._b = np.full(n, 2.0, dtype=np.float64)
        self._c = np.zeros(n, dtype=np.float64)
        super().prepare()

    def bytes_per_query(self) -> int:
        """Deterministic traffic estimate for the intensity axis."""
        return self._passes * 3 * self._size_mb * (1 << 20)

    def run(self, query: Query) -> dict[int, Query]:
        scale = 3.0
        for _ in range(self._passes):
            # STREAM triad: reads a and b, writes c — out= avoids allocation.
            np.multiply(self._b, scale, out=self._c)
            np.add(self._c, self._a, out=self._c)
        query.data = None  # co-runner produces no payload downstream
        return {idx: query for idx in self.output_queues}

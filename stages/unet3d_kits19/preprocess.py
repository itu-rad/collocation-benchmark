from stages.stage import Stage, log_phase, marker_span
from utils.schemas import Query

from . import kits19_lib as kl


class KiTS19Preprocess(Stage):
    """The resample → clip/normalize → pad chain MLPerf runs offline and excludes
    from its timed section. In Choreo it is a first-class timed stage, which is
    the point of the workload: for KiTS19 this preprocessing is heavy (trilinear
    resample of a multi-hundred-slice CT volume) and dominates end-to-end latency
    on many cases, so a benchmark that hides it (MLPerf) misreports the real cost
    of serving the model.

    Input  : query.data = {case, image_path, label_path} (from KiTS19CaseLoader)
    Output : query.data = {case, image[1,D,H,W] float32, label|None, aux,
                           n_subvolumes}
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._keep_label = self.extra_config.get("keep_label", True)

    @log_phase
    def prepare(self):
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        payload = query.data
        label_path = payload.get("label_path") if self._keep_label else None
        image, label, aux = kl.preprocess_volume(
            payload["image_path"], label_path)

        n_sub = kl.count_subvolumes(image[None, ...] if image.ndim == 4 else image)
        query.data = {
            "case": payload["case"],
            "image": image,
            "label": label,
            "aux": aux,
            "n_subvolumes": n_sub,
        }
        # The independent variable of the whole experiment, recorded as a
        # property of this query's own trace. KiTS19 volumes differ by ~18x in
        # sliding-window count, which is what makes the preprocessing share
        # vary case to case; without this the analysis has to join sizes in
        # from a side file produced by a DIFFERENT run, which is how the
        # previous version of E3 did it and is not defensible.
        #
        # Emitted unconditionally, exactly like the LLM token counts: the perf
        # config sets `disable_logs` everywhere, and marker_span deliberately
        # ignores that flag so the size survives with the CSV instrument off.
        marker_span(self, "case_size", {
            "case": payload["case"],
            "n_subvolumes": n_sub,
            # Post-resample shape -- what inference actually tiles over. The
            # raw shape is a different number and the two must not be confused.
            "image_shape": "x".join(str(d) for d in image.shape[-3:]),
        })
        return {idx: query for idx in self.output_queues}

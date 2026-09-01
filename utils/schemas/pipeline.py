from pydantic import BaseModel
from .loadgen import LoadGenModel
from .stage import StageModel


class PipelineModel(BaseModel):
    """
    Pipeline configuration parsed from yaml file.
    """

    inputs: list[int]
    outputs: list[int]
    dataset_stage_id: int
    loadgen: LoadGenModel
    stages: list[StageModel]
    name: str = "Unknown pipeline name"
    # Suppress the pipeline's PER-QUERY CSV rows (`pipeline - <split>, run,
    # start|end`). This is the pipeline-level twin of StageModel.disable_logs,
    # which only ever silenced the stages -- Pipeline had no such flag, so every
    # run wrote two rows per query through a synchronous FileHandler no matter
    # what, and that write lands inside the span-measured `exit` interval:
    # measured at 142-217 us/query on GB10 and ~81 on Apple silicon, i.e. 42-50%
    # of `exit`.
    #
    # Default False (rows ON) deliberately: E1's `uninstrumented` arm has no
    # spans by construction, so these rows are its ONLY instrument, and E3/E4/E5
    # parse them too. Experiments that measure from spans (E2) set it True.
    # The per-RUN `prepare` rows are never suppressed -- they are the session
    # marker the analyzers use to find the last session in an appended file.
    disable_logs: bool = False
    # When True, the pipeline waits for the previous query to fully exit
    # before admitting the next one. Useful for isolating per-query
    # latency / removing inter-query bandwidth contention.
    serialize_queries: bool = False

from stages.stage import Stage
from utils.schemas import Query

class DataInjector(Stage):
    """
    A stage that injects a fixed-size payload into the query data.
    Useful for measuring the overhead of passing large objects through the pipeline.
    """
    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self.payload_size = self.extra_config.get("payload_size", 0)
        # Pre-allocate data to avoid measuring allocation time during 'run'
        self.payload = bytearray(self.payload_size)

    def run(self, query: Query) -> dict[int, Query]:
        # Inject payload if not already present
        if query.data is None:
            query.data = self.payload
        
        # Standard identity forward
        return {idx: query for idx in self.output_queues}

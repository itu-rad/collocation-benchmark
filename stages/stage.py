from enum import Enum
import time
from functools import wraps


def log_phase(f):
    @wraps(f)
    def wrapper(self, *args, **kw):
        print(f"{time.perf_counter}\t{self.name}\tSTART\t{f.__name__}")
        result = f(self, *args, **kw)
        print(f"{time.perf_counter}\t{self.name}\tSTART\t{f.__name__}")
        return result

    return wrapper


class Stage:
    """This is the building block of the pipelines. A stage can perform tasks such as data
    loading, data preprocessing or model execution. The stages are separated in order to
    make the development of specific part of a pipeline and subsequent evaluation as
    easy as possible."""

    name = "Generic stage"

    def __init__(self, stage_config):
        self.name = stage_config.get("name", "Unknown stage")

    @log_phase
    def prepare(self):
        """This function contains one-time setup of the stage, such as loading/building a model."""
        pass

    @log_phase
    def run(self, data):
        """Invokes the stage of the pipeline, such as model inference or preprocessing."""
        pass

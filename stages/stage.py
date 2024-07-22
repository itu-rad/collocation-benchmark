import time
from functools import wraps
import logging


def log_phase(f):
    @wraps(f)
    def wrapper(self, *args, **kw):
        # TODO: the print statements here should be offloaded to custom logger
        # https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
        # use this thread to implement the logging. Here, loadgen should start up a new process for
        # the logger and new process for each new pipeline (have to think about how to initialize
        # this).
        if not self.disable_logs:
            logging.info("%s, %s, %s, start", self.parent_name, self.name, f.__name__)
        result = f(self, *args, **kw)
        if not self.disable_logs:
            logging.info("%s, %s, %s, end", self.parent_name, self.name, f.__name__)
        return result

    return wrapper


class Stage:
    """This is the building block of the pipelines. A stage can perform tasks such as data
    loading, data preprocessing or model execution. The stages are separated in order to
    make the development of specific part of a pipeline and subsequent evaluation as
    easy as possible."""

    name = "Generic stage"
    parent_name = "Unknown parent"
    disable_logs = False

    def __init__(self, stage_config, parent_name):
        self.name = stage_config.get("name", "Unknown stage")
        if parent_name is not None and len(parent_name) > 0:
            self.parent_name = parent_name
        self.disable_logs = stage_config.get("disable_logs", False)

    @log_phase
    def prepare(self):
        """This function contains one-time setup of the stage, such as loading/building a model."""
        pass

    @log_phase
    def run(self, data):
        """Invokes the stage of the pipeline, such as model inference or preprocessing."""
        pass

from queue import Queue
from threading import Thread
import time
from functools import wraps
import logging


def log_phase(f):
    @wraps(f)
    def wrapper(self, *args, **kw):
        if not self.disable_logs:
            logging.info("%s, %s, %s, start", self.parent_name, self.name, f.__name__)
        result = f(self, *args, **kw)
        if not self.disable_logs:
            logging.info("%s, %s, %s, end", self.parent_name, self.name, f.__name__)
        return result

    return wrapper


def log_phase_single(parent_name, name, phase, start):
    logging.info("%s, %s, %s, %s", parent_name, name, phase, start)


class Stage:
    """This is the building block of the pipelines. A stage can perform tasks such as data
    loading, data preprocessing or model execution. The stages are separated in order to
    make the development of specific part of a pipeline and subsequent evaluation as
    easy as possible."""

    def __init__(self, stage_config, parent_name):
        self.id = stage_config["id"]
        self.name = stage_config.get("name", "Unknown stage")
        if parent_name is not None and len(parent_name) > 0:
            self.parent_name = parent_name
        self.disable_logs = stage_config.get("disable_logs", False)
        self.previous_stages = []
        self.next_stages = []
        self.input_queues = {}
        self.output_queues = {}
        self.thread = None

    def get_id(self):
        return self.id

    def add_previous_stage(self, stage):
        self.previous_stages.append(stage)

    def add_next_stage(self, stage):
        self.next_stages.append(stage)

    def get_input_queues(self):
        return self.input_queues

    def set_output_queue(self, queue):
        self.output_queues = {0: queue}

    def get_next_from_queues(self):
        inputs = dict()
        for idx, input_queue in self.input_queues.items():
            inputs[idx] = input_queue.get()

        return inputs

    def push_to_output(self, output):
        for output_queue in self.output_queues.values():
            output_queue.put(output)

    def is_done(self, inputs):
        counter = 0
        for ins in inputs.values():
            if ins is None:
                counter += 1
        if counter == len(inputs):
            return True
        return False

    def join_thread(self):
        self.thread.join()

    def create_input_queues(self):
        for previous_stage in self.previous_stages:
            self.input_queues[previous_stage.get_id()] = Queue()
        # if no previous stages present (only the case for input stages),
        # create input queue
        if len(self.previous_stages) == 0:
            self.input_queues[0] = Queue()

    def prepare(self):
        """This function contains one-time setup of the stage, such as loading/building a model."""
        for next_stage in self.next_stages:
            idx = next_stage.get_id()
            queues = next_stage.get_input_queues()
            self.output_queues[idx] = queues[self.id]
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        """Invokes the stage of the pipeline, such as model inference or preprocessing."""
        pass

from queue import Queue
from threading import Thread
from functools import wraps
import logging


def log_phase(f):
    """Wraps the function execution with logging functionality.
    The wrapper automatically parses the function, pipeline and stage names."""

    @wraps(f)
    def wrapper(self, *args, **kw):
        if not self._disable_logs:
            logging.info("%s, %s, %s, start", self._parent_name, self._name, f.__name__)
        result = f(self, *args, **kw)
        if not self._disable_logs:
            logging.info("%s, %s, %s, end", self._parent_name, self._name, f.__name__)
        return result

    return wrapper


def log_phase_single(parent_name, name, phase, start):
    """Logs the stage execution status

    Args:
        parent_name (str): Name of the pipeline
        name (str): Stage name
        phase (str): Name of phase (prepare or run)
        start (str): Execution status (start or end)
    """
    logging.info("%s, %s, %s, %s", parent_name, name, phase, start)


class Stage:
    """This is the building block of the pipelines. A stage can perform tasks such as data
    loading, data preprocessing or model execution. The stages are separated in order to
    make the development of specific part of a pipeline and subsequent evaluation as
    easy as possible."""

    def __init__(self, stage_config, pipeline_config):
        self._id = stage_config["id"]
        self._name = stage_config.get("name", "Unknown stage")
        self._parent_name = pipeline_config.get("name", "Unknown pipeline")
        self._disable_logs = stage_config.get("disable_logs", False)
        self._previous_stages: list[Stage] = []
        self._next_stages: list[Stage] = []
        self._input_queues: dict[int, Queue] = {}
        self._output_queues: dict[int, Queue] = {}

    def get_id(self):
        """Getter for the ID of the stage

        Returns:
            int: Stage ID
        """
        return self._id

    def add_previous_stage(self, stage):
        """Add an incoming edge (preceeding stage) to the DAG execution graph.

        Args:
            stage (Stage): Preceeding stage
        """
        self._previous_stages.append(stage)

    def add_next_stage(self, stage):
        """Add an outgoing edge (following stage) to the DAG execution graph.

        Args:
            stage (Stage): Following stage
        """
        self._next_stages.append(stage)

    def add_stage_dict(self, stage_dict):
        """Set the stage dictionary, which is used for dynamic method invocation.

        Args:
            stage_dict (dict): Dictionary mapping stage IDs to their corresponding Stage objects.
        """
        self._stage_dict = stage_dict

    def dispatch_call(self, stage_id, method_name, *args, **kwargs):
        """Invoke a method on a stage by its ID.

        The method is invoked on the stage with the given ID. The method to invoke
        is specified by the method_name parameter. The arguments to the method are
        passed in as *args and **kwargs.

        Args:
            stage_id (int): The ID of the stage to invoke the method on.
            method_name (str): The name of the method to invoke.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the invoked method.
        """
        return getattr(self._stage_dict[stage_id], method_name)(*args, **kwargs)

    def get_input_queues(self):
        """Getter for input queues

        Returns:
            list[queue.Queue]: Input queues
        """
        return self._input_queues

    def set_output_queue(self, queue):
        """Setter for output queue. (only applicable in case of the last layer,
        where the queue does not come from the following layer but rather from the pipeline itself)

        Args:
            queue (queue.Queue): Output queue
        """
        self._output_queues = {0: queue}

    def get_next_from_queues(self):
        """Retrieve items from all input queues

        Returns:
            dict[int, any]: dictionary of inputs.
        """
        inputs = dict()
        for idx, input_queue in self._input_queues.items():
            inputs[idx] = input_queue.get()

        return inputs

    def push_to_output(self, output):
        """Push to output queues

        Args:
            output (dict[str, any]): Element to be pushed to output queues
        """
        for output_queue in self._output_queues.values():
            output_queue.put(output)

    def is_done(self, inputs):
        """Check for termination elements from all input queues.

        Args:
            inputs (dict[int, any]): Inputs retrieved from input queues

        Returns:
            bool: Boolean representing whether execution has been termianted from all queues.
        """
        counter = 0
        for ins in inputs.values():
            if ins is None:
                counter += 1
        if counter == len(inputs):
            return True
        return False

    def join_thread(self):
        """Wait for the stage thread to join."""
        self._thread.join()

    def create_input_queues(self):
        """Initializes all of the input queries, based on the number of preceeding stages."""
        for previous_stage in self._previous_stages:
            self._input_queues[previous_stage.get_id()] = Queue()
        # if no previous stages present (only the case for input stages),
        # create input queue
        if len(self._previous_stages) == 0:
            self._input_queues[0] = Queue()

    def prepare(self):
        """Iterates through the following stages, gets their input queue and sets
        it as its output queue. Additionally, starts the thread of the run phase execution.
        """
        for next_stage in self._next_stages:
            idx = next_stage.get_id()
            queues = next_stage.get_input_queues()
            self._output_queues[idx] = queues[self._id]
        self._thread = Thread(target=self.run_wrapper)
        self._thread.start()

    def run(self, inputs):
        """Pass the input onto the output (no action to be performed on the data)"""
        return next(iter(inputs.values()))

    def run_wrapper(self):
        """Continuously poll for the incoming data in the input queues,
        perform actions on them and push the results onto the output queues."""
        while True:
            inputs = self.get_next_from_queues()
            if self.is_done(inputs):
                self.push_to_output(list(inputs.values())[0])
                break

            if not self._disable_logs:
                log_phase_single(self._parent_name, self._name, "run", "start")

            new_data = self.run(inputs)

            self.push_to_output(new_data)
            if not self._disable_logs:
                log_phase_single(self._parent_name, self._name, "run", "end")

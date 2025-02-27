from __future__ import annotations  # postponed evaluation of annotations
from queue import Queue
from threading import Thread
from functools import wraps
import logging
from typing import Any
import json

from utils.queues.polling.polling_policy import PollingPolicy
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


def log_phase(f):
    """
    Decorator to log the start and end of a function's execution.

    This decorator wraps a function to log its execution phases (start and end) using the `logging` module.
    It logs the parent name, stage name, and function name at the start and end of the function execution.

    Args:
        f (function): The function to be wrapped with logging functionality.

    Returns:
        function: The wrapped function with logging.

    Usage:
        @log_phase
        def some_function(self, *args, **kwargs):
            # Function implementation
    """

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

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        self.id = stage_config.id
        self.name = stage_config.name
        self.parent_name = pipeline_config.name
        self.disable_logs = stage_config.disable_logs
        self._stage_config = stage_config
        self._polling_policy = stage_config.polling_policy
        self._output_stage_ids = stage_config.outputs
        self.extra_config = stage_config.config
        self._stage_dict: dict[int, Stage] = {}
        self._input_queues: dict[int, Queue] = {}
        self.output_queues: dict[int, Queue] = {}

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        The string includes the object's ID, name, and a JSON dump of its stage configuration.
        Additionally, it lists all output stage IDs in the format "current_id -> output_stage_id".
        Returns:
            str: A formatted string representing the object.
        """
        encoded_config = (
            json.dumps(self._stage_config.__dict__, indent=4)
            .replace('"', "'")
            .replace("    ", "&emsp;")
        )
        s = f'{self.id}["`{self.name}\n{encoded_config}`"]\nstyle {self.id} text-align:left\n'
        for output_stage_id in self._output_stage_ids:
            s += f"{self.id} --> {output_stage_id}\n"
        return s

    def set_stage_dict(self, stage_dict: dict[int, Stage]) -> None:
        """Set the stage dictionary, which is used for dynamic method invocation.

        Args:
            stage_dict (dict[int, Stage]): Dictionary mapping stage IDs (int) to their corresponding Stage objects.
        """
        self._stage_dict = stage_dict

    def set_output_queues(self):
        """Set the output queues of the stage by calling get_input_queue on the outgoing stages.

        Note: This method is automatically called by the pipeline after setting the stage_dict.
        """
        for out_stage_id in self._output_stage_ids:
            self.output_queues[out_stage_id] = self.dispatch_call(
                out_stage_id, "get_input_queue", self.id
            )

    def set_output_queue(self, queue: Queue):
        """Set the output queue of the stage manually. Only used for output stages.

        Args:
            queue (Queue):
        """
        self.output_queues = {-1: queue}

    def get_input_queue(self, idx: int) -> Queue:
        """Get the input queue for the given stage ID.

        If the queue does not exist yet, it is created.

        Args:
            id (int): The ID of the stage to get the input queue for.

        Returns:
            queue.Queue: The input queue for the given stage ID.
        """
        if idx not in self._input_queues:
            self._input_queues[idx] = Queue()
        return self._input_queues[idx]

    def dispatch_call(
        self, stage_id: int, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke a method on a stage by its ID.

        The method is invoked on the stage with the given ID. The method to invoke
        is specified by the method_name parameter. The arguments to the method are
        passed in as *args and **kwargs.

        Args:
            stage_id (int): The ID of the stage to invoke the method on.
            method_name (str): The name of the method to invoke.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the invoked method.
        """
        return getattr(self._stage_dict[stage_id], method_name)(*args, **kwargs)

    def join_thread(self) -> None:
        """Wait for the stage thread to join."""
        self._thread.join()

    def prepare(self) -> None:
        """
        Prepare the stage for execution.
        """
        if (
            len(self._input_queues) > 1
            and self._polling_policy == "stages.queues.polling.SingleQueuePolicy"
        ):
            raise ValueError("SingleQueuePolicy only works with one input queue")
        self._polling_policy_obj: PollingPolicy = get_component(self._polling_policy)(
            self._input_queues
        )
        self._thread = Thread(target=self.run_wrapper)
        self._thread.start()

    def _get_input_from_queues(self) -> Query | None:
        """Retrieve items from all input queues

        Returns:
            Query | None: The first query from all input queues or None if terminating character is received.
        """
        return self._polling_policy_obj.get_input_from_queues()

    def _push_to_all_outputs(self, output: any) -> None:
        """Push the same data to all output queues

        Args:
            output (any): Element to be pushed to all output queues
        """
        for output_queue in self.output_queues.values():
            output_queue.put(output)

    def _push_to_outputs(self, outputs: dict[int, Query]) -> None:
        """Push each output to its corresponding output queue.

        Args:
            outputs (dict[int, any]): dictionary of outputs, where keys are the stage IDs
                and values are the outputs to be pushed to the corresponding output queue.
        """
        for idx, output in outputs.items():
            self.output_queues[idx].put(output)

    def run(self, query: Query) -> dict[int, Query]:
        """
        Run function of the Identity stage.

        This function simply returns the first value it receives from any of the input queues.
        It does not perform any operation on the inputs.

        Args:
            query (Query): Inputs retrieved from input queues

        Returns:
            dict[int, Query]: Dictionary of queries, where keys are the stage IDs
                and values are the queries to be pushed to the corresponding output queue.
        """
        return {idx: query for idx in self.output_queues}

    def pre_run(self) -> None:
        """
        This function is run at the very beginning of the run wrapper function.
        This can be used to perform any setup operation that requires to happen in the same thread as run (prepare function is run in the main pipeline thread).
        """
        pass

    def post_run(self) -> None:
        """
        This function is run at the very end of the run wrapper function.
        This can be used to perform any cleanup operations such as closing files or connections.
        """
        pass

    def run_wrapper(self) -> None:
        """Continuously poll for the incoming data in the input queues,
        perform actions on them and push the results onto the output queues."""
        self.pre_run()
        while True:
            query = self._get_input_from_queues()
            if not query:
                # received terminating element (None)
                self._push_to_all_outputs(None)
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")

            outputs = self.run(query)

            self._push_to_outputs(outputs)
            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
        self.post_run()

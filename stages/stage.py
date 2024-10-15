from __future__ import annotations  # postponed evaluation of annotations
from queue import Queue
from threading import Thread
from functools import wraps
import logging
from typing import Any

from utils.schemas import StageModel, PipelineModel


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

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        self.id = stage_config.id
        self._name = stage_config.name
        self._parent_name = pipeline_config.name
        self._disable_logs = stage_config.disable_logs
        self._output_stage_ids = stage_config.outputs
        self.extra_config = stage_config.config
        self._stage_dict: dict[int, Stage] = {}
        self._input_queues: dict[int, Queue] = {}
        self._output_queues: dict[int, Queue] = {}

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
        self._output_queues: dict[int, Queue] = {}
        for out_stage_id in self._output_stage_ids:
            self._output_queues[out_stage_id] = self._dispatch_call(
                out_stage_id, "get_input_queue", self.id
            )

    def set_output_queue(self, queue: Queue):
        """Set the output queue of the stage manually. Only used for output stages.

        Args:
            queue (Queue):
        """
        self._output_queues = {-1: queue}

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

    def _dispatch_call(
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
        self._thread = Thread(target=self.run_wrapper)
        self._thread.start()

    def _get_input_from_queues(self) -> dict[int, Any]:
        """Retrieve items from all input queues

        Returns:
            dict[int, Any]: dictionary of inputs.
        """
        inputs: dict[int, Any] = {}
        for idx, input_queue in self._input_queues.items():
            inputs[idx] = input_queue.get()

        return inputs

    def _push_to_all_outputs(self, output: any) -> None:
        """Push the same data to all output queues

        Args:
            output (any): Element to be pushed to all output queues
        """
        for output_queue in self._output_queues.values():
            output_queue.put(output)

    def _push_to_outputs(self, outputs: dict[int, any]) -> None:
        """Push each output to its corresponding output queue.

        Args:
            outputs (dict[int, any]): dictionary of outputs, where keys are the stage IDs
                and values are the outputs to be pushed to the corresponding output queue.
        """
        for idx, output in outputs.items():
            self._output_queues[idx].put(output)

    def _is_done(self, inputs) -> bool:
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

    def run(self, inputs) -> dict[int, any]:
        """
        Run function of the Identity stage.

        This function simply returns the first value it receives from any of the input queues.
        It does not perform any operation on the inputs.

        Args:
            inputs (dict[int, any]): Inputs retrieved from input queues

        Returns:
            dict[int, any]: Dictionary of outputs, where keys are the stage IDs
                and values are the outputs to be pushed to the corresponding output queue.
        """
        val = next(iter(inputs.values()))
        return {idx: val for idx in self._output_queues}

    def run_wrapper(self):
        """Continuously poll for the incoming data in the input queues,
        perform actions on them and push the results onto the output queues."""
        while True:
            inputs = self._get_input_from_queues()
            if self._is_done(inputs):
                self._push_to_all_outputs(None)
                break

            if not self._disable_logs:
                log_phase_single(self._parent_name, self._name, "run", "start")

            new_data = self.run(inputs)

            self._push_to_all_outputs(new_data)
            if not self._disable_logs:
                log_phase_single(self._parent_name, self._name, "run", "end")

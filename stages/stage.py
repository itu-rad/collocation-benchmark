from __future__ import annotations  # postponed evaluation of annotations
from queue import Queue
from threading import Thread
from functools import wraps
import logging
import threading
import uuid
from typing import Any
import json

import mlflow
from utils.trace_span import trace_span, SPANS_ENABLED

from utils.queues.polling.polling_policy import PollingPolicy
from utils.queues.peekable_queue import PeekableQueue
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
            logging.getLogger("benchmark").info(
                "%s, %s, %s, start", self.parent_name, self.name, f.__name__
            )
        result = f(self, *args, **kw)
        if not self.disable_logs:
            logging.getLogger("benchmark").info(
                "%s, %s, %s, end", self.parent_name, self.name, f.__name__
            )
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
    logging.getLogger("benchmark").info(
        "%s, %s, %s, %s", parent_name, name, phase, start
    )


def marker_span(stage: "Stage", name: str, attributes: dict | None = None) -> None:
    """Emit a zero-duration span marking an INSTANT in `stage`'s current query.

    Span attributes are fixed at span start, so a span's START is the only
    instant it can report on the monotonic clock (utils/trace_span.PERF_ATTR).
    An instant that is not bracketed by a span therefore cannot be recovered
    from the span artifacts at all -- which is why first_token, and the token
    counts that normalise it, were previously CSV-only and invisible to any
    span-based analysis.

    NOT gated by `disable_logs`: that flag turns off the CSV instrument, and
    the whole point of these markers is that the timing survives without it.
    Cheap when tracing is off (SPANS_ENABLED short-circuits before the dict is
    built) and ~2 events when it is on -- once per query, not per stage.

    Names must be unique per query, hence `first_token_start` / `first_token_end`
    rather than one name with a start/end attribute: SpanTrace.by_query() treats
    a repeated (name, query) as a shape error rather than silently keeping one.
    """
    if not SPANS_ENABLED:
        return
    attrs = {"stage": stage.name, "query_id": stage.current_query_id}
    if attributes:
        attrs.update(attributes)
    with trace_span(name=f"{stage.name}.{name}", attributes=attrs):
        pass


def log_first_token(stage: "Stage", event: str) -> None:
    """Log one edge of a generator stage's "first_token" sub-phase.

    Goes through the exact same benchmark-logger path as log_phase_single, so
    (with utils/logger.py PERF_FORMAT) the trace CSV row is

        wall, <pipeline>, <stage>, first_token, {start|end}, perf

    which evaluation/contention/staged_lib.py keys as the
    "<stage>::first_token" sub-phase and pairs start/end; the Step D analyzer
    then derives ttft = pair_end_perf - stage_run_start_perf. Emit "start"
    immediately before the generate call and "end" at the first produced
    token, exactly once per run() invocation, so sub-phase pairs stay 1:1
    with the stage's run start/end pairs (staged_lib aligns them by index).
    Gated by disable_logs identically to the per-query run rows.
    """
    marker_span(stage, f"first_token_{event}")
    if not stage.disable_logs:
        log_phase_single(stage.parent_name, stage.name, "first_token", event)


def log_generated_tokens(stage: "Stage", n_tokens: int) -> None:
    """Log the real per-query generated-token count as a companion trace row:

        wall, <pipeline>, <stage>, n_generated_tokens, <int>, perf

    Field 5 is the count, not start/end, so existing parsers
    (staged_lib.parse_trace_files filters on start|end) skip the row —
    backward compatible. Analyzers should prefer this over the config's
    gen_kwargs.max_tokens when present (early EOS makes max_tokens an
    overestimate). Gated by disable_logs like all per-query stage rows.
    """
    marker_span(stage, "generated_tokens", {"n_generated_tokens": n_tokens})
    if not stage.disable_logs:
        logging.getLogger("benchmark").info(
            "%s, %s, n_generated_tokens, %d", stage.parent_name, stage.name,
            n_tokens,
        )


def log_prompt_tokens(stage: "Stage", n_tokens: int) -> None:
    """Log the per-query PROMPT (prefill) token count as a companion trace row:

        wall, <pipeline>, <stage>, n_prompt_tokens, <int>, perf

    Field 5 is the count, not start/end, so parsers that filter on start|end
    skip the row — backward compatible, exactly like log_generated_tokens.

    Without this, prefill is an unnormalised duration: it cannot be converted to
    a rate, roofline-checked, or compared across devices whose prompts differ.
    It is also the direct measurement of "decomposition re-reads the retrieved
    context in every sub-call", which was previously only inferred. Gated by
    disable_logs like all per-query stage rows.
    """
    marker_span(stage, "prompt_tokens", {"n_prompt_tokens": n_tokens})
    if not stage.disable_logs:
        logging.getLogger("benchmark").info(
            "%s, %s, n_prompt_tokens, %d", stage.parent_name, stage.name,
            n_tokens,
        )


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
        self._input_queues: dict[int, PeekableQueue] = {}
        # One condition per consuming stage, shared by all of its input queues
        # (which notify it on put) and its polling policy (which waits on it),
        # so fan-in polling blocks instead of busy-waiting.
        self._input_cond = threading.Condition()
        self.output_queues: dict[int, Queue] = {}
        self._logger = logging.getLogger("benchmark")
        # The query this stage is currently processing, per THREAD: most stages
        # are one thread, but a stage that dispatches queries concurrently
        # (stages.llm_server.Inference) runs _process_query on several workers
        # at once, and a plain attribute would let one worker's marker spans be
        # attributed to another's query. Read via `current_query_id`.
        self._current = threading.local()

    @property
    def current_query_id(self):
        """Query id this thread is processing, or None outside a query."""
        return getattr(self._current, "query_id", None)

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

        If the queue does not exist yet, it is created. Honors
        `max_input_queue_depth` from the stage config (None / 0 == unbounded).

        Args:
            id (int): The ID of the stage to get the input queue for.

        Returns:
            queue.Queue: The input queue for the given stage ID.
        """
        if idx not in self._input_queues:
            maxsize = self._stage_config.max_input_queue_depth or 0
            self._input_queues[idx] = PeekableQueue(
                maxsize=maxsize, notify_condition=self._input_cond
            )
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

    def get_dataset_splits(self) -> dict[str, int]:
        """
        Get the number of batches in each dataset split.
        Default implementation returns a single batch for 'train' split.
        """
        return {'train': 1}

    def prepare(self) -> None:
        """
        Prepare the stage for execution.
        """
        # Match on the class name, not a hard-coded dotted path: the schema
        # default (utils/schemas/stage.py) is "utils.queues.polling.
        # SingleQueuePolicy" and every config uses that path, while this guard
        # previously compared against "stages.queues.polling..." — so the
        # strings never matched and the guard could NEVER fire. A misconfigured
        # multi-input stage then silently starved all but one upstream instead
        # of raising.
        if (
            len(self._input_queues) > 1
            and self._polling_policy.rsplit(".", 1)[-1] == "SingleQueuePolicy"
        ):
            raise ValueError(
                f"SingleQueuePolicy only works with one input queue "
                f"(stage {self.name!r} has {len(self._input_queues)}); "
                f"use a multi-queue polling policy"
            )
        self._polling_policy_obj: PollingPolicy = get_component(self._polling_policy)(
            self._input_queues, self._input_cond
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
        # query_id and stage are carried so this span can be CORRELATED, not just
        # counted: the hand-off cost between stage k and k+1 is
        #   start(stage k+1 ".run") - start(stage k ".push_to_outputs")
        # for one query, which is unrecoverable if the push span cannot say
        # which query it pushed. All outputs are the same query object, so any
        # of them names it.
        qid = next((getattr(o, "query_id", None) for o in outputs.values()
                    if o is not None), None)
        with trace_span(
            name=f"{self.name}.push_to_outputs",
            attributes={"thread_id": threading.get_ident(),
                        "stage": self.name, "query_id": qid},
        ):
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

    def _process_query(self, query: Query) -> None:
        """Run a single query through self.run, wrapped in its tracing span,
        and push the results to the output queues.

        Factored out of run_wrapper so stages that dispatch queries
        concurrently (e.g. stages.llm_server.Inference) can reuse the exact
        same span/flow/logging contract from a worker thread.
        """
        if not self.disable_logs:
            log_phase_single(self.parent_name, self.name, "run", "start")

        # Published before run() so marker spans emitted from deep inside a
        # stage's inference code (first_token, token counts) can name the query
        # they belong to without every call site having to thread it through.
        self._current.query_id = query.query_id
        in_flow_id = str(query.out_flow_id) if query.out_flow_id else None
        out_flow_id = uuid.uuid4()
        with trace_span(
            name=f"{self.name}.run",
            attributes={
                "in_flow_id": in_flow_id,
                "out_flow_id": str(out_flow_id),
                "thread_id": threading.get_ident(),
                "stage": self.name,
                "epoch": query.epoch,
                "split": query.split,
                "batch": query.batch,
                "query_id": query.query_id,
            },
        ):
            query.out_flow_id = out_flow_id
            outputs = self.run(query)

        self._push_to_outputs(outputs)
        self._current.query_id = None
        if not self.disable_logs:
            log_phase_single(self.parent_name, self.name, "run", "end")

    def run_wrapper(self) -> None:
        """Continuously poll for the incoming data in the input queues,
        perform actions on them and push the results onto the output queues."""
        self.pre_run()
        while True:
            # No query_id here: attributes are fixed at span START, and at that
            # instant this stage has not received a query yet -- that is the
            # whole point of the span. Its START is when the stage began
            # waiting; the query it eventually got is named by the ".run" span
            # that follows on the same thread.
            with trace_span(
                name=f"{self.name}.get_input",
                attributes={"thread_id": threading.get_ident(),
                            "stage": self.name},
            ):
                query = self._get_input_from_queues()
            if not query:
                # received terminating element (None)
                self._push_to_all_outputs(None)
                break

            try:
                self._process_query(query)
            except Exception:
                # A dead stage thread must not silently hang the pipeline until
                # the run timeout (it burned a 1h pilot slot on 2026-07-14):
                # log loudly, propagate the terminator so downstream drains and
                # the run FAILS FAST, then re-raise for the traceback.
                import traceback
                print(f"[stage FATAL] {self.name}: exception in run(); "
                      f"terminating pipeline", flush=True)
                traceback.print_exc()
                self._push_to_all_outputs(None)
                raise
        self.post_run()

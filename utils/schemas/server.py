from pydantic import BaseModel
from typing import Literal


class ServerModel(BaseModel):
    """Configuration for a local inference engine server (vLLM / Ollama).

    Lives under an inference stage's ``config.server`` block. Only the *owner*
    inference stage (the one without ``depends_on_id``) carries it; dependent
    stages reuse the already-running server via ``get_server_handle()``.

    The server is launched as a subprocess inside the benchmark's process tree
    so its GPU / host memory is captured by the system-wide RadT listeners.
    """

    # Which engine to launch. vLLM is NVIDIA-only; Ollama runs cross-platform.
    engine: Literal["vllm", "ollama"]
    # Served model id. For vLLM this is the HF repo / local path; for Ollama
    # this is the model tag (e.g. "llama3.1:8b").
    model: str
    host: str = "127.0.0.1"
    # 0 => auto-pick a free port (avoids collisions across colocated pipelines).
    port: int = 8000
    # If set, do NOT launch anything — attach to an already-running server at
    # this base URL (e.g. a system Ollama daemon).
    api_base: str | None = None
    # Max time to wait for the server's health endpoint after launch.
    startup_timeout_s: float = 600.0
    # Raw extra CLI args passed through verbatim to the engine command.
    launch_args: list[str] = []
    # vLLM convenience knobs (also expressible via launch_args).
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    # Extra environment variables for the subprocess.
    env: dict[str, str] = {}

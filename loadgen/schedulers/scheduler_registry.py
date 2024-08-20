from .scheduler import LoadScheduler
from .offline_scheduler import OfflineLoadScheduler
from .poisson_scheduler import PoissonLoadScheduler


SCHEDULER_REGISTRY: dict[str, LoadScheduler] = {
    "poisson": PoissonLoadScheduler,
    "offline": OfflineLoadScheduler,
}

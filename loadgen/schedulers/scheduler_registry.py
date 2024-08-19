from .offline_scheduler import OfflineLoadScheduler
from .poisson_scheduler import PoissonLoadScheduler


SCHEDULER_REGISTRY = {"poisson": PoissonLoadScheduler, "offline": OfflineLoadScheduler}

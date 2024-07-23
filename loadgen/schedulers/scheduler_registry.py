from loadgen.schedulers.offline_scheduler import OfflineLoadScheduler
from loadgen.schedulers.poisson_scheduler import PoissionLoadScheduler


SCHEDULER_REGISTRY = {"poisson": PoissionLoadScheduler, "offline": OfflineLoadScheduler}

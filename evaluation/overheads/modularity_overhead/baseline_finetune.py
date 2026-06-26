import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import time
import logging
from logging.handlers import QueueHandler, QueueListener
import queue
import radt

# Global listener to ensure we can stop it at the end
log_listener = None

# Trace column layout must match the framework (utils/logger.py): wall-clock
# column 0 (RadT alignment) + monotonic perf_counter_ns trailing column
# (per-step latency). Inlined here so this standalone baseline stays importable
# without the repo's package on sys.path.
PERF_FORMAT = "%(created)f, %(message)s, %(perf)d"


def _install_perf_clock():
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = _orig_factory(*args, **kwargs)
        record.perf = time.perf_counter_ns()
        return record

    logging.setLogRecordFactory(_factory)


def setup_logging(label="baseline_finetune"):
    global log_listener
    # Use absolute path for log directory (this experiment's own results dir)
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{label}.csv")

    _install_perf_clock()
    formatter = logging.Formatter(PERF_FORMAT)

    # These handlers will be used by the Listener in a background thread
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Create a queue for the QueueHandler to push to
    log_queue = queue.Queue(-1)
    q_handler = QueueHandler(log_queue)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(q_handler)

    # Start the listener in a background thread to handle the actual I/O
    log_listener = QueueListener(log_queue, file_handler, stream_handler)
    log_listener.start()

    return logger


def parse_args():
    ap = argparse.ArgumentParser(
        description="Hand-written PyTorch baseline for the modularity-overhead "
        "experiment: EfficientNetV2-S Imagenette transfer-learning fine-tune, "
        "monolithic (no Choreo framework)."
    )
    ap.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"],
                    default="auto", help="compute device (auto: cuda>mps>cpu)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader worker processes (match the Choreo config)")
    ap.add_argument("--max-batches", type=int, default=1000,
                    help="number of training steps to run")
    ap.add_argument("--label", default="baseline_finetune",
                    help="output CSV basename in results/ (so R runs don't collide)")
    ap.add_argument("--no-radt", action="store_true",
                    help="skip the RADTBenchmark telemetry wrapper -> a true "
                         "zero-framework control (used by run_modularity.py)")
    ap.add_argument("--run", type=int, default=None,
                    help="run index, for provenance only")
    return ap.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training(args, logger):
    batch_size = args.batch_size
    num_classes = 10
    lr = 0.001
    max_batches = args.max_batches

    device = resolve_device(args.device)

    logger.info(f"baseline_finetune, system, setup, start, device={device}")

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Same location the Choreo TorchVisionDataLoader uses (cwd/tmp/...), so both
    # implementations share one downloaded copy; auto-download if missing.
    dataset_path = os.path.join(os.getcwd(), "tmp", "torchvision_dataset", "Imagenette")
    os.makedirs(dataset_path, exist_ok=True)
    already = os.path.isdir(os.path.join(dataset_path, "imagenette2"))

    try:
        train_dataset = datasets.Imagenette(
            root=dataset_path,
            split="train",
            size="full",
            download=not already,
            transform=preprocess,
        )
    except Exception as e:
        logger.error(f"baseline_finetune, system, error, {str(e)}")
        return

    # drop_last=True to match TorchVisionDataLoader exactly
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = models.efficientnet_v2_s(weights=None)

    # Freezing logic to match Choreo's Transfer Learning
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"baseline_finetune, model, info, trainable_params={trainable_params}, total_params={total_params}"
    )
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")

    logger.info("baseline_finetune, system, setup, end")

    batch_count = 0

    logger.info(f"baseline_finetune, training_loop, run, start")

    try:
        for inputs, labels in train_loader:
            if batch_count >= max_batches:
                break

            logger.info("baseline_finetune, training_step, run, start")

            # Match Choreo's per-batch state management
            model.train()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device)
                # Explicitly cast to LongTensor to match Choreo's classification.py
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Synchronize GPU to ensure timing captures actual hardware execution
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            logger.info("baseline_finetune, training_step, run, end")

            batch_count += 1

    except KeyboardInterrupt:
        logger.info("baseline_finetune, training_loop, run, interrupted")

    logger.info(f"baseline_finetune, training_loop, run, end")


def main():
    args = parse_args()
    logger = setup_logging(args.label)

    try:
        if args.no_radt:
            # True zero-framework control: no RadT telemetry/listeners.
            run_training(args, logger)
        else:
            # Wrap execution in RADTBenchmark for hardware metrics
            with radt.run.RADTBenchmark() as run:
                run_training(args, logger)
    finally:
        if log_listener:
            log_listener.stop()


if __name__ == "__main__":
    main()
